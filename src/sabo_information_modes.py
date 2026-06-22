import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Optional

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel
from gpytorch.constraints import Interval


# ============================================================
# 0) Configuration
# ============================================================
# MODE = "full_known"   # "no_sensor" | "full_known" | "partial_known"
# MODE = "partial_known"   # "no_sensor" | "full_known" | "partial_known"
MODE = "no_sensor"   # "no_sensor" | "full_known" | "partial_known"
RUN_DIR = "../data/optimization_results"
CSV_NAME = {
    "no_sensor": "qnehvi_no_repeat.csv",
    "partial_known": "sabo_feature_completion.csv",
    "full_known": "sabo_full_library.csv",
}[MODE]
STATE_NAME = f"{MODE}_state.json"

DATA_ROOT = "../data/Data"

DEVICE = "cpu"
DTYPE = torch.double

USE_MEDIAN_OF_3 = False

REF_POINT = torch.tensor([0.1, 0.01], dtype=DTYPE, device=DEVICE)

FLOWMAP_RBF_LENGTHSCALE = 0.25
FLOWMAP_RIDGE_ALPHA = 1e-4

PENDING_IS_ZERO_ZERO = True

FLOW_KEY_ROUND = 6

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1) Discrete search space
# ============================================================
SPACE = {
    "C_aq":    [0.1, 0.2, 0.3, 0.4, 0.5],
    "C_org":   [0.01, 0.02, 0.03, 0.04, 0.05],
    "Phi":     [0.3, 0.4, 0.5, 0.6, 0.7],
    "Q_total": [0.2, 0.4, 0.6, 0.8, 1.0],
    "L":       [1.0, 2.0, 4.0],
}

PARAM_ORDER = ["C_aq", "C_org", "Phi", "Q_total", "L"]


def make_all_candidates():
    rows = []
    for C_aq in SPACE["C_aq"]:
        for C_org in SPACE["C_org"]:
            for Phi in SPACE["Phi"]:
                for Q_total in SPACE["Q_total"]:
                    for L in SPACE["L"]:
                        rows.append({"C_aq": C_aq, "C_org": C_org, "Phi": Phi, "Q_total": Q_total, "L": L})
    return pd.DataFrame(rows)


CANDIDATES_DF = make_all_candidates()

# ============================================================
# 2) Cold-flow feature columns
# ============================================================
FEATURE_COLS_16 = [
    "Mean", "Std", "Skewness", "Kurtosis", "Dom_Freq", "Spec_Energy", "Wave_Eng_D1",
    "Entropy", "Hurst", "Cross_Rate", "Peak_Dist", "Num_Peaks", "Num_Valleys",
    "Avg_Peak_H", "Avg_Valley_H", "Cross_Count"
]


def make_flow_key(phi, qtot, L):
    return (round(float(phi), FLOW_KEY_ROUND),
            round(float(qtot), FLOW_KEY_ROUND),
            round(float(L), FLOW_KEY_ROUND))


def _aggregate_reps(group: pd.DataFrame) -> np.ndarray:
    X = group[FEATURE_COLS_16].to_numpy(dtype=float)
    if USE_MEDIAN_OF_3:
        return np.median(X, axis=0)
    return X.mean(axis=0)


def load_flow_feature_lookup(data_root: str) -> dict:
    lookup = {}
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[FlowData] DATA_ROOT not found: {data_root}")

    for sub in os.listdir(data_root):
        folder = os.path.join(data_root, sub)
        if not os.path.isdir(folder):
            continue

        if "1m flow regime mapping" in sub:
            L = 1.0
        elif "2m flow regime mapping" in sub:
            L = 2.0
        elif "4m flow regime mapping" in sub:
            L = 4.0
        else:
            continue

        feat_path = os.path.join(folder, "features_table.csv")
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(f"[FlowData] missing {feat_path}")

        df = pd.read_csv(feat_path)
        required = {"phi_aq", "Q_total"}.union(set(FEATURE_COLS_16))
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"[FlowData] Missing columns in {feat_path}: {missing}")

        for (phi, qtot), g in df.groupby(["phi_aq", "Q_total"]):
            vec = _aggregate_reps(g)
            lookup[make_flow_key(phi, qtot, L)] = vec

    if not lookup:
        raise RuntimeError("[FlowData] No features loaded. Check Data folder naming and features_table.csv exists.")
    return lookup


# ============================================================
# 3) Partial-known flow map: RBF KRR for standardized features
# ============================================================
def normalize_v(v: np.ndarray) -> np.ndarray:
    phi = (v[..., 0] - min(SPACE["Phi"])) / (max(SPACE["Phi"]) - min(SPACE["Phi"]))
    q   = (v[..., 1] - min(SPACE["Q_total"])) / (max(SPACE["Q_total"]) - min(SPACE["Q_total"]))
    L   = (v[..., 2] - min(SPACE["L"])) / (max(SPACE["L"]) - min(SPACE["L"]))
    return np.stack([phi, q, L], axis=-1)


def rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float) -> np.ndarray:
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True).T
    sq = A2 + B2 - 2.0 * (A @ B.T)
    return np.exp(-0.5 * sq / (lengthscale**2 + 1e-12))


class FlowMapKRR:
    def __init__(self, lengthscale=0.25, alpha=1e-4):
        self.ls = float(lengthscale)
        self.alpha = float(alpha)
        self.V_train = None
        self.A = None

    def fit(self, V: np.ndarray, Zs: np.ndarray):
        Vn = normalize_v(V)
        K = rbf_kernel(Vn, Vn, self.ls)
        n = K.shape[0]
        K_reg = K + self.alpha * np.eye(n)
        self.A = np.linalg.solve(K_reg, Zs)
        self.V_train = Vn

    def predict(self, V: np.ndarray) -> np.ndarray:
        Vn = normalize_v(V)
        K_star = rbf_kernel(Vn, self.V_train, self.ls)
        return K_star @ self.A


# ============================================================
# 4) SABO Kernel：k_chem * ((1-λ)k_hyd + λ k_sensor)
# ============================================================
class SABOKernel(gpytorch.kernels.Kernel):
    def __init__(self, sensor_dim: int, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.k_chem = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        self.k_hyd  = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        sensor_dims = list(range(5, 5 + sensor_dim))
        self.k_sen  = ScaleKernel(RBFKernel(ard_num_dims=sensor_dim, active_dims=sensor_dims))

        self.register_parameter(name="raw_lambda", parameter=torch.nn.Parameter(torch.tensor(0.0)))
        self.register_constraint("raw_lambda", Interval(-10.0, 10.0))

    @property
    def lam(self):
        return torch.sigmoid(self.raw_lambda)

    def forward(self, x1, x2, **params):
        kchem = self.k_chem(x1, x2, **params)
        khyd  = self.k_hyd(x1, x2, **params)
        ksen  = self.k_sen(x1, x2, **params)
        mix = (1.0 - self.lam) * khyd + self.lam * ksen
        return kchem * mix


def build_model(train_X: torch.Tensor, train_Y: torch.Tensor, mode: str, sensor_dim: int):
    gp1 = SingleTaskGP(train_X, train_Y[:, [0]], outcome_transform=Standardize(m=1))
    gp2 = SingleTaskGP(train_X, train_Y[:, [1]], outcome_transform=Standardize(m=1))

    if mode == "no_sensor":
        kchem = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        khyd  = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        gp1.covar_module = ProductKernel(kchem, khyd)

        kchem2 = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        khyd2  = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        gp2.covar_module = ProductKernel(kchem2, khyd2)
    else:
        gp1.covar_module = SABOKernel(sensor_dim=sensor_dim)
        gp2.covar_module = SABOKernel(sensor_dim=sensor_dim)

    return ModelListGP(gp1, gp2)


def fit_model(model: ModelListGP):
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ============================================================
# 5) qNEHVI compatibility layer
# ============================================================
def build_qnehvi_acq(model, ref_point, X_baseline, sampler, partitioning):
    """Construct qNEHVI across the BoTorch versions used during development."""
    # Newer versions support partitioning and cache_root.
    try:
        return qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=True,
            cache_root=True,
            partitioning=partitioning,
        )
    except TypeError:
        pass

    # Some versions support cache_root but not partitioning.
    try:
        return qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=True,
            cache_root=True,
        )
    except TypeError:
        pass

    # Older versions support neither optional argument.
    return qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,
    )


# ============================================================
# 6) CSV/state helpers
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def csv_path():
    return os.path.join(RUN_DIR, CSV_NAME)


def state_path():
    return os.path.join(RUN_DIR, STATE_NAME)


def load_state():
    sp = state_path()
    if not os.path.exists(sp):
        return {"tried_keys": [], "trial_index": 0}
    with open(sp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def key_from_row(row) -> str:
    return f"Caq{row['C_aq']}_Corg{row['C_org']}_Phi{row['Phi']}_Q{row['Q_total']}_L{row['L']}"


def is_pending(row) -> bool:
    if PENDING_IS_ZERO_ZERO:
        return (float(row["yield"]) == 0.0) and (float(row["sty"]) == 0.0)
    return pd.isna(row["yield"]) or pd.isna(row["sty"])


# ============================================================
# 7) Training and candidate tensors
# ============================================================
def build_train_tensors(df_done: pd.DataFrame, flow_lookup: dict, mode: str, flowmap: Optional[FlowMapKRR]):
    X_base = df_done[PARAM_ORDER].to_numpy(dtype=float)
    Y = df_done[["yield", "sty"]].to_numpy(dtype=float)

    if mode == "no_sensor":
        return (
            torch.tensor(X_base, dtype=DTYPE, device=DEVICE),
            torch.tensor(Y, dtype=DTYPE, device=DEVICE),
            0,
            None,
        )

    Z = []
    for _, r in df_done.iterrows():
        key = make_flow_key(r["Phi"], r["Q_total"], r["L"])
        if key not in flow_lookup:
            raise KeyError(f"[FlowLookup] Missing key {key} in flow tables.")
        Z.append(flow_lookup[key])
    Z = np.stack(Z, axis=0)

    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True) + 1e-12
    Zs = (Z - mu) / sd

    X_aug = np.concatenate([X_base, Zs], axis=1)
    sensor_dim = Zs.shape[1]

    if mode == "partial_known" and flowmap is not None:
        V = df_done[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        flowmap.fit(V, Zs)

    train_X = torch.tensor(X_aug, dtype=DTYPE, device=DEVICE)
    train_Y = torch.tensor(Y, dtype=DTYPE, device=DEVICE)
    scaler = (mu, sd)
    return train_X, train_Y, sensor_dim, scaler


def build_candidate_tensor(df_cand: pd.DataFrame, flow_lookup: dict, mode: str,
                           flowmap: Optional[FlowMapKRR], scaler):
    X_base = df_cand[PARAM_ORDER].to_numpy(dtype=float)

    if mode == "no_sensor":
        return torch.tensor(X_base, dtype=DTYPE, device=DEVICE)

    mu, sd = scaler

    if mode == "full_known":
        Z = []
        for _, r in df_cand.iterrows():
            key = make_flow_key(r["Phi"], r["Q_total"], r["L"])
            if key not in flow_lookup:
                raise KeyError(f"[FlowLookup] Missing candidate key {key}")
            Z.append(flow_lookup[key])
        Z = np.stack(Z, axis=0)
        Zs = (Z - mu) / sd
    else:
        if flowmap is None or flowmap.V_train is None:
            raise RuntimeError("[PartialKnown] flowmap not trained yet (need completed trials).")
        V = df_cand[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        Zs = flowmap.predict(V)

    X_aug = np.concatenate([X_base, Zs], axis=1)
    return torch.tensor(X_aug, dtype=DTYPE, device=DEVICE)


# ============================================================
# 8) qNEHVI recommendation
# ============================================================
def recommend_next(df: pd.DataFrame, state: dict, flow_lookup: dict):
    df_done = df[~df.apply(is_pending, axis=1)].copy()
    if len(df_done) < 2:
        raise RuntimeError("Need at least 2 completed trials to fit GP robustly.")

    flowmap = None
    if MODE == "partial_known":
        flowmap = FlowMapKRR(lengthscale=FLOWMAP_RBF_LENGTHSCALE, alpha=FLOWMAP_RIDGE_ALPHA)

    train_X, train_Y, sensor_dim, scaler = build_train_tensors(df_done, flow_lookup, MODE, flowmap)

    model = build_model(train_X, train_Y, MODE, sensor_dim).to(dtype=DTYPE, device=DEVICE)
    model.train()
    fit_model(model)
    model.eval()

    tried = set(state["tried_keys"])
    pool = CANDIDATES_DF[~CANDIDATES_DF.apply(lambda r: key_from_row(r) in tried, axis=1)].copy()
    if len(pool) == 0:
        raise RuntimeError("All candidates tried.")

    X_cand = build_candidate_tensor(pool, flow_lookup, MODE, flowmap, scaler)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    partitioning = NondominatedPartitioning(ref_point=REF_POINT, Y=train_Y)

    acq = build_qnehvi_acq(
        model=model,
        ref_point=REF_POINT.tolist(),
        X_baseline=train_X,
        sampler=sampler,
        partitioning=partitioning,
    )

    with torch.no_grad():
        values = acq(X_cand.unsqueeze(1)).view(-1)

    best_idx = int(torch.argmax(values).item())
    best_row = pool.iloc[best_idx].to_dict()

    if MODE in ("full_known", "partial_known"):
        lam1 = float(model.models[0].covar_module.lam.detach().cpu().item())
        lam2 = float(model.models[1].covar_module.lam.detach().cpu().item())
        print(f"[lambda] gp_yield={lam1:.3f}  gp_sty={lam2:.3f}")

    return best_row


# ============================================================
# 9) Main workflow; the initial CSV must already exist
# ============================================================
def run_workflow():
    ensure_dir(RUN_DIR)
    cp = csv_path()
    state = load_state()

    if not os.path.exists(cp):
        raise FileNotFoundError(
            f"[Init] Required CSV not found: {cp}\n"
            f"Required columns: trial_index, {', '.join(PARAM_ORDER)}, yield, sty\n"
            "All existing rows must contain completed yield and sty values."
        )

    flow_lookup = {}
    if MODE in ("full_known", "partial_known"):
        flow_lookup = load_flow_feature_lookup(DATA_ROOT)
        print(f"[FlowData] Loaded keys={len(flow_lookup)} from {DATA_ROOT}")

    df = pd.read_csv(cp)

    required_cols = set(["trial_index"] + PARAM_ORDER + ["yield", "sty"])
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"[CSV] Missing columns: {missing}")

    if (not state.get("tried_keys")) and (state.get("trial_index", 0) == 0):
        state["tried_keys"] = [key_from_row(r) for _, r in df.iterrows()]
        state["trial_index"] = int(df["trial_index"].max()) + 1
        save_state(state)
        print(f"[StateInit] Initialized state from existing CSV. next_trial_index={state['trial_index']}")

    pending = df[df.apply(is_pending, axis=1)]
    if len(pending) > 0:
        print(f"[Stop] {len(pending)} pending rows have zero or missing objectives. Complete them first.")
        return

    print("[Fit+Recommend] Fitting model and recommending next...")
    best = recommend_next(df, state, flow_lookup)

    new_row = best.copy()
    new_row["trial_index"] = int(state["trial_index"])
    new_row["yield"] = 0.0
    new_row["sty"] = 0.0

    df_new = pd.DataFrame([new_row])[df.columns]
    df_new.to_csv(cp, mode="a", header=False, index=False)

    state["tried_keys"].append(key_from_row(new_row))
    state["trial_index"] += 1
    save_state(state)

    print(f"[New Trial] trial_index={new_row['trial_index']}  params={best}")
    print("[Done] The next candidate was appended. Run the experiment, enter yield and sty, then rerun.")


if __name__ == "__main__":
    run_workflow()
