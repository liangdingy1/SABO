#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sabo_partial_variants.py

Improved partial-known SABO design tester.

Based on sabo_information_modes.py, but adds switchable
partial-known designs through PARTIAL_MODE.

Main interaction is kept the same:
1) Prepare a CSV with initial/completed points.
2) Run this script.
3) It fits the model, recommends one new pending row with yield=0, sty=0.
4) You perform/fill the experiment result.
5) Run again.

Implemented PARTIAL_MODE options:
- baseline_krr
- confidence_weighted
- pca_krr
- pca_confidence_weighted
- full_flow_lookup
- lambda_warmup
- confidence_lambda_warmup
- pca_lambda_warmup
- pca_confidence_lambda_warmup

A GP-based flow-map is not included here. That would be a separate uncertainty-aware
design and is mainly more complex/less robust with small data and 16 sensor features,
rather than just computationally expensive.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any

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
# MODE = "full_known"       # "no_sensor" | "full_known" | "partial_known"
MODE = "partial_known"      # "no_sensor" | "full_known" | "partial_known"
# MODE = "no_sensor"        # "no_sensor" | "full_known" | "partial_known"

# Ignored unless MODE == "partial_known".
# PARTIAL_MODE = "baseline_krr"
# PARTIAL_MODE = "confidence_weighted"  # 1 done
# PARTIAL_MODE = "pca_krr"
# PARTIAL_MODE = "pca_confidence_weighted"  # 2 done
# PARTIAL_MODE = "full_flow_lookup"
# PARTIAL_MODE = "lambda_warmup"
PARTIAL_MODE = "confidence_lambda_warmup"  # 3 done
# PARTIAL_MODE = "pca_lambda_warmup"
# PARTIAL_MODE = "pca_confidence_lambda_warmup"  # 4 done

RUN_DIR = "../data/optimization_results"

if MODE == "partial_known":
    CSV_NAME = {
        "baseline_krr": "sabo_feature_completion.csv",
        "confidence_weighted": "sabo_fc_confidence.csv",
        "pca_confidence_weighted": "sabo_fc_pca_confidence.csv",
        "confidence_lambda_warmup": "sabo_fc_confidence_warmup.csv",
        "pca_confidence_lambda_warmup": "sabo_fc_pca_confidence_warmup.csv",
    }.get(PARTIAL_MODE, f"sabo_partial_{PARTIAL_MODE}.csv")
    STATE_NAME = f"sabo_partial_{PARTIAL_MODE}_state.json"
else:
    CSV_NAME = f"sabo_{MODE}.csv"
    STATE_NAME = f"sabo_{MODE}_state.json"

DATA_ROOT = "../data/Data"

DEVICE = "cpu"
DTYPE = torch.double

USE_MEDIAN_OF_3 = False

REF_POINT = torch.tensor([0.1, 0.01], dtype=DTYPE, device=DEVICE)

FLOWMAP_RBF_LENGTHSCALE = 0.25
FLOWMAP_RIDGE_ALPHA = 1e-4

# B1 confidence weighting
CONFIDENCE_RBF_LENGTHSCALE = 0.25
CONFIDENCE_POWER = 1.0
CONFIDENCE_FLOOR = 0.05

# B2 PCA compression
PCA_N_COMPONENTS = 4
PCA_MIN_VARIANCE_EPS = 1e-12

# B5 lambda warm-up
# warmup factor = clip((n_train - START_N) / (FULL_N - START_N), 0, 1)
# Effective lambda = sigmoid(raw_lambda) * warmup_factor
WARMUP_START_N = 10
WARMUP_FULL_N = 20

PENDING_IS_ZERO_ZERO = True
FLOW_KEY_ROUND = 6

warnings.filterwarnings("ignore", category=FutureWarning)

ALLOWED_PARTIAL_MODES = {
    "baseline_krr",
    "confidence_weighted",
    "pca_krr",
    "pca_confidence_weighted",
    "full_flow_lookup",
    "lambda_warmup",
    "confidence_lambda_warmup",
    "pca_lambda_warmup",
    "pca_confidence_lambda_warmup",
}

if MODE == "partial_known" and PARTIAL_MODE not in ALLOWED_PARTIAL_MODES:
    raise ValueError(f"Unknown PARTIAL_MODE={PARTIAL_MODE}. Allowed: {sorted(ALLOWED_PARTIAL_MODES)}")


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
# 3) partial-known flow-map：RBF KRR
# ============================================================
def normalize_v(v: np.ndarray) -> np.ndarray:
    phi = (v[..., 0] - min(SPACE["Phi"])) / (max(SPACE["Phi"]) - min(SPACE["Phi"]))
    q   = (v[..., 1] - min(SPACE["Q_total"])) / (max(SPACE["Q_total"]) - min(SPACE["Q_total"]))
    L   = (v[..., 2] - min(SPACE["L"])) / (max(SPACE["L"]) - min(SPACE["L"]))
    return np.stack([phi, q, L], axis=-1)


def rbf_kernel_np(A: np.ndarray, B: np.ndarray, lengthscale: float) -> np.ndarray:
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

    def fit(self, V: np.ndarray, Target: np.ndarray):
        Vn = normalize_v(V)
        K = rbf_kernel_np(Vn, Vn, self.ls)
        n = K.shape[0]
        K_reg = K + self.alpha * np.eye(n)
        self.A = np.linalg.solve(K_reg, Target)
        self.V_train = Vn

    def predict(self, V: np.ndarray) -> np.ndarray:
        if self.V_train is None or self.A is None:
            raise RuntimeError("FlowMapKRR has not been fitted.")
        Vn = normalize_v(V)
        K_star = rbf_kernel_np(Vn, self.V_train, self.ls)
        return K_star @ self.A


def confidence_from_observed(V_query: np.ndarray, V_observed: np.ndarray) -> np.ndarray:
    """
    B1 uncertainty proxy: confidence = max RBF similarity to observed hydrodynamic points.
    Observed/training points get confidence=1 separately.
    """
    Vq = normalize_v(V_query)
    Vo = normalize_v(V_observed)
    K = rbf_kernel_np(Vq, Vo, CONFIDENCE_RBF_LENGTHSCALE)
    conf = np.max(K, axis=1)
    conf = np.clip(conf, 0.0, 1.0)
    conf = conf ** CONFIDENCE_POWER
    conf = np.maximum(conf, CONFIDENCE_FLOOR)
    return conf.reshape(-1, 1)


# ============================================================
# 4) PCA helper for B2
# ============================================================
class SensorProjector:
    """
    kind="raw_std": standardize original 16 features.
    kind="pca": standardize original 16 features and project to PCA scores.
    """
    def __init__(self, kind: str = "raw_std", n_components: int = 4):
        self.kind = kind
        self.n_components = int(n_components)
        self.mu = None
        self.sd = None
        self.components = None
        self.actual_dim = None
        self.explained_variance_ratio = None

    def fit_transform(self, Z_raw: np.ndarray) -> np.ndarray:
        self.mu = Z_raw.mean(axis=0, keepdims=True)
        self.sd = Z_raw.std(axis=0, keepdims=True) + 1e-12
        Zs = (Z_raw - self.mu) / self.sd

        if self.kind == "raw_std":
            self.actual_dim = Zs.shape[1]
            return Zs

        if self.kind != "pca":
            raise ValueError(f"Unknown projector kind: {self.kind}")

        max_comp = min(self.n_components, Zs.shape[0], Zs.shape[1])
        if max_comp < 1:
            raise RuntimeError("Not enough data to fit PCA sensor projector.")

        U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
        self.components = Vt[:max_comp, :]
        scores = Zs @ self.components.T
        self.actual_dim = scores.shape[1]

        var = S**2
        total = float(np.sum(var)) + PCA_MIN_VARIANCE_EPS
        self.explained_variance_ratio = var[:max_comp] / total
        return scores

    def transform(self, Z_raw: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sd is None:
            raise RuntimeError("SensorProjector has not been fitted.")
        Zs = (Z_raw - self.mu) / self.sd
        if self.kind == "raw_std":
            return Zs
        return Zs @ self.components.T


# ============================================================
# 5) PARTIAL_MODE switches
# ============================================================
def uses_pca() -> bool:
    return MODE == "partial_known" and ("pca" in PARTIAL_MODE)


def uses_confidence() -> bool:
    return MODE == "partial_known" and ("confidence" in PARTIAL_MODE)


def uses_lambda_warmup() -> bool:
    return MODE == "partial_known" and ("warmup" in PARTIAL_MODE)


def uses_full_flow_lookup_partial() -> bool:
    return MODE == "partial_known" and PARTIAL_MODE == "full_flow_lookup"


def lambda_warmup_scale(n_train: int) -> float:
    if not uses_lambda_warmup():
        return 1.0
    if WARMUP_FULL_N <= WARMUP_START_N:
        return 1.0
    s = (float(n_train) - float(WARMUP_START_N)) / (float(WARMUP_FULL_N) - float(WARMUP_START_N))
    return float(np.clip(s, 0.0, 1.0))


# ============================================================
# 6) SABO Kernel variants
# ============================================================
class SABOKernel(gpytorch.kernels.Kernel):
    """
    k_chem * ((1-lambda_eff) k_hyd + lambda_eff k_sensor)

    If use_confidence=True, an extra input column is appended after sensor features.
    The sensor kernel is multiplied by c(x)c(x').
    """
    def __init__(self, sensor_dim: int, use_confidence: bool = False, lambda_scale: float = 1.0, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.sensor_dim = int(sensor_dim)
        self.use_confidence = bool(use_confidence)
        self.lambda_scale_value = float(lambda_scale)

        self.k_chem = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        self.k_hyd  = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        sensor_dims = list(range(5, 5 + sensor_dim))
        self.k_sen  = ScaleKernel(RBFKernel(ard_num_dims=sensor_dim, active_dims=sensor_dims))

        self.conf_dim = 5 + sensor_dim

        self.register_parameter(name="raw_lambda", parameter=torch.nn.Parameter(torch.tensor(0.0)))
        self.register_constraint("raw_lambda", Interval(-10.0, 10.0))

    @property
    def lam_base(self):
        return torch.sigmoid(self.raw_lambda)

    @property
    def lam(self):
        return self.lam_base * self.lambda_scale_value

    def forward(self, x1, x2, **params):
        kchem = self.k_chem(x1, x2, **params)
        khyd  = self.k_hyd(x1, x2, **params)
        ksen  = self.k_sen(x1, x2, **params)

        if self.use_confidence:
            c1 = torch.clamp(x1[..., self.conf_dim], min=0.0, max=1.0).unsqueeze(-1)
            c2 = torch.clamp(x2[..., self.conf_dim], min=0.0, max=1.0).unsqueeze(-2)
            conf_outer = c1 * c2
            ksen = ksen * conf_outer

        mix = (1.0 - self.lam) * khyd + self.lam * ksen
        return kchem * mix


def build_model(train_X: torch.Tensor, train_Y: torch.Tensor, mode: str, sensor_dim: int, n_train: int):
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
        lam_scale = lambda_warmup_scale(n_train)
        gp1.covar_module = SABOKernel(sensor_dim=sensor_dim, use_confidence=uses_confidence(), lambda_scale=lam_scale)
        gp2.covar_module = SABOKernel(sensor_dim=sensor_dim, use_confidence=uses_confidence(), lambda_scale=lam_scale)

    return ModelListGP(gp1, gp2)


def fit_model(model: ModelListGP):
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ============================================================
# 7) qNEHVI initialization compatibility layer
# ============================================================
def build_qnehvi_acq(model, ref_point, X_baseline, sampler, partitioning):
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

    return qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,
    )


# ============================================================
# 8) CSV/state helpers
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
# 9) Training and candidate tensors
# ============================================================
def raw_sensor_matrix_for_df(df_points: pd.DataFrame, flow_lookup: dict) -> np.ndarray:
    Z = []
    for _, r in df_points.iterrows():
        key = make_flow_key(r["Phi"], r["Q_total"], r["L"])
        if key not in flow_lookup:
            raise KeyError(f"[FlowLookup] Missing key {key} in flow tables.")
        Z.append(flow_lookup[key])
    return np.stack(Z, axis=0)


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

    Z_raw = raw_sensor_matrix_for_df(df_done, flow_lookup)

    if mode == "full_known":
        projector = SensorProjector(kind="raw_std")
    else:
        projector = SensorProjector(kind="pca" if uses_pca() else "raw_std", n_components=PCA_N_COMPONENTS)

    Z_rep = projector.fit_transform(Z_raw)
    sensor_dim = Z_rep.shape[1]

    X_parts = [X_base, Z_rep]

    if mode == "partial_known" and uses_confidence():
        conf_train = np.ones((len(df_done), 1), dtype=float)
        X_parts.append(conf_train)

    X_aug = np.concatenate(X_parts, axis=1)

    if mode == "partial_known" and flowmap is not None and not uses_full_flow_lookup_partial():
        V = df_done[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        flowmap.fit(V, Z_rep)

    train_X = torch.tensor(X_aug, dtype=DTYPE, device=DEVICE)
    train_Y = torch.tensor(Y, dtype=DTYPE, device=DEVICE)

    context = {
        "projector": projector,
        "observed_V": df_done[["Phi", "Q_total", "L"]].to_numpy(dtype=float),
        "sensor_dim": sensor_dim,
    }
    return train_X, train_Y, sensor_dim, context


def build_candidate_tensor(df_cand: pd.DataFrame, flow_lookup: dict, mode: str,
                           flowmap: Optional[FlowMapKRR], context: Optional[Dict[str, Any]]):
    X_base = df_cand[PARAM_ORDER].to_numpy(dtype=float)

    if mode == "no_sensor":
        return torch.tensor(X_base, dtype=DTYPE, device=DEVICE)

    projector = context["projector"]
    observed_V = context["observed_V"]

    if mode == "full_known":
        Z_raw = raw_sensor_matrix_for_df(df_cand, flow_lookup)
        Z_rep = projector.transform(Z_raw)
        X_aug = np.concatenate([X_base, Z_rep], axis=1)
        return torch.tensor(X_aug, dtype=DTYPE, device=DEVICE)

    # partial-known
    if uses_full_flow_lookup_partial():
        Z_raw = raw_sensor_matrix_for_df(df_cand, flow_lookup)
        Z_rep = projector.transform(Z_raw)
    else:
        if flowmap is None or flowmap.V_train is None:
            raise RuntimeError("[PartialKnown] flowmap not trained yet (need completed trials).")
        V = df_cand[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        Z_rep = flowmap.predict(V)

    X_parts = [X_base, Z_rep]

    if uses_confidence():
        V = df_cand[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        conf = confidence_from_observed(V, observed_V)
        X_parts.append(conf)

    X_aug = np.concatenate(X_parts, axis=1)
    return torch.tensor(X_aug, dtype=DTYPE, device=DEVICE)


# ============================================================
# 10) qNEHVI recommendation
# ============================================================
def recommend_next(df: pd.DataFrame, state: dict, flow_lookup: dict):
    df_done = df[~df.apply(is_pending, axis=1)].copy()
    if len(df_done) < 2:
        raise RuntimeError("Need at least 2 completed trials to fit GP robustly.")

    flowmap = None
    if MODE == "partial_known" and (not uses_full_flow_lookup_partial()):
        flowmap = FlowMapKRR(lengthscale=FLOWMAP_RBF_LENGTHSCALE, alpha=FLOWMAP_RIDGE_ALPHA)

    train_X, train_Y, sensor_dim, context = build_train_tensors(df_done, flow_lookup, MODE, flowmap)

    model = build_model(train_X, train_Y, MODE, sensor_dim, n_train=len(df_done)).to(dtype=DTYPE, device=DEVICE)
    model.train()
    fit_model(model)
    model.eval()

    tried = set(state["tried_keys"])
    pool = CANDIDATES_DF[~CANDIDATES_DF.apply(lambda r: key_from_row(r) in tried, axis=1)].copy()
    if len(pool) == 0:
        raise RuntimeError("All candidates tried.")

    X_cand = build_candidate_tensor(pool, flow_lookup, MODE, flowmap, context)

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

    print(f"[Mode] MODE={MODE}")
    if MODE == "partial_known":
        print(f"[PartialMode] PARTIAL_MODE={PARTIAL_MODE}")
        print(f"[PartialMode] uses_pca={uses_pca()}  uses_confidence={uses_confidence()}  "
              f"uses_lambda_warmup={uses_lambda_warmup()}  full_flow_lookup={uses_full_flow_lookup_partial()}")
        if uses_lambda_warmup():
            print(f"[Warmup] n_train={len(df_done)}  lambda_scale={lambda_warmup_scale(len(df_done)):.3f}")
        if context is not None and context.get("projector") is not None:
            projector = context["projector"]
            if projector.kind == "pca":
                evr = projector.explained_variance_ratio
                print(f"[PCA] actual_dim={projector.actual_dim}  explained_variance_ratio="
                      f"{np.round(evr, 3).tolist()}")

    if MODE in ("full_known", "partial_known"):
        cov1 = model.models[0].covar_module
        cov2 = model.models[1].covar_module
        if hasattr(cov1, "lam"):
            lam1_eff = float(cov1.lam.detach().cpu().item())
            lam1_base = float(cov1.lam_base.detach().cpu().item())
            lam2_eff = float(cov2.lam.detach().cpu().item())
            lam2_base = float(cov2.lam_base.detach().cpu().item())
            print(f"[lambda] yield base={lam1_base:.3f}, eff={lam1_eff:.3f} | "
                  f"sty base={lam2_base:.3f}, eff={lam2_eff:.3f}")

    print(f"[BestAcq] value={float(values[best_idx].detach().cpu().item()):.6g}")
    return best_row


# ============================================================
# 11) Main workflow
# ============================================================
def run_workflow():
    ensure_dir(RUN_DIR)
    cp = csv_path()
    state = load_state()

    if not os.path.exists(cp):
        raise FileNotFoundError(
            f"[Init] Required CSV not found: {cp}\n"
            f"Required columns: trial_index, {', '.join(PARAM_ORDER)}, yield, sty\n"
            f"All existing rows must contain completed yield/sty values.\n\n"
            f"Copy an existing 10-point initialization CSV, rename it to {CSV_NAME}, "
            f"and place it in {RUN_DIR}/."
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
        print(f"[Stop] {len(pending)} pending row(s) have yield/sty equal to 0 or NaN. Complete them before rerunning.")
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
    print(f"[Done] Appended the next point to the CSV: {cp}")
    print("[Next] Run the experiment, enter yield/sty, and rerun this script.")


if __name__ == "__main__":
    run_workflow()
