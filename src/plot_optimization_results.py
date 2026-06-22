#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.constraints import Interval
from gpytorch.kernels import ProductKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Input data is not contained to the unit cube.*")
warnings.filterwarnings("ignore", message="`scipy_minimize` terminated with status 3.*")

# ============================================================
# 0) USER CONFIG
# ============================================================
INITIAL_N = 10

# All 8 files correspond to the final manuscript legend order.
CSV_BY_MODE = {
    "qnehvi": "qnehvi_ax.csv",
    "qnehvi_nr": "qnehvi_no_repeat.csv",
    "sabo_fc": "sabo_feature_completion.csv",
    "sabo_fc_pca_cw_lw": "sabo_fc_pca_confidence_warmup.csv",
    "sabo_fc_pca_cw": "sabo_fc_pca_confidence.csv",
    "sabo_fc_cw_lw": "sabo_fc_confidence_warmup.csv",
    "sabo_fc_cw": "sabo_fc_confidence.csv",
    "sabo_fl": "sabo_full_library.csv",
}

# The folder where the 8 CSVs are placed.
SCRIPT_BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_BASE_DIR.parent / "data" / "optimization_results"

# Pointing to the flow data from sa_09.
DATA_ROOT = SCRIPT_BASE_DIR.parent / "data" / "Data"

DEVICE = "cpu"
DTYPE = torch.double

USE_MEDIAN_OF_3 = False
PENDING_IS_ZERO_ZERO = True
FLOW_KEY_ROUND = 6
REF_POINT = torch.tensor([0.1, 0.01], dtype=DTYPE, device=DEVICE)

FLOWMAP_RBF_LENGTHSCALE = 0.25
FLOWMAP_RIDGE_ALPHA = 1e-4
QMC_SAMPLES = 256

SPACE = {
    "C_aq": [0.1, 0.2, 0.3, 0.4, 0.5],
    "C_org": [0.01, 0.02, 0.03, 0.04, 0.05],
    "Phi": [0.3, 0.4, 0.5, 0.6, 0.7],
    "Q_total": [0.2, 0.4, 0.6, 0.8, 1.0],
    "L": [1.0, 2.0, 4.0],
}
PARAM_ORDER = ["C_aq", "C_org", "Phi", "Q_total", "L"]
OBJECTIVE_COLS = ["yield", "sty"]

FEATURE_COLS_16 = [
    "Mean", "Std", "Skewness", "Kurtosis", "Dom_Freq", "Spec_Energy", "Wave_Eng_D1",
    "Entropy", "Hurst", "Cross_Rate", "Peak_Dist", "Num_Peaks", "Num_Valleys",
    "Avg_Peak_H", "Avg_Valley_H", "Cross_Count",
]

DPI = 300
# SHOW_FIGURES = True
SHOW_FIGURES = False
SAVE_FIGURES = True

PLOT_DATA_DIR = SCRIPT_BASE_DIR / "optimization_replay_cache"
PLOT_DATA_FILES = {
    "summary": PLOT_DATA_DIR / "summary.csv",
    "visited": PLOT_DATA_DIR / "visited.csv",
    "pareto_final": PLOT_DATA_DIR / "pareto_final.csv",
}
ITERATION_XTICKS = np.arange(INITIAL_N + 1, 26, 2)

NO_SENSOR_MODES = {"qnehvi", "qnehvi_nr"}
PARTIAL_MODES = {
    "sabo_fc",
    "sabo_fc_pca_cw_lw",
    "sabo_fc_pca_cw",
    "sabo_fc_cw_lw",
    "sabo_fc_cw",
}
FULL_MODES = {"sabo_fl"}
SABO_MODES = PARTIAL_MODES | FULL_MODES

MODE_ORDER = [
    "qnehvi",
    "qnehvi_nr",
    "sabo_fc",
    "sabo_fc_pca_cw_lw",
    "sabo_fc_pca_cw",
    "sabo_fc_cw_lw",
    "sabo_fc_cw",
    "sabo_fl",
]

MODE_LABELS = {
    "qnehvi": "qNEHVI",
    "qnehvi_nr": "qNEHVI-NR",
    "sabo_fc": "SABO-FC",
    "sabo_fc_pca_cw_lw": "SABO-FC-PCA-CW-LW",
    "sabo_fc_pca_cw": "SABO-FC-PCA-CW",
    "sabo_fc_cw_lw": "SABO-FC-CW-LW",
    "sabo_fc_cw": "SABO-FC-CW",
    "sabo_fl": "SABO-FL",
}

# Manuscript color families:
# no-sensor red, partial-known feature-completion split into FC / PCA / CW groups,
# and full-library green.
MODE_COLORS = {
    "qnehvi": "#ffb6a6",
    "qnehvi_nr": "#ff725f",
    "sabo_fc": "#46b6ff",
    "sabo_fc_pca_cw_lw": "#8b5cf6",
    "sabo_fc_pca_cw": "#d946ef",
    "sabo_fc_cw_lw": "#fbbf24",
    "sabo_fc_cw": "#f59e0b",
    "sabo_fl": "#62c98f",
}

MODE_MARKERS = {
    "qnehvi": "o",
    "qnehvi_nr": "X",
    "sabo_fc": "^",
    "sabo_fc_pca_cw_lw": "s",
    "sabo_fc_pca_cw": "D",
    "sabo_fc_cw_lw": "v",
    "sabo_fc_cw": "P",
    "sabo_fl": "*",
}

INITIAL_COLOR = "#B3B3B3"
FLOWMAP_RMSE_COLOR = "#A78BFA"
FLOWMAP_MAE_COLOR = "#FFA95A"

_RUN_SAVE_DIR: Path | None = None


# ============================================================
# 1) Basic helpers
# ============================================================
def key_from_row(row) -> str:
    return f"Caq{row['C_aq']}_Corg{row['C_org']}_Phi{row['Phi']}_Q{row['Q_total']}_L{row['L']}"


def make_flow_key(phi, qtot, L):
    return (
        round(float(phi), FLOW_KEY_ROUND),
        round(float(qtot), FLOW_KEY_ROUND),
        round(float(L), FLOW_KEY_ROUND),
    )


def is_pending(row) -> bool:
    if PENDING_IS_ZERO_ZERO:
        return (float(row["yield"]) == 0.0) and (float(row["sty"]) == 0.0)
    return pd.isna(row["yield"]) or pd.isna(row["sty"])


def make_all_candidates() -> pd.DataFrame:
    rows = []
    for C_aq in SPACE["C_aq"]:
        for C_org in SPACE["C_org"]:
            for Phi in SPACE["Phi"]:
                for Q_total in SPACE["Q_total"]:
                    for L in SPACE["L"]:
                        rows.append({
                            "C_aq": C_aq,
                            "C_org": C_org,
                            "Phi": Phi,
                            "Q_total": Q_total,
                            "L": L,
                        })
    return pd.DataFrame(rows)


CANDIDATES_DF = make_all_candidates()


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
        raise RuntimeError("[FlowData] No features loaded. Check Data folder naming and features_table.csv.")
    return lookup


# ============================================================
# 2) Partial-known feature-completion diagnostic model
# ============================================================
def normalize_v(v: np.ndarray) -> np.ndarray:
    phi = (v[..., 0] - min(SPACE["Phi"])) / (max(SPACE["Phi"]) - min(SPACE["Phi"]))
    q = (v[..., 1] - min(SPACE["Q_total"])) / (max(SPACE["Q_total"]) - min(SPACE["Q_total"]))
    L = (v[..., 2] - min(SPACE["L"])) / (max(SPACE["L"]) - min(SPACE["L"]))
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

    def fit(self, V: np.ndarray, Zs: np.ndarray):
        Vn = normalize_v(V)
        K = rbf_kernel_np(Vn, Vn, self.ls)
        n = K.shape[0]
        K_reg = K + self.alpha * np.eye(n)
        self.A = np.linalg.solve(K_reg, Zs)
        self.V_train = Vn

    def predict(self, V: np.ndarray) -> np.ndarray:
        Vn = normalize_v(V)
        K_star = rbf_kernel_np(Vn, self.V_train, self.ls)
        return K_star @ self.A


# ============================================================
# 3) Model definition
# ============================================================
class SABOKernel(gpytorch.kernels.Kernel):
    def __init__(self, sensor_dim: int, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.k_chem = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        self.k_hyd = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        sensor_dims = list(range(5, 5 + sensor_dim))
        self.k_sen = ScaleKernel(RBFKernel(ard_num_dims=sensor_dim, active_dims=sensor_dims))

        self.register_parameter(name="raw_lambda", parameter=torch.nn.Parameter(torch.tensor(0.0)))
        self.register_constraint("raw_lambda", Interval(-10.0, 10.0))

    @property
    def lam(self):
        return torch.sigmoid(self.raw_lambda)

    def forward(self, x1, x2, **params):
        kchem = self.k_chem(x1, x2, **params)
        khyd = self.k_hyd(x1, x2, **params)
        ksen = self.k_sen(x1, x2, **params)
        mix = (1.0 - self.lam) * khyd + self.lam * ksen
        return kchem * mix


def build_model(train_X: torch.Tensor, train_Y: torch.Tensor, mode: str, sensor_dim: int):
    gp1 = SingleTaskGP(train_X, train_Y[:, [0]], outcome_transform=Standardize(m=1))
    gp2 = SingleTaskGP(train_X, train_Y[:, [1]], outcome_transform=Standardize(m=1))

    if mode in NO_SENSOR_MODES:
        kchem = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        khyd = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        gp1.covar_module = ProductKernel(kchem, khyd)

        kchem2 = ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[0, 1]))
        khyd2 = ScaleKernel(RBFKernel(ard_num_dims=3, active_dims=[2, 3, 4]))
        gp2.covar_module = ProductKernel(kchem2, khyd2)
    else:
        gp1.covar_module = SABOKernel(sensor_dim=sensor_dim)
        gp2.covar_module = SABOKernel(sensor_dim=sensor_dim)

    return ModelListGP(gp1, gp2)


def fit_model(model: ModelListGP):
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


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
# 4) Tensor builders
# ============================================================
def build_train_tensors(df_done: pd.DataFrame, flow_lookup: dict, mode: str, flowmap: Optional[FlowMapKRR]):
    X_base = df_done[PARAM_ORDER].to_numpy(dtype=float)
    Y = df_done[OBJECTIVE_COLS].to_numpy(dtype=float)

    if mode in NO_SENSOR_MODES:
        return (
            torch.tensor(X_base, dtype=DTYPE, device=DEVICE),
            torch.tensor(Y, dtype=DTYPE, device=DEVICE),
            0,
            None,
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

    if mode in PARTIAL_MODES and flowmap is not None:
        V = df_done[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        flowmap.fit(V, Zs)

    train_X = torch.tensor(X_aug, dtype=DTYPE, device=DEVICE)
    train_Y = torch.tensor(Y, dtype=DTYPE, device=DEVICE)
    scaler = (mu, sd)
    return train_X, train_Y, sensor_dim, scaler, Zs


def build_candidate_tensor(df_cand: pd.DataFrame, flow_lookup: dict, mode: str, flowmap: Optional[FlowMapKRR], scaler):
    X_base = df_cand[PARAM_ORDER].to_numpy(dtype=float)

    if mode in NO_SENSOR_MODES:
        return torch.tensor(X_base, dtype=DTYPE, device=DEVICE), None

    mu, sd = scaler

    if mode in FULL_MODES:
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
            raise RuntimeError("[PartialKnown] flowmap not trained yet.")
        V = df_cand[["Phi", "Q_total", "L"]].to_numpy(dtype=float)
        Zs = flowmap.predict(V)

    X_aug = np.concatenate([X_base, Zs], axis=1)
    return torch.tensor(X_aug, dtype=DTYPE, device=DEVICE), Zs


# ============================================================
# 5) Diagnostics helpers
# ============================================================
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b) and np.any(a > b)


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    n = len(Y)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominates(Y[j], Y[i]):
                keep[i] = False
                break
    return keep


def hypervolume_2d_max(Y: np.ndarray, ref_point: np.ndarray) -> float:
    Y = np.asarray(Y, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if Y.size == 0:
        return 0.0
    P = Y[(Y[:, 0] > ref_point[0]) & (Y[:, 1] > ref_point[1])].copy()
    if len(P) == 0:
        return 0.0
    mask = pareto_mask(P)
    P = P[mask]
    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    prev_x = ref_point[0]
    for x, y in P:
        width = x - prev_x
        height = y - ref_point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = max(prev_x, x)
    return float(hv)


def safe_lambda(gp) -> float:
    if hasattr(gp.covar_module, "lam"):
        return float(gp.covar_module.lam.detach().cpu().item())
    return float("nan")


def compute_partial_flowmap_metrics(flow_lookup: dict, flowmap: FlowMapKRR, scaler) -> Dict[str, float]:
    mu, sd = scaler
    V_all = []
    Z_true = []
    for (phi, qtot, L), vec in flow_lookup.items():
        V_all.append([phi, qtot, L])
        Z_true.append(vec)
    V_all = np.asarray(V_all, dtype=float)
    Z_true = np.asarray(Z_true, dtype=float)
    Z_true_std = (Z_true - mu) / sd
    Z_pred_std = flowmap.predict(V_all)
    diff = Z_pred_std - Z_true_std
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((Z_true_std - Z_true_std.mean(axis=0, keepdims=True))**2))
    r2 = float("nan") if ss_tot <= 1e-20 else 1.0 - ss_res / ss_tot
    return {
        "flowmap_rmse_std": rmse,
        "flowmap_mae_std": mae,
        "flowmap_r2_std": r2,
    }


def validate_input_csv(df: pd.DataFrame, mode: str, csv_file: str) -> None:
    required = set(["trial_index", *PARAM_ORDER, *OBJECTIVE_COLS])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"[{mode}] Missing columns in {csv_file}: {sorted(missing)}")
    if len(df) <= INITIAL_N:
        raise ValueError(f"[{mode}] CSV has <= INITIAL_N rows; nothing to replay: {csv_file}")
    pending = df.apply(is_pending, axis=1)
    if pending.any():
        bad = df.loc[pending, "trial_index"].tolist()
        raise ValueError(f"[{mode}] Pending zero-zero objective rows found at trial_index={bad}")


# ============================================================
# 6) Replay engine
# ============================================================
@dataclass
class ReplayOutput:
    summary: pd.DataFrame
    visited: pd.DataFrame
    pareto_all: pd.DataFrame
    pareto_final: pd.DataFrame


def replay_one_mode(mode: str, csv_file: str, flow_lookup: dict) -> ReplayOutput:
    df = pd.read_csv(csv_file).copy()
    validate_input_csv(df, mode, csv_file)

    df = df.sort_values("trial_index").reset_index(drop=True)
    visited_rows = []
    summary_rows = []

    for _, row in df.iterrows():
        visited_rows.append({
            "mode": mode,
            "label": MODE_LABELS.get(mode, mode),
            "trial_index": int(row["trial_index"]),
            **{p: float(row[p]) for p in PARAM_ORDER},
            "yield": float(row["yield"]),
            "sty": float(row["sty"]),
            "is_initial": int(int(row["trial_index"]) < INITIAL_N),
        })

    for replay_pos in range(INITIAL_N, len(df)):
        df_done = df.iloc[:replay_pos].copy()
        df_attained = df.iloc[:replay_pos + 1].copy()
        Y_attained = df_attained[OBJECTIVE_COLS].to_numpy(dtype=float)
        hv = hypervolume_2d_max(Y_attained, ref_point=REF_POINT.detach().cpu().numpy())
        pareto_count = int(pareto_mask(Y_attained).sum())

        lam_y = float("nan")
        lam_s = float("nan")
        flowmap_metrics = {
            "flowmap_rmse_std": float("nan"),
            "flowmap_mae_std": float("nan"),
            "flowmap_r2_std": float("nan"),
        }

        if mode in SABO_MODES:
            flowmap = None
            if mode in PARTIAL_MODES:
                flowmap = FlowMapKRR(lengthscale=FLOWMAP_RBF_LENGTHSCALE, alpha=FLOWMAP_RIDGE_ALPHA)

            train_X, train_Y, sensor_dim, scaler, _ = build_train_tensors(
                df_done, flow_lookup, mode, flowmap
            )

            if mode in PARTIAL_MODES and flowmap is not None and scaler is not None:
                flowmap_metrics = compute_partial_flowmap_metrics(flow_lookup, flowmap, scaler)

            try:
                model = build_model(train_X, train_Y, mode, sensor_dim).to(dtype=DTYPE, device=DEVICE)
                model.train()
                fit_model(model)
                model.eval()

                lam_y = safe_lambda(model.models[0])
                lam_s = safe_lambda(model.models[1])
            except Exception as exc:
                print(f"[Warning] {MODE_LABELS.get(mode, mode)} fit failed at replay_pos={replay_pos}: {exc}")

        summary_rows.append({
            "mode": mode,
            "label": MODE_LABELS.get(mode, mode),
            "replay_iteration": replay_pos - INITIAL_N + 1,
            "replay_pos": replay_pos,
            "n_train": len(df_done),
            "n_attained": len(df_attained),
            "hypervolume": hv,
            "pareto_count": pareto_count,
            "lambda_yield": lam_y,
            "lambda_sty": lam_s,
            **flowmap_metrics,
        })

    summary_df = pd.DataFrame(summary_rows)
    visited_df = pd.DataFrame(visited_rows)

    all_Y = df[OBJECTIVE_COLS].to_numpy(dtype=float)
    final_mask = pareto_mask(all_Y)
    pareto_all = df.copy()
    pareto_all["mode"] = mode
    pareto_all["label"] = MODE_LABELS.get(mode, mode)
    pareto_all["is_pareto_final"] = final_mask.astype(int)
    pareto_final = pareto_all[pareto_all["is_pareto_final"] == 1].copy()

    return ReplayOutput(
        summary=summary_df,
        visited=visited_df,
        pareto_all=pareto_all,
        pareto_final=pareto_final,
    )


# ============================================================
# 7) Plotting logic
# ============================================================
def get_save_dir() -> Path:
    global _RUN_SAVE_DIR
    if _RUN_SAVE_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        _RUN_SAVE_DIR = SCRIPT_BASE_DIR / f"optimization_results_{timestamp}"
    return _RUN_SAVE_DIR


def configure_matplotlib() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    rc_updates = {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
        "axes.linewidth": 2,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    if "Times New Roman" in available_fonts:
        rc_updates.update({
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        })
    plt.rcParams.update(rc_updates)


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(direction="in", length=5, width=1.3)
    ax.tick_params(which="minor", direction="in", length=3, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if grid:
        ax.grid(alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def style_iteration_axis(ax: plt.Axes) -> None:
    ax.set_xlim(ITERATION_XTICKS[0] - 0.5, ITERATION_XTICKS[-1] + 0.5)
    ax.set_xticks(ITERATION_XTICKS)


def add_panel_letter(
    ax: plt.Axes,
    letter: str,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = 24,
) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=fontsize,
        va="bottom",
    )


def add_absolute_iteration(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    if "n_attained" in df.columns:
        df["iteration"] = df["n_attained"]
    return df


def ordered_legend(ax: plt.Axes):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [MODE_LABELS[m] for m in MODE_ORDER if MODE_LABELS[m] in by_label]
    ordered_handles = [by_label[label] for label in ordered_labels]
    return ordered_handles, ordered_labels


def plot_hv_yield_sty(data: dict[str, pd.DataFrame]) -> plt.Figure:
    summary = add_absolute_iteration(data["summary"])
    visited = data["visited"]
    pareto_final = data["pareto_final"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    ax0, ax1 = axes

    for idx, mode in enumerate(MODE_ORDER):
        sub = summary.loc[summary["mode"] == mode].sort_values("iteration")
        if sub.empty:
            continue
        ax0.plot(
            sub["iteration"].to_numpy(),
            sub["hypervolume"].to_numpy(),
            label=MODE_LABELS.get(mode, mode),
            marker=MODE_MARKERS.get(mode, "o"),
            markersize=5,
            linewidth=1.9,
            color=MODE_COLORS.get(mode, f"C{idx}"),
        )

    ax0.set_xlabel("Experiment number")
    ax0.set_ylabel("Hypervolume")
    ax0.set_title("Hypervolume progression")
    style_iteration_axis(ax0)
    style_axes(ax0, grid=True)

    for idx, mode in enumerate(MODE_ORDER):
        sub_all = visited.loc[visited["mode"] == mode].copy()
        sub_pf = pareto_final.loc[pareto_final["mode"] == mode].copy()
        color = MODE_COLORS.get(mode, f"C{idx}")
        marker = MODE_MARKERS.get(mode, "o")
        label = MODE_LABELS.get(mode, mode)

        if not sub_all.empty:
            ax1.scatter(
                sub_all["yield"],
                sub_all["sty"],
                s=28,
                color=color,
                alpha=0.20,
                marker=marker,
                linewidths=0,
            )
        if not sub_pf.empty:
            sub_pf = sub_pf.sort_values("yield")
            ax1.plot(sub_pf["yield"], sub_pf["sty"], color=color, linewidth=1.7, alpha=0.85)
            ax1.scatter(
                sub_pf["yield"],
                sub_pf["sty"],
                s=62,
                color=color,
                marker=marker,
                label=label,
                edgecolors="none",
                alpha=0.98,
            )

    ax1.set_xlabel("Yield")
    ax1.set_ylabel("STY")
    ax1.set_title("Final attained fronts")
    ax1.xaxis.set_major_locator(MultipleLocator(0.1))
    style_axes(ax1, grid=True)

    add_panel_letter(ax0, "A")
    add_panel_letter(ax1, "B")

    handles, labels = ordered_legend(ax1)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(MODE_ORDER),
        fontsize=12,
        frameon=False,
        columnspacing=0.75,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def plot_lambda_flowmap(data: dict[str, pd.DataFrame]) -> plt.Figure:
    summary = add_absolute_iteration(data["summary"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    ax0, ax1 = axes

    for idx, mode in enumerate(MODE_ORDER):
        if mode in NO_SENSOR_MODES:
            continue
        sub = summary.loc[summary["mode"] == mode].sort_values("iteration")
        if sub.empty:
            continue
        ax0.plot(
            sub["iteration"],
            sub["lambda_yield"],
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, f"C{idx}"),
            linestyle="-",
            marker=MODE_MARKERS.get(mode, "o"),
            markersize=4.6,
            linewidth=1.7,
        )

    ax0.set_xlabel("Experiment number")
    ax0.set_ylabel(r"Learned $\lambda$ (Yield)")
    ax0.set_ylim(-0.03, 1.03)
    ax0.set_title(r"Learned sensor-weight parameter ($\lambda$) for Yield")
    style_iteration_axis(ax0)
    style_axes(ax0, grid=True)
    ax0.legend(frameon=False, loc="best", fontsize=8)

    for idx, mode in enumerate(MODE_ORDER):
        if mode in NO_SENSOR_MODES:
            continue
        sub = summary.loc[summary["mode"] == mode].sort_values("iteration")
        if sub.empty:
            continue
        ax1.plot(
            sub["iteration"],
            sub["lambda_sty"],
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, f"C{idx}"),
            linestyle="--",
            marker=MODE_MARKERS.get(mode, "o"),
            markersize=4.6,
            linewidth=1.7,
        )

    ax1.set_xlabel("Experiment number")
    ax1.set_ylabel(r"Learned $\lambda$ (STY)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title(r"Learned sensor-weight parameter ($\lambda$) for STY")
    style_iteration_axis(ax1)
    style_axes(ax1, grid=True)
    ax1.legend(frameon=False, loc="best", fontsize=8)

    add_panel_letter(ax0, "A")
    add_panel_letter(ax1, "B")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def print_final_metrics(summary: pd.DataFrame, pareto_final: pd.DataFrame, visited: pd.DataFrame) -> None:
    summary = add_absolute_iteration(summary)
    print("\n[Final metrics]")
    for mode in MODE_ORDER:
        label = MODE_LABELS[mode]
        sub = summary.loc[summary["mode"] == mode].sort_values("iteration")
        vis = visited.loc[visited["mode"] == mode]
        pf = pareto_final.loc[pareto_final["mode"] == mode]
        if sub.empty or vis.empty:
            print(f"  {label}: no data")
            continue
        final_hv = float(sub.iloc[-1]["hypervolume"])
        max_yield = float(vis["yield"].max())
        max_sty = float(vis["sty"].max())
        print(
            f"  {label}: final HV={final_hv:.8g}, "
            f"Pareto points={len(pf)}, max Yield={max_yield:.6g}, max STY={max_sty:.6g}"
        )


def load_cached_plot_data() -> Optional[dict[str, pd.DataFrame]]:
    missing = [path for path in PLOT_DATA_FILES.values() if not path.exists()]
    if missing:
        return None

    print(f"[Cache] Loading plot data from {PLOT_DATA_DIR}")
    data = {
        key: pd.read_csv(path)
        for key, path in PLOT_DATA_FILES.items()
    }

    if data["summary"].empty or data["visited"].empty:
        print("[Cache] Cached plot data is empty; recomputing replay data.")
        return None

    return data


def save_plot_data(data: dict[str, pd.DataFrame]) -> None:
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, path in PLOT_DATA_FILES.items():
        df = data.get(key, pd.DataFrame())
        df.to_csv(path, index=False)
    print(f"[Cache] Plot data saved to {PLOT_DATA_DIR}")


def load_and_process_csvs() -> dict[str, pd.DataFrame]:
    flow_lookup = load_flow_feature_lookup(str(DATA_ROOT))
    summary_frames = []
    visited_frames = []
    pareto_frames = []

    for mode in MODE_ORDER:
        filename = CSV_BY_MODE[mode]
        csv_path = CSV_DIR / filename
        if not csv_path.exists():
            print(f"[Warning] Missing {csv_path}, skipping mode: {mode}")
            continue

        print(f"[Replay] {MODE_LABELS.get(mode, mode):<22} csv={csv_path.name}")
        out = replay_one_mode(mode, str(csv_path), flow_lookup)
        summary_frames.append(out.summary)
        visited_frames.append(out.visited)
        pareto_frames.append(out.pareto_final)

    return {
        "summary": pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        "visited": pd.concat(visited_frames, ignore_index=True) if visited_frames else pd.DataFrame(),
        "pareto_final": pd.concat(pareto_frames, ignore_index=True) if pareto_frames else pd.DataFrame(),
    }


def main() -> None:
    configure_matplotlib()
    data = load_cached_plot_data()
    if data is None:
        print("[Cache] Cached plot data not found. Running replay calculations.")
        data = load_and_process_csvs()
        if not data["summary"].empty:
            save_plot_data(data)

    if data["summary"].empty:
        print("No valid data processed. Exiting.")
        return

    fig1 = plot_hv_yield_sty(data)
    fig2 = plot_lambda_flowmap(data)

    if SAVE_FIGURES:
        save_dir = get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(save_dir / "Fig6_Comparison_HV.pdf", bbox_inches="tight")
        fig2.savefig(save_dir / "Fig7_Comparison_Lambda.pdf", bbox_inches="tight")
        print(f"\nFigures saved to {save_dir}")

    print_final_metrics(data["summary"], data["pareto_final"], data["visited"])

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
