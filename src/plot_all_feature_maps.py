from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# ============================================================
# SI Figure 5: complete 16-feature heatmaps
# ============================================================
# This script is intentionally independent of earlier exploratory plotting
# scripts. It reads the flow-mapping feature tables generated during sa_08
# acquisition, averages the three saved windows at each hydrodynamic point,
# and plots all 16 sensor-feature maps for 1 m, 2 m, and 4 m reactors.


SCRIPT_DIR = Path(__file__).resolve().parent
SABO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = SABO_ROOT / "data" / "Data"

LENGTHS = ("1m", "2m", "4m")
FEATURE_COLS = [
    "Mean",
    "Std",
    "Skewness",
    "Kurtosis",
    "Dom_Freq",
    "Spec_Energy",
    "Wave_Eng_D1",
    "Entropy",
    "Hurst",
    "Cross_Rate",
    "Peak_Dist",
    "Num_Peaks",
    "Num_Valleys",
    "Avg_Peak_H",
    "Avg_Valley_H",
    "Cross_Count",
]

AXIS_COLUMNS = ["phi_aq", "Q_total"]
REQUIRED_COLUMNS = ["Replicate_ID", "Q_aq", "Q_org", *AXIS_COLUMNS, *FEATURE_COLS]

DPI = 300
# SHOW_FIGURES = True
SHOW_FIGURES = False
SAVE_FIGURES = True
# SAVE_FIGURES = False
AGGREGATION_METHOD = "mean"  # "mean" or "median"; experiments use mean.
CMAP = "viridis"
CONTOURF_LEVELS = 21
CONTOUR_LINE_EVERY = 2

OUTPUT_DIR = SCRIPT_DIR / f"all_feature_maps_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def configure_matplotlib() -> None:
    rc_updates = {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 1.4,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "stixsans",
    }
    plt.rcParams.update(rc_updates)


def style_map_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", length=4, width=1.0, labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax.tick_params(which="minor", direction="in", length=2.5, width=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.2, 1.0)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def center_to_edges(values: Iterable[float]) -> np.ndarray:
    centers = np.asarray(sorted(values), dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("Cannot build grid edges from empty coordinate values.")
    if centers.size == 1:
        delta = 0.1
        return np.array([centers[0] - delta / 2, centers[0] + delta / 2])

    midpoints = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def latest_folder_for_length(length_label: str) -> Path:
    pattern = f"*_{length_label} flow regime mapping"
    folders = sorted([p for p in DATA_ROOT.glob(pattern) if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"No folder found for {length_label}: {DATA_ROOT / pattern}")
    return folders[-1]


def feature_table_path(folder: Path) -> Path:
    candidates = [folder / "features_table.csv", folder / "features_table_v2.csv"]
    for path in candidates:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if all(col in header for col in REQUIRED_COLUMNS):
            return path
    existing = [str(path) for path in candidates if path.exists()]
    raise FileNotFoundError(
        f"No valid feature table found in {folder}. Existing candidates: {existing}"
    )


def load_length_data(length_label: str) -> pd.DataFrame:
    if AGGREGATION_METHOD not in {"mean", "median"}:
        raise ValueError('AGGREGATION_METHOD must be "mean" or "median".')

    folder = latest_folder_for_length(length_label)
    table_path = feature_table_path(folder)
    df = pd.read_csv(table_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table_path}: {missing}")

    for col in [*AXIS_COLUMNS, *FEATURE_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped_obj = df.groupby(AXIS_COLUMNS, as_index=False)[FEATURE_COLS]
    if AGGREGATION_METHOD == "mean":
        grouped = grouped_obj.mean(numeric_only=True)
    else:
        grouped = grouped_obj.median(numeric_only=True)
    grouped = grouped.sort_values(AXIS_COLUMNS).reset_index(drop=True)
    grouped.insert(0, "length", length_label)
    grouped.attrs["folder"] = str(folder)
    grouped.attrs["table_path"] = str(table_path)
    return grouped


def load_all_data() -> Dict[str, pd.DataFrame]:
    return {length: load_length_data(length) for length in LENGTHS}


def global_feature_ranges(data_by_length: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    for feature in FEATURE_COLS:
        values = np.concatenate(
            [df[feature].to_numpy(dtype=float) for df in data_by_length.values()]
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            ranges[feature] = (0.0, 1.0)
            continue
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            eps = 1e-9 if np.isclose(vmin, 0.0) else abs(vmin) * 1e-6
            vmin -= eps
            vmax += eps
        ranges[feature] = (vmin, vmax)
    return ranges


def contour_data_for_feature(
    df: pd.DataFrame,
    feature: str,
) -> Tuple[mtri.Triangulation, np.ndarray, np.ndarray, np.ndarray]:
    clean = df[["phi_aq", "Q_total", feature]].dropna().copy()
    if clean.empty:
        raise ValueError(f"No finite data available for feature: {feature}")

    x = clean["phi_aq"].to_numpy(dtype=float)
    y = clean["Q_total"].to_numpy(dtype=float)
    z = clean[feature].to_numpy(dtype=float)
    triangulation = mtri.Triangulation(x, y)
    return triangulation, x, y, z


def plot_all_length_feature_maps(
    data_by_length: Dict[str, pd.DataFrame],
    ranges: Dict[str, Tuple[float, float]],
) -> plt.Figure:
    fig = plt.figure(figsize=(24.0, 8.6), constrained_layout=False)
    outer_grid = GridSpec(
        nrows=1,
        ncols=3,
        figure=fig,
        left=0.035,
        right=0.992,
        bottom=0.105,
        top=0.885,
        wspace=0.145,
    )

    fig.supxlabel(r"$\phi_{\mathrm{aq}}$", y=0.030, fontsize=14)
    fig.supylabel(r"$Q_{\mathrm{total}}$ (mL/min)", x=0.010, fontsize=14)

    panel_labels = ("(A)", "(B)", "(C)")
    block_bounds = []
    for block_idx, length_label in enumerate(LENGTHS):
        df = data_by_length[length_label]
        block_grid = outer_grid[0, block_idx].subgridspec(
            nrows=4,
            ncols=4,
            wspace=0.18,
            hspace=0.30,
        )
        block_axes = []

        for idx, feature in enumerate(FEATURE_COLS):
            row = idx // 4
            col = idx % 4
            ax = fig.add_subplot(block_grid[row, col])
            block_axes.append(ax)

            triangulation, x, y, z = contour_data_for_feature(df, feature)
            vmin, vmax = ranges[feature]
            norm = Normalize(vmin=vmin, vmax=vmax)
            levels = np.linspace(vmin, vmax, CONTOURF_LEVELS)

            ax.tricontourf(
                triangulation,
                z,
                levels=levels,
                cmap=CMAP,
                norm=norm,
                extend="both",
            )
            ax.tricontour(
                triangulation,
                z,
                levels=levels[::CONTOUR_LINE_EVERY],
                colors="black",
                linewidths=0.45,
                alpha=0.32,
            )

            style_map_axis(ax)
            ax.set_title(feature, pad=4)

        block_left = min(ax.get_position().x0 for ax in block_axes)
        block_right = max(ax.get_position().x1 for ax in block_axes)
        block_bottom = min(ax.get_position().y0 for ax in block_axes)
        block_top = max(ax.get_position().y1 for ax in block_axes)
        block_bounds.append((block_left, block_right, block_bottom, block_top))
        header_y = block_top + 0.026
        fig.text(
            block_left,
            header_y,
            panel_labels[block_idx],
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            (block_left + block_right) / 2,
            header_y,
            length_label,
            ha="center",
            va="bottom",
            fontsize=14,
        )

    for left_bounds, right_bounds in zip(block_bounds[:-1], block_bounds[1:]):
        x_sep = (left_bounds[1] + right_bounds[0]) / 2
        y0 = min(left_bounds[2], right_bounds[2]) - 0.012
        y1 = max(left_bounds[3], right_bounds[3]) + 0.055
        fig.add_artist(
            plt.Line2D(
                [x_sep, x_sep],
                [y0, y1],
                transform=fig.transFigure,
                color="0.65",
                linewidth=0.8,
                alpha=0.75,
            )
        )

    return fig


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def save_aggregated_csv(data_by_length: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    combined = pd.concat(data_by_length.values(), ignore_index=True)
    combined.to_csv(
        out_dir / f"fig5_si_aggregated_{AGGREGATION_METHOD}_features.csv",
        index=False,
    )


def write_readme(data_by_length: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    lines = [
        "# Fig. 5 SI complete feature maps",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Data root: {DATA_ROOT}",
        "",
        f"Aggregation: {AGGREGATION_METHOD} of the three saved windows at each (phi_aq, Q_total) point.",
        "Color scaling: each feature uses one global range shared by 1 m, 2 m, and 4 m.",
        "",
        "Input tables:",
    ]
    for length, df in data_by_length.items():
        lines.append(f"- {length}: {df.attrs.get('table_path', '')}")
    lines.extend(
        [
            "",
            "Outputs:",
            "- FigS5_all_lengths_16_feature_heatmaps.[png/pdf]",
            f"- fig5_si_aggregated_{AGGREGATION_METHOD}_features.csv",
        ]
    )
    (out_dir / "README_fig5_si.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    data_by_length = load_all_data()
    ranges = global_feature_ranges(data_by_length)

    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_aggregated_csv(data_by_length, OUTPUT_DIR)
        write_readme(data_by_length, OUTPUT_DIR)

    figures = []
    try:
        fig = plot_all_length_feature_maps(data_by_length, ranges)
        try:
            fig.canvas.manager.set_window_title("FigS5 all lengths complete feature maps")
        except Exception:
            pass
        figures.append(fig)

        if SAVE_FIGURES:
            save_figure(fig, OUTPUT_DIR, "FigS5_all_lengths_16_feature_heatmaps")

        if SHOW_FIGURES:
            plt.show()
    finally:
        for fig in figures:
            plt.close(fig)

    if SAVE_FIGURES:
        print(f"[OK] Figure 5 SI outputs saved to: {OUTPUT_DIR}")
    elif SHOW_FIGURES:
        print("[OK] Figure 5 SI preview shown; no files were saved.")
    else:
        print("[OK] Figure 5 SI figures generated; SHOW_FIGURES and SAVE_FIGURES are both False.")
    print(f"[AGGREGATION] {AGGREGATION_METHOD}")
    for length in LENGTHS:
        print(f"[DATA] {length}: {data_by_length[length].attrs.get('table_path', '')}")

if __name__ == "__main__":
    main()
