from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FuncFormatter


# ============================================================
# Figure 5: representative sensor-feature maps
# ============================================================
# Main-text version of the complete SI feature maps. It reads the sa_08
# flow-mapping feature tables, aggregates the three saved windows at each
# hydrodynamic condition, and plots four representative features for all
# three reactor lengths with one horizontal shared colorbar below each feature column.


SCRIPT_DIR = Path(__file__).resolve().parent
SABO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = SABO_ROOT / "data" / "Data"

LENGTHS = ("1m", "2m", "4m")
SELECTED_FEATURES = ["Mean", "Std", "Spec_Energy", "Num_Peaks"]
AXIS_COLUMNS = ["phi_aq", "Q_total"]
REQUIRED_COLUMNS = ["Replicate_ID", "Q_aq", "Q_org", *AXIS_COLUMNS, *SELECTED_FEATURES]

DPI = 300
SHOW_FIGURES = True
# SHOW_FIGURES = False
SAVE_FIGURES = True
# SAVE_FIGURES = False
AGGREGATION_METHOD = "mean"  # "mean" or "median"; experiments use mean.
CMAP = "viridis"
CONTOURF_LEVELS = 21
CONTOUR_LINE_EVERY = 2

OUTPUT_DIR = SCRIPT_DIR / f"representative_feature_maps_{datetime.now().strftime('%Y%m%d%H%M%S')}"


FEATURE_TITLES = {
    "Mean": "Mean",
    "Std": "Std",
    "Spec_Energy": "Spec_Energy",
    "Num_Peaks": "Num_Peaks",
}

ROW_LABELS = {
    "1m": "(A) 1m",
    "2m": "(B) 2m",
    "4m": "(C) 4m",
}


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
    ax.tick_params(
        direction="in",
        length=4,
        width=1.0,
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
    )
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


def style_colorbar(cbar: plt.colorbar, vmin: float, vmax: float, feature: str) -> None:
    mid = (vmin + vmax) / 2.0
    cbar.set_ticks([vmin, mid, vmax])
    if feature == "Num_Peaks":
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0f}"))
    else:
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.2g}"))
    cbar.ax.tick_params(direction="in", length=3.5, width=0.8, labelsize=8)
    cbar.outline.set_linewidth(1.0)


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

    for col in [*AXIS_COLUMNS, *SELECTED_FEATURES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped_obj = df.groupby(AXIS_COLUMNS, as_index=False)[SELECTED_FEATURES]
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
    for feature in SELECTED_FEATURES:
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


def plot_figure5(
    data_by_length: Dict[str, pd.DataFrame],
    ranges: Dict[str, Tuple[float, float]],
) -> plt.Figure:
    fig = plt.figure(figsize=(11.4, 9.2), constrained_layout=False)
    grid = GridSpec(
        nrows=4,
        ncols=4,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 0.060],
        left=0.075,
        right=0.985,
        bottom=0.125,
        top=0.900,
        wspace=0.20,
        hspace=0.24,
    )


    column_mappables = {}
    for row_idx, length_label in enumerate(LENGTHS):
        df = data_by_length[length_label]
        row_axes = []

        for feature_idx, feature in enumerate(SELECTED_FEATURES):
            ax = fig.add_subplot(grid[row_idx, feature_idx])
            row_axes.append(ax)

            triangulation, x, y, z = contour_data_for_feature(df, feature)
            vmin, vmax = ranges[feature]
            norm = Normalize(vmin=vmin, vmax=vmax)
            levels = np.linspace(vmin, vmax, CONTOURF_LEVELS)

            contour = ax.tricontourf(
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
            column_mappables[feature] = contour

            style_map_axis(ax)
            if row_idx == 0:
                ax.set_title(FEATURE_TITLES[feature], pad=7, fontsize=12)
            if feature_idx == 0:
                ax.set_ylabel(r"$Q_{\mathrm{total}}$ (mL/min)", fontsize=12, labelpad=8)
            if row_idx == len(LENGTHS) - 1:
                ax.set_xlabel(r"$\phi_{\mathrm{aq}}$", fontsize=12, labelpad=7)

        row_left = min(ax.get_position().x0 for ax in row_axes)
        row_top = max(ax.get_position().y1 for ax in row_axes)
        fig.text(
            row_left - 0.07,
            row_top,
            ROW_LABELS[length_label],
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    for feature_idx, feature in enumerate(SELECTED_FEATURES):
        cax = fig.add_subplot(grid[3, feature_idx])
        cbar = fig.colorbar(
            column_mappables[feature],
            cax=cax,
            orientation="horizontal",
        )
        vmin, vmax = ranges[feature]
        style_colorbar(cbar, vmin, vmax, feature)

    return fig


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def save_aggregated_csv(data_by_length: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    combined = pd.concat(data_by_length.values(), ignore_index=True)
    combined.to_csv(
        out_dir / f"fig5_aggregated_{AGGREGATION_METHOD}_representative_features.csv",
        index=False,
    )


def write_readme(data_by_length: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    lines = [
        "# Fig. 5 representative feature maps",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Data root: {DATA_ROOT}",
        "",
        f"Selected features: {', '.join(SELECTED_FEATURES)}",
        f"Aggregation: {AGGREGATION_METHOD} of the three saved windows at each (phi_aq, Q_total) point.",
        "Color scaling: each feature column uses one global range shared by 1 m, 2 m, and 4 m.",
        "Layout: 3 reactor lengths x 4 representative features, with one horizontal colorbar below each feature column.",
        "",
        "Input tables:",
    ]
    for length, df in data_by_length.items():
        lines.append(f"- {length}: {df.attrs.get('table_path', '')}")
    lines.extend(
        [
            "",
            "Outputs:",
            "- Fig5_representative_feature_maps_v2.[png/pdf]",
            f"- fig5_aggregated_{AGGREGATION_METHOD}_representative_features.csv",
        ]
    )
    (out_dir / "README_fig5.txt").write_text("\n".join(lines), encoding="utf-8")


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
        fig = plot_figure5(data_by_length, ranges)
        try:
            fig.canvas.manager.set_window_title("Fig5 representative feature maps v2")
        except Exception:
            pass
        figures.append(fig)

        if SAVE_FIGURES:
            save_figure(fig, OUTPUT_DIR, "Fig5_representative_feature_maps_v2")

        if SHOW_FIGURES:
            plt.show()
    finally:
        for fig in figures:
            plt.close(fig)

    if SAVE_FIGURES:
        print(f"[OK] Figure 5 outputs saved to: {OUTPUT_DIR}")
    elif SHOW_FIGURES:
        print("[OK] Figure 5 preview shown; no files were saved.")
    else:
        print("[OK] Figure 5 generated; SHOW_FIGURES and SAVE_FIGURES are both False.")
    print(f"[AGGREGATION] {AGGREGATION_METHOD}")
    for length in LENGTHS:
        print(f"[DATA] {length}: {data_by_length[length].attrs.get('table_path', '')}")


if __name__ == "__main__":
    main()
