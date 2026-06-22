from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator


# ============================================================
# Figure 4: representative photodiode waveforms
# ============================================================
# Main-text Figure 4. The selected conditions follow draft_md:
# six hydrodynamic points, each shown at 1 m, 2 m, and 4 m.


SCRIPT_DIR = Path(__file__).resolve().parent
SABO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = SABO_ROOT / "data" / "Data"

LENGTHS = ("1m", "2m", "4m")
REPRESENTATIVE_CONDITIONS = (
    (0.2, 0.7),
    (0.8, 0.5),
    (1.0, 0.3),
    (1.0, 0.5),
    (1.0, 0.6),
    (1.0, 0.7),
)
DEFAULT_REPLICATE = 2

N_SAMPLES = 16384
SAMPLE_RATE_HZ = 1000.0
DPI = 300
# SAVE_FIGURES = True
SAVE_FIGURES = False
SHOW_FIGURES = True
# SHOW_FIGURES = False

PANEL_ASPECT = 5.0  # width / height
NORMALIZE_TRACES = True
ROBUST_SCALE_PERCENTILE = 99.0
SHOW_X_TICK_LABELS = True
WAVEFORM_LINEWIDTH = 0.42
PLOT_WINDOW_START = 4096
PLOT_WINDOW_SAMPLES = 4096  # Use None to show the full 16384-point waveform.
YLIM_PERCENTILES = (0.5, 99.5)
YLIM_MARGIN_FRACTION = 0.10

OUTPUT_DIR = SCRIPT_DIR / f"representative_waveforms_{datetime.now().strftime('%Y%m%d%H%M%S')}"
FIGURE_STEM = "Fig4_representative_waveforms"

LENGTH_COLORS = {
    "1m": "#4BAFE8",
    "2m": "#1F78B4",
    "4m": "#0B4F8A",
}

FILENAME_RE = re.compile(
    r"^(?P<time>\d{6})_aq(?P<q_aq>\d+(?:\.\d+)?)-org(?P<q_org>\d+(?:\.\d+)?)-"
    r"phi(?P<phi>\d+(?:\.\d+)?)-total(?P<q_total>\d+(?:\.\d+)?)-(?P<rep>\d+)\.csv$"
)


@dataclass(frozen=True)
class WaveformRecord:
    length: str
    folder: Path
    filename: str
    q_aq: float
    q_org: float
    phi: float
    q_total: float
    replicate: int

    @property
    def path(self) -> Path:
        return self.folder / self.filename


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "axes.linewidth": 1.2,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
        }
    )


def latest_folder_for_length(length_label: str) -> Path:
    pattern = f"*_{length_label} flow regime mapping"
    folders = sorted(path for path in DATA_ROOT.glob(pattern) if path.is_dir())
    if not folders:
        raise FileNotFoundError(f"No folder found for {length_label}: {DATA_ROOT / pattern}")
    return folders[-1]


def build_waveform_index() -> Dict[Tuple[str, float, float, int], WaveformRecord]:
    index: Dict[Tuple[str, float, float, int], WaveformRecord] = {}
    for length in LENGTHS:
        folder = latest_folder_for_length(length)
        for path in sorted(folder.glob("*.csv")):
            if path.name.startswith("features_table"):
                continue
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            groups = match.groupdict()
            record = WaveformRecord(
                length=length,
                folder=folder,
                filename=path.name,
                q_aq=float(groups["q_aq"]),
                q_org=float(groups["q_org"]),
                phi=float(groups["phi"]),
                q_total=float(groups["q_total"]),
                replicate=int(groups["rep"]),
            )
            key = (length, round(record.q_total, 6), round(record.phi, 6), record.replicate)
            index[key] = record
    return index


def select_records(replicate: int) -> Dict[Tuple[float, float, str], WaveformRecord]:
    index = build_waveform_index()
    selected: Dict[Tuple[float, float, str], WaveformRecord] = {}
    missing: list[str] = []

    for q_total, phi in REPRESENTATIVE_CONDITIONS:
        for length in LENGTHS:
            key = (length, round(q_total, 6), round(phi, 6), replicate)
            record = index.get(key)
            if record is None:
                missing.append(f"{length}, Q_total={q_total}, phi={phi}, rep={replicate}")
                continue
            selected[(q_total, phi, length)] = record

    if missing:
        raise FileNotFoundError("Missing selected waveform files:\n  " + "\n  ".join(missing))
    return selected


def read_waveform(path: Path) -> np.ndarray:
    values: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[0]))
            except ValueError:
                continue
    waveform = np.asarray(values, dtype=float)
    if waveform.size == 0:
        raise ValueError(f"No numeric waveform values found: {path}")
    return waveform


def normalize_waveform(values: np.ndarray) -> np.ndarray:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return values
    centered = values - float(np.nanmedian(clean))
    if not NORMALIZE_TRACES:
        return centered

    scale = float(np.nanpercentile(np.abs(centered[np.isfinite(centered)]), ROBUST_SCALE_PERCENTILE))
    if not np.isfinite(scale) or np.isclose(scale, 0.0):
        scale = float(np.nanstd(centered))
    if not np.isfinite(scale) or np.isclose(scale, 0.0):
        scale = 1.0
    return centered / scale


def style_waveform_axis(
    ax: plt.Axes,
    show_tick_labels: bool,
    x_start: int,
    x_stop: int,
) -> None:
    x_mid = (x_start + x_stop) / 2.0
    ax.set_xlim(x_start, x_stop)
    ax.set_xticks([x_start, x_mid, x_stop])
    if show_tick_labels:
        ax.set_xticklabels([f"{int(x_start)}", f"{int(x_mid)}", f"{int(x_stop)}"])
    ax.xaxis.set_minor_locator(FixedLocator([(x_start + x_mid) / 2.0, (x_mid + x_stop) / 2.0]))
    ax.tick_params(axis="x", direction="in", length=4.0, width=0.9, labelbottom=show_tick_labels)
    ax.tick_params(axis="x", which="minor", direction="in", length=2.4, width=0.8)
    ax.tick_params(axis="y", left=False, right=False, labelleft=False)
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_box_aspect(1.0 / PANEL_ASPECT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def row_label(q_total: float, phi: float) -> str:
    return (
        rf"$Q_{{\mathrm{{total}}}}$={q_total:.1f}"
        + "\n"
        + rf"$\phi_{{\mathrm{{aq}}}}$={phi:.1f}"
    )


def condition_label(q_total: float, phi: float) -> str:
    return f"Q_total={q_total:.1f}, phi_aq={phi:.1f}"


def load_selected_waveforms(
    selected: Dict[Tuple[float, float, str], WaveformRecord],
) -> Dict[Tuple[float, float, str], np.ndarray]:
    data: Dict[Tuple[float, float, str], np.ndarray] = {}
    for key, record in selected.items():
        data[key] = normalize_waveform(read_waveform(record.path))
    return data


def window_waveform(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if PLOT_WINDOW_SAMPLES is None:
        start = 0
        stop = waveform.size
    else:
        n_window = min(int(PLOT_WINDOW_SAMPLES), waveform.size)
        start = max(0, min(int(PLOT_WINDOW_START), waveform.size - n_window))
        stop = start + n_window
    return np.arange(start, stop, dtype=float), waveform[start:stop], start, stop


def compute_trace_ylim(waveform: np.ndarray) -> Tuple[float, float]:
    values = waveform[np.isfinite(waveform)]
    if values.size == 0:
        return (-1.0, 1.0)
    y_min, y_max = np.nanpercentile(values, YLIM_PERCENTILES)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or np.isclose(y_min, y_max):
        center = float(np.nanmedian(values))
        spread = float(np.nanstd(values))
        if not np.isfinite(spread) or np.isclose(spread, 0.0):
            spread = 1.0
        y_min = center - spread
        y_max = center + spread
    margin = (float(y_max) - float(y_min)) * YLIM_MARGIN_FRACTION
    if not np.isfinite(margin) or np.isclose(margin, 0.0):
        margin = 0.1
    return (float(y_min) - margin, float(y_max) + margin)


def plot_figure4(
    selected: Dict[Tuple[float, float, str], WaveformRecord],
    waveforms: Dict[Tuple[float, float, str], np.ndarray],
) -> plt.Figure:
    fig, axes = plt.subplots(
        nrows=len(REPRESENTATIVE_CONDITIONS),
        ncols=len(LENGTHS),
        figsize=(12.6, 6.2),
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.940, bottom=0.075, wspace=0.16, hspace=0.50)

    for row_idx, (q_total, phi) in enumerate(REPRESENTATIVE_CONDITIONS):
        for col_idx, length in enumerate(LENGTHS):
            ax = axes[row_idx, col_idx]
            key = (q_total, phi, length)
            x_values, waveform, x_start, x_stop = window_waveform(waveforms[key])

            ax.plot(
                x_values,
                waveform,
                color=LENGTH_COLORS[length],
                linewidth=WAVEFORM_LINEWIDTH,
                solid_capstyle="butt",
            )
            ax.set_ylim(*compute_trace_ylim(waveform))
            style_waveform_axis(ax, SHOW_X_TICK_LABELS, x_start, x_stop)

            if row_idx == 0:
                ax.set_title(length, pad=8, fontsize=12, fontweight="bold")

            if col_idx == 0:
                ax.text(
                    -0.13,
                    0.50,
                    row_label(q_total, phi),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                )

    return fig


def selected_records_table(selected: Dict[Tuple[float, float, str], WaveformRecord]) -> pd.DataFrame:
    rows = []
    for q_total, phi in REPRESENTATIVE_CONDITIONS:
        for length in LENGTHS:
            record = selected[(q_total, phi, length)]
            rows.append(
                {
                    "condition": condition_label(q_total, phi),
                    "Q_total": q_total,
                    "phi_aq": phi,
                    "length": length,
                    "replicate": record.replicate,
                    "Q_aq": record.q_aq,
                    "Q_org": record.q_org,
                    "filename": record.filename,
                    "path": str(record.path),
                }
            )
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, replicate: int, selected: Dict[Tuple[float, float, str], WaveformRecord]) -> None:
    lines = [
        "# Fig. 4 representative waveforms",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Data root: {DATA_ROOT}",
        f"Replicate: rep{replicate}",
        f"Layout: {len(REPRESENTATIVE_CONDITIONS)} hydrodynamic conditions x {len(LENGTHS)} reactor lengths.",
        f"Waveform length: {N_SAMPLES} points at {SAMPLE_RATE_HZ:g} Hz.",
        f"Plotted window: samples {PLOT_WINDOW_START} to {PLOT_WINDOW_START + PLOT_WINDOW_SAMPLES if PLOT_WINDOW_SAMPLES is not None else N_SAMPLES}.",
        f"Panel aspect ratio: {PANEL_ASPECT:g}:1 (width:height).",
        f"Trace normalization: {'median-centered and robust-scaled per waveform' if NORMALIZE_TRACES else 'median-centered only'}.",
        f"Waveform line width: {WAVEFORM_LINEWIDTH:g} pt.",
        "Y limits: automatically set per panel from robust percentiles.",
        f"Axes: y-axis ticks and labels are hidden; x-axis keeps start, middle, and end major ticks with one minor tick between adjacent majors; x tick labels are {'shown' if SHOW_X_TICK_LABELS else 'hidden'}.",
        "",
        "Conditions:",
    ]
    for q_total, phi in REPRESENTATIVE_CONDITIONS:
        lines.append(f"- Q_total={q_total:.1f}, phi_aq={phi:.1f}")
    lines.extend(
        [
            "",
            "Outputs:",
            f"- {FIGURE_STEM}.png",
            f"- {FIGURE_STEM}.pdf",
            "- fig4_selected_waveforms.csv",
            "",
            "Input files:",
        ]
    )
    for q_total, phi in REPRESENTATIVE_CONDITIONS:
        for length in LENGTHS:
            record = selected[(q_total, phi, length)]
            lines.append(f"- {length}, Q_total={q_total:.1f}, phi_aq={phi:.1f}: {record.path}")
    (out_dir / "README_fig4.txt").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    fig: plt.Figure,
    selected: Dict[Tuple[float, float, str], WaveformRecord],
    out_dir: Path,
    replicate: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_records_table(selected).to_csv(out_dir / "fig4_selected_waveforms.csv", index=False)
    write_readme(out_dir, replicate, selected)
    fig.savefig(out_dir / f"{FIGURE_STEM}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{FIGURE_STEM}.pdf", bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw SABO main-text Figure 4 representative waveforms.")
    parser.add_argument("--replicate", type=int, default=DEFAULT_REPLICATE, help="Waveform replicate to plot.")
    parser.add_argument("--show", action="store_true", help="Show the figure window after generation.")
    parser.add_argument("--no-save", action="store_true", help="Preview only; do not save outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    selected = select_records(args.replicate)
    waveforms = load_selected_waveforms(selected)

    figures = []
    try:
        fig = plot_figure4(selected, waveforms)
        try:
            fig.canvas.manager.set_window_title("Fig4 representative waveforms")
        except Exception:
            pass
        figures.append(fig)

        if SAVE_FIGURES and not args.no_save:
            save_outputs(fig, selected, OUTPUT_DIR, args.replicate)

        if SHOW_FIGURES or args.show:
            plt.show()
    finally:
        for fig in figures:
            plt.close(fig)

    if SAVE_FIGURES and not args.no_save:
        print(f"[OK] Figure 4 outputs saved to: {OUTPUT_DIR}")
    else:
        print("[OK] Figure 4 generated without saving outputs.")
    print(f"[DATA] {DATA_ROOT}")
    print(f"[REPLICATE] rep{args.replicate}")
    print(f"[NORMALIZE] {NORMALIZE_TRACES}")


if __name__ == "__main__":
    main()
