#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.fftpack as fft
import scipy.signal as signal
import scipy.stats as stats
from hurst import compute_Hc
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# Window-length robustness analysis using final cold-flow
# feature definitions.
# ============================================================
# The feature formulas, names, and order follow the final cold-flow
# acquisition script:
#   sa_08/chatgpt.../sa_08_gpt...v2.py::calculate_all_features
#
# Only the fixed 16384-point guard is relaxed here so that the same
# formulas can be evaluated over multiple clip lengths.


SCRIPT_DIR = Path(__file__).resolve().parent
SABO_ROOT = SCRIPT_DIR.parent
DATA_DIR = SABO_ROOT / "data" / "window_length_pretest"

FS = 1000.0
WINDOW_POWERS = list(range(5, 16))
WINDOW_LENGTHS = [2**p for p in WINDOW_POWERS]
SELECTED_WINDOW = 2**14

SOURCE_FILES = [
    ("Pretest 1", "08_start150300_clip32768.txt"),
    ("Pretest 2", "09_start55700_clip32768.txt"),
    ("Pretest 3", "10_start47100_clip32768.txt"),
    ("Pretest 4", "11_start300_clip32768.txt"),
]

FEATURE_KEYS = [
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

FEATURE_TITLES = {
    "Mean": "Mean",
    "Std": "Std",
    "Skewness": "Skewness",
    "Kurtosis": "Kurtosis",
    "Dom_Freq": "Dom_Freq",
    "Spec_Energy": "Spec_Energy",
    "Wave_Eng_D1": "Wave_Eng_D1",
    "Entropy": "Entropy",
    "Hurst": "Hurst",
    "Cross_Rate": "Cross_Rate",
    "Peak_Dist": "Peak_Dist",
    "Num_Peaks": "Num_Peaks",
    "Num_Valleys": "Num_Valleys",
    "Avg_Peak_H": "Avg_Peak_H",
    "Avg_Valley_H": "Avg_Valley_H",
    "Cross_Count": "Cross_Count",
}

# Palette adapted from the manuscript plotting utilities in sa_11.
TRACE_COLORS = {
    "Pretest 1": "#46b6ff",
    "Pretest 2": "#8b5cf6",
    "Pretest 3": "#f59e0b",
    "Pretest 4": "#62c98f",
}
TRACE_MARKERS = {
    "Pretest 1": "o",
    "Pretest 2": "s",
    "Pretest 3": "^",
    "Pretest 4": "D",
}

OUTPUT_COMBINED_CSV = SCRIPT_DIR / "SI_window_length_features.csv"
OUTPUT_FIG_PNG = SCRIPT_DIR / "SI_window_length_analysis.png"
OUTPUT_FIG_PDF = SCRIPT_DIR / "SI_window_length_analysis.pdf"


def configure_matplotlib() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    rc_updates = {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 13,
        "axes.linewidth": 1.6,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if "Times New Roman" in available_fonts:
        rc_updates.update({
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        })
    plt.rcParams.update(rc_updates)


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(direction="in", length=4.5, width=1.2)
    ax.tick_params(which="minor", direction="in", length=2.8, width=0.9)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if grid:
        ax.grid(alpha=0.22, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def calculate_all_features_variable_window(data: np.ndarray) -> Dict[str, float]:
    """Final cold-flow feature formulas evaluated for a variable clip length."""
    feats: Dict[str, float] = {}
    feats["Mean"] = float(np.mean(data))
    feats["Std"] = float(np.std(data))
    feats["Skewness"] = float(stats.skew(data))
    feats["Kurtosis"] = float(stats.kurtosis(data))

    data_centered = data - np.mean(data)
    n_points = len(data_centered)
    fft_vals = fft.fft(data_centered)
    fft_freqs = fft.fftfreq(n_points, 1 / FS)
    power_spectrum = np.abs(fft_vals[: n_points // 2]) / n_points
    idx_peak = int(np.argmax(power_spectrum))
    feats["Dom_Freq"] = float(fft_freqs[idx_peak])
    feats["Spec_Energy"] = float(power_spectrum[idx_peak])

    try:
        coeffs = pywt.wavedec(data, "db4", level=3)
        feats["Wave_Eng_D1"] = float(np.sum(np.square(coeffs[-1])) / len(coeffs[-1]))
    except Exception:
        feats["Wave_Eng_D1"] = 0.0

    try:
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]
        feats["Entropy"] = float(-np.sum(hist * np.log2(hist)))
    except Exception:
        feats["Entropy"] = 0.0

    try:
        h_value, _, _ = compute_Hc(data, kind="change", simplified=False)
        feats["Hurst"] = float(h_value)
    except Exception:
        feats["Hurst"] = 0.0

    zero_crossings = np.where(np.diff(np.sign(data - np.mean(data))))[0]
    feats["Cross_Rate"] = float(len(zero_crossings) / (len(data) / FS))
    feats["Cross_Count"] = float(len(zero_crossings))

    peaks, _ = signal.find_peaks(data, distance=20, prominence=0.1)
    valleys, _ = signal.find_peaks(-data, distance=20, prominence=0.1)
    feats["Num_Peaks"] = float(len(peaks))
    feats["Num_Valleys"] = float(len(valleys))
    feats["Peak_Dist"] = float(np.mean(np.diff(peaks))) if len(peaks) > 1 else 0.0
    feats["Avg_Peak_H"] = float(np.mean(data[peaks])) if len(peaks) > 0 else 0.0
    feats["Avg_Valley_H"] = float(np.mean(data[valleys])) if len(valleys) > 0 else 0.0

    return {key: feats[key] for key in FEATURE_KEYS}


def compute_feature_table() -> pd.DataFrame:
    rows: List[Dict[str, float | str | int]] = []
    for trace_label, filename in SOURCE_FILES:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing source waveform: {file_path}")

        raw_data = np.loadtxt(file_path)
        for clip_len in WINDOW_LENGTHS:
            if len(raw_data) < clip_len:
                print(f"[Skip] {filename}: length {len(raw_data)} < {clip_len}")
                continue

            segment = raw_data[:clip_len]
            features = calculate_all_features_variable_window(segment)
            row: Dict[str, float | str | int] = {
                "Trace_Label": trace_label,
                "Source_File": filename,
                "Clip_Length": int(clip_len),
                "Clip_Power": int(np.log2(clip_len)),
                "Duration_s": float(clip_len / FS),
            }
            row.update(features)
            rows.append(row)

    if not rows:
        raise RuntimeError("No feature rows were generated.")

    df = pd.DataFrame(rows)
    ordered_cols = ["Trace_Label", "Source_File", "Clip_Length", "Clip_Power", "Duration_s", *FEATURE_KEYS]
    return df[ordered_cols]


def save_per_trace_csvs(df: pd.DataFrame) -> None:
    for trace_label, filename in SOURCE_FILES:
        sub = df.loc[df["Source_File"] == filename].copy()
        stem = Path(filename).stem
        out_path = SCRIPT_DIR / f"SI_window_length_{stem}_features.csv"
        sub.to_csv(out_path, index=False)
        print(f"[CSV] {out_path}")


def plot_window_length_robustness(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 4, figsize=(15.2, 10.8), sharex=True)
    axes_flat = axes.ravel()

    for ax, feature in zip(axes_flat, FEATURE_KEYS):
        for trace_label, _ in SOURCE_FILES:
            sub = df.loc[df["Trace_Label"] == trace_label].sort_values("Clip_Length")
            ax.plot(
                sub["Clip_Length"],
                sub[feature],
                label=trace_label,
                color=TRACE_COLORS[trace_label],
                marker=TRACE_MARKERS[trace_label],
                markersize=4.4,
                linewidth=1.7,
                alpha=0.96,
            )

        ax.axvline(
            SELECTED_WINDOW,
            color="#4b5563",
            linestyle="--",
            linewidth=1.2,
            alpha=0.82,
        )
        ax.set_xscale("log", base=2)
        ax.set_title(FEATURE_TITLES[feature], pad=6)
        style_axes(ax, grid=True)

    xticks = WINDOW_LENGTHS
    xticklabels = [rf"$2^{{{power}}}$" for power in WINDOW_POWERS]
    for ax in axes_flat:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.045, top=0.925, wspace=0.20, hspace=0.30)
    return fig


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pywt")
    configure_matplotlib()

    df = compute_feature_table()
    df.to_csv(OUTPUT_COMBINED_CSV, index=False)
    print(f"[CSV] {OUTPUT_COMBINED_CSV}")
    save_per_trace_csvs(df)

    fig = plot_window_length_robustness(df)
    fig.savefig(OUTPUT_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"[Figure] {OUTPUT_FIG_PNG}")
    print(f"[Figure] {OUTPUT_FIG_PDF}")
    print("[Done] Window-length robustness analysis completed.")


if __name__ == "__main__":
    main()
