"""Biomechanical gait assessment pipeline for frame-by-frame lower-limb kinematics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

INPUT_FILE = "input.csv"
OUTPUT_DIR = "results"

FRAME_COL = "Frame"
ANGLE_COLUMNS = [
    "LEFT HIP (degrees)",
    "LEFT KNEE (degrees)",
    "LEFT ANKLE (degrees)",
    "RIGHT HIP (degrees)",
    "RIGHT KNEE (degrees)",
    "RIGHT ANKLE (degrees)",
]
STEP_WIDTH_COL = "STEP WIDTH (px)"
REQUIRED_COLUMNS = [FRAME_COL, *ANGLE_COLUMNS, STEP_WIDTH_COL]

SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 2
ROLLING_WINDOW = 5

ASYMMETRY_PERCENT_THRESHOLD = 15.0
STEP_WIDTH_CV_THRESHOLD = 20.0
INSTABILITY_MAD_MULTIPLIER = 6.0


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def load_data(input_file: str) -> pd.DataFrame:
    """Load gait kinematics CSV into a DataFrame.

    Raises:
        FileNotFoundError: if the input file does not exist.
        ValueError: if the file is empty or unreadable as tabular data.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty.") from exc

    if df.empty:
        raise ValueError("Input CSV has headers but no rows.")

    LOGGER.info("Loaded %d rows from %s", len(df), input_file)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Validate schema, duplicates, and numeric convertibility."""
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
        "missing_columns": [],
        "duplicate_frames": 0,
        "non_numeric_counts": {},
        "missing_value_counts": {},
    }

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    validation["missing_columns"] = missing_cols
    if missing_cols:
        validation["is_valid"] = False
        validation["errors"].append(f"Missing required columns: {missing_cols}")
        return validation

    duplicate_frames = int(df[FRAME_COL].duplicated().sum())
    validation["duplicate_frames"] = duplicate_frames
    if duplicate_frames > 0:
        validation["warnings"].append(
            f"Found {duplicate_frames} duplicated frame index values."
        )

    for col in REQUIRED_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        invalid_count = int(coerced.isna().sum() - df[col].isna().sum())
        if invalid_count > 0:
            validation["warnings"].append(
                f"Column '{col}' has {invalid_count} non-numeric values coerced to NaN."
            )
        validation["non_numeric_counts"][col] = max(0, invalid_count)
        validation["missing_value_counts"][col] = int(coerced.isna().sum())

    if validation["row_count"] == 0:
        validation["is_valid"] = False
        validation["errors"].append("No rows found in dataset.")

    return validation


def _smooth_series(series: pd.Series) -> pd.Series:
    """Smooth a series using Savitzky-Golay when feasible, else rolling mean."""
    n = len(series)
    if n >= SAVGOL_WINDOW and SAVGOL_WINDOW % 2 == 1 and SAVGOL_WINDOW > SAVGOL_POLYORDER:
        return pd.Series(
            savgol_filter(series.to_numpy(), window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER),
            index=series.index,
        )
    window = min(ROLLING_WINDOW, max(1, n))
    return series.rolling(window=window, min_periods=1, center=True).mean()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, coerce numerics, impute missing values, and smooth signals."""
    proc = df.copy()

    for col in REQUIRED_COLUMNS:
        proc[col] = pd.to_numeric(proc[col], errors="coerce")

    proc = proc.sort_values(FRAME_COL).drop_duplicates(subset=[FRAME_COL], keep="first")
    proc = proc.reset_index(drop=True)

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != FRAME_COL]
    proc[numeric_cols] = proc[numeric_cols].interpolate(method="linear", limit_direction="both")
    proc[numeric_cols] = proc[numeric_cols].ffill().bfill()

    for col in numeric_cols:
        proc[col] = _smooth_series(proc[col])

    return proc


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive, ROM, smoothness, asymmetry, and variability metrics."""
    metrics: List[dict] = []
    signal_cols = [*ANGLE_COLUMNS, STEP_WIDTH_COL]

    for col in signal_cols:
        series = df[col].astype(float)
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            metrics.append({"metric": f"{col} | {stat}", "value": float(desc[stat])})

        rom = float(series.max() - series.min())
        smoothness = float(series.diff().abs().dropna().mean()) if len(series) > 1 else 0.0
        metrics.append({"metric": f"{col} | ROM", "value": rom})
        metrics.append({"metric": f"{col} | mean_abs_frame_change", "value": smoothness})

    lr_pairs = {
        "HIP": ("LEFT HIP (degrees)", "RIGHT HIP (degrees)"),
        "KNEE": ("LEFT KNEE (degrees)", "RIGHT KNEE (degrees)"),
        "ANKLE": ("LEFT ANKLE (degrees)", "RIGHT ANKLE (degrees)"),
    }

    for joint, (left_col, right_col) in lr_pairs.items():
        left = df[left_col].astype(float)
        right = df[right_col].astype(float)
        abs_diff = (left - right).abs()
        denom = ((left.abs() + right.abs()) / 2.0).replace(0, np.nan)
        pct_diff = (abs_diff / denom) * 100.0
        pct_diff = pct_diff.replace([np.inf, -np.inf], np.nan)

        metrics.append(
            {"metric": f"{joint} asymmetry | abs_mean_difference", "value": float(abs_diff.mean())}
        )
        metrics.append(
            {
                "metric": f"{joint} asymmetry | pct_mean_difference",
                "value": float(pct_diff.dropna().mean()) if pct_diff.notna().any() else np.nan,
            }
        )

    step_width = df[STEP_WIDTH_COL].astype(float)
    sw_mean = float(step_width.mean())
    sw_std = float(step_width.std(ddof=1)) if len(step_width) > 1 else 0.0
    sw_cv = (sw_std / sw_mean * 100.0) if sw_mean != 0 else np.nan
    metrics.append({"metric": "STEP WIDTH | coefficient_of_variation_pct", "value": float(sw_cv) if not np.isnan(sw_cv) else np.nan})

    return pd.DataFrame(metrics)


def detect_asymmetry(metrics_df: pd.DataFrame) -> dict:
    """Flag unusually high bilateral asymmetry based on percentage thresholds."""
    results = {
        "threshold_pct": ASYMMETRY_PERCENT_THRESHOLD,
        "flags": {},
        "summary": [],
    }

    for joint in ["HIP", "KNEE", "ANKLE"]:
        metric_name = f"{joint} asymmetry | pct_mean_difference"
        row = metrics_df.loc[metrics_df["metric"] == metric_name, "value"]
        value = float(row.iloc[0]) if not row.empty else np.nan
        is_flagged = bool(np.isfinite(value) and value > ASYMMETRY_PERCENT_THRESHOLD)
        results["flags"][joint] = {
            "pct_mean_difference": value,
            "flagged": is_flagged,
        }
        if is_flagged:
            results["summary"].append(
                f"{joint} asymmetry exceeds threshold ({value:.2f}% > {ASYMMETRY_PERCENT_THRESHOLD:.2f}%)."
            )

    if not results["summary"]:
        results["summary"].append("No asymmetry metrics exceeded configured thresholds.")

    return results


def detect_signal_instability(df: pd.DataFrame) -> dict:
    """Detect abrupt jumps via robust first-difference thresholds and step-width instability."""
    instability = {
        "abrupt_jumps": {},
        "step_width_cv_pct": np.nan,
        "step_width_unstable": False,
        "step_width_threshold_pct": STEP_WIDTH_CV_THRESHOLD,
        "summary": [],
    }

    signal_cols = [*ANGLE_COLUMNS, STEP_WIDTH_COL]
    for col in signal_cols:
        diffs = df[col].astype(float).diff().dropna().abs()
        if diffs.empty:
            instability["abrupt_jumps"][col] = {
                "threshold": np.nan,
                "jump_count": 0,
                "jump_frames": [],
            }
            continue

        med = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - med)))
        robust_sigma = 1.4826 * mad
        threshold = med + INSTABILITY_MAD_MULTIPLIER * robust_sigma

        jump_mask = diffs > threshold
        jump_frames = df.loc[jump_mask.index[jump_mask], FRAME_COL].astype(int).tolist()
        jump_count = int(jump_mask.sum())

        instability["abrupt_jumps"][col] = {
            "threshold": float(threshold),
            "jump_count": jump_count,
            "jump_frames": jump_frames,
        }

        if jump_count > 0:
            instability["summary"].append(f"{col}: {jump_count} abrupt jump(s) detected.")

    sw = df[STEP_WIDTH_COL].astype(float)
    sw_mean = float(sw.mean())
    sw_std = float(sw.std(ddof=1)) if len(sw) > 1 else 0.0
    sw_cv = (sw_std / sw_mean * 100.0) if sw_mean != 0 else np.nan
    instability["step_width_cv_pct"] = float(sw_cv) if np.isfinite(sw_cv) else np.nan
    instability["step_width_unstable"] = bool(np.isfinite(sw_cv) and sw_cv > STEP_WIDTH_CV_THRESHOLD)
    if instability["step_width_unstable"]:
        instability["summary"].append(
            f"Step width variability exceeds threshold ({sw_cv:.2f}% > {STEP_WIDTH_CV_THRESHOLD:.2f}%)."
        )

    if not instability["summary"]:
        instability["summary"].append("No instability flags triggered under current thresholds.")

    return instability


def _plot_pair(df: pd.DataFrame, left_col: str, right_col: str, title: str, ylabel: str, output_file: Path) -> None:
    """Create a clean two-line plot for paired left/right signals."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(df[FRAME_COL], df[left_col], label="Left", linewidth=2)
    ax.plot(df[FRAME_COL], df[right_col], label="Right", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def plot_signals(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate publication-style plots for hip, knee, ankle, and step width."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_pair(
        df,
        "LEFT HIP (degrees)",
        "RIGHT HIP (degrees)",
        "Hip Angles Across Frames",
        "Angle (degrees)",
        output_dir / "hip_angles.png",
    )
    _plot_pair(
        df,
        "LEFT KNEE (degrees)",
        "RIGHT KNEE (degrees)",
        "Knee Angles Across Frames",
        "Angle (degrees)",
        output_dir / "knee_angles.png",
    )
    _plot_pair(
        df,
        "LEFT ANKLE (degrees)",
        "RIGHT ANKLE (degrees)",
        "Ankle Angles Across Frames",
        "Angle (degrees)",
        output_dir / "ankle_angles.png",
    )

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(df[FRAME_COL], df[STEP_WIDTH_COL], color="tab:purple", linewidth=2)
    ax.set_title("Step Width Across Frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Step Width (px)")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_dir / "step_width.png")
    plt.close(fig)


def generate_report(
    validation: dict,
    metrics_df: pd.DataFrame,
    asymmetry: dict,
    instability: dict,
) -> str:
    """Create a conservative technical markdown report without clinical diagnosis."""
    lines: List[str] = []
    lines.append("# Gait Analysis Report")
    lines.append("")
    lines.append("## Data Validation")
    lines.append(f"- Rows processed: {validation.get('row_count', 0)}")
    lines.append(f"- Duplicate frames detected: {validation.get('duplicate_frames', 0)}")

    if validation.get("errors"):
        lines.append("- Errors:")
        for err in validation["errors"]:
            lines.append(f"  - {err}")
    if validation.get("warnings"):
        lines.append("- Warnings:")
        for warning in validation["warnings"]:
            lines.append(f"  - {warning}")
    if not validation.get("errors") and not validation.get("warnings"):
        lines.append("- No validation issues detected.")

    lines.append("")
    lines.append("## Key Metrics")

    key_metrics = [
        "HIP asymmetry | abs_mean_difference",
        "HIP asymmetry | pct_mean_difference",
        "KNEE asymmetry | abs_mean_difference",
        "KNEE asymmetry | pct_mean_difference",
        "ANKLE asymmetry | abs_mean_difference",
        "ANKLE asymmetry | pct_mean_difference",
        "STEP WIDTH | coefficient_of_variation_pct",
    ]
    for metric_name in key_metrics:
        row = metrics_df.loc[metrics_df["metric"] == metric_name, "value"]
        if not row.empty:
            value = row.iloc[0]
            lines.append(f"- {metric_name}: {value:.4f}" if np.isfinite(value) else f"- {metric_name}: NaN")

    lines.append("")
    lines.append("## Asymmetry Flags")
    for item in asymmetry.get("summary", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Signal Instability Flags")
    for item in instability.get("summary", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Technical Interpretation")
    lines.append(
        "- This report provides technical signal-based descriptors (distribution, variability, symmetry, and temporal smoothness) only."
    )
    lines.append(
        "- Findings should be interpreted in context of acquisition conditions and measurement uncertainty; this script does not perform diagnosis."
    )

    return "\n".join(lines) + "\n"


def save_results(metrics_df: pd.DataFrame, report_text: str, output_dir: Path) -> None:
    """Persist tabular metrics and markdown report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    """Run full gait analysis pipeline from input CSV to saved results."""
    LOGGER.info("Starting gait analysis pipeline")

    try:
        raw_df = load_data(INPUT_FILE)
    except Exception as exc:
        LOGGER.error("Failed to load input data: %s", exc)
        raise

    validation = validate_data(raw_df)
    if not validation.get("is_valid", False):
        msg = "; ".join(validation.get("errors", ["Unknown validation failure."]))
        LOGGER.error("Validation failed: %s", msg)
        raise ValueError(msg)

    clean_df = preprocess_data(raw_df)
    if clean_df.empty:
        raise ValueError("No usable data after preprocessing.")

    metrics_df = compute_metrics(clean_df)
    asymmetry = detect_asymmetry(metrics_df)
    instability = detect_signal_instability(clean_df)

    output_dir = Path(OUTPUT_DIR)
    plot_signals(clean_df, output_dir)
    report_text = generate_report(validation, metrics_df, asymmetry, instability)
    save_results(metrics_df, report_text, output_dir)

    LOGGER.info("Analysis complete. Outputs saved to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
