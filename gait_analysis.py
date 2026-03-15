"""Biomechanical gait assessment pipeline for frame-by-frame lower-limb kinematics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
INSTABILITY_FALLBACK_MULTIPLIER = 3.0

MAX_MISSING_RATIO = 0.3
MIN_REQUIRED_ROWS = 10
OUTLIER_ROBUST_Z_THRESHOLD = 3.5
SEVERE_DUPLICATE_FRAME_RATIO = 0.5


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


LR_PAIRS: Dict[str, Tuple[str, str]] = {
    "HIP": ("LEFT HIP (degrees)", "RIGHT HIP (degrees)"),
    "KNEE": ("LEFT KNEE (degrees)", "RIGHT KNEE (degrees)"),
    "ANKLE": ("LEFT ANKLE (degrees)", "RIGHT ANKLE (degrees)"),
}


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
    """Validate schema and data quality for reliable downstream analysis."""
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
        "missing_columns": [],
        "duplicate_frames": 0,
        "duplicate_frame_ratio": 0.0,
        "frame_non_numeric_count": 0,
        "frame_missing_count": 0,
        "frame_is_monotonic": True,
        "non_numeric_counts": {},
        "missing_value_counts": {},
        "missing_ratios": {},
        "max_missing_ratio_allowed": MAX_MISSING_RATIO,
        "min_required_rows": MIN_REQUIRED_ROWS,
        "valid_rows_after_coercion": 0,
    }

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    validation["missing_columns"] = missing_cols
    if missing_cols:
        validation["is_valid"] = False
        validation["errors"].append(f"Missing required columns: {missing_cols}")
        return validation

    frame_numeric = pd.to_numeric(df[FRAME_COL], errors="coerce")
    frame_non_numeric = int(frame_numeric.isna().sum() - df[FRAME_COL].isna().sum())
    frame_missing = int(frame_numeric.isna().sum())
    validation["frame_non_numeric_count"] = max(0, frame_non_numeric)
    validation["frame_missing_count"] = frame_missing

    if frame_non_numeric > 0:
        validation["warnings"].append(
            f"Frame column has {frame_non_numeric} non-numeric values coerced to NaN."
        )

    frame_original_order = frame_numeric.dropna()
    frame_is_monotonic = bool(frame_original_order.is_monotonic_increasing)
    validation["frame_is_monotonic"] = frame_is_monotonic
    if not frame_is_monotonic:
        validation["warnings"].append(
            "Frame values are not monotonic in original input order; temporal ordering may be inconsistent."
        )

    duplicate_frames = int(frame_numeric.dropna().duplicated().sum())
    validation["duplicate_frames"] = duplicate_frames
    frame_count = max(len(df), 1)
    duplicate_ratio = duplicate_frames / frame_count
    validation["duplicate_frame_ratio"] = duplicate_ratio
    if duplicate_frames > 0:
        validation["warnings"].append(
            f"Found {duplicate_frames} duplicated frame index values ({duplicate_ratio:.1%})."
        )
        if duplicate_ratio > SEVERE_DUPLICATE_FRAME_RATIO:
            validation["is_valid"] = False
            validation["errors"].append(
                "Duplicate frame ratio is too high for stable temporal analysis."
            )

    coerced_df = df.copy()
    for col in REQUIRED_COLUMNS:
        coerced = pd.to_numeric(coerced_df[col], errors="coerce")
        invalid_count = int(coerced.isna().sum() - coerced_df[col].isna().sum())
        missing_count = int(coerced.isna().sum())
        missing_ratio = float(missing_count / frame_count)

        validation["non_numeric_counts"][col] = max(0, invalid_count)
        validation["missing_value_counts"][col] = missing_count
        validation["missing_ratios"][col] = missing_ratio

        if invalid_count > 0:
            validation["warnings"].append(
                f"Column '{col}' has {invalid_count} non-numeric values coerced to NaN."
            )

        if missing_ratio > MAX_MISSING_RATIO:
            validation["is_valid"] = False
            validation["errors"].append(
                f"Column '{col}' exceeds missing/invalid threshold "
                f"({missing_ratio:.1%} > {MAX_MISSING_RATIO:.1%})."
            )

        coerced_df[col] = coerced

    valid_rows_mask = coerced_df[REQUIRED_COLUMNS].notna().all(axis=1)
    valid_rows_after_coercion = int(valid_rows_mask.sum())
    validation["valid_rows_after_coercion"] = valid_rows_after_coercion
    if valid_rows_after_coercion < MIN_REQUIRED_ROWS:
        validation["is_valid"] = False
        validation["errors"].append(
            f"Too few valid rows after coercion ({valid_rows_after_coercion} < {MIN_REQUIRED_ROWS})."
        )

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
    """Sort, de-duplicate, coerce numerics, and impute missing values (no smoothing)."""
    proc = df.copy()

    for col in REQUIRED_COLUMNS:
        proc[col] = pd.to_numeric(proc[col], errors="coerce")

    proc = proc.sort_values(FRAME_COL).drop_duplicates(subset=[FRAME_COL], keep="first")
    proc = proc.reset_index(drop=True)

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != FRAME_COL]
    proc[numeric_cols] = proc[numeric_cols].interpolate(method="linear", limit_direction="both")
    proc[numeric_cols] = proc[numeric_cols].ffill().bfill()

    return proc


def smooth_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Create plot-ready smoothed signals; keeps step width unsmoothed to preserve variability."""
    smoothed = df.copy()
    for col in ANGLE_COLUMNS:
        smoothed[col] = _smooth_series(smoothed[col].astype(float))
    # STEP WIDTH is intentionally left unchanged because its variability is itself informative.
    return smoothed


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

    global_pct_asymmetry_values: List[float] = []
    for joint, (left_col, right_col) in LR_PAIRS.items():
        left = df[left_col].astype(float)
        right = df[right_col].astype(float)

        left_mean = float(left.mean())
        right_mean = float(right.mean())
        abs_diff_of_means = abs(left_mean - right_mean)

        denom_global = (left_mean + right_mean) / 2.0
        if denom_global == 0:
            pct_diff_of_means = np.nan
        else:
            pct_diff_of_means = abs_diff_of_means / abs(denom_global) * 100.0
            global_pct_asymmetry_values.append(float(pct_diff_of_means))

        framewise_abs_diff = (left - right).abs()
        framewise_denom = ((left + right) / 2.0).abs().replace(0, np.nan)
        framewise_pct_diff = (framewise_abs_diff / framewise_denom) * 100.0
        framewise_pct_diff = framewise_pct_diff.replace([np.inf, -np.inf], np.nan)

        metrics.append(
            {
                "metric": f"{joint} asymmetry | abs_difference_of_means",
                "value": float(abs_diff_of_means),
            }
        )
        metrics.append(
            {
                "metric": f"{joint} asymmetry | pct_difference_of_means",
                "value": float(pct_diff_of_means) if np.isfinite(pct_diff_of_means) else np.nan,
            }
        )
        metrics.append(
            {
                "metric": f"{joint} asymmetry | mean_framewise_abs_difference",
                "value": float(framewise_abs_diff.mean()),
            }
        )
        metrics.append(
            {
                "metric": f"{joint} asymmetry | mean_framewise_pct_difference",
                "value": float(framewise_pct_diff.dropna().mean()) if framewise_pct_diff.notna().any() else np.nan,
            }
        )

    if global_pct_asymmetry_values:
        symmetry_score = max(0.0, 100.0 - float(np.mean(global_pct_asymmetry_values)))
    else:
        symmetry_score = np.nan
    metrics.append(
        {
            "metric": "GLOBAL symmetry | score_0_to_100",
            "value": float(symmetry_score) if np.isfinite(symmetry_score) else np.nan,
        }
    )

    step_width = df[STEP_WIDTH_COL].astype(float)
    sw_mean = float(step_width.mean())
    sw_std = float(step_width.std(ddof=1)) if len(step_width) > 1 else 0.0
    sw_cv = (sw_std / sw_mean * 100.0) if sw_mean != 0 else np.nan
    metrics.append(
        {
            "metric": "STEP WIDTH | coefficient_of_variation_pct",
            "value": float(sw_cv) if np.isfinite(sw_cv) else np.nan,
        }
    )

    return pd.DataFrame(metrics)


def detect_asymmetry(metrics_df: pd.DataFrame) -> dict:
    """Flag unusually high bilateral asymmetry based on global percentage thresholds."""
    results = {
        "threshold_pct": ASYMMETRY_PERCENT_THRESHOLD,
        "flags": {},
        "summary": [],
    }

    for joint in ["HIP", "KNEE", "ANKLE"]:
        global_metric_name = f"{joint} asymmetry | pct_difference_of_means"
        framewise_metric_name = f"{joint} asymmetry | mean_framewise_pct_difference"

        global_row = metrics_df.loc[metrics_df["metric"] == global_metric_name, "value"]
        framewise_row = metrics_df.loc[metrics_df["metric"] == framewise_metric_name, "value"]

        global_pct = float(global_row.iloc[0]) if not global_row.empty else np.nan
        framewise_pct = float(framewise_row.iloc[0]) if not framewise_row.empty else np.nan

        is_flagged = bool(np.isfinite(global_pct) and global_pct > ASYMMETRY_PERCENT_THRESHOLD)
        results["flags"][joint] = {
            "pct_difference_of_means": global_pct,
            "mean_framewise_pct_difference": framewise_pct,
            "flagged": is_flagged,
        }
        if is_flagged:
            results["summary"].append(
                f"{joint} global asymmetry exceeds threshold ({global_pct:.2f}% > {ASYMMETRY_PERCENT_THRESHOLD:.2f}%)."
            )

    if not results["summary"]:
        results["summary"].append("No global asymmetry metrics exceeded configured thresholds.")

    return results


def _fallback_instability_threshold(diffs: pd.Series, median_diff: float) -> float:
    """Fallback threshold when MAD is zero or unavailable."""
    if diffs.empty:
        return np.nan
    p95 = float(np.percentile(diffs, 95))
    scale = max(median_diff, float(diffs.mean()), 1e-6)
    return max(p95, median_diff + INSTABILITY_FALLBACK_MULTIPLIER * scale)


def detect_signal_instability(df: pd.DataFrame) -> dict:
    """Detect abrupt jumps on clean (non-smoothed) signals and step-width variability."""
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
                "method": "insufficient_data",
                "jump_count": 0,
                "jump_frames": [],
            }
            continue

        med = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - med)))
        if mad > 0:
            robust_sigma = 1.4826 * mad
            threshold = med + INSTABILITY_MAD_MULTIPLIER * robust_sigma
            method = "mad_based"
        else:
            threshold = _fallback_instability_threshold(diffs, med)
            method = "fallback_non_mad"

        jump_mask = diffs > threshold
        jump_frames = df.loc[jump_mask.index[jump_mask], FRAME_COL].tolist()
        jump_count = int(jump_mask.sum())

        instability["abrupt_jumps"][col] = {
            "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
            "method": method,
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


def detect_outliers(df: pd.DataFrame) -> dict:
    """Flag conservative outliers per signal using robust z-scores without removing points."""
    outliers = {
        "threshold": OUTLIER_ROBUST_Z_THRESHOLD,
        "signals": {},
        "summary": [],
    }

    signal_cols = [*ANGLE_COLUMNS, STEP_WIDTH_COL]
    for col in signal_cols:
        series = df[col].astype(float)
        median = float(series.median())
        mad = float(np.median(np.abs(series - median)))

        if mad > 0:
            robust_z = 0.6745 * (series - median) / mad
            mask = robust_z.abs() > OUTLIER_ROBUST_Z_THRESHOLD
            method = "mad_robust_z"
        else:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (series < lower) | (series > upper)
                method = "iqr_fallback"
            else:
                mask = pd.Series(False, index=series.index)
                method = "no_variation"

        count = int(mask.sum())
        outlier_frames = df.loc[mask, FRAME_COL].tolist()
        outliers["signals"][col] = {
            "method": method,
            "count": count,
            "frames": outlier_frames,
        }
        if count > 0:
            outliers["summary"].append(f"{col}: {count} outlier frame(s) flagged.")

    if not outliers["summary"]:
        outliers["summary"].append("No conservative outlier flags were triggered.")

    return outliers


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


def _get_metric_value(metrics_df: pd.DataFrame, metric_name: str) -> float:
    """Safely fetch a scalar metric value from long-format metric table."""
    row = metrics_df.loc[metrics_df["metric"] == metric_name, "value"]
    return float(row.iloc[0]) if not row.empty else np.nan


def generate_report(
    validation: dict,
    metrics_df: pd.DataFrame,
    asymmetry: dict,
    instability: dict,
    outliers: dict,
) -> str:
    """Create a cautious markdown report with technical (non-diagnostic) interpretation."""
    lines: List[str] = []
    lines.append("# Gait Analysis Report")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(f"- Rows in input dataset: {validation.get('row_count', 0)}")
    lines.append(f"- Valid rows after numeric coercion: {validation.get('valid_rows_after_coercion', 0)}")
    lines.append(f"- Duplicate frames: {validation.get('duplicate_frames', 0)}")

    lines.append("")
    lines.append("## Validation Summary")
    lines.append(f"- Validation status: {'PASS' if validation.get('is_valid') else 'FAIL'}")
    lines.append(f"- Maximum allowed missing/invalid ratio: {validation.get('max_missing_ratio_allowed', MAX_MISSING_RATIO):.0%}")
    lines.append(f"- Minimum required valid rows: {validation.get('min_required_rows', MIN_REQUIRED_ROWS)}")

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
    lines.append("## Joint Descriptive Metrics")
    for joint, (left_col, right_col) in LR_PAIRS.items():
        lines.append(f"### {joint}")
        for side_label, col in [("Left", left_col), ("Right", right_col)]:
            mean_v = _get_metric_value(metrics_df, f"{col} | mean")
            std_v = _get_metric_value(metrics_df, f"{col} | std")
            min_v = _get_metric_value(metrics_df, f"{col} | min")
            max_v = _get_metric_value(metrics_df, f"{col} | max")
            rom_v = _get_metric_value(metrics_df, f"{col} | ROM")
            lines.append(
                f"- {side_label}: mean={mean_v:.4f}, std={std_v:.4f}, min={min_v:.4f}, max={max_v:.4f}, ROM={rom_v:.4f}"
            )

    lines.append("")
    lines.append("## Step Width Metrics")
    for metric_name in [
        f"{STEP_WIDTH_COL} | mean",
        f"{STEP_WIDTH_COL} | std",
        f"{STEP_WIDTH_COL} | min",
        f"{STEP_WIDTH_COL} | max",
        f"{STEP_WIDTH_COL} | ROM",
        "STEP WIDTH | coefficient_of_variation_pct",
    ]:
        value = _get_metric_value(metrics_df, metric_name)
        lines.append(f"- {metric_name}: {value:.4f}" if np.isfinite(value) else f"- {metric_name}: NaN")

    lines.append("")
    lines.append("## Global Asymmetry Metrics (Difference of Means)")
    for joint in ["HIP", "KNEE", "ANKLE"]:
        abs_val = _get_metric_value(metrics_df, f"{joint} asymmetry | abs_difference_of_means")
        pct_val = _get_metric_value(metrics_df, f"{joint} asymmetry | pct_difference_of_means")
        lines.append(
            f"- {joint}: abs_difference_of_means={abs_val:.4f}, pct_difference_of_means={pct_val:.4f}%"
            if np.isfinite(pct_val)
            else f"- {joint}: abs_difference_of_means={abs_val:.4f}, pct_difference_of_means=NaN"
        )
    symmetry_score = _get_metric_value(metrics_df, "GLOBAL symmetry | score_0_to_100")
    lines.append(
        f"- GLOBAL symmetry | score_0_to_100={symmetry_score:.4f}"
        if np.isfinite(symmetry_score)
        else "- GLOBAL symmetry | score_0_to_100=NaN"
    )
    lines.append(
        "- Note: percentage asymmetry can become unstable when the underlying bilateral mean signal is near zero."
    )

    lines.append("")
    lines.append("## Framewise Asymmetry Metrics")
    for joint in ["HIP", "KNEE", "ANKLE"]:
        abs_val = _get_metric_value(metrics_df, f"{joint} asymmetry | mean_framewise_abs_difference")
        pct_val = _get_metric_value(metrics_df, f"{joint} asymmetry | mean_framewise_pct_difference")
        lines.append(
            f"- {joint}: mean_framewise_abs_difference={abs_val:.4f}, mean_framewise_pct_difference={pct_val:.4f}%"
            if np.isfinite(pct_val)
            else f"- {joint}: mean_framewise_abs_difference={abs_val:.4f}, mean_framewise_pct_difference=NaN"
        )

    lines.append("")
    lines.append("## Asymmetry Flags")
    for item in asymmetry.get("summary", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Instability Findings")
    lines.append(f"- Step width CV (%): {instability.get('step_width_cv_pct', np.nan):.4f}")
    for col, details in instability.get("abrupt_jumps", {}).items():
        lines.append(
            f"- {col}: jump_count={details.get('jump_count', 0)}, "
            f"threshold={details.get('threshold', np.nan):.4f}, method={details.get('method', 'unknown')}"
        )
    lines.append("- Summary: see per-signal jump counts above for detailed instability findings.")

    lines.append("")
    lines.append("## Outlier Counts")
    lines.append(f"- Robust z threshold: {outliers.get('threshold', OUTLIER_ROBUST_Z_THRESHOLD):.2f}")
    for col, details in outliers.get("signals", {}).items():
        lines.append(f"- {col}: outlier_count={details.get('count', 0)} (method={details.get('method', 'unknown')})")
    lines.append("- Summary: outlier findings are reported per signal above.")

    lines.append("")
    lines.append("## Technical Interpretation")
    lines.append(
        "- This report provides technical signal descriptors (distribution, variability, asymmetry, and abrupt temporal changes) from the provided kinematic data."
    )
    lines.append(
        "- Metrics are sensitive to capture quality, preprocessing choices, and coordinate definitions; interpret in acquisition context."
    )
    lines.append(
        "- Non-diagnostic disclaimer: this script does not provide medical diagnosis, clinical classification, or treatment recommendations."
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
    if len(clean_df) < MIN_REQUIRED_ROWS:
        raise ValueError(
            f"Too few usable rows after preprocessing ({len(clean_df)} < {MIN_REQUIRED_ROWS})."
        )

    smoothed_df = smooth_signals(clean_df)
    metrics_df = compute_metrics(clean_df)
    asymmetry = detect_asymmetry(metrics_df)
    instability = detect_signal_instability(clean_df)
    outliers = detect_outliers(clean_df)

    output_dir = Path(OUTPUT_DIR)
    plot_signals(smoothed_df, output_dir)
    report_text = generate_report(validation, metrics_df, asymmetry, instability, outliers)
    save_results(metrics_df, report_text, output_dir)

    LOGGER.info("Analysis complete. Outputs saved to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
