"""
generate_oracle_labels.py  Stage 2: compute per-row reactive oracle labels.

Logic:
  if max(cpu, gpu) < SAFE_TARGET_MS  -> target = current_bias + BIAS_STEP  (nudge up)
  if max(cpu, gpu) < TARGET_FRAME_MS -> target = current_bias               (hold)
  else                               -> target = current_bias - BIAS_STEP  (nudge down)

Labels are rounded to clean 0.25 steps and clipped to [BIAS_MIN, BIAS_MAX].
"""

import pandas as pd
import numpy as np
from config import (
    TRAINING_MERGED, TRAINING_LABELED,
    TARGET_FRAME_MS, SAFE_TARGET_MS,
)

BIAS_STEP = 0.25
BIAS_MIN  = 0.25
BIAS_MAX  = 2.0


def run():
    if not TRAINING_MERGED.exists():
        raise FileNotFoundError(
            f"Merged file not found: {TRAINING_MERGED}\n"
            "Run Stage 1 (merge_training_data.py) first."
        )

    df = pd.read_csv(TRAINING_MERGED)
    print(f"Loaded {len(df)} rows from {TRAINING_MERGED}")
    print(f"Budget         : {TARGET_FRAME_MS:.2f} ms")
    print(f"Safe budget    : {SAFE_TARGET_MS:.2f} ms")
    print(f"Bias step      : {BIAS_STEP}")
    print(f"Bias range     : [{BIAS_MIN}, {BIAS_MAX}]")

    print(f"\nlod_bias_current distribution:")
    print(df["lod_bias_current"].value_counts().sort_index())

    df = df.drop(columns=["target_lod_bias"], errors="ignore")

    # per-row bottleneck
    df["bottleneck_ms"] = df[["cpu_frame_time_ms", "gpu_frame_time_ms"]].max(axis=1)

    under_budget = df["bottleneck_ms"] < SAFE_TARGET_MS
    at_budget    = (df["bottleneck_ms"] >= SAFE_TARGET_MS) & (df["bottleneck_ms"] < TARGET_FRAME_MS)
    over_budget  = df["bottleneck_ms"] >= TARGET_FRAME_MS

    current = df["lod_bias_current"]

    df["target_lod_bias"] = np.where(
        under_budget, (current + BIAS_STEP).clip(BIAS_MIN, BIAS_MAX),
        np.where(
            at_budget,  current.clip(BIAS_MIN, BIAS_MAX),
                        (current - BIAS_STEP).clip(BIAS_MIN, BIAS_MAX)
        )
    )

    # round to clean 0.25 steps
    df["target_lod_bias"] = (df["target_lod_bias"] / 0.25).round() * 0.25
    df["target_lod_bias"] = df["target_lod_bias"].clip(BIAS_MIN, BIAS_MAX)

    df = df.drop(columns=["bottleneck_ms"])

    print(f"\nZone distribution:")
    print(f"  Under budget (nudge up)  : {under_budget.sum():>8}  ({under_budget.mean()*100:.1f}%)")
    print(f"  At budget    (hold)      : {at_budget.sum():>8}  ({at_budget.mean()*100:.1f}%)")
    print(f"  Over budget  (nudge down): {over_budget.sum():>8}  ({over_budget.mean()*100:.1f}%)")

    print(f"\nRounded label distribution:")
    print(df["target_lod_bias"].value_counts().sort_index())

    print(f"\ntarget_lod_bias stats:")
    print(df["target_lod_bias"].describe().round(4))

    std  = df["target_lod_bias"].std()
    flag = "COLLAPSE RISK" if std < 0.15 else "good spread"
    print(f"target_lod_bias std: {std:.4f}  ({flag})")

    TRAINING_LABELED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_LABELED, index=False)
    print(f"\nSaved -> {TRAINING_LABELED}")


if __name__ == "__main__":
    run()