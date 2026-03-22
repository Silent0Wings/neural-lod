"""
generate_oracle_labels.py  Stage 2: compute soft oracle labels from merged CSV.
"""

import pandas as pd
import numpy as np
from config import (
    TRAINING_MERGED, TRAINING_LABELED,
    TARGET_FRAME_MS, SAFE_TARGET_MS,
    POSITION_BINS, ROTATION_BINS,
    LAMBDA, SOFTMAX_TEMP, HEADROOM_WEIGHT
)

def run():
    if not TRAINING_MERGED.exists():
        raise FileNotFoundError(
            f"Merged file not found: {TRAINING_MERGED}\n"
            "Run Stage 1 (merge_training_data.py) first."
        )

    df = pd.read_csv(TRAINING_MERGED)
    print(f"Loaded {len(df)} rows from {TRAINING_MERGED}")

    bias_values = sorted(df['lod_bias_current'].unique())
    bias_min    = min(bias_values)
    bias_max    = max(bias_values)

    print(f"Bias values    : {bias_values}")
    print(f"Safe budget    : {SAFE_TARGET_MS:.2f} ms | Lambda: {LAMBDA} | Softmax temp: {SOFTMAX_TEMP} | Headroom weight: {HEADROOM_WEIGHT}")
    print(f"path_progress  : {df['path_progress'].min():.4f} -> {df['path_progress'].max():.4f}")
    print(f"cam_rot_y      : {df['cam_rot_y'].min():.4f} -> {df['cam_rot_y'].max():.4f}")
    print(f"\nRows per bias:")
    print(df['lod_bias_current'].value_counts().sort_index())

    df['pos_bin']   = pd.cut(df['path_progress'], bins=POSITION_BINS, labels=False)
    df['rot_bin']   = pd.cut(df['cam_rot_y'],     bins=ROTATION_BINS, labels=False)
    df['state_bin'] = df['pos_bin'].astype(str) + "_" + df['rot_bin'].astype(str)

    bin_counts = df.groupby('state_bin').size()
    sparse     = bin_counts[bin_counts < 30]
    print(f"\nTotal state bins : {len(bin_counts)}")
    if len(sparse) > 0:
        print(f"WARNING {len(sparse)} sparse bins (<30 rows)")
    else:
        print(f"YES No sparse bins")

    def compute_soft_oracle(group):
        scores, present_biases = [], []
        for bias in bias_values:
            subset = group[group['lod_bias_current'] == bias]
            if len(subset) == 0:
                continue
            p95_cpu        = subset['cpu_frame_time_ms'].quantile(0.95)
            p95_gpu        = subset['gpu_frame_time_ms'].quantile(0.95)
            max_frame_time = max(p95_cpu, p95_gpu)
            quality        = (bias - bias_min) / (bias_max - bias_min)
            perf_penalty   = max(0.0, (max_frame_time - SAFE_TARGET_MS) / SAFE_TARGET_MS)
            headroom_ratio = max(0.0, (SAFE_TARGET_MS - max_frame_time) / SAFE_TARGET_MS)
            headroom_bonus = HEADROOM_WEIGHT * headroom_ratio * quality
            score          = quality - LAMBDA * perf_penalty + headroom_bonus
            scores.append(score)
            present_biases.append(bias)

        scores_arr  = np.array(scores) / SOFTMAX_TEMP
        scores_arr -= scores_arr.max()
        probs       = np.exp(scores_arr) / np.exp(scores_arr).sum()
        expected_bias = float(np.dot(probs, present_biases))

        soft_dict = {f"soft_{b}": p for b, p in zip(present_biases, probs)}
        soft_dict["target_lod_bias"] = expected_bias
        return pd.Series(soft_dict)

    print("\nComputing soft oracle labels per state bin...")
    soft_map = df.groupby('state_bin').apply(compute_soft_oracle, include_groups=False)
    if isinstance(soft_map, pd.Series):
        soft_map = soft_map.unstack()

    df       = df.drop(columns=['target_lod_bias'], errors='ignore')
    soft_cols = [c for c in soft_map.columns if c.startswith("soft_")]

    for col in soft_cols + ["target_lod_bias"]:
        df[col] = df['state_bin'].map(soft_map[col])

    df = df.drop(columns=['pos_bin', 'rot_bin', 'state_bin'])

    before  = len(df)
    df      = df.dropna(subset=["target_lod_bias"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"WARNING Dropped {dropped} rows with NaN oracle (sparse bins)")
    else:
        print(f"YES No rows dropped")

    print(f"\nLabeled rows: {df['target_lod_bias'].notna().sum()} / {len(df)}")
    print(f"target_lod_bias stats:")
    print(df['target_lod_bias'].describe().round(4))

    std  = df['target_lod_bias'].std()
    flag = "COLLAPSE RISK" if std < 0.15 else "good spread"
    print(f"target_lod_bias std: {std:.4f}  ({flag})")

    df.to_csv(TRAINING_LABELED, index=False)
    print(f"\nSaved -> {TRAINING_LABELED}")

if __name__ == "__main__":
    run()
