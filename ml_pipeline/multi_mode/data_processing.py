"""
data_processing.py  Core logic for multi-mode data merging and oracle labeling.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Ensure we can find scripts/config.py
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

from config import (
    TRAINING_LABELED,
    TARGET_FRAME_MS, SAFE_TARGET_MS,
    POSITION_BINS, ROTATION_BINS,
    LAMBDA, SOFTMAX_TEMP, HEADROOM_WEIGHT
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def parse_metadata(filename):
    """
    Parse (bias, spd, rot) from filename like:
    training_data_bias_0.3_spd_0.5_rot_1.5.csv
    """
    pattern = r"bias_([\d.]+)_spd_([\d.]+)_rot_([\d.]+)\.csv"
    match = re.search(pattern, filename)
    if match:
        return {
            "bias": float(match.group(1)),
            "spd": float(match.group(2)),
            "rot": float(match.group(3))
        }
    return None

def merge_multi_mode_data(base_dir, drop_rot_zero=True):
    """
    Step 1 & 2: Merge data with power mode tags and handle grid inconsistency.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logging.warning(f"Base directory {base_dir} does not exist.")
        return None

    # Find all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        logging.warning(f"No subdirectories found in {base_dir}")
        return None

    all_dfs = []
    
    for subdir in subdirs:
        power_mode = subdir.name
        csv_files = list(subdir.glob("training_data_*.csv"))
        
        if not csv_files:
            logging.warning(f"No CSV files found in {subdir}. Skipping power mode: {power_mode}")
            continue
            
        logging.info(f"Processing power mode: {power_mode} ({len(csv_files)} files)")
        
        for csv_f in csv_files:
            metadata = parse_metadata(csv_f.name)
            if not metadata:
                logging.warning(f"Could not parse metadata from {csv_f.name}. Skipping.")
                continue
                
            if drop_rot_zero and metadata['rot'] == 0.0:
                logging.info(f"  Skipping rot_0.0 file: {csv_f.name}")
                continue
                
            df = pd.read_csv(csv_f)
            df['power_mode'] = power_mode
            df['bias'] = metadata['bias']
            df['spd'] = metadata['spd']
            df['rot'] = metadata['rot']
            
            all_dfs.append(df)
            
    if not all_dfs:
        logging.error("No data loaded across any power modes.")
        return None
        
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Parity with merge_training_data.py: recompute headroom if configured
    from config import FIX_HEADROOM
    if FIX_HEADROOM:
        merged_df['frame_headroom_ms'] = TARGET_FRAME_MS - merged_df['cpu_frame_time_ms']
        logging.info(f"Recomputed frame_headroom_ms using TARGET_FRAME_MS={TARGET_FRAME_MS}")
        
    logging.info(f"Merged all modes. Total rows: {len(merged_df)}")
    return merged_df

def compute_soft_oracle_for_group(group, bias_values, bias_min, bias_max):
    """
    Logic adapted from generate_oracle_labels.py
    """
    scores, present_biases = [], []
    for bias in bias_values:
        subset = group[group['lod_bias_current'] == bias]
        if len(subset) == 0:
            continue
            
        p95_cpu        = subset['cpu_frame_time_ms'].quantile(0.95)
        p95_gpu        = subset['gpu_frame_time_ms'].quantile(0.95)
        max_frame_time = max(p95_cpu, p95_gpu)
        
        quality        = (bias - bias_min) / (bias_max - bias_min) if bias_max > bias_min else 1.0
        perf_penalty   = max(0.0, (max_frame_time - SAFE_TARGET_MS) / SAFE_TARGET_MS)
        headroom_ratio = max(0.0, (SAFE_TARGET_MS - max_frame_time) / SAFE_TARGET_MS)
        headroom_bonus = HEADROOM_WEIGHT * headroom_ratio * quality
        
        score          = quality - LAMBDA * perf_penalty + headroom_bonus
        scores.append(score)
        present_biases.append(bias)

    if not scores:
        return pd.Series(dtype=float)

    scores_arr  = np.array(scores) / SOFTMAX_TEMP
    scores_arr -= scores_arr.max()
    probs       = np.exp(scores_arr) / np.exp(scores_arr).sum()
    expected_bias = float(np.dot(probs, present_biases))

    soft_dict = {f"soft_{b}": p for b, p in zip(present_biases, probs)}
    soft_dict["target_lod_bias"] = expected_bias
    return pd.Series(soft_dict)

def compute_multi_mode_oracle(df):
    """
    Step 3: Group by power_mode and compute oracle labels.
    """
    logging.info("Computing soft oracle labels per power mode and state bin...")
    
    bias_values = sorted(df['lod_bias_current'].unique())
    bias_min    = min(bias_values)
    bias_max    = max(bias_values)
    
    # Binning logic
    df['pos_bin']   = pd.cut(df['path_progress'], bins=POSITION_BINS, labels=False)
    df['rot_bin']   = pd.cut(df['cam_rot_y'],     bins=ROTATION_BINS, labels=False)
    df['state_bin'] = df['pos_bin'].astype(str) + "_" + df['rot_bin'].astype(str)
    
    def process_mode_group(mode_df):
        return mode_df.groupby('state_bin').apply(
            compute_soft_oracle_for_group, 
            bias_values=bias_values, 
            bias_min=bias_min, 
            bias_max=bias_max,
            include_groups=False
        )

    # Group by power_mode and state_bin
    soft_map = df.groupby('power_mode').apply(process_mode_group)
    
    # Reset index for easier mapping
    # Resulting soft_map will have index [power_mode, state_bin] if it worked as expected
    if isinstance(soft_map, pd.Series):
        soft_map = soft_map.unstack()

    # Map results back to main df
    soft_cols = [c for c in soft_map.columns if c.startswith("soft_")]
    target_col = "target_lod_bias"
    
    df = df.drop(columns=[target_col], errors='ignore')
    
    # Efficient mapping using multi-index
    soft_map_indexed = soft_map.copy()
    
    # We need to map (power_mode, state_bin) pairs to soft_map values
    # Let's create a lookup key in both
    df['lookup_key'] = df['power_mode'] + "|" + df['state_bin']
    soft_map_indexed.index = [f"{i[0]}|{i[1]}" for i in soft_map_indexed.index]
    
    for col in soft_cols + [target_col]:
        df[col] = df['lookup_key'].map(soft_map_indexed[col])
        
    df = df.drop(columns=['pos_bin', 'rot_bin', 'state_bin', 'lookup_key'])
    return df

def run_sanity_checks(df):
    """
    Step 5: Print stats per power mode.
    """
    logging.info("=== SANITY CHECKS ===")
    
    # 1. Mean GPU frame time per mode
    stats = df.groupby('power_mode')['gpu_frame_time_ms'].mean()
    logging.info("Mean gpu_frame_time_ms per power mode:")
    for mode, value in stats.items():
        logging.info(f"  {mode:30} : {value:.4f} ms")
        
    # 2. Row counts and balance
    counts = df['power_mode'].value_counts()
    total = len(df)
    logging.info(f"Total rows: {total}")
    for mode, count in counts.items():
        share = (count / total) * 100
        flag = " [LOW SHARE!]" if share < 10.0 else ""
        logging.info(f"  {mode:30} : {count:8} ({share:5.1f}%){flag}")
    
    # 3. Label spread
    if 'target_lod_bias' in df.columns:
        std = df['target_lod_bias'].std()
        flag = "COLLAPSE RISK" if std < 0.15 else "good spread"
        logging.info(f"target_lod_bias std: {std:.4f} ({flag})")

def finalize_and_save(df, output_path):
    """
    Step 4: Cleanup and save.
    """
    # Drop metadata columns
    final_df = df.drop(columns=['power_mode', 'bias', 'spd', 'rot'], errors='ignore')
    
    # Drop NaNs from oracle computation (sparse bins)
    before = len(final_df)
    final_df = final_df.dropna(subset=['target_lod_bias'])
    dropped = before - len(final_df)
    if dropped > 0:
        logging.warning(f"Dropped {dropped} rows due to NaN oracle (sparse bins)")
        
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(output_path, index=False)
    logging.info(f"Final labeled data saved to: {output_path}")
    return final_df
