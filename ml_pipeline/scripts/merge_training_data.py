"""
merge_training_data.py  Stage 1: merge raw Unity training CSVs into one file.
"""

import glob
import pandas as pd
from config import (
    TRAINING_RAW_PATTERN, TRAINING_MERGED,
    DROP_SLOW, SLOW_SPEED, FIX_HEADROOM, TARGET_FRAME_MS
)

def run():
    files = sorted(glob.glob(TRAINING_RAW_PATTERN))
    files = [f for f in files if "merged" not in f and "labeled" not in f]

    if not files:
        raise FileNotFoundError(
            f"No training_data_*.csv files found.\n"
            f"Pattern: {TRAINING_RAW_PATTERN}\n"
            f"Copy Unity output CSVs into ml_pipeline/data/"
        )

    print(f"Found {len(files)} raw CSV files.")

    dfs, skipped = [], []

    for f in files:
        df    = pd.read_csv(f)
        fname = str(f)

        if DROP_SLOW and df['move_speed'].iloc[0] <= SLOW_SPEED:
            skipped.append(fname)
            continue

        df['source_file'] = fname
        dfs.append(df)
        print(f"  Loaded {fname}  ({len(df)} rows)")

    if skipped:
        print(f"\n  Skipped {len(skipped)} slow-speed files.")

    if not dfs:
        raise ValueError("No files loaded. Check DROP_SLOW and data directory.")

    merged = pd.concat(dfs, ignore_index=True)

    if FIX_HEADROOM:
        merged['frame_headroom_ms'] = TARGET_FRAME_MS - merged['cpu_frame_time_ms']
        print(f"\n  Recomputed frame_headroom_ms = {TARGET_FRAME_MS} - cpu_frame_time_ms")

    print(f"\nTotal rows : {len(merged)}")
    print(f"Columns    : {list(merged.columns)}")
    print(f"\nPer-bias row counts:")
    print(merged['lod_bias_current'].value_counts().sort_index())

    merged.to_csv(TRAINING_MERGED, index=False)
    print(f"\nSaved -> {TRAINING_MERGED}")

if __name__ == "__main__":
    run()
