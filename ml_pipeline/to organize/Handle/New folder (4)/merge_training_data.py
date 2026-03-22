import os
import glob
import pandas as pd

# -----------------------------------------
# CONFIG
# -----------------------------------------
INPUT_DIR   = "."                          # folder containing training_data_*.csv
OUTPUT_FILE = "training_data_merged.csv"
DROP_SLOW   = False                         # set False to include spd=0.5 runs
SLOW_SPEED  = 0.5                          # move_speed value to drop if DROP_SLOW=True

# Fix frame_headroom to use CPU instead of GPU
FIX_HEADROOM    = True
TARGET_FRAME_MS = 16.6
# -----------------------------------------

files = sorted(glob.glob(os.path.join(INPUT_DIR, "training_data_*.csv")))
files = [f for f in files if os.path.basename(f) != OUTPUT_FILE]

if not files:
    raise FileNotFoundError(f"No training_data_*.csv files found in {INPUT_DIR}")

print(f"Found {len(files)} files.")

dfs     = []
skipped = []

for f in files:
    df    = pd.read_csv(f)
    fname = os.path.basename(f)

    if DROP_SLOW and df['move_speed'].iloc[0] <= SLOW_SPEED:
        skipped.append(fname)
        continue

    df['source_file'] = fname
    dfs.append(df)
    print(f"  Loaded {fname} - {len(df)} rows")

if skipped:
    print(f"\n  Skipped {len(skipped)} slow-speed files:")
    for s in skipped:
        print(f"     {s}")

if not dfs:
    raise ValueError("No files loaded - check DROP_SLOW and INPUT_DIR settings.")

merged = pd.concat(dfs, ignore_index=True)

if FIX_HEADROOM:
    merged['frame_headroom_ms'] = TARGET_FRAME_MS - merged['cpu_frame_time_ms']
    print(f"\n  Recomputed frame_headroom_ms = {TARGET_FRAME_MS} - cpu_frame_time_ms")

print(f"\nTotal rows: {len(merged)}")
print(f"Columns:    {list(merged.columns)}")
print(f"\nPer-bias row counts:")
print(merged['lod_bias_current'].value_counts().sort_index())

merged.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")