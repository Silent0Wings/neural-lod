"""
config.py  central config for the Adaptive LOD ML pipeline.
All scripts import from here. Edit paths and constants in one place only.
"""

from pathlib import Path
import glob

# root of ml_pipeline folder
PIPELINE_ROOT = Path(__file__).resolve().parent.parent

# data folder that holds all CSVs
DATA_DIR = PIPELINE_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# plots folder for all generated figures
PLOTS_DIR = PIPELINE_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---- file paths -------------------------------------------------------

# merge stage: input raw Unity training CSVs
TRAINING_RAW_DIR     = DATA_DIR                          # put training_data_*.csv here
TRAINING_RAW_PATTERN = str(DATA_DIR / "training_data_*.csv")

# merge stage: output
TRAINING_MERGED = DATA_DIR / "training_data_merged.csv"

# label stage: input + output
TRAINING_LABELED = DATA_DIR / "training_data_labeled.csv"

# eval compare stage: latest eval CSV is auto-detected (most recent by name)
def get_latest_eval_csv() -> Path:
    matches = sorted(glob.glob(str(DATA_DIR / "eval_neural_*.csv")))
    if not matches:
        raise FileNotFoundError(
            f"No eval_neural_*.csv found in {DATA_DIR}\n"
            "Copy your EvaluationLogger output CSV into ml_pipeline/data/ first."
        )
    return Path(matches[-1])

# eval compare stage: output figure
EVAL_COMPARE_FIG = PLOTS_DIR / "compare_eval_vs_train.png"

# ---- shared constants -------------------------------------------------

TARGET_FPS      = 60
TARGET_FRAME_MS = 1000.0 / TARGET_FPS   # 16.66 ms
SAFE_TARGET_MS  = TARGET_FRAME_MS * 0.90

# merge stage
DROP_SLOW  = False
SLOW_SPEED = 0.5
FIX_HEADROOM = True

# oracle label stage
POSITION_BINS  = 50
ROTATION_BINS  = 8
LAMBDA         = 6
SOFTMAX_TEMP   = 0.5
HEADROOM_WEIGHT = 0.3
