"""
Stage 4: RL LOD Policy Training - Configuration
All constants and hyperparameters matching the notebook's config cell.
"""
import torch
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent.parent.parent  # ml_pipeline
DATA_DIR        = BASE_DIR / 'data' / 'RL' / 'Train' / 'RLRollouts'
MODEL_DIR       = BASE_DIR / 'models' / 'Stage_4'
PLOTS_DIR       = BASE_DIR / 'plots' / 'Stage_4' / 'Train'

# ── Reward shaping ─────────────────────────────────────────────────────
BONUS_SCALE           = 1.0
REWARD_CLIP           = 5.0
GPU_VALID_MIN_MS      = 0.10
GPU_VALID_MAX_MS      = 33.3
UNDER_BUDGET_MARGIN   = 1.10
OVER_BUDGET_COEF      = 0.70
UNDER_BUDGET_COEF     = 0.60
QUALITY_REWARD_COEF   = 0.85
TARGET_ZONE_COEF      = 1.50
RECOVERY_REWARD_COEF  = 0.60
CONTROL_DIRECTION_REWARD_COEF = 0.65
NEAR_TARGET_ACTION_PENALTY_COEF = 0.08
LOW_BIAS_PENALTY_COEF = 1.40
FLOOR_MARGIN          = 0.18
CEILING_MARGIN        = 0.18
RECOVERY_BIAS_TARGET  = 1.10
RECOVERY_BIAS_TOL     = 0.10
SAFE_RECOVERY_GPU_MARGIN = 0.25
FLOOR_RECOVERY_MIN_STRENGTH = 1.00
CEILING_CORRECTION_MIN_STRENGTH = 0.22
NEAR_TARGET_FLOOR_RECOVERY_SCALE = 1.35
NEAR_TARGET_CEILING_CORRECTION_SCALE = 0.25
TARGET_PROX_SIGMA     = 1.75

# ── Runtime guardrails (must match Unity deployment) ───────────────────
DEAD_ZONE        = 0.02
DWELL_FRAMES     = 5
BIAS_MIN         = 0.30
BIAS_MAX         = 2.00
DWELL_ACCUMULATION_RATE = 0.30
DWELL_DECAY            = 0.92
DWELL_ACTIVE_THRESHOLD = 0.28
FLOOR_DWELL_RECOVERY_GAIN = 1.35
CEILING_DWELL_CORRECTION_GAIN = 0.40
RECOVERY_ELIGIBLE_BIAS = 1.05
CORRECTION_ELIGIBLE_BIAS = 1.60
RECOVERY_GROWTH_THRESHOLD = 0.035
RECOVERY_TRIGGER_FRAMES = 2
RECOVERY_FORCE_MULTIPLIER = 2.50
RECOVERY_FORCE_MAX = 9.0
RECOVERY_BOOST_BASE = 0.12
RECOVERY_BUDGET_RESET_MARGIN = 0.25

# ── Legacy weights (not used in reward; retained for report parity) ────
ALPHA            = 1.0
BETA             = 0.5
GAMMA_W          = 0.0
N_MAX            = 30.0

# ── REINFORCE / rollout-policy training hyperparameters ────────────────
GAMMA_RL           = 0.99
TRAIN_SIGMA        = 0.10
ACTION_HEAD_SCALE  = 0.30
PG_COEF            = 1.10
BC_COEF_START      = 0.30
BC_COEF_END        = 0.10
ZERO_BC_WEIGHT     = 0.08
NONZERO_BC_WEIGHT  = 1.0
SUPPORT_MARGIN     = 0.015
SUPPORT_COEF_START = 0.10
SUPPORT_COEF_END   = 0.00
SAT_WARN_THRESHOLD = ACTION_HEAD_SCALE - 0.01
SAT_COEF              = 1.0
NEG_SAT_COEF          = 6.0
FLOOR_COEF            = 1.5
HEADROOM_PUSH_COEF    = 3.0
OVER_BUDGET_POS_COEF  = 8.0
NEAR_TARGET_ZERO_COEF = 0.25
NEAR_TARGET_NEG_COEF  = 2.0
LOW_BIAS_RECOVERY_COEF = 15.0
HIGH_BIAS_TRIM_COEF    = 2.0
CONTROL_COEF          = 1.0
CONTROL_OVER_WEIGHT   = 1.8
CONTROL_UNDER_WEIGHT  = 2.6
CONTROL_NEAR_WEIGHT   = 1.5
CONTROL_FLOOR_BOOST   = 4.0
CONTROL_DEADBAND_MS   = 0.25
CONTROL_RESPONSE_MS   = 2.25
CONTROL_TARGET_TOL    = 0.010
ENTROPY_COEF          = 0.010
DEAD_ZONE_COEF        = 3.0
POS_SAT_COEF          = 2.5
DEPLOY_ACTIVE_TARGET  = 3.0
VAL_GROUP_FRAC        = 0.20
GRAD_CLIP             = 1.0
BATCH_SIZE            = 256
RANDOM_SEED           = 42

# ── Seed ───────────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Feature specification ──────────────────────────────────────────────
FEATURE_COUNT = 13
FEATURE_COLS  = [
    'cpu_frame_time',
    'gpu_frame_time',
    'fps',
    'visible_renderer_count',
    'triangle_count',
    'draw_call_count',
    'camera_speed',
    'camera_rotation_speed',
    'avg_screen_coverage',
    'previous_bias',
    'recent_lod_switch_count',
    'floor_dwell_score',
    'ceiling_dwell_score',
]

# ── Device ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Derived constants (set after data load) ────────────────────────────
T_TARGET = None  # Set dynamically by data_loader / reward

SOFT_SUPPORT_LIMIT = max(DEAD_ZONE, ACTION_HEAD_SCALE - SUPPORT_MARGIN)

# Feature indices (convenience)
GPU_FEATURE_IDX = FEATURE_COLS.index('gpu_frame_time')
BIAS_FEATURE_IDX = FEATURE_COLS.index('previous_bias')
FLOOR_DWELL_FEATURE_IDX = FEATURE_COLS.index('floor_dwell_score')
CEILING_DWELL_FEATURE_IDX = FEATURE_COLS.index('ceiling_dwell_score')
