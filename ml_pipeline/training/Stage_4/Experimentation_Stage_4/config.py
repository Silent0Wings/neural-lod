"""
Stage 4: RL LOD Policy Training - Configuration
All constants and hyperparameters matching the notebook's config cell.
"""
# Pipeline stage: shared configuration used by all Stage 4 experiment scripts.
import numpy as np
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent.parent.parent  # ml_pipeline
DATA_DIR        = BASE_DIR / 'data' / 'RL' / 'Train' / 'RLRollouts'
MODEL_DIR       = BASE_DIR / 'models' / 'Stage_4'
PLOTS_DIR       = BASE_DIR / 'plots' / 'Stage_4' / 'Train'

# ── Unity runtime contract (must match collection and deployment JSON) ─
#
# The Stage 4 Unity runtime now owns more than the scaler mean/scale:
# action envelope, dwell/dead-zone guardrails, feature-extractor dwell
# settings, scene target warmup, and recovery-assist constants all affect the
# labels written by null/fallback collection.  Keep those values in one Python
# contract so training, diagnostics, generated JSON, and ONNX export agree.
_CONTRACT_JSON_CANDIDATES = (
    Path(__file__).resolve().parent / 'rl_null_collection_constants.json',
    Path(__file__).resolve().parent.parent.parent.parent / 'Assets' / 'StreamingAssets' / 'rl_null_collection_constants.json',
)

_RUNTIME_CONTRACT_BASE = {
    'action_head_scale': 0.20,
    'max_action_delta': 0.20,
    'dead_zone': 0.02,
    'dwell_frames': 5,
    'bias_min': 0.30,
    'bias_max': 2.00,
    'inference_interval': 2,
    'coverage_sample_interval': 2,
    'lod_switch_window': 30,
    'floor_margin': 0.18,
    'ceiling_margin': 0.18,
    'dwell_accumulation_rate': 0.30,
    'dwell_decay': 0.92,
    'recovery_eligible_bias': 0.85,
    'correction_eligible_bias': 1.45,
    'recovery_growth_threshold': 0.02,
    'recovery_trigger_frames': 5,
    'recovery_force_multiplier': 1.35,
    'recovery_force_max': 4.0,
    'recovery_boost_base': 0.02,
    'recovery_budget_reset_margin': 0.10,
    'scene_target_warmup_frames': 64,
    'dwell_seconds': 0.5,
    'ema_alpha': 0.2,
}
_RUNTIME_CONTRACT_KEYS = tuple(_RUNTIME_CONTRACT_BASE.keys())


def _load_runtime_contract_overrides() -> dict:
    """Load Unity-owned runtime constants from the local null-collection JSON."""
    import json

    for path in _CONTRACT_JSON_CANDIDATES:
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            key: data[key]
            for key in _RUNTIME_CONTRACT_KEYS
            if key in data and data[key] is not None
        }
    return {}


RUNTIME_CONTRACT_DEFAULTS = dict(_RUNTIME_CONTRACT_BASE)
RUNTIME_CONTRACT_DEFAULTS.update(_load_runtime_contract_overrides())
RUNTIME_CONTRACT_KEYS = tuple(RUNTIME_CONTRACT_DEFAULTS.keys())

# ── Reward shaping ─────────────────────────────────────────────────────
BONUS_SCALE           = 1.0
REWARD_CLIP           = 5.0
GPU_VALID_MIN_MS      = 0.10
GPU_VALID_MAX_MS      = 33.3
UNDER_BUDGET_MARGIN   = 0.05
OVER_BUDGET_COEF      = 1.40
UNDER_BUDGET_COEF     = 0.60
QUALITY_REWARD_COEF   = 0.85
TARGET_ZONE_COEF      = 1.50
RECOVERY_REWARD_COEF  = 0.60
CONTROL_DIRECTION_REWARD_COEF = 2.00
NEAR_TARGET_ACTION_PENALTY_COEF = 0.08
LOW_BIAS_PENALTY_COEF = 1.40
FLOOR_MARGIN          = RUNTIME_CONTRACT_DEFAULTS['floor_margin']
CEILING_MARGIN        = RUNTIME_CONTRACT_DEFAULTS['ceiling_margin']
RECOVERY_BIAS_TARGET  = 1.10
RECOVERY_BIAS_TOL     = 0.10
SAFE_RECOVERY_GPU_MARGIN = 0.25
FLOOR_RECOVERY_MIN_STRENGTH = 1.00
CEILING_CORRECTION_MIN_STRENGTH = 0.22
NEAR_TARGET_FLOOR_RECOVERY_SCALE = 1.35
NEAR_TARGET_CEILING_CORRECTION_SCALE = 0.25
TARGET_PROX_SIGMA     = 1.75

# ── Runtime guardrails (must match Unity deployment) ───────────────────
DEAD_ZONE        = RUNTIME_CONTRACT_DEFAULTS['dead_zone']
DWELL_FRAMES     = RUNTIME_CONTRACT_DEFAULTS['dwell_frames']
DWELL_SECONDS    = RUNTIME_CONTRACT_DEFAULTS['dwell_seconds']
BIAS_MIN         = RUNTIME_CONTRACT_DEFAULTS['bias_min']
BIAS_MAX         = RUNTIME_CONTRACT_DEFAULTS['bias_max']
EMA_ALPHA        = RUNTIME_CONTRACT_DEFAULTS['ema_alpha']
INFERENCE_INTERVAL = RUNTIME_CONTRACT_DEFAULTS['inference_interval']
COVERAGE_SAMPLE_INTERVAL = RUNTIME_CONTRACT_DEFAULTS['coverage_sample_interval']
LOD_SWITCH_WINDOW = RUNTIME_CONTRACT_DEFAULTS['lod_switch_window']
DWELL_ACCUMULATION_RATE = RUNTIME_CONTRACT_DEFAULTS['dwell_accumulation_rate']
DWELL_DECAY            = RUNTIME_CONTRACT_DEFAULTS['dwell_decay']
DWELL_ACTIVE_THRESHOLD = 0.28
FLOOR_DWELL_RECOVERY_GAIN = 1.35
CEILING_DWELL_CORRECTION_GAIN = 0.40
RECOVERY_ELIGIBLE_BIAS = RUNTIME_CONTRACT_DEFAULTS['recovery_eligible_bias']
CORRECTION_ELIGIBLE_BIAS = RUNTIME_CONTRACT_DEFAULTS['correction_eligible_bias']
RECOVERY_GROWTH_THRESHOLD = RUNTIME_CONTRACT_DEFAULTS['recovery_growth_threshold']
RECOVERY_TRIGGER_FRAMES = RUNTIME_CONTRACT_DEFAULTS['recovery_trigger_frames']
RECOVERY_FORCE_MULTIPLIER = RUNTIME_CONTRACT_DEFAULTS['recovery_force_multiplier']
RECOVERY_FORCE_MAX = RUNTIME_CONTRACT_DEFAULTS['recovery_force_max']
RECOVERY_BOOST_BASE = RUNTIME_CONTRACT_DEFAULTS['recovery_boost_base']
RECOVERY_BUDGET_RESET_MARGIN = RUNTIME_CONTRACT_DEFAULTS['recovery_budget_reset_margin']
SCENE_TARGET_WARMUP_FRAMES = RUNTIME_CONTRACT_DEFAULTS['scene_target_warmup_frames']

# ── Legacy weights (not used in reward; retained for report parity) ────
ALPHA            = 1.0
BETA             = 0.5
GAMMA_W          = 0.0
N_MAX            = 30.0

# ── REINFORCE / rollout-policy training hyperparameters ────────────────
GAMMA_RL           = 0.99
TRAIN_SIGMA        = 0.150
ACTION_HEAD_SCALE  = float(RUNTIME_CONTRACT_DEFAULTS['action_head_scale'])
MAX_ACTION_DELTA   = float(RUNTIME_CONTRACT_DEFAULTS['max_action_delta'])
PG_COEF            = 1.10
BC_COEF_START      = 0.30
BC_COEF_END        = 0.15
ZERO_BC_WEIGHT     = 0.08
NONZERO_BC_WEIGHT  = 1.0
SUPPORT_MARGIN     = 0.015
SUPPORT_COEF_START = 0.10
SUPPORT_COEF_END   = 0.00
SAT_WARN_THRESHOLD = MAX_ACTION_DELTA - 0.01
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
CONTROL_DEADBAND_MS   = 0.05
CONTROL_RESPONSE_MS   = 2.25
CONTROL_TARGET_TOL    = 0.010
ENTROPY_COEF          = 0.040
DEAD_ZONE_COEF        = 3.0
POS_SAT_COEF          = 2.5
DEPLOY_ACTIVE_TARGET  = 3.0
VAL_GROUP_FRAC        = 0.20
GRAD_CLIP             = 1.0
BATCH_SIZE            = 256
RANDOM_SEED           = 42

# ── Seed ───────────────────────────────────────────────────────────────
if torch is not None:
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if torch is not None else None

# ── Derived constants (set after data load) ────────────────────────────
T_TARGET = None  # Set dynamically by data_loader / reward

SOFT_SUPPORT_LIMIT = max(DEAD_ZONE, MAX_ACTION_DELTA - SUPPORT_MARGIN)

# Feature indices (convenience)
GPU_FEATURE_IDX = FEATURE_COLS.index('gpu_frame_time')
BIAS_FEATURE_IDX = FEATURE_COLS.index('previous_bias')
FLOOR_DWELL_FEATURE_IDX = FEATURE_COLS.index('floor_dwell_score')
CEILING_DWELL_FEATURE_IDX = FEATURE_COLS.index('ceiling_dwell_score')
