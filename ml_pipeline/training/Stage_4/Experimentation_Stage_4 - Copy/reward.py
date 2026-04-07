"""
Stage 4: Feature Scaling & Reward Computation (Notebook §3-4)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from config import (
    MODEL_DIR, PLOTS_DIR, FEATURE_COLS, FEATURE_COUNT,
    BIAS_MIN, BIAS_MAX, UNDER_BUDGET_MARGIN, BONUS_SCALE,
    OVER_BUDGET_COEF, UNDER_BUDGET_COEF, RECOVERY_REWARD_COEF,
    REWARD_CLIP, TARGET_PROX_SIGMA, CONTROL_DEADBAND_MS,
    ACTION_HEAD_SCALE, DEAD_ZONE, PG_COEF, BC_COEF_START, BC_COEF_END,
)


def fit_scaler(df_clean: pd.DataFrame, t_target: float):
    """Fit StandardScaler on features and save scaler constants JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_raw = df_clean[FEATURE_COLS].values.astype(np.float32)
    if X_raw.shape[0] == 0:
        raise ValueError('No samples available for scaling')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    zero_scale = [
        (name, float(scale))
        for name, scale in zip(FEATURE_COLS, scaler.scale_)
        if scale < 1e-8
    ]
    if zero_scale:
        raise ValueError(f'Zero-scale features detected: {zero_scale}')

    scaler_data = {
        'feature_names': FEATURE_COLS,
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        't_target_ms': float(t_target),
        'action_head_scale': ACTION_HEAD_SCALE,
        'max_action_delta': ACTION_HEAD_SCALE,
        'dead_zone': 0.02,
        'dwell_frames': 5,
        'bias_min': 0.30,
        'bias_max': 2.00,
        'inference_interval': 2,
        'pg_coef': float(PG_COEF),
        'bc_coef_start': BC_COEF_START,
        'bc_coef_end': BC_COEF_END,
        'gpu_target_ms_base': t_target,
        'gpu_target_ms_min': 4.0,
        'gpu_target_ms_max': 6.5,
        'scene_complexity_scale': 0.8,
        'scene_complexity_normalization': 50.0,
        'mode_headroom_margin_ms': 0.5,
        'mode_headroom_fps_floor': 48,
        'mode_budget_fps_floor': 45,
        'exploration_upward_weight': 0.4,
        'recovery_downward_weight': 0.6,
        'thrash_penalty_weight': 0.1,
        'nominal_action_regularization': 0.08,
        'exploration_action_regularization': 0.05,
        'recovery_action_regularization': 0.1,
    }

    scaler_path = MODEL_DIR / 'rl_scaler_constants.json'
    with open(scaler_path, 'w', encoding='utf-8') as f:
        json.dump(scaler_data, f, indent=2)

    print('Scaling OK:', X_scaled.shape)
    print('Saved:', scaler_path)

    # Plot feature distributions
    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        col_data = df_clean[col]
        clipped = col_data.clip(lower=col_data.quantile(0.01), upper=col_data.quantile(0.99))
        axes[i].hist(clipped, bins=40, alpha=0.8)
        axes[i].set_title(col, fontsize=9)
    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_distributions.png', dpi=150)
    plt.show()

    print('Final rows:', len(df_clean))
    print('Episodes  :', df_clean['episode'].nunique())

    return scaler, X_scaled


def print_data_distribution(df_clean: pd.DataFrame, t_target: float):
    """Print per-file and overall budget distribution analysis."""
    print("\n=== DATA DISTRIBUTION ANALYSIS ===")
    for source_file in df_clean['source_file'].unique():
        subset = df_clean[df_clean['source_file'] == source_file]
        gpu = subset['gpu_frame_time']
        print(f"\n{source_file}:")
        print(f"  Frames: {len(subset):,}")
        print(f"  GPU mean: {gpu.mean():.2f} ms")
        print(f"  Over-budget (>{t_target:.3f}ms): {(gpu > t_target).mean() * 100:.1f}%")
        print(f"  Under-budget (<={t_target:.3f}ms): {(gpu <= t_target).mean() * 100:.1f}%")

    print(f"\nTotal frames: {len(df_clean):,}")
    overall_over = (df_clean['gpu_frame_time'] > t_target).mean() * 100
    overall_under = (df_clean['gpu_frame_time'] <= t_target).mean() * 100
    print(f"Overall over-budget (>{t_target:.3f}ms): {overall_over:.1f}%")
    print(f"Overall under-budget (<={t_target:.3f}ms): {overall_under:.1f}%")


def compute_rewards(df_clean: pd.DataFrame, t_target: float) -> pd.DataFrame:
    """Compute budget-tracking rewards with quality preservation."""
    df_clean = df_clean.sort_values(['episode', 'step']).reset_index(drop=True)

    gpu = df_clean['gpu_frame_time'].values.astype('float32')
    coverage = df_clean['avg_screen_coverage'].values.astype('float32')
    bias_before = df_clean['lod_bias_before_action'].values.astype('float32')
    bias_after = df_clean['lod_bias_after_action'].values.astype('float32')

    coverage_q95 = float(np.quantile(coverage, 0.95)) if len(coverage) else 1.0
    coverage_scale = max(coverage_q95, 1e-4)
    bias_norm = np.clip((bias_after - BIAS_MIN) / max(BIAS_MAX - BIAS_MIN, 1e-6), 0.0, 1.0).astype('float32')

    over_budget = np.clip(gpu - t_target, 0.0, None).astype('float32')
    under_budget = np.clip(t_target - gpu, 0.0, None).astype('float32')
    under_budget_headroom = (under_budget >= UNDER_BUDGET_MARGIN).astype('float32')

    target_proximity = np.exp(-((gpu - t_target) ** 2) / (2.0 * TARGET_PROX_SIGMA ** 2)).astype('float32')
    r_budget = (
        BONUS_SCALE * target_proximity
        - OVER_BUDGET_COEF * over_budget
        - UNDER_BUDGET_COEF * under_budget
    )
    r_recovery = RECOVERY_REWARD_COEF * under_budget_headroom * np.clip(bias_after - bias_before, 0.0, None).astype('float32')

    rewards = np.clip(r_budget + r_recovery, -REWARD_CLIP, REWARD_CLIP).astype('float32')

    if len(rewards) == 0:
        raise ValueError('rewards is empty -- df_clean is empty')

    pos_pct = float((rewards > 0).mean() * 100)
    print(f'Reward | mean={rewards.mean():.4f} std={rewards.std():.4f} min={rewards.min():.4f} max={rewards.max():.4f}')
    print(f'Positive steps: {(rewards > 0).sum():,}/{len(rewards):,} ({pos_pct:.1f}%)')
    print(f'Under-budget headroom steps: {under_budget_headroom.sum():,.0f}/{len(rewards):,}')

    if pos_pct < 1.0:
        raise ValueError(
            f'Only {pos_pct:.1f}% positive rewards -- check GPU data quality.\n'
            'Ensure training_data_*.csv has valid per-frame gpu_frame_time_ms values.'
        )

    df_clean = df_clean.copy()
    df_clean['reward'] = rewards
    return df_clean
