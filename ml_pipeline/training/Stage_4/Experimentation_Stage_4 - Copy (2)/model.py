"""
Stage 4: Policy Network & Helper Functions (Notebook §5-6 utilities)
"""
# Pipeline stage: notebook model/runtime helper layer used by training and diagnostics.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    MAX_ACTION_DELTA, FEATURE_COLS, FEATURE_COUNT,
    DEAD_ZONE, DWELL_FRAMES, BIAS_MIN, BIAS_MAX, FLOOR_MARGIN, CEILING_MARGIN,
    DWELL_ACTIVE_THRESHOLD, FLOOR_DWELL_RECOVERY_GAIN, CEILING_DWELL_CORRECTION_GAIN,
    RECOVERY_ELIGIBLE_BIAS, CORRECTION_ELIGIBLE_BIAS,
    RECOVERY_GROWTH_THRESHOLD, RECOVERY_TRIGGER_FRAMES,
    RECOVERY_FORCE_MULTIPLIER, RECOVERY_FORCE_MAX, RECOVERY_BOOST_BASE,
    RECOVERY_BUDGET_RESET_MARGIN, RECOVERY_BIAS_TARGET,
    FLOOR_RECOVERY_MIN_STRENGTH, CEILING_CORRECTION_MIN_STRENGTH,
    SAFE_RECOVERY_GPU_MARGIN, UNDER_BUDGET_MARGIN, CONTROL_DEADBAND_MS,
    CONTROL_RESPONSE_MS, SOFT_SUPPORT_LIMIT, SUPPORT_MARGIN,
    SAT_WARN_THRESHOLD, DEPLOY_ACTIVE_TARGET,
    GPU_FEATURE_IDX, BIAS_FEATURE_IDX,
    FLOOR_DWELL_FEATURE_IDX, CEILING_DWELL_FEATURE_IDX,
    PG_COEF, device,
)


class PolicyMLP(nn.Module):
    def __init__(self, in_dim, h1, h2, h3, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, 1),
        )
        self.log_sigma = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        mu = self.net(x)
        log_sigma = self.log_sigma
        mu = MAX_ACTION_DELTA * torch.tanh(mu)
        sigma = torch.exp(self.log_sigma).clamp(0.1, 1.0)
        if self.training:
            mu = mu + torch.randn_like(mu) * sigma * 0.05
        return mu.squeeze(-1), log_sigma


def linear_schedule(start, end, progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return start + (end - start) * progress


def apply_runtime_guardrails_np(raw_mu, gpu_ms, start_bias, t_target):
    """Simulate Unity runtime guardrails (dead zone, dwell, recovery)."""
    raw_mu = np.clip(np.asarray(raw_mu, dtype=np.float32), -MAX_ACTION_DELTA, MAX_ACTION_DELTA)
    gpu_ms = np.asarray(gpu_ms, dtype=np.float32)
    applied = np.zeros_like(raw_mu, dtype=np.float32)
    bias_trace = np.zeros_like(raw_mu, dtype=np.float32)
    bias = float(start_bias)
    frames_since_switch = DWELL_FRAMES
    weak_upward_count = 0
    recovery_scalar = 1.0
    weak_downward_count = 0
    correction_scalar = 1.0

    for i, delta in enumerate(raw_mu):
        gpu_t = float(gpu_ms[i]) if i < len(gpu_ms) else float(t_target)
        low_bias = bias <= float(RECOVERY_ELIGIBLE_BIAS)
        high_bias = bias >= float(CORRECTION_ELIGIBLE_BIAS)
        upward_budget_reached = gpu_t >= float(t_target - RECOVERY_BUDGET_RESET_MARGIN)
        downward_budget_reached = gpu_t <= float(t_target + RECOVERY_BUDGET_RESET_MARGIN)

        if upward_budget_reached or not low_bias:
            weak_upward_count = 0
            recovery_scalar = 1.0
        else:
            if float(delta) < float(RECOVERY_GROWTH_THRESHOLD):
                weak_upward_count += 1
                if weak_upward_count >= int(RECOVERY_TRIGGER_FRAMES):
                    recovery_scalar = min(float(RECOVERY_FORCE_MAX), recovery_scalar * float(RECOVERY_FORCE_MULTIPLIER))
                    weak_upward_count = 0
            else:
                weak_upward_count = 0

        if downward_budget_reached or not high_bias:
            weak_downward_count = 0
            correction_scalar = 1.0
        else:
            if float(delta) > -float(RECOVERY_GROWTH_THRESHOLD):
                weak_downward_count += 1
                if weak_downward_count >= int(RECOVERY_TRIGGER_FRAMES):
                    correction_scalar = min(float(RECOVERY_FORCE_MAX), correction_scalar * float(RECOVERY_FORCE_MULTIPLIER))
                    weak_downward_count = 0
            else:
                weak_downward_count = 0

        if recovery_scalar > 1.0:
            delta = max(float(delta), 0.0) + float(RECOVERY_BOOST_BASE) * recovery_scalar
        if correction_scalar > 1.0:
            delta = min(float(delta), 0.0) - float(RECOVERY_BOOST_BASE) * correction_scalar
        delta = float(np.clip(delta, -MAX_ACTION_DELTA, MAX_ACTION_DELTA))

        if abs(delta) < DEAD_ZONE:
            bias_trace[i] = bias
            frames_since_switch += 1
            continue
        if frames_since_switch < DWELL_FRAMES:
            bias_trace[i] = bias
            frames_since_switch += 1
            continue

        new_bias = float(np.clip(bias + float(delta), BIAS_MIN, BIAS_MAX))
        actual = new_bias - bias
        if abs(actual) < DEAD_ZONE:
            bias = new_bias
            bias_trace[i] = bias
            frames_since_switch += 1
            continue

        applied[i] = actual
        bias = new_bias
        bias_trace[i] = bias
        frames_since_switch = 0

    bias_trace = np.where(bias_trace == 0.0, bias, bias_trace).astype(np.float32)
    return applied, bias_trace


def build_deployment_frame(model, df_split, scaler, t_target, group_col='source_file'):
    """Simulate deployment: run model + guardrails on sequence groups."""
    rows = []
    model.eval()
    with torch.no_grad():
        for _, grp in df_split.sort_values([group_col, 'step']).groupby(group_col, sort=False):
            X_seq = scaler.transform(grp[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)
            X_seq_t = torch.tensor(X_seq, dtype=torch.float32, device=device)
            mu, _ = model(X_seq_t)
            start_bias = float(grp['previous_bias'].iloc[0]) if 'previous_bias' in grp.columns else 1.0
            raw_mu = mu.detach().cpu().numpy().astype(np.float32)
            applied_mu, deployed_bias = apply_runtime_guardrails_np(
                raw_mu, grp['gpu_frame_time'].values.astype(np.float32), start_bias, t_target
            )
            part = grp[['episode', 'step', 'gpu_frame_time', 'avg_screen_coverage', 'action_delta', 'previous_bias']].copy()
            part['raw_mu'] = raw_mu
            part['applied_mu'] = applied_mu
            part['deployed_bias'] = deployed_bias
            rows.append(part)

    if not rows:
        return pd.DataFrame(columns=[
            'episode', 'step', 'gpu_frame_time', 'avg_screen_coverage',
            'action_delta', 'previous_bias', 'raw_mu', 'applied_mu', 'deployed_bias',
        ])
    return pd.concat(rows, ignore_index=True)


def deployment_metrics(model, df_split, scaler, t_target, group_col='source_file'):
    """Compute deployment simulation metrics."""
    deploy_df = build_deployment_frame(model, df_split, scaler, t_target, group_col)
    if deploy_df.empty:
        return {k: np.nan for k in [
            'deploy_mae', 'deploy_active_pct', 'deploy_std', 'mean_deployed_bias',
            'floor_pct', 'r_budget_mean', 'r_quality_mean', 'r_recovery_mean',
            'r_floor_penalty_mean', 'r_target_zone_mean',
            'correct_dir_pct', 'corr_mu_gpu_next', 'r_total_std',
        ]}

    actual = deploy_df['action_delta'].values.astype(np.float32)
    applied = deploy_df['applied_mu'].values.astype(np.float32)
    gpu = deploy_df['gpu_frame_time'].values.astype(np.float32)
    bias_after = deploy_df['deployed_bias'].values.astype(np.float32)

    alpha = 1.5
    gpu_next = np.clip(gpu + alpha * applied, 2.0, 16.0)
    error = gpu_next - float(t_target)
    r_dir = 2.0 * np.where(error > 0, -applied, applied)
    r_floor_pen = -0.3 * (bias_after <= (BIAS_MIN + FLOOR_MARGIN)).astype(np.float32)

    correct = ((error > 0) & (applied < 0)) | ((error < 0) & (applied > 0))
    if len(gpu_next) > 1 and np.std(applied) > 1e-6 and np.std(gpu_next) > 1e-6:
        corr = np.corrcoef(applied, gpu_next)[0, 1]
    else:
        corr = 0.0
    r_total_std = float((r_dir + r_floor_pen).std())
    correct_dir_pct = float(correct.mean() * 100.0)
    print(f"[Phase 4] Corr(mu, gpu_next): {corr:.3f} | r_total.std: {r_total_std:.3f} | CorrectDir%: {correct_dir_pct:.3f}")

    return {
        'deploy_mae': float(np.mean(np.abs(applied - actual))),
        'deploy_active_pct': float((np.abs(applied) > 1e-6).mean() * 100.0),
        'deploy_std': float(np.std(applied)),
        'mean_deployed_bias': float(np.mean(bias_after)),
        'floor_pct': float((bias_after <= (BIAS_MIN + FLOOR_MARGIN)).mean() * 100.0),
        'r_budget_mean': float(np.mean(r_dir)),
        'r_quality_mean': 0.0,
        'r_recovery_mean': 0.0,
        'r_floor_penalty_mean': float(np.mean(r_floor_pen)),
        'r_target_zone_mean': 0.0,
        'correct_dir_pct': correct_dir_pct,
        'corr_mu_gpu_next': float(corr),
        'r_total_std': r_total_std,
    }


def build_control_target_torch(gpu_raw, prev_bias_raw, floor_dwell_raw, ceiling_dwell_raw, t_target):
    """Three-regime control target: low bias→recover, mid→hold, high→trim."""
    bias_span = max(BIAS_MAX - BIAS_MIN, 1e-6)
    over_budget_mask = gpu_raw > (t_target + CONTROL_DEADBAND_MS)
    under_budget_mask = gpu_raw < (t_target - UNDER_BUDGET_MARGIN)
    near_target_mask = (~over_budget_mask) & (~under_budget_mask)

    up_room = ((BIAS_MAX - prev_bias_raw) / bias_span).clamp(0.0, 1.0)
    down_room = ((prev_bias_raw - BIAS_MIN) / bias_span).clamp(0.0, 1.0)
    over_strength = ((gpu_raw - (t_target + CONTROL_DEADBAND_MS)) / CONTROL_RESPONSE_MS).clamp(0.0, 1.0)
    under_strength = (((t_target - UNDER_BUDGET_MARGIN) - gpu_raw) / CONTROL_RESPONSE_MS).clamp(0.0, 1.0)

    floor_dwell_raw = floor_dwell_raw.clamp(0.0, 1.0)
    ceiling_dwell_raw = ceiling_dwell_raw.clamp(0.0, 1.0)
    floor_dwell_mask = floor_dwell_raw >= DWELL_ACTIVE_THRESHOLD
    ceiling_dwell_mask = ceiling_dwell_raw >= DWELL_ACTIVE_THRESHOLD

    floor_mask = prev_bias_raw <= (BIAS_MIN + FLOOR_MARGIN)
    low_bias_mask = prev_bias_raw <= RECOVERY_ELIGIBLE_BIAS
    mid_bias_mask = (prev_bias_raw > RECOVERY_ELIGIBLE_BIAS) & (prev_bias_raw < CORRECTION_ELIGIBLE_BIAS)
    high_bias_mask = prev_bias_raw >= CORRECTION_ELIGIBLE_BIAS
    ceiling_mask = prev_bias_raw >= (BIAS_MAX - CEILING_MARGIN)
    safe_recovery_mask = gpu_raw <= (t_target + SAFE_RECOVERY_GPU_MARGIN)
    ceiling_guard_mask = gpu_raw >= (t_target - SAFE_RECOVERY_GPU_MARGIN)

    recovery_deficit = ((RECOVERY_BIAS_TARGET - prev_bias_raw) / bias_span).clamp(0.0, 1.0)
    recovery_strength = torch.clamp(
        recovery_deficit * (1.0 + FLOOR_DWELL_RECOVERY_GAIN * floor_dwell_raw),
        FLOOR_RECOVERY_MIN_STRENGTH, 1.0,
    )
    ceiling_excess = ((prev_bias_raw - CORRECTION_ELIGIBLE_BIAS) / bias_span).clamp(0.0, 1.0)
    ceiling_strength = torch.clamp(
        ceiling_excess * (1.0 + CEILING_DWELL_CORRECTION_GAIN * ceiling_dwell_raw),
        CEILING_CORRECTION_MIN_STRENGTH, 1.0,
    )

    positive_floor_target = MAX_ACTION_DELTA * recovery_strength * up_room
    positive_under_target = MAX_ACTION_DELTA * under_strength * up_room
    negative_over_target = -MAX_ACTION_DELTA * over_strength * down_room
    negative_ceiling_target = -MAX_ACTION_DELTA * ceiling_strength * down_room

    control_target = torch.zeros_like(gpu_raw)

    # REGIME 1: LOW BIAS - Recover upward when under budget
    low_bias_under_budget = low_bias_mask & under_budget_mask
    min_directional_action = min(0.1, float(MAX_ACTION_DELTA))
    control_target = torch.where(low_bias_under_budget, positive_under_target.clamp(min_directional_action, MAX_ACTION_DELTA), control_target)

    # REGIME 2: MID BIAS - Hold when near target
    mid_bias_near_target = mid_bias_mask & near_target_mask
    control_target = torch.where(mid_bias_near_target, torch.zeros_like(control_target), control_target)

    # REGIME 3: HIGH BIAS - Trim downward when over budget
    high_bias_over_budget = high_bias_mask & over_budget_mask
    control_target = torch.where(high_bias_over_budget, negative_over_target.clamp(-MAX_ACTION_DELTA, -min_directional_action), control_target)

    # Fallbacks
    control_target = torch.where(over_budget_mask & ~high_bias_over_budget, negative_over_target, control_target)
    control_target = torch.where(under_budget_mask & ~low_bias_under_budget, positive_under_target, control_target)

    # Recovery floors & ceiling guards
    control_target = torch.where(
        (low_bias_mask | floor_mask | floor_dwell_mask) & safe_recovery_mask,
        torch.maximum(control_target, positive_floor_target), control_target,
    )
    control_target = torch.where(
        (high_bias_mask | ceiling_mask | ceiling_dwell_mask) & ceiling_guard_mask,
        torch.minimum(control_target, negative_ceiling_target), control_target,
    )

    near_target_hold_mask = mid_bias_mask & near_target_mask
    near_target_low_bias_mask = low_bias_mask & near_target_mask
    near_target_high_bias_mask = high_bias_mask & near_target_mask

    return (
        control_target,
        over_budget_mask.float(),
        under_budget_mask.float(),
        near_target_hold_mask.float(),
        near_target_low_bias_mask.float(),
        near_target_high_bias_mask.float(),
    )
