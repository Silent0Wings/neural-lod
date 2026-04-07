"""
Stage 4: Policy Diagnostics (Notebook §8)
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import (
    FEATURE_COLS, ACTION_HEAD_SCALE, SOFT_SUPPORT_LIMIT,
    SAT_WARN_THRESHOLD, TRAIN_SIGMA, BIAS_MIN, FLOOR_MARGIN, PLOTS_DIR, device,
)
from model import build_deployment_frame


def run_diagnostics(model, df_clean, X_scaled, scaler, t_target, group_col='source_file', run_plots=True):
    """Run full action diagnostics and generate plots."""
    model = model.to(device)
    for name, buf in model.named_buffers():
        model._buffers[name] = buf.to(device)

    X_all_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    logging.info(f'Model device: {next(model.parameters()).device}')
    logging.info(f'Input device: {X_all_t.device}')

    model.eval()
    with torch.no_grad():
        mu_all = model(X_all_t)[0].cpu().numpy().flatten()

    deployment_frame = build_deployment_frame(model, df_clean, scaler, t_target, group_col)
    raw_mu_all = deployment_frame['raw_mu'].values.astype('float32')
    applied_mu_all = deployment_frame['applied_mu'].values.astype('float32')
    actual_actions = deployment_frame['action_delta'].values.astype('float32')

    mae_all = float(np.mean(np.abs(raw_mu_all - actual_actions)))
    deploy_mae_all = float(np.mean(np.abs(applied_mu_all - actual_actions)))
    support_viol_pct = float((np.abs(raw_mu_all) > SOFT_SUPPORT_LIMIT).mean() * 100)
    sat_pct = float((np.abs(raw_mu_all) >= SAT_WARN_THRESHOLD).mean() * 100)
    pos_sat_pct = float((raw_mu_all >= SAT_WARN_THRESHOLD).mean() * 100)
    neg_sat_pct = float((raw_mu_all <= -SAT_WARN_THRESHOLD).mean() * 100)
    zero_pct = float((np.abs(raw_mu_all) < 0.005).mean() * 100)
    deploy_active_pct = float((np.abs(applied_mu_all) > 1e-6).mean() * 100)
    mean_deployed_bias = float(np.mean(deployment_frame['deployed_bias'].values.astype('float32')))
    floor_pct = float((deployment_frame['deployed_bias'].values.astype('float32') <= (BIAS_MIN + FLOOR_MARGIN)).mean() * 100)

    logging.info(f'Fixed training sigma: {TRAIN_SIGMA:.6f}')
    logging.info(f'Deployable action envelope: +/-{ACTION_HEAD_SCALE:.3f}')
    logging.info(
        f'Predicted mu stats | mean={raw_mu_all.mean():.6f} | std={raw_mu_all.std():.6f} | '
        f'min={raw_mu_all.min():.6f} | max={raw_mu_all.max():.6f}'
    )
    logging.info(
        f'Action fit | raw_MAE={mae_all:.6f} | deploy_MAE={deploy_mae_all:.6f} | '
        f'support%={support_viol_pct:.2f} | sat%={sat_pct:.2f} | '
        f'pos_sat%={pos_sat_pct:.2f} | neg_sat%={neg_sat_pct:.2f} | '
        f'zero%={zero_pct:.2f} | deploy_active%={deploy_active_pct:.2f} | '
        f'mean_bias={mean_deployed_bias:.3f} | floor%={floor_pct:.2f}'
    )

    if run_plots:
        # Diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        ax = axes[0, 0]
        ax.hist(raw_mu_all, bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
        for v in sorted(np.unique(actual_actions)):
            ax.axvline(v, color='red', linestyle='--', alpha=0.3)
        ax.axvline(SOFT_SUPPORT_LIMIT, color='blue', linestyle=':', alpha=0.7)
        ax.axvline(-SOFT_SUPPORT_LIMIT, color='blue', linestyle=':', alpha=0.7)
        ax.set_title('Predicted Action (raw mu) Distribution')

        ax = axes[0, 1]
        ax.hist(actual_actions, bins=21, color='#4CAF50', alpha=0.6, edgecolor='white', label='actual rollout')
        ax.hist(applied_mu_all, bins=21, color='#FF9800', alpha=0.6, edgecolor='white', label='simulated deploy')
        ax.set_title('Applied Action Distribution'); ax.legend()

        ax = axes[1, 0]
        ax.scatter(actual_actions[:2000], applied_mu_all[:2000], alpha=0.2, s=6, color='#9C27B0')
        ax.plot([-ACTION_HEAD_SCALE, ACTION_HEAD_SCALE], [-ACTION_HEAD_SCALE, ACTION_HEAD_SCALE], 'r--', alpha=0.5)
        ax.set_title('Deploy Simulation vs. Logged Actions')

        ax = axes[1, 1]
        ax.scatter(deployment_frame['gpu_frame_time'].values[:3000], applied_mu_all[:3000], alpha=0.2, s=6, color='#FF9800')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(t_target, color='blue', linestyle='--', alpha=0.5, label=f'Target={t_target}ms')
        ax.set_title('Policy Response: GPU Time vs. Deploy Action'); ax.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'action_diagnostics.png', dpi=150)
        plt.show()
        logging.info('Saved: action_diagnostics.png')
    else:
        logging.info('Skipped plot: action_diagnostics.png')

    # Warnings
    if pos_sat_pct > 10.0:
        logging.warning('positive mu saturation too high')
    if neg_sat_pct > 35.0:
        logging.warning('negative mu saturation too high')
    if floor_pct > 50.0:
        logging.warning('deployed bias near minimum floor too often')
    if deploy_mae_all > 0.08:
        logging.warning('deploy MAE still high')

    return {
        'deploy_mae_all': deploy_mae_all,
        'pos_sat_pct': pos_sat_pct,
        'neg_sat_pct': neg_sat_pct,
        'floor_pct': floor_pct,
        'deploy_active_pct': deploy_active_pct,
        'mean_deployed_bias': mean_deployed_bias,
        'deployment_frame': deployment_frame,
    }
