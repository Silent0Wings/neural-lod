"""
Stage 4: Optuna Tuning + REINFORCE Training Loop (Notebook §6-7)
"""
# Pipeline stage: notebook sections 6-7, returns, split, Optuna, and final training.
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import GroupShuffleSplit

from config import (
    FEATURE_COLS, FEATURE_COUNT, GAMMA_RL, ACTION_HEAD_SCALE, DEAD_ZONE,
    PG_COEF, BC_COEF_START, BC_COEF_END, SUPPORT_COEF_START, SUPPORT_COEF_END,
    SUPPORT_MARGIN, SAT_WARN_THRESHOLD, DEPLOY_ACTIVE_TARGET,
    VAL_GROUP_FRAC, GRAD_CLIP, BATCH_SIZE, RANDOM_SEED,
    BIAS_MIN, FLOOR_MARGIN, CEILING_MARGIN,
    GPU_FEATURE_IDX, BIAS_FEATURE_IDX,
    FLOOR_DWELL_FEATURE_IDX, CEILING_DWELL_FEATURE_IDX,
    SOFT_SUPPORT_LIMIT, PLOTS_DIR, device,
)
from model import (
    PolicyMLP, linear_schedule, deployment_metrics,
    build_control_target_torch,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_returns(rewards_ep, gamma):
    G = np.zeros_like(rewards_ep, dtype='float32')
    running = 0.0
    for t in reversed(range(len(rewards_ep))):
        running = float(rewards_ep[t]) + gamma * running
        G[t] = running
    return G


def prepare_returns(df_clean):
    """Compute discounted returns per episode, normalize globally."""
    if len(df_clean) == 0:
        raise ValueError('df_clean is empty')
    if 'reward' not in df_clean.columns:
        raise ValueError("'reward' column missing -- run reward computation first")

    attrs = dict(df_clean.attrs)
    df_clean = df_clean.sort_values(['episode', 'step']).reset_index(drop=True)
    df_clean.attrs.update(attrs)
    returns_list = []
    for _, grp in df_clean.groupby('episode', sort=False):
        returns_list.append(compute_returns(grp['reward'].values, GAMMA_RL))

    df_clean['G_t'] = np.concatenate(returns_list)
    G_mean = df_clean['G_t'].mean()
    G_std = df_clean['G_t'].std() + 1e-8
    df_clean['G_t_norm'] = (df_clean['G_t'] - G_mean) / G_std

    print(f'Returns shape: {df_clean["G_t_norm"].shape}')
    print(f'G_t   | mean={df_clean["G_t"].mean():.4f} std={df_clean["G_t"].std():.4f}')
    print(f'G_norm| mean={df_clean["G_t_norm"].mean():.4f} std={df_clean["G_t_norm"].std():.4f}')
    return df_clean


def split_data(df_clean, X_scaled):
    """Group-based holdout split."""
    A_all = df_clean['action_delta'].values.astype('float32')
    G_all = df_clean['G_t_norm'].values.astype('float32')
    group_col = 'source_file' if 'source_file' in df_clean.columns and df_clean['source_file'].nunique() >= 2 else 'episode'
    groups = df_clean[group_col].astype(str).values
    unique_groups = pd.Index(pd.Series(groups).unique())
    if len(unique_groups) < 2:
        raise ValueError(f'Need at least 2 unique {group_col} groups; found {len(unique_groups)}.')

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_GROUP_FRAC, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X_scaled, A_all, groups=groups))

    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    A_tr, A_val = A_all[train_idx], A_all[val_idx]
    G_tr, G_val = G_all[train_idx], G_all[val_idx]
    df_train = df_clean.iloc[train_idx].copy()
    df_val = df_clean.iloc[val_idx].copy()

    print(f'Holdout split uses: {group_col}')
    print(f'Train groups      : {df_train[group_col].nunique()}')
    print(f'Val groups        : {df_val[group_col].nunique()}')
    print(f'Train rows        : {len(df_train):,}')
    print(f'Val rows          : {len(df_val):,}')

    # Action support
    ACTION_SUPPORT = np.array(sorted(df_clean['action_delta'].dropna().unique().tolist()), dtype=np.float32)
    BOOTSTRAP_ACTION_LIMIT = float(np.max(np.abs(ACTION_SUPPORT))) if ACTION_SUPPORT.size else 0.0
    print(f'Observed rollout action support: {ACTION_SUPPORT.tolist()}')
    print(f'Observed support abs max      : +/-{BOOTSTRAP_ACTION_LIMIT:.3f}')
    print(f'Deployable action envelope    : +/-{ACTION_HEAD_SCALE:.3f}')
    print(f'Soft support warning limit    : +/-{SOFT_SUPPORT_LIMIT:.3f}')

    return (
        group_col,
        torch.tensor(X_tr, dtype=torch.float32, device=device),
        torch.tensor(A_tr, dtype=torch.float32, device=device),
        torch.tensor(G_tr, dtype=torch.float32, device=device),
        torch.tensor(X_val, dtype=torch.float32, device=device),
        torch.tensor(A_val, dtype=torch.float32, device=device),
        torch.tensor(G_val, dtype=torch.float32, device=device),
        df_train, df_val,
    )


def reinforce_loss(model, X, A, G, scaler, t_target, bc_coef=0.05, support_coef=0.1, progress=1.0):
    mu, _ = model(X)
    if model.training:
        mu = mu + torch.randn_like(mu) * 0.05

    gpu_raw = X[:, GPU_FEATURE_IDX] * float(scaler.scale_[GPU_FEATURE_IDX]) + float(scaler.mean_[GPU_FEATURE_IDX])
    prev_bias_raw = X[:, BIAS_FEATURE_IDX] * float(scaler.scale_[BIAS_FEATURE_IDX]) + float(scaler.mean_[BIAS_FEATURE_IDX])

    alpha = 1.5
    gpu_next = torch.clamp(gpu_raw + alpha * mu, min=2.0, max=16.0)
    error = gpu_next - t_target

    r_dir = 2.0 * torch.where(error > 0, -mu, mu)
    floor_distance = torch.clamp(FLOOR_MARGIN - (prev_bias_raw - BIAS_MIN), min=0)
    r_floor_pen = -2.0 * (floor_distance ** 2)
    r_total = r_dir + r_floor_pen
    loss = -torch.mean(r_total)

    # BC loss toward control target
    floor_dwell_raw = X[:, FLOOR_DWELL_FEATURE_IDX] * float(scaler.scale_[FLOOR_DWELL_FEATURE_IDX]) + float(scaler.mean_[FLOOR_DWELL_FEATURE_IDX])
    ceiling_dwell_raw = X[:, CEILING_DWELL_FEATURE_IDX] * float(scaler.scale_[CEILING_DWELL_FEATURE_IDX]) + float(scaler.mean_[CEILING_DWELL_FEATURE_IDX])
    with torch.no_grad():
        control_target, *_ = build_control_target_torch(gpu_raw, prev_bias_raw, floor_dwell_raw, ceiling_dwell_raw, t_target)
    bc_loss = torch.mean((mu - control_target) ** 2)
    outside_support_penalty = torch.mean(torch.relu(torch.abs(mu) - SOFT_SUPPORT_LIMIT) ** 2)

    return (PG_COEF * loss) + (bc_coef * bc_loss) + (support_coef * outside_support_penalty)


def eval_metrics(model, X, A, G, df_split, scaler, t_target, group_col, progress=1.0):
    model.eval()
    with torch.no_grad():
        mu, _ = model(X)
        gpu_raw = X[:, GPU_FEATURE_IDX] * float(scaler.scale_[GPU_FEATURE_IDX]) + float(scaler.mean_[GPU_FEATURE_IDX])
        prev_bias_raw = X[:, BIAS_FEATURE_IDX] * float(scaler.scale_[BIAS_FEATURE_IDX]) + float(scaler.mean_[BIAS_FEATURE_IDX])
        floor_dwell_raw = X[:, FLOOR_DWELL_FEATURE_IDX] * float(scaler.scale_[FLOOR_DWELL_FEATURE_IDX]) + float(scaler.mean_[FLOOR_DWELL_FEATURE_IDX])
        ceiling_dwell_raw = X[:, CEILING_DWELL_FEATURE_IDX] * float(scaler.scale_[CEILING_DWELL_FEATURE_IDX]) + float(scaler.mean_[CEILING_DWELL_FEATURE_IDX])

        control_target, over_budget_mask, under_budget_mask, near_target_hold_mask, near_target_low_bias_mask, near_target_high_bias_mask = build_control_target_torch(
            gpu_raw, prev_bias_raw, floor_dwell_raw, ceiling_dwell_raw, t_target,
        )
        headroom_mask = under_budget_mask > 0
        near_target_mask = (near_target_hold_mask + near_target_low_bias_mask + near_target_high_bias_mask) > 0
        over_budget_mask = over_budget_mask > 0

        val_loss = reinforce_loss(model, X, A, G, scaler, t_target, progress=progress).item()
        mae = torch.mean(torch.abs(mu - A)).item()
        control_mae = torch.mean(torch.abs(mu - control_target)).item()
        support_viol_pct = torch.mean((mu.abs() > SOFT_SUPPORT_LIMIT).float()).item() * 100.0
        sat_pct = torch.mean((mu.abs() >= SAT_WARN_THRESHOLD).float()).item() * 100.0
        pos_sat_pct = torch.mean((mu >= SAT_WARN_THRESHOLD).float()).item() * 100.0
        neg_sat_pct = torch.mean((mu <= -SAT_WARN_THRESHOLD).float()).item() * 100.0
        zero_pct = torch.mean((mu.abs() < 0.005).float()).item() * 100.0
        mean_mu = torch.mean(mu).item()
        neg_mu_pct = torch.mean((mu < 0).float()).item() * 100.0
        pos_mu_pct = torch.mean((mu > 0).float()).item() * 100.0
        headroom_neg_pct = torch.mean((mu[headroom_mask] < 0).float()).item() * 100.0 if headroom_mask.any() else np.nan
        near_target_neg_pct = torch.mean((mu[near_target_mask] < 0).float()).item() * 100.0 if near_target_mask.any() else np.nan
        over_budget_pos_pct = torch.mean((mu[over_budget_mask] > 0).float()).item() * 100.0 if over_budget_mask.any() else np.nan

    deploy = deployment_metrics(model, df_split, scaler, t_target, group_col)
    return (
        val_loss, mae, support_viol_pct, sat_pct, pos_sat_pct, neg_sat_pct, zero_pct,
        deploy['deploy_mae'], deploy['deploy_active_pct'], deploy['mean_deployed_bias'],
        deploy['floor_pct'], control_mae, mean_mu, neg_mu_pct, pos_mu_pct,
        headroom_neg_pct, near_target_neg_pct, over_budget_pos_pct,
        deploy['r_budget_mean'], deploy['r_quality_mean'], deploy['r_recovery_mean'],
        deploy['r_floor_penalty_mean'], deploy['r_target_zone_mean'],
        deploy['correct_dir_pct'], deploy['corr_mu_gpu_next'], deploy['r_total_std'],
    )


def run_optuna(X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col, n_trials=2):
    """Run Optuna hyperparameter search."""
    def run_trial(h1, h2, h3, lr, dropout, epochs=30):
        model_trial = PolicyMLP(FEATURE_COUNT, h1, h2, h3, dropout).to(device)
        optimizer = optim.AdamW(model_trial.parameters(), lr=lr)
        n = len(X_tr_t)
        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            model_trial.train()
            idx = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                b = idx[start:start + BATCH_SIZE]
                optimizer.zero_grad()
                loss = reinforce_loss(model_trial, X_tr_t[b], A_tr_t[b], G_tr_t[b], scaler, t_target, progress=progress)
                loss.backward()
                nn.utils.clip_grad_norm_(model_trial.parameters(), GRAD_CLIP)
                optimizer.step()

        metrics = eval_metrics(model_trial, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col, progress=1.0)
        val_loss, mae, support_viol_pct, sat_pct, pos_sat_pct, neg_sat_pct, zero_pct, deploy_mae, deploy_active_pct, mean_deployed_bias, floor_pct, control_mae, mean_mu, neg_mu_pct, pos_mu_pct, headroom_neg_pct, near_target_neg_pct, over_budget_pos_pct, *_ = metrics
        deploy_inactive_penalty = max(0.0, DEPLOY_ACTIVE_TARGET - deploy_active_pct)
        headroom_neg_penalty = 0.0 if np.isnan(headroom_neg_pct) else headroom_neg_pct
        near_target_neg_penalty = 0.0 if np.isnan(near_target_neg_pct) else near_target_neg_pct
        over_budget_pos_penalty = 0.0 if np.isnan(over_budget_pos_pct) else over_budget_pos_pct
        score = (
            val_loss + 0.35 * deploy_mae + 0.75 * control_mae + 0.02 * sat_pct
            + 0.05 * pos_sat_pct + 0.06 * neg_sat_pct + 0.03 * floor_pct
            + 0.07 * near_target_neg_penalty + 0.05 * headroom_neg_penalty
            + 0.05 * over_budget_pos_penalty + 0.25 * deploy_inactive_penalty
            + 0.20 * max(0.0, (BIAS_MIN + FLOOR_MARGIN) - mean_deployed_bias)
        )
        logging.info(f'  score={score:.4f} val_loss={val_loss:.4f} control_mae={control_mae:.4f} deploy_mae={deploy_mae:.4f}')
        return score

    def objective(trial):
        h1 = trial.suggest_categorical('h1', [128, 256, 512])
        h2 = trial.suggest_categorical('h2', [64, 128, 256])
        h3 = trial.suggest_categorical('h3', [32, 64, 128])
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.05, 0.30)
        return run_trial(h1, h2, h3, lr, dropout)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    logging.info(f'Best objective : {study.best_value:.6f}')
    logging.info(f'Best params    : {study.best_params}')
    return study


def train_final(study, X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col, epochs=150, run_plots=True):
    """Full REINFORCE training with best Optuna params."""
    bp = study.best_params
    logging.info(f'Best params loaded: {bp}')

    model = PolicyMLP(FEATURE_COUNT, h1=bp['h1'], h2=bp['h2'], h3=bp['h3'], dropout=bp['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=bp['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {k: [] for k in [
        'train', 'val', 'mae', 'control_mae', 'deploy_mae', 'support_pct',
        'sat_pct', 'pos_sat_pct', 'neg_sat_pct', 'deploy_active_pct',
        'mean_deployed_bias', 'floor_pct', 'mean_mu', 'neg_mu_pct', 'pos_mu_pct',
        'headroom_neg_pct', 'near_target_neg_pct', 'over_budget_pos_pct',
        'r_budget_mean', 'r_quality_mean', 'r_recovery_mean',
        'r_floor_penalty_mean', 'r_target_zone_mean', 'bc_coef', 'support_coef',
        'correct_dir_pct', 'corr_mu_gpu_next', 'r_total_std',
    ]}

    n = len(X_tr_t)
    logging.info(f'TRAINING START | epochs={epochs} lr={bp["lr"]:.5g}')

    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        bc_coef = linear_schedule(BC_COEF_START, BC_COEF_END, progress)
        support_coef = linear_schedule(SUPPORT_COEF_START, SUPPORT_COEF_END, progress)

        model.train()
        idx = torch.randperm(n)
        t_losses = []
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = reinforce_loss(model, X_tr_t[b], A_tr_t[b], G_tr_t[b], scaler, t_target, bc_coef=bc_coef, support_coef=support_coef, progress=progress)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            t_losses.append(loss.item())

        scheduler.step()

        metrics = eval_metrics(model, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col)
        v_loss, mae, support_viol_pct, sat_pct, pos_sat_pct, neg_sat_pct, zero_pct, deploy_mae, deploy_active_pct, mean_deployed_bias, floor_pct, control_mae, mean_mu, neg_mu_pct, pos_mu_pct, headroom_neg_pct, near_target_neg_pct, over_budget_pos_pct, r_budget_mean, r_quality_mean, r_recovery_mean, r_floor_penalty_mean, r_target_zone_mean, correct_dir_pct, corr_mu_gpu_next, r_total_std = metrics

        train_loss = float(np.mean(t_losses))
        vals = [train_loss, v_loss, mae, control_mae, deploy_mae, support_viol_pct,
                sat_pct, pos_sat_pct, neg_sat_pct, deploy_active_pct,
                mean_deployed_bias, floor_pct, mean_mu, neg_mu_pct, pos_mu_pct,
                headroom_neg_pct, near_target_neg_pct, over_budget_pos_pct,
                r_budget_mean, r_quality_mean, r_recovery_mean,
                r_floor_penalty_mean, r_target_zone_mean, bc_coef, support_coef,
                correct_dir_pct, corr_mu_gpu_next, r_total_std]
        for key, val in zip(history.keys(), vals):
            history[key].append(val)

        logging.info(
            f'Epoch {epoch+1}/{epochs} | train={train_loss:.6f} | val={v_loss:.6f} | '
            f'mae={mae:.6f} | control_mae={control_mae:.6f} | deploy_mae={deploy_mae:.6f} | '
            f'sat%={sat_pct:.2f} | floor%={floor_pct:.2f} | mean_mu={mean_mu:.4f} | '
            f'neg_mu%={neg_mu_pct:.2f} | bc={bc_coef:.3f}'
        )

        # Warnings
        if epoch >= 10 and pos_sat_pct > 10.0:
            logging.warning(f'Epoch {epoch+1}: pos_sat%={pos_sat_pct:.1f} > 10%')
        if epoch >= 10 and neg_sat_pct > 35.0:
            logging.warning(f'Epoch {epoch+1}: neg_sat%={neg_sat_pct:.1f} > 35%')
        if epoch >= 10 and floor_pct > 50.0:
            logging.warning(f'Epoch {epoch+1}: floor%={floor_pct:.1f} > 50%')

    if run_plots:
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(18, 8))
        ax = axes[0, 0]
        ax.plot(history['train'], label='Train', color='#2196F3')
        ax.plot(history['val'], label='Val', color='#FF9800')
        ax.set_title('REINFORCE Loss Curve'); ax.legend()

        ax = axes[0, 1]
        ax.plot(history['mae'], color='#4CAF50', label='raw mu MAE')
        ax.plot(history['control_mae'], color='#03A9F4', label='control target MAE')
        ax.plot(history['deploy_mae'], color='#F44336', label='deploy MAE')
        ax.set_title('Validation Action Error'); ax.legend()

        ax = axes[1, 0]
        ax.plot(history['sat_pct'], color='#9C27B0', label='sat%')
        ax.plot(history['pos_sat_pct'], color='#E91E63', label='pos_sat%')
        ax.plot(history['neg_sat_pct'], color='#673AB7', label='neg_sat%')
        ax.plot(history['floor_pct'], color='#FF5722', label='floor%')
        ax.axhline(10.0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Deployability Warnings'); ax.legend()

        ax = axes[1, 1]
        ax.plot(history['bc_coef'], color='#607D8B', label='BC coef')
        ax.plot(history['deploy_active_pct'], color='#8BC34A', label='Deploy active%')
        ax.plot(history['mean_deployed_bias'], color='#3F51B5', label='Mean deployed bias')
        ax.set_title('Schedules & Deploy Activity'); ax.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'loss_curve.png', dpi=150)
        plt.show()
    else:
        logging.info('Skipped plot: loss_curve.png')

    return model, history
