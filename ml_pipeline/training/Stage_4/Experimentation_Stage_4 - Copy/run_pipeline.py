"""
Stage 4: RL LOD Policy Training Pipeline - Main Entry Point
Orchestrates the full notebook pipeline: load → clean → reward → train → diagnose → export.

Usage:
    python run_pipeline.py
"""
import warnings
warnings.filterwarnings('ignore')

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sympy==1.12', 'optuna', '--quiet'])

import config
from config import MODEL_DIR, PLOTS_DIR, device, FEATURE_COLS, ACTION_HEAD_SCALE, TRAIN_SIGMA
from data_loader import load_rollouts, clean_data, compute_t_target
from reward import fit_scaler, print_data_distribution, compute_rewards
from train import prepare_returns, split_data, run_optuna, train_final
from diagnostics import run_diagnostics
from export_onnx import quality_gate, export_onnx


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Print config summary
    print(f'Device:              {device}')
    print(f'ACTION_HEAD_SCALE:   {ACTION_HEAD_SCALE:.3f}')
    print(f'TRAIN_SIGMA:         {TRAIN_SIGMA:.3f}')

    # §1-2: Load and clean rollout data
    print('\n' + '='*60)
    print('§1-2: Loading and cleaning rollout data')
    print('='*60)
    raw = load_rollouts()
    df_clean = clean_data(raw)

    # §3: Compute GPU target and fit scaler
    print('\n' + '='*60)
    print('§3: Feature scaling')
    print('='*60)
    t_target = compute_t_target(df_clean)
    config.T_TARGET = t_target
    scaler, X_scaled = fit_scaler(df_clean, t_target)

    # §3b: Data distribution analysis
    print_data_distribution(df_clean, t_target)

    # §4: Compute rewards
    print('\n' + '='*60)
    print('§4: Reward computation')
    print('='*60)
    df_clean = compute_rewards(df_clean, t_target)

    # §5-6: Prepare returns and split data
    print('\n' + '='*60)
    print('§5-6: Returns & train/val split')
    print('='*60)
    df_clean = prepare_returns(df_clean)
    group_col, X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t, df_train, df_val = split_data(df_clean, X_scaled)

    # §6: Optuna hyperparameter search
    print('\n' + '='*60)
    print('§6: Optuna hyperparameter tuning')
    print('='*60)
    study = run_optuna(X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col)

    # §7: Final training
    print('\n' + '='*60)
    print('§7: Final REINFORCE training')
    print('='*60)
    model, history = train_final(study, X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col)

    # §8: Diagnostics
    print('\n' + '='*60)
    print('§8: Policy diagnostics')
    print('='*60)
    diag_results = run_diagnostics(model, df_clean, X_scaled, scaler, t_target, group_col)

    # §9-10: Quality gates and ONNX export
    print('\n' + '='*60)
    print('§9-10: Quality gates & ONNX export')
    print('='*60)
    quality_gate(df_clean, diag_results, history)
    onnx_path = export_onnx(model, history)

    print('\n' + '='*60)
    print('PIPELINE COMPLETE')
    print('='*60)
    return model, history, onnx_path


if __name__ == '__main__':
    main()
