"""
Stage 4: RL LOD Policy Training Pipeline - Main Entry Point
Orchestrates the full notebook pipeline: load → clean → reward → train → diagnose → export.

Usage:
    uv run run_pipeline.py
"""
# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "pandas",
#   "numpy",
#   "scikit-learn",
#   "optuna",
#   "sympy==1.12",
#   "onnx",
#   "onnxruntime",
# ]
# ///
# Pipeline stage: main orchestrator that runs all Stage 4 experiment stages in order.
import warnings
warnings.filterwarnings('ignore')


import argparse

from pipeline_logging import setup_pipeline_logging

RUN_LOG_PATH = setup_pipeline_logging()

import config
from config import MODEL_DIR, PLOTS_DIR, device, FEATURE_COLS, ACTION_HEAD_SCALE, MAX_ACTION_DELTA, TRAIN_SIGMA, RUNTIME_CONTRACT_DEFAULTS
from data_loader import load_rollouts, clean_data, compute_t_target
from reward import fit_scaler, print_data_distribution, compute_rewards
from train import prepare_returns, split_data, run_optuna, train_final
from diagnostics import run_diagnostics
from health_report import write_run_health_report
from export_onnx import quality_gate, export_onnx


skip_optuna = False

skip_diagnostics = False

skip_export = True

def parse_args():
    parser = argparse.ArgumentParser(description='Run the Stage 4 RL policy training pipeline.')
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip plot creation in feature scaling, final training, and diagnostics.',
    )
    parser.add_argument(
        '--run-plots',
        action='store_true',
        help='Explicitly run plot creation. This is the default.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_plots = not args.skip_plots

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Print config summary
    print(f'Run log:             {RUN_LOG_PATH}')
    print(f'Run plots:           {run_plots}')
    print(f'Device:              {device}')
    print(f'ACTION_HEAD_SCALE:   {ACTION_HEAD_SCALE:.3f}')
    print(f'MAX_ACTION_DELTA:    {MAX_ACTION_DELTA:.3f}')
    print(f'TRAIN_SIGMA:         {TRAIN_SIGMA:.3f}')
    print(
        'Runtime contract:    '
        f'dead_zone={RUNTIME_CONTRACT_DEFAULTS["dead_zone"]} '
        f'dwell_frames={RUNTIME_CONTRACT_DEFAULTS["dwell_frames"]} '
        f'inference_interval={RUNTIME_CONTRACT_DEFAULTS["inference_interval"]} '
        f'recovery=({RUNTIME_CONTRACT_DEFAULTS["recovery_eligible_bias"]}, '
        f'{RUNTIME_CONTRACT_DEFAULTS["correction_eligible_bias"]}, '
        f'{RUNTIME_CONTRACT_DEFAULTS["recovery_growth_threshold"]}, '
        f'{RUNTIME_CONTRACT_DEFAULTS["recovery_trigger_frames"]})'
    )

    # §1-2: Load and clean rollout data
    print('\n' + '='*60)
    print('§1-2: Loading and cleaning rollout data')
    print('='*60)
    raw = load_rollouts()
    df_clean = clean_data(raw)
    training_target_source = (
        'rollout_reward_target_ms'
        if 'reward_target_ms' in df_clean.columns and df_clean['reward_target_ms'].notna().any()
        else 'legacy_gpu_frame_median'
    )

    # §3: Compute GPU target and fit scaler
    print('\n' + '='*60)
    print('§3: Feature scaling')
    print('='*60)
    t_target = compute_t_target(df_clean)
    config.T_TARGET = t_target
    scaler, X_scaled = fit_scaler(df_clean, t_target, run_plots=run_plots)

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

    if skip_optuna:
        print('skip_optuna=True; stopping before Optuna tuning and final training.')
        return df_clean

    # §6: Optuna hyperparameter search
    print('\n' + '='*60)
    print('§6: Optuna hyperparameter tuning')
    print('='*60)
    study = run_optuna(X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t, df_val, scaler, t_target, group_col)

    # §7: Final training
    print('\n' + '='*60)
    print('§7: Final REINFORCE training')
    print('='*60)
    model, history = train_final(
        study, X_tr_t, A_tr_t, G_tr_t, X_val_t, A_val_t, G_val_t,
        df_val, scaler, t_target, group_col, run_plots=run_plots,
    )

    if skip_diagnostics:
        print('skip_diagnostics=True; stopping before diagnostics and export.')
        return model, history

    # §8: Diagnostics
    print('\n' + '='*60)
    print('§8: Policy diagnostics')
    print('='*60)
    diag_results = run_diagnostics(model, df_clean, X_scaled, scaler, t_target, group_col, run_plots=run_plots)
    health_report_path, is_healthy = write_run_health_report(
        df_clean, history, diag_results, t_target, run_log_path=RUN_LOG_PATH, target_source=training_target_source,
    )

    if skip_export:
        print('skip_export=True; stopping before quality gates and ONNX export.')
        return model, history, diag_results, health_report_path, is_healthy

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
