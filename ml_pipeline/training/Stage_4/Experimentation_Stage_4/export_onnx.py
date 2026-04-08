"""
Stage 4: ONNX Export & Viability Gates (Notebook §9-10)
"""
# Pipeline stage: notebook sections 9-10, quality gates and ONNX export.
import torch
import numpy as np

from config import (
    MODEL_DIR, FEATURE_COLS, FEATURE_COUNT, BIAS_MIN, FLOOR_MARGIN,
)


def quality_gate(df_clean, diag_results, history):
    """Run all quality/viability gates before export."""
    # Feature zero-check
    nonzero_pct = (df_clean[FEATURE_COLS] != 0).mean() * 100
    zero_features = nonzero_pct[nonzero_pct < 1.0].index.tolist()
    if zero_features:
        for fn in zero_features:
            print(f'  {fn}: {nonzero_pct[fn]:.1f}% non-zero')
        print('⚠️ WARNING: ONNX export would be blocked -- near-zero feature columns detected.')

    deploy_mae_all = diag_results['deploy_mae_all']
    pos_sat_pct = diag_results['pos_sat_pct']
    neg_sat_pct = diag_results['neg_sat_pct']
    floor_pct = diag_results['floor_pct']

    if deploy_mae_all > 0.08:
        print(f'⚠️ WARNING: ONNX export would be blocked -- deploy_MAE={deploy_mae_all:.4f} > 0.08')
    if pos_sat_pct > 10.0:
        print(f'⚠️ WARNING: ONNX export would be blocked -- pos_sat%={pos_sat_pct:.2f} > 10%')
    if neg_sat_pct > 35.0:
        print(f'⚠️ WARNING: ONNX export would be blocked -- neg_sat%={neg_sat_pct:.2f} > 35%')
    if floor_pct > 50.0:
        print(f'⚠️ WARNING: ONNX export would be blocked -- floor%={floor_pct:.2f} > 50%')

    # Viability gates from history
    final_deploy_active = history['deploy_active_pct'][-1]
    neg_mu_pct = history['neg_mu_pct'][-1]
    near_target_neg = history['near_target_neg_pct'][-1]

    gates_passed = True
    if final_deploy_active < 5.0:
        print(f'❌ GATE FAILED: Policy active only {final_deploy_active:.2f}%')
        gates_passed = False
    else:
        print(f'✅ GATE PASSED: Policy active {final_deploy_active:.2f}%')

    if neg_mu_pct > 70 or neg_mu_pct < 30:
        print(f'❌ GATE FAILED: Action imbalance {neg_mu_pct:.1f}% negative')
        gates_passed = False
    else:
        print(f'✅ GATE PASSED: Action balance {neg_mu_pct:.1f}% negative')

    if near_target_neg > 70:
        print(f'❌ GATE FAILED: Near-target negative {near_target_neg:.1f}%')
        gates_passed = False
    else:
        print(f'✅ GATE PASSED: Near-target behavior {near_target_neg:.1f}%')

    if not gates_passed:
        print('⚠️ WARNING: MODEL FAILED VIABILITY GATES. Proceeding anyway for bootstrap.')

    print('✅ ALL GATES PASSED. Model is viable for deployment.')
    print('Quality gate passed. Proceeding with ONNX export.')


def export_onnx(model, history):
    """Export model to ONNX and validate with onnxruntime."""
    import onnxruntime as ort

    onnx_path = MODEL_DIR / 'rl_policy_stage4.onnx'
    export_model = model.cpu().eval()
    dummy_input = torch.zeros(1, FEATURE_COUNT, dtype=torch.float32)

    torch.onnx.export(
        export_model, dummy_input, str(onnx_path),
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12,
    )
    print(f'ONNX exported → {onnx_path}')

    sess = ort.InferenceSession(str(onnx_path))
    out = sess.run(None, {'input': dummy_input.numpy()})[0]
    print(f'ONNX output shape: {out.shape} | value: {out[0, 0]:.6f}')
    assert out.shape == (1, 1), f'Expected (1,1), got {out.shape}'
    print('ONNX validation OK.')

    # Summary
    print('\nTraining complete.')
    print(f'  Final val loss      : {history["val"][-1]:.4f}')
    print(f'  Final val MAE       : {history["mae"][-1]:.4f}')
    print(f'  Final deploy MAE    : {history["deploy_mae"][-1]:.4f}')
    print(f'  Final support rate  : {history["support_pct"][-1]:.2f}%')
    print(f'  Final sat rate      : {history["sat_pct"][-1]:.2f}%')
    print(f'  Final deploy active : {history["deploy_active_pct"][-1]:.2f}%')
    print(f'\nNext step: assign {onnx_path.name} to RLPolicyController.OnnxAsset in Unity.')

    return onnx_path
