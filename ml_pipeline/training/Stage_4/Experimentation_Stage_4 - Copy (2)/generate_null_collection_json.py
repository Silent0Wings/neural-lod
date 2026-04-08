"""
Stage 4: RL Null Collection JSON Generator
"""
# Pipeline utility: writes a Unity-readable scaler/runtime JSON for the null RL data collection phase.
import argparse
import json
from pathlib import Path

from config import FEATURE_COLS, RUNTIME_CONTRACT_DEFAULTS


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[3]
DEFAULT_SOURCE_SCALER = REPO_DIR / 'Assets' / 'StreamingAssets' / 'rl_scaler_constants.json'
UNITY_NULL_JSON = REPO_DIR / 'Assets' / 'StreamingAssets' / 'rl_null_collection_constants.json'
EXPERIMENT_NULL_JSON = SCRIPT_DIR / 'rl_null_collection_constants.json'
DEFAULT_OUTPUT = EXPERIMENT_NULL_JSON


NULL_COLLECTION_DEFAULTS = {
    # Bootstrap target until Unity finishes scene-warmup calibration.
    't_target_ms': 4.5,
    'gpu_target_ms_base': 4.5,
    'gpu_target_ms_min': 4.0,
    'gpu_target_ms_max': 6.5,
    # C# fallback-era scaler defaults used when old JSON files omitted these fields.
    'action_head_scale': 0.20,
    'max_action_delta': 0.20,
    # Training/reference values carried by RLFeatureExtractor.ScalerData.
    'pg_coef': 0.50,
    'bc_coef_start': 1.0,
    'bc_coef_end': 0.5,
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
    'target_source': 'scene_warmup_median',
    'collection_mode': 'null_rl',
}
NULL_COLLECTION_DEFAULTS.update(RUNTIME_CONTRACT_DEFAULTS)


def load_source_scaler(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'Source scaler JSON not found: {path}')

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing = [key for key in ('feature_names', 'mean', 'scale') if key not in data]
    if missing:
        raise ValueError(f'Source scaler JSON is missing required fields: {missing}')

    if data['feature_names'] != FEATURE_COLS:
        raise ValueError(
            'Source scaler feature order does not match Stage 4 FEATURE_COLS. '
            f'Expected {FEATURE_COLS}, got {data["feature_names"]}'
        )

    if len(data['mean']) != len(FEATURE_COLS) or len(data['scale']) != len(FEATURE_COLS):
        raise ValueError('Source scaler mean/scale lengths do not match feature count')

    return data


def build_null_collection_json(source_scaler: dict, overrides: dict | None = None) -> dict:
    out = {
        'feature_names': source_scaler['feature_names'],
        'mean': source_scaler['mean'],
        'scale': source_scaler['scale'],
    }
    out.update(NULL_COLLECTION_DEFAULTS)
    if overrides:
        out.update({key: value for key, value in overrides.items() if value is not None})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a Unity-readable JSON for the Stage 4 RL null data collection phase.'
    )
    parser.add_argument(
        '--source-scaler',
        type=Path,
        default=DEFAULT_SOURCE_SCALER,
        help=f'Existing scaler JSON to copy feature_names/mean/scale from. Default: {DEFAULT_SOURCE_SCALER}',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output JSON path. Default writes the in-scope experiment JSON: {DEFAULT_OUTPUT}',
    )
    parser.add_argument(
        '--write-experiment-copy',
        action='store_true',
        help=f'Also write a review copy beside this script: {EXPERIMENT_NULL_JSON}',
    )
    parser.add_argument('--t-target-ms', type=float, default=NULL_COLLECTION_DEFAULTS['t_target_ms'])
    parser.add_argument('--action-head-scale', type=float, default=NULL_COLLECTION_DEFAULTS['action_head_scale'])
    parser.add_argument('--max-action-delta', type=float, default=NULL_COLLECTION_DEFAULTS['max_action_delta'])
    parser.add_argument('--dead-zone', type=float, default=NULL_COLLECTION_DEFAULTS['dead_zone'])
    parser.add_argument('--dwell-frames', type=int, default=NULL_COLLECTION_DEFAULTS['dwell_frames'])
    parser.add_argument('--bias-min', type=float, default=NULL_COLLECTION_DEFAULTS['bias_min'])
    parser.add_argument('--bias-max', type=float, default=NULL_COLLECTION_DEFAULTS['bias_max'])
    parser.add_argument('--inference-interval', type=int, default=NULL_COLLECTION_DEFAULTS['inference_interval'])
    parser.add_argument('--scene-target-warmup-frames', type=int, default=NULL_COLLECTION_DEFAULTS['scene_target_warmup_frames'])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_scaler = load_source_scaler(args.source_scaler)
    data = build_null_collection_json(
        source_scaler,
        overrides={
            't_target_ms': args.t_target_ms,
            'gpu_target_ms_base': args.t_target_ms,
            'action_head_scale': args.action_head_scale,
            'max_action_delta': args.max_action_delta,
            'dead_zone': args.dead_zone,
            'dwell_frames': args.dwell_frames,
            'bias_min': args.bias_min,
            'bias_max': args.bias_max,
            'inference_interval': args.inference_interval,
            'scene_target_warmup_frames': args.scene_target_warmup_frames,
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')

    wrote_experiment_copy = False
    if args.write_experiment_copy and args.output.resolve() != EXPERIMENT_NULL_JSON.resolve():
        EXPERIMENT_NULL_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENT_NULL_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        wrote_experiment_copy = True

    print('Wrote RL null-mode JSON:', args.output)
    if wrote_experiment_copy:
        print('Wrote experiment review copy:', EXPERIMENT_NULL_JSON)
    print('Source scaler:', args.source_scaler)
    print(f't_target_ms={data["t_target_ms"]} action_head_scale={data["action_head_scale"]} '
          f'max_action_delta={data["max_action_delta"]} dead_zone={data["dead_zone"]} '
          f'dwell_frames={data["dwell_frames"]} inference_interval={data["inference_interval"]} '
          f'recovery_eligible_bias={data["recovery_eligible_bias"]} '
          f'correction_eligible_bias={data["correction_eligible_bias"]} '
          f'target_source={data["target_source"]} scene_target_warmup_frames={data["scene_target_warmup_frames"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
