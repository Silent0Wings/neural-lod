"""
Stage 4: Data Loading & Cleaning (Notebook §1-2)
Loads rollout_ep*.csv files, cleans timing data, adds dwell features.
"""
# Pipeline stage: notebook sections 1-2, plus dynamic GPU target setup.
import numpy as np
import pandas as pd
from config import (
    DATA_DIR, GPU_VALID_MIN_MS, GPU_VALID_MAX_MS, BIAS_MIN, BIAS_MAX,
    FLOOR_MARGIN, CEILING_MARGIN, DWELL_DECAY, DWELL_ACCUMULATION_RATE,
    FEATURE_COLS,
)

REQUIRED_ROLLOUT_COLUMNS = {
    'episode', 'step', 'cpu_frame_time', 'gpu_frame_time', 'fps',
    'visible_renderer_count', 'triangle_count', 'draw_call_count',
    'camera_speed', 'camera_rotation_speed', 'avg_screen_coverage',
    'previous_bias', 'recent_lod_switch_count',
    'lod_bias_before_action', 'action_delta', 'lod_bias_after_action',
}


def add_bias_dwell_features(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute floor/ceiling dwell scores per sequence group."""
    floor_scores = np.zeros(len(df), dtype=np.float32)
    ceiling_scores = np.zeros(len(df), dtype=np.float32)

    for _, grp in df.groupby(group_col, sort=False):
        floor_score = 0.0
        ceiling_score = 0.0
        idx = grp.index.to_numpy()
        bias_values = grp['previous_bias'].to_numpy(dtype=np.float32)

        for pos, bias in zip(idx, bias_values):
            floor_proximity = np.clip(
                ((BIAS_MIN + FLOOR_MARGIN) - bias) / max(FLOOR_MARGIN, 1e-6), 0.0, 1.0
            )
            ceiling_proximity = np.clip(
                (bias - (BIAS_MAX - CEILING_MARGIN)) / max(CEILING_MARGIN, 1e-6), 0.0, 1.0
            )
            floor_score = np.clip(
                floor_score * DWELL_DECAY + floor_proximity * DWELL_ACCUMULATION_RATE, 0.0, 1.0
            )
            ceiling_score = np.clip(
                ceiling_score * DWELL_DECAY + ceiling_proximity * DWELL_ACCUMULATION_RATE, 0.0, 1.0
            )
            floor_scores[pos] = floor_score
            ceiling_scores[pos] = ceiling_score

    out = df.copy()
    out['floor_dwell_score'] = floor_scores
    out['ceiling_dwell_score'] = ceiling_scores
    return out


def load_rollouts() -> pd.DataFrame:
    """Load and validate rollout CSV files."""
    rollout_files = sorted(DATA_DIR.glob('rollout_ep*.csv'))
    print(f'rollout_ep*.csv: {len(rollout_files)}')

    if not rollout_files:
        raise FileNotFoundError(
            'Stage 4 training now requires native rollout_ep*.csv files. '
            'Audit findings block training_data_*.csv / inference_eval_*.csv fallbacks. '
            f'Expected files under: {DATA_DIR}'
        )

    dfs = []
    for f in rollout_files:
        df = pd.read_csv(f)
        missing = sorted(REQUIRED_ROLLOUT_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f'{f.name} is missing required rollout columns: {missing}')
        df['source_file'] = f.name
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        'episode', 'step', 'cpu_frame_time', 'gpu_frame_time', 'fps',
        'visible_renderer_count', 'triangle_count', 'draw_call_count',
        'camera_speed', 'camera_rotation_speed', 'avg_screen_coverage',
        'previous_bias', 'recent_lod_switch_count',
        'lod_bias_before_action', 'action_delta', 'lod_bias_after_action',
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

    print(f'Raw rows   : {len(raw):,}')
    print(f'Files      : {raw["source_file"].nunique()}')

    if raw['episode'].isna().any():
        missing_count = int(raw['episode'].isna().sum())
        print(f'Fixing missing episode values: {missing_count:,}')
        raw['episode'] = rebuild_episodes_from_steps(raw)

    print(f'Episodes   : {raw["episode"].nunique()}')
    print(f'Steps / ep : {raw.groupby("episode").size().mean():.0f}')

    if raw['source_file'].nunique() < 2 and raw['episode'].nunique() < 2:
        raise ValueError(
            'Need at least 2 rollout files or 2 episodes for an audit-safe holdout split. '
            f'Found files={raw["source_file"].nunique()} episodes={raw["episode"].nunique()}.'
        )

    print("Episodes after fix:", raw['episode'].nunique())
    return raw


def rebuild_episodes_from_steps(raw: pd.DataFrame) -> pd.Series:
    """Rebuild stable episode ids from step resets, per source file."""
    rebuilt = pd.Series(index=raw.index, dtype='float64')
    next_episode_id = 0

    for _, grp in raw.groupby('source_file', sort=False):
        steps = grp['step'].fillna(-1)
        local_episode = (steps == 0).cumsum()
        if local_episode.max() == 0:
            local_episode = pd.Series(1, index=grp.index)

        local_episode = local_episode.astype('int64') - 1
        rebuilt.loc[grp.index] = local_episode + next_episode_id
        next_episode_id = int(rebuilt.loc[grp.index].max()) + 1

    if rebuilt.isna().any():
        raise ValueError('Episode rebuild failed -- some episode ids are still missing.')
    if rebuilt.nunique() < 2 and raw['source_file'].nunique() < 2:
        raise ValueError('Episode rebuild produced only one episode and one file; holdout split is unsafe.')

    return rebuilt.astype('int64')


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid timing rows, add dwell features, validate."""
    df_clean = raw[
        (raw['cpu_frame_time'] > 0)
        & (raw['gpu_frame_time'] >= GPU_VALID_MIN_MS)
        & (raw['gpu_frame_time'] <= GPU_VALID_MAX_MS)
    ].copy()

    print(f'After timing filter: {len(df_clean):,} rows ({100 * len(df_clean) / len(raw):.1f}% kept)')

    if len(df_clean) == 0:
        raise ValueError('All rows removed by timing filter -- check rollout collection')

    base_feature_cols = [
        'cpu_frame_time', 'gpu_frame_time', 'fps', 'visible_renderer_count', 'triangle_count',
        'draw_call_count', 'camera_speed', 'camera_rotation_speed', 'avg_screen_coverage',
        'previous_bias', 'recent_lod_switch_count',
    ]
    df_clean = df_clean.dropna(subset=base_feature_cols + ['action_delta']).copy()
    print(f'After NaN drop:      {len(df_clean):,} rows')

    if len(df_clean) == 0:
        raise ValueError('All rows removed after NaN drop')

    sequence_group_col = 'source_file' if 'source_file' in df_clean.columns else 'episode'
    df_clean = df_clean.sort_values([sequence_group_col, 'step']).reset_index(drop=True)

    # Add dwell features
    df_clean = add_bias_dwell_features(df_clean, sequence_group_col)

    # Validate features
    a = pd.to_numeric(df_clean['action_delta'], errors='coerce').fillna(0).astype('float32')
    print(f'Non-zero action % : {(a.abs() > 1e-6).mean() * 100:.1f}')
    print(f'Near-zero action %: {(a.abs() < 0.005).mean() * 100:.1f}')

    feature_nonzero = (df_clean[FEATURE_COLS] != 0).mean() * 100
    bad_features = feature_nonzero[feature_nonzero < 1.0]
    if len(bad_features):
        lines = [f'  {name}: {pct:.2f}% non-zero' for name, pct in bad_features.items()]
        raise ValueError(
            'Audit guardrail tripped: near-all-zero Stage 4 features remain after cleaning:\n'
            + '\n'.join(lines)
        )

    print('Dwell feature summary:')
    print(df_clean[['floor_dwell_score', 'ceiling_dwell_score']].describe().round(4).to_string())

    return df_clean


def compute_t_target(df_clean: pd.DataFrame) -> float:
    """Compute a realistic main GPU target from data distribution."""
    gpu_series = df_clean['gpu_frame_time'].dropna().astype('float32')
    if gpu_series.empty:
        raise ValueError('gpu_frame_time has no valid rows')

    p25 = gpu_series.quantile(0.25)
    p40 = gpu_series.quantile(0.40)
    median = gpu_series.quantile(0.50)
    stretch_target = float(0.5 * (p25 + p40))
    t_target = float(median)

    under_target_pct = float((gpu_series <= t_target).mean() * 100)
    under_stretch_pct = float((gpu_series <= stretch_target).mean() * 100)

    print(f'T_TARGET (median/main) = {t_target:.3f} ms')
    print(f'Stretch target p25/p40 = {stretch_target:.3f} ms ({under_stretch_pct:.1f}% under)')
    print(f'Under main target      = {under_target_pct:.1f}%')
    print(f'GPU stats | mean={gpu_series.mean():.3f} '
          f'median={median:.3f} '
          f'p25={p25:.3f} p40={p40:.3f} p75={gpu_series.quantile(0.75):.3f}')
    return t_target
