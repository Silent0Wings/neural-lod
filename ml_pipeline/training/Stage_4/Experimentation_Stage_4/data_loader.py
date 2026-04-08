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

OPTIONAL_PROVENANCE_COLUMNS = {
    'raw_action_delta': np.nan,
    'selected_target_ms': np.nan,
    'target_source': 'legacy_unknown',
    'scene_target_ready': 1,
    'collection_mode': 'legacy_unknown',
    'gpu_ms_at_target_lock': np.nan,
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
    for col, default in OPTIONAL_PROVENANCE_COLUMNS.items():
        if col not in raw.columns:
            raw[col] = default

    numeric_cols = [
        'episode', 'step', 'cpu_frame_time', 'gpu_frame_time', 'fps',
        'visible_renderer_count', 'triangle_count', 'draw_call_count',
        'camera_speed', 'camera_rotation_speed', 'avg_screen_coverage',
        'previous_bias', 'recent_lod_switch_count',
        'lod_bias_before_action', 'action_delta', 'lod_bias_after_action',
        'raw_action_delta', 'selected_target_ms', 'scene_target_ready', 'gpu_ms_at_target_lock',
    ]
    for optional_feature_col in ('floor_dwell_score', 'ceiling_dwell_score'):
        if optional_feature_col in raw.columns:
            numeric_cols.append(optional_feature_col)
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

    raw['raw_action_delta'] = raw['raw_action_delta'].fillna(raw['action_delta'])
    raw['scene_target_ready'] = raw['scene_target_ready'].fillna(1).astype('int32')
    raw['target_source'] = raw['target_source'].fillna('legacy_unknown').astype(str)
    raw['collection_mode'] = raw['collection_mode'].fillna('legacy_unknown').astype(str)

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
    dropped_pre_target_lock_rows = 0

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

    collection_mode = df_clean['collection_mode'].fillna('legacy_unknown').astype(str).str.lower()
    scene_target_ready = pd.to_numeric(df_clean['scene_target_ready'], errors='coerce').fillna(1)
    pre_target_lock = collection_mode.eq('null_rl') & scene_target_ready.eq(0)
    dropped_pre_target_lock_rows = int(pre_target_lock.sum())
    if dropped_pre_target_lock_rows:
        df_clean = df_clean.loc[~pre_target_lock].copy()
        print(f'Dropped pre-target-lock null rows: {dropped_pre_target_lock_rows:,}')
    else:
        print('Dropped pre-target-lock null rows: 0')

    if len(df_clean) == 0:
        raise ValueError('All rows removed after dropping pre-target-lock null rows')

    sequence_group_col = 'source_file' if 'source_file' in df_clean.columns else 'episode'
    df_clean = df_clean.sort_values([sequence_group_col, 'step']).reset_index(drop=True)

    # Current Unity rollouts log dwell scores as part of the 13-feature policy
    # vector. Use them directly so training sees the same state the controller
    # saw; recompute only for older rollout files that did not have the fields.
    missing_dwell_cols = [
        col for col in ('floor_dwell_score', 'ceiling_dwell_score')
        if col not in df_clean.columns or df_clean[col].isna().all()
    ]
    if missing_dwell_cols:
        print(f'Recomputing missing dwell feature(s): {missing_dwell_cols}')
        df_clean = add_bias_dwell_features(df_clean, sequence_group_col)
    else:
        recomputed_dwell = add_bias_dwell_features(df_clean, sequence_group_col)
        for col in ('floor_dwell_score', 'ceiling_dwell_score'):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(recomputed_dwell[col]).astype('float32')
        print('Using logged Unity dwell features: floor_dwell_score, ceiling_dwell_score')
    df_clean['reward_target_ms'] = pd.to_numeric(df_clean['selected_target_ms'], errors='coerce').astype('float32')
    df_clean.attrs['dropped_pre_target_lock_rows'] = dropped_pre_target_lock_rows

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
    """Compute the main GPU target from rollout metadata, falling back to GPU median."""
    gpu_series = df_clean['gpu_frame_time'].dropna().astype('float32')
    if gpu_series.empty:
        raise ValueError('gpu_frame_time has no valid rows')

    if 'reward_target_ms' in df_clean.columns:
        reward_target = pd.to_numeric(df_clean['reward_target_ms'], errors='coerce')
        valid_reward_target = reward_target.dropna().astype('float32')
    else:
        valid_reward_target = pd.Series(dtype='float32')

    p25 = gpu_series.quantile(0.25)
    p40 = gpu_series.quantile(0.40)
    median = gpu_series.quantile(0.50)
    stretch_target = float(0.5 * (p25 + p40))

    if len(valid_reward_target):
        t_target = float(valid_reward_target.median())
        target_basis = 'rollout reward_target_ms median'
    else:
        t_target = float(median)
        target_basis = 'legacy gpu_frame_time median'

    under_target_pct = float((gpu_series <= t_target).mean() * 100)
    under_stretch_pct = float((gpu_series <= stretch_target).mean() * 100)

    print(f'T_TARGET ({target_basis}) = {t_target:.3f} ms')
    print(f'Stretch target p25/p40 = {stretch_target:.3f} ms ({under_stretch_pct:.1f}% under)')
    print(f'Under main target      = {under_target_pct:.1f}%')
    if len(valid_reward_target):
        print(f'Reward target stats | min={valid_reward_target.min():.3f} '
              f'median={valid_reward_target.median():.3f} '
              f'max={valid_reward_target.max():.3f} '
              f'valid_rows={len(valid_reward_target):,}/{len(df_clean):,}')
    print(f'GPU stats | mean={gpu_series.mean():.3f} '
          f'median={median:.3f} '
          f'p25={p25:.3f} p40={p40:.3f} p75={gpu_series.quantile(0.75):.3f}')
    return t_target
