import json
import re

filepath = 'train_rl_policy_stage4.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update reinforce_loss
        if 'def reinforce_loss(model, X, A, G' in source:
            old_start = source.index('def reinforce_loss(model, X, A, G')
            old_end = source.index('def eval_metrics(')
            
            before_loss = source[:old_start]
            rest = source[old_end:]
            
            new_loss = """def reinforce_loss(model, X, A, G, bc_coef=0.05, support_coef=0.1, progress=1.0):
    mu, _ = model(X)
    
    if model.training:
        mu = mu + torch.randn_like(mu) * 0.05
        
    GPU_FEATURE_IDX = FEATURE_COLS.index('gpu_frame_time')
    BIAS_FEATURE_IDX = FEATURE_COLS.index('previous_bias')
    
    gpu_raw = X[:, GPU_FEATURE_IDX] * float(scaler.scale_[GPU_FEATURE_IDX]) + float(scaler.mean_[GPU_FEATURE_IDX])
    prev_bias_raw = X[:, BIAS_FEATURE_IDX] * float(scaler.scale_[BIAS_FEATURE_IDX]) + float(scaler.mean_[BIAS_FEATURE_IDX])
    
    # 1.1 Add transition model
    alpha = 1.5
    gpu_next = gpu_raw + alpha * mu
    
    # 1.2 Clamp realistic bounds
    gpu_next = torch.clamp(gpu_next, min=2.0, max=16.0)
    
    # 1.3 Target tracking
    error = gpu_next - T_TARGET
    
    # 2.1 Directional reward
    r_budget = -error * mu
    
    # 2.2 Floor penalty
    r_floor_pen = -2.0 * (prev_bias_raw <= (BIAS_MIN + FLOOR_MARGIN)).float()
    
    # 2.3 Smoothness
    r_smooth = -0.05 * (mu ** 2)
    
    # 2.4 Final reward
    r_total = r_budget + r_floor_pen + r_smooth
    
    # Maximize r_total
    loss = -torch.mean(r_total)
    
    bc_loss = torch.mean((mu - A)**2)
    outside_support_penalty = torch.mean(torch.relu(torch.abs(mu) - SOFT_SUPPORT_LIMIT) ** 2)
    
    total_loss = loss + (bc_coef * bc_loss) + (support_coef * outside_support_penalty)
    
    return total_loss

"""
            combined = before_loss + new_loss + rest
            cell['source'] = [line + '\n' for line in combined.split('\n') if not (line == '' and combined.endswith('\n\n'))]
            source = "".join(cell['source']) # update for next check
            
        # 2. Update deployment_metrics
        if 'def deployment_metrics(model, df_split):' in source:
            old_start = source.index('def deployment_metrics(model, df_split):')
            old_end = source.index('def build_control_target_torch')
            
            before_deploy = source[:old_start]
            rest = source[old_end:]
            
            new_deploy = """def deployment_metrics(model, df_split):
    deploy_df = build_deployment_frame(model, df_split)
    if deploy_df.empty:
        return {
            'deploy_mae': np.nan, 'deploy_active_pct': np.nan, 'deploy_std': np.nan,
            'mean_deployed_bias': np.nan, 'floor_pct': np.nan, 'r_budget_mean': np.nan,
            'r_quality_mean': np.nan, 'r_recovery_mean': np.nan, 'r_floor_penalty_mean': np.nan,
            'r_target_zone_mean': np.nan,
        }

    actual = deploy_df['action_delta'].values.astype(np.float32)
    applied = deploy_df['applied_mu'].values.astype(np.float32)
    gpu = deploy_df['gpu_frame_time'].values.astype(np.float32)
    bias_after = deploy_df['deployed_bias'].values.astype(np.float32)
    
    alpha = 1.5
    gpu_next = gpu + alpha * applied
    gpu_next = np.clip(gpu_next, 2.0, 16.0)
    
    error = gpu_next - float(T_TARGET)
    r_budget = -error * applied
    r_floor_pen = -2.0 * (bias_after <= (BIAS_MIN + FLOOR_MARGIN)).astype(np.float32)
    r_smooth = -0.05 * (applied ** 2)
    r_total = r_budget + r_floor_pen + r_smooth
    
    # Phase 4 Validation Print
    correct = ((error > 0) & (applied < 0)) | ((error < 0) & (applied > 0))
    if len(gpu_next) > 1 and np.std(applied) > 1e-6 and np.std(gpu_next) > 1e-6:
        corr = np.corrcoef(applied, gpu_next)[0, 1]
    else:
        corr = 0.0
    
    print(f"[Phase 4] Corr(mu, gpu_next): {corr:.3f} | r_total.std: {r_total.std():.3f} | CorrectDir%: {correct.mean():.3f}")

    return {
        'deploy_mae': float(np.mean(np.abs(applied - actual))),
        'deploy_active_pct': float((np.abs(applied) > 1e-6).mean() * 100.0),
        'deploy_std': float(np.std(applied)),
        'mean_deployed_bias': float(np.mean(deploy_df['deployed_bias'].values.astype(np.float32))),
        'floor_pct': float((deploy_df['deployed_bias'].values.astype(np.float32) <= (BIAS_MIN + FLOOR_MARGIN)).mean() * 100.0),
        'r_budget_mean': float(np.mean(r_budget)),
        'r_quality_mean': 0.0,
        'r_recovery_mean': 0.0,
        'r_floor_penalty_mean': float(np.mean(r_floor_pen)),
        'r_target_zone_mean': 0.0,
    }

"""
            combined = before_deploy + new_deploy + rest
            cell['source'] = [line + '\n' for line in combined.split('\n') if not (line == '' and combined.endswith('\n\n'))]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("Transition model fix successfully applied to notebook.")
