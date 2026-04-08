import re
import pandas as pd
from pathlib import Path

LOG_DIR = Path(r'C:\Users\Gica\neural-lod\ml_pipeline\training\Stage_4\Experimentation_Stage_4\logs')

def analyze_logs():
    results = []
    log_files = sorted(list(LOG_DIR.glob('*.log')))
    
    for log_path in log_files:
        content = log_path.read_text(errors='ignore')
        
        # Extract ID
        run_id = log_path.stem.split('_')[-1]
        
        # Extract Target MS (Reward target ms: min=..., median=6.467)
        target_match = re.search(r'median=([\d\.]+)', content)
        target = float(target_match.group(1)) if target_match else None
        
        # Extract Best Params (Best params loaded: {...})
        params_match = re.search(r"Best params loaded: ({.*?})", content)
        params = eval(params_match.group(1)) if params_match else {}
        
        # Extract Final Metrics (Final deploy MAE: 0.0283)
        mae_match = re.search(r'Final deploy MAE\s+:\s+([\d\.]+)', content)
        mae = float(mae_match.group(1)) if mae_match else None
        
        # Extract Final CorrectDir (from health report or Phase 4 prints)
        # We'll look for the last Phase 4 print
        phase4_matches = re.findall(r'CorrectDir%: ([\d\.]+)', content)
        # Also check the newer Intent-based prints
        intent_matches = re.findall(r'CorrectDir% \(Intent\): ([\d\.]+)', content)
        # And the newest split prints
        split_matches = re.findall(r'UnderBudget: ([\d\.]+)%', content)
        
        dir_pct = None
        if split_matches: dir_pct = float(split_matches[-1])
        elif intent_matches: dir_pct = float(intent_matches[-1])
        elif phase4_matches: dir_pct = float(phase4_matches[-1])
        
        # Extract mean_mu
        mu_match = re.search(r'mean_mu=([\d\.\-]+)', content)
        mean_mu = float(mu_match.group(1)) if mu_match else None
        
        results.append({
            'ID': run_id,
            'Target': target,
            'LR': params.get('lr'),
            'H1': params.get('h1'),
            'CorrectDir%': dir_pct,
            'MeanMu': mean_mu,
            'MAE': mae
        })
        
    df = pd.DataFrame(results)
    df = df.dropna(subset=['CorrectDir%']) # Only interested in completed/training runs
    df = df.sort_values('CorrectDir%', ascending=False)
    
    print("\n=== TOP PERFORMING RUNS ANALYSIS ===")
    print(df.to_string(index=False))
    
    # Analyze Patterns
    winners = df[df['CorrectDir%'] > 70]
    if not winners.empty:
        print("\n=== WINNER PATTERNS (ACC > 70%) ===")
        print(f"Avg LR:     {winners['LR'].mean():.6f}")
        print(f"Avg Target: {winners['Target'].mean():.3f}")
        print(f"Avg MeanMu: {winners['MeanMu'].mean():.4f}")

if __name__ == '__main__':
    analyze_logs()
