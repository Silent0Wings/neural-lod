"""
Stage 4 run health report.
Summarizes whether a training run learned usable control or collapsed.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean


THRESHOLDS = {
    "correct_dir_pct_min": 25.0,
    "corr_abs_min": 0.30,
    "neg_mu_pct_max": 55.0,
    "negative_mean_mu_limit": -0.005,
    "floor_pct_max": 35.0,
    "sat_pct_min": 0.10,
    "r_total_std_min": 0.20,
    "bc_decay_min": 0.10,
    "near_zero_action_pct_max": 60.0,
}


def _tail(values, n=20):
    values = [float(v) for v in values if v == v]
    return values[-min(n, len(values)):] if values else []


def _avg_tail(history, key, n=20, default=0.0):
    values = _tail(history.get(key, []), n=n)
    return float(mean(values)) if values else default


def _final(history, key, default=0.0):
    values = _tail(history.get(key, []), n=1)
    return values[-1] if values else default


def _status(failed):
    return "FAIL" if failed else "PASS"


def build_run_health_report(df_clean, history, diag_results, t_target, run_log_path=None) -> tuple[str, bool]:
    """Return markdown report and overall healthy flag."""
    final_mean_mu = _final(history, "mean_mu")
    final_neg_mu_pct = _final(history, "neg_mu_pct")
    final_sat_pct = _final(history, "sat_pct")
    final_bc = _final(history, "bc_coef")
    start_bc = float(history.get("bc_coef", [final_bc])[0]) if history.get("bc_coef") else final_bc

    avg_correct_dir = _avg_tail(history, "correct_dir_pct")
    avg_corr = _avg_tail(history, "corr_mu_gpu_next")
    avg_floor_pct = _avg_tail(history, "floor_pct")
    avg_sat_pct = _avg_tail(history, "sat_pct")
    avg_r_total_std = _avg_tail(history, "r_total_std")
    avg_deploy_mae = _avg_tail(history, "deploy_mae")

    actions = df_clean["action_delta"].astype("float32")
    near_zero_action_pct = float((actions.abs() < 0.005).mean() * 100.0)

    checks = []
    checks.append((
        "Direction learning",
        avg_correct_dir < THRESHOLDS["correct_dir_pct_min"] or abs(avg_corr) < THRESHOLDS["corr_abs_min"],
        f"CorrectDir% tail={avg_correct_dir:.1f}%, Corr(mu,gpu_next) tail={avg_corr:.3f}",
    ))
    checks.append((
        "Action collapse",
        final_mean_mu < THRESHOLDS["negative_mean_mu_limit"] and final_neg_mu_pct > THRESHOLDS["neg_mu_pct_max"],
        f"mean_mu={final_mean_mu:.4f}, neg_mu%={final_neg_mu_pct:.1f}%",
    ))
    checks.append((
        "Floor problem",
        avg_floor_pct > THRESHOLDS["floor_pct_max"],
        f"floor% tail={avg_floor_pct:.1f}%",
    ))
    checks.append((
        "Zero saturation",
        avg_sat_pct < THRESHOLDS["sat_pct_min"],
        f"sat% tail={avg_sat_pct:.2f}%",
    ))
    checks.append((
        "Reward signal strength",
        avg_r_total_std < THRESHOLDS["r_total_std_min"],
        f"r_total.std tail={avg_r_total_std:.3f}",
    ))
    checks.append((
        "BC dominance",
        (start_bc - final_bc) < THRESHOLDS["bc_decay_min"] or near_zero_action_pct > THRESHOLDS["near_zero_action_pct_max"],
        f"bc={start_bc:.3f}->{final_bc:.3f}, near-zero actions={near_zero_action_pct:.1f}%",
    ))

    failed_checks = [name for name, failed, _ in checks if failed]
    healthy = not failed_checks
    verdict = "HEALTHY - RUN LOOKS USABLE" if healthy else "NOT FIXED - SYSTEM STILL COLLAPSED"

    lines = [
        "# Stage 4 Run Health Report",
        "",
        "## Core Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Summary",
        "",
        f"- T_TARGET: `{float(t_target):.3f} ms`",
        f"- Final deploy_MAE: `{float(diag_results.get('deploy_mae_all', avg_deploy_mae)):.4f}`",
        f"- Final deploy_active%: `{float(diag_results.get('deploy_active_pct', 0.0)):.2f}%`",
        f"- Final floor%: `{float(diag_results.get('floor_pct', avg_floor_pct)):.2f}%`",
        f"- Final mean deployed bias: `{float(diag_results.get('mean_deployed_bias', 0.0)):.3f}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]

    for name, failed, evidence in checks:
        lines.append(f"| {name} | {_status(failed)} | {evidence} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if healthy:
        lines.append("The run does not show the configured collapse signals.")
    else:
        lines.append("The run still shows collapse or weak-control signals:")
        for name in failed_checks:
            lines.append(f"- {name}")

    lines.extend([
        "",
        "## KISS Next Step",
        "",
        "If this report fails, inspect reward design before trusting export.",
    ])

    if run_log_path:
        lines.extend(["", f"Run log: `{run_log_path}`"])

    return "\n".join(lines) + "\n", healthy


def write_run_health_report(df_clean, history, diag_results, t_target, run_log_path=None) -> tuple[Path, bool]:
    report, healthy = build_run_health_report(df_clean, history, diag_results, t_target, run_log_path)
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_log_path:
        stem = Path(run_log_path).stem
        report_path = log_dir / f"{stem}_health.md"
    else:
        report_path = log_dir / "stage4_pipeline_health.md"

    report_path.write_text(report, encoding="utf-8")
    print("\n" + "=" * 60)
    print("Run Health Report")
    print("=" * 60)
    print(report)
    print(f"Saved health report: {report_path}")
    return report_path, healthy
