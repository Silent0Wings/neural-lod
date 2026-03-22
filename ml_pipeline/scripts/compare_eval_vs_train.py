"""
compare_eval_vs_train.py  Stage 3: cross-compare neural eval CSV vs training data.
Auto-detects the latest eval_neural_*.csv in data/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import (
    TRAINING_LABELED, EVAL_COMPARE_FIG,
    TARGET_FRAME_MS, get_latest_eval_csv
)

def run(save_individual=False):
    if not TRAINING_LABELED.exists():
        raise FileNotFoundError(
            f"Labeled training file not found: {TRAINING_LABELED}\n"
            "Run Stage 2 (generate_oracle_labels.py) first."
        )

    eval_path = get_latest_eval_csv()
    print(f"Using eval file : {eval_path.name}")

    train = pd.read_csv(TRAINING_LABELED)
    eval_ = pd.read_csv(eval_path)

    print(f"Train rows : {len(train)}")
    print(f"Eval rows  : {len(eval_)}")

    train = train.rename(columns={
        "cpu_frame_time_ms": "cpu_ms",
        "gpu_frame_time_ms": "gpu_ms",
        "lod_bias_current":  "lod_bias",
    })
    eval_ = eval_.rename(columns={
        "cpu_frame_ms": "cpu_ms",
        "gpu_frame_ms": "gpu_ms",
    })

    for df in [train, eval_]:
        df["over_budget"] = df["cpu_ms"] > TARGET_FRAME_MS

    def stats(df, label):
        cpu = df["cpu_ms"]
        print(f"\n{'='*40}\n  {label}\n{'='*40}")
        print(f"  CPU mean   : {cpu.mean():.2f} ms")
        print(f"  CPU median : {cpu.median():.2f} ms")
        print(f"  CPU P95    : {cpu.quantile(0.95):.2f} ms")
        print(f"  CPU P99    : {cpu.quantile(0.99):.2f} ms")
        print(f"  CPU max    : {cpu.max():.2f} ms")
        print(f"  Over budget: {df['over_budget'].mean()*100:.1f}%")
        print(f"  lod_bias mean : {df['lod_bias'].mean():.3f}")
        print(f"  lod_bias std  : {df['lod_bias'].std():.3f}")

    stats(train, "TRAINING DATA")
    stats(eval_,  "EVAL / INFERENCE")

    if "bias_switched" in eval_.columns:
        print(f"\n  Bias switch rate: {eval_['bias_switched'].mean()*100:.2f}% of frames")

    # ---- plot -----------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Neural LOD  Eval Inference vs Training Data", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    TC   = "#4C72B0"
    EC   = "#DD8452"
    BINS = 60

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(train["cpu_ms"].clip(0, 60), bins=BINS, alpha=0.6, color=TC, label="Train", density=True)
    ax1.hist(eval_["cpu_ms"].clip(0, 60),  bins=BINS, alpha=0.6, color=EC, label="Eval",  density=True)
    ax1.axvline(TARGET_FRAME_MS, color="red", linestyle="--", linewidth=1.2, label=f"Budget")
    ax1.set_title("CPU Frame Time Distribution")
    ax1.set_xlabel("CPU ms (clipped 60)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(train["gpu_ms"].clip(0, 20), bins=BINS, alpha=0.6, color=TC, label="Train", density=True)
    ax2.hist(eval_["gpu_ms"].clip(0, 20),  bins=BINS, alpha=0.6, color=EC, label="Eval",  density=True)
    ax2.set_title("GPU Frame Time Distribution")
    ax2.set_xlabel("GPU ms (clipped 20)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(train["lod_bias"], bins=30, alpha=0.6, color=TC, label="Train (discrete)", density=True)
    ax3.hist(eval_["lod_bias"],  bins=60, alpha=0.6, color=EC, label="Eval (continuous)", density=True)
    ax3.set_title("LOD Bias Distribution")
    ax3.set_xlabel("lod_bias")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    train_grp = train.groupby("lod_bias")["over_budget"].mean() * 100
    ax4.bar(train_grp.index.astype(str), train_grp.values, color=TC, alpha=0.8)
    ax4.set_title("Train: % Over Budget per Bias")
    ax4.set_xlabel("lod_bias")
    ax4.set_ylabel("% frames > budget")
    ax4.set_ylim(0, 100)

    ax5 = fig.add_subplot(gs[1, 1])
    if "elapsed_s" in eval_.columns:
        es = eval_.sort_values("elapsed_s")
        # Downsample to ~400 points to prevent unreadable visual plotting
        step = max(1, len(es) // 400)
        es_sub = es.iloc[::step]
        ax5.plot(es_sub["elapsed_s"], es_sub["cpu_ms"].clip(0, 80), color=EC, alpha=0.6, linewidth=0.6)
        ax5.axhline(TARGET_FRAME_MS, color="red", linestyle="--", linewidth=1.2)
        ax5.set_title("Eval: CPU Frame Time over Run")
        ax5.set_xlabel("Elapsed (s)")
        ax5.set_ylabel("CPU ms")
    else:
        ax5.text(0.5, 0.5, "No elapsed_s column", ha="center", va="center")
        ax5.set_title("Eval: CPU over time (N/A)")

    ax6 = fig.add_subplot(gs[1, 2])
    if "elapsed_s" in eval_.columns:
        # Use the same downsampled subset for bias over time
        ax6.plot(es_sub["elapsed_s"], es_sub["lod_bias"], color=EC, alpha=0.7, linewidth=0.8)
        ax6.set_title("Eval: LOD Bias over Run")
        ax6.set_xlabel("Elapsed (s)")
        ax6.set_ylabel("lod_bias")
    else:
        ax6.plot(eval_["lod_bias"].values, color=EC, alpha=0.7, linewidth=0.6)
        ax6.set_title("Eval: LOD Bias (frame index)")
        ax6.set_xlabel("Frame")
        ax6.set_ylabel("lod_bias")

    ax7 = fig.add_subplot(gs[2, 0])
    percentiles = [50, 75, 90, 95, 99]
    tp = [train["cpu_ms"].quantile(p/100) for p in percentiles]
    ep = [eval_["cpu_ms"].quantile(p/100)  for p in percentiles]
    x  = np.arange(len(percentiles))
    w  = 0.35
    ax7.bar(x - w/2, tp, w, label="Train", color=TC, alpha=0.8)
    ax7.bar(x + w/2, ep, w, label="Eval",  color=EC, alpha=0.8)
    ax7.axhline(TARGET_FRAME_MS, color="red", linestyle="--", linewidth=1.2)
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"P{p}" for p in percentiles])
    ax7.set_title("CPU Percentiles Comparison")
    ax7.set_ylabel("ms")
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(gs[2, 1])
    sample = eval_.sample(min(1000, len(eval_)), random_state=42)
    ax8.scatter(sample["lod_bias"], sample["cpu_ms"].clip(0, 60), alpha=0.3, s=8, color=EC)
    ax8.axhline(TARGET_FRAME_MS, color="red", linestyle="--", linewidth=1.2)
    ax8.set_title("Eval: LOD Bias vs CPU ms")
    ax8.set_xlabel("lod_bias")
    ax8.set_ylabel("CPU ms (clipped 60)")

    ax9 = fig.add_subplot(gs[2, 2])
    if "path_progress" in train.columns and "path_progress" in eval_.columns:
        ax9.hist(train["path_progress"], bins=50, alpha=0.6, color=TC, label="Train", density=True)
        ax9.hist(eval_["path_progress"],  bins=50, alpha=0.6, color=EC, label="Eval",  density=True)
        ax9.set_title("Path Progress Coverage")
        ax9.set_xlabel("path_progress")
        ax9.set_ylabel("Density")
        ax9.legend(fontsize=8)
    else:
        ax9.text(0.5, 0.5, "path_progress not in both", ha="center", va="center")
        ax9.set_title("Path Coverage (N/A)")

    plt.savefig(EVAL_COMPARE_FIG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {EVAL_COMPARE_FIG}")

    if save_individual:
        for i, ax in enumerate(fig.axes):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            out_path = EVAL_COMPARE_FIG.parent / f"{EVAL_COMPARE_FIG.stem}_plot_{i+1}.png"
            fig.savefig(out_path, bbox_inches=extent.expanded(1.2, 1.25), dpi=150)
        print(f"Saved {len(fig.axes)} separated individual plots to {EVAL_COMPARE_FIG.parent}")

if __name__ == "__main__":
    import sys
    do_separate = "--separate-plots" in sys.argv
    run(save_individual=do_separate)
