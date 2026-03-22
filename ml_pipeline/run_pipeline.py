"""
run_pipeline.py  Master runner for the Adaptive LOD ML pipeline.

Usage:
  python run_pipeline.py              run all stages
  python run_pipeline.py --stage merge
  python run_pipeline.py --stage label
  python run_pipeline.py --stage eval_compare
  python run_pipeline.py --stage all

Stages:
  1. merge         merge_training_data.py      -> training_data_merged.csv
  2. label         generate_oracle_labels.py   -> training_data_labeled.csv
  3. eval_compare  compare_eval_vs_train.py    -> compare_eval_vs_train.png
"""

import argparse
import sys
import time
from pathlib import Path

# Make sure the `scripts` folder is in sys.path so they can find `config.py`
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

# import stage modules from the scripts folder
from scripts import merge_training_data
from scripts import generate_oracle_labels
from scripts import compare_eval_vs_train

from config import (
    TRAINING_MERGED, TRAINING_LABELED,
    EVAL_COMPARE_FIG, DATA_DIR,
    get_latest_eval_csv
)

DIVIDER = "=" * 55

def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def check_data_dir():
    section("DATA DIRECTORY CHECK")
    print(f"  data/ -> {DATA_DIR}")
    csvs = list(DATA_DIR.glob("training_data_*.csv"))
    csvs = [f for f in csvs if "merged" not in f.name and "labeled" not in f.name]
    print(f"  Raw training CSVs found : {len(csvs)}")
    if len(csvs) == 0:
        print(f"\n  WARNING No raw training CSVs in {DATA_DIR}")
        print( "  Copy your Unity training_data_*.csv files into ml_pipeline/data/")
    try:
        ef = get_latest_eval_csv()
        print(f"  Latest eval CSV         : {ef.name}")
    except FileNotFoundError:
        print( "  WARNING No eval_neural_*.csv found")
        print( "  Stage 3 (eval_compare) will be skipped if missing")

def run_stage(name: str, fn, required_inputs: list[Path] = None):
    section(f"STAGE: {name}")
    if required_inputs:
        for p in required_inputs:
            if not p.exists():
                print(f"  NO required input missing: {p.name}")
                print(f"  Skipping {name}.")
                return False
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  YES {name} completed in {elapsed:.1f}s")
        return True
    except FileNotFoundError as e:
        print(f"\n  NO {name} failed: {e}")
        return False
    except Exception as e:
        print(f"\n  NO {name} failed with error:")
        print(f"     {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive LOD ML Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=["all", "merge", "label", "eval_compare"],
        default="all",
        help=(
            "all          run all stages in order (default)\n"
            "merge        Stage 1: merge raw Unity CSVs\n"
            "label        Stage 2: generate oracle labels\n"
            "eval_compare Stage 3: compare eval vs training\n"
        )
    )
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        help="Save each plot in Stage 3 as an individual image"
    )
    args = parser.parse_args()

    print(f"\n{'#'*55}")
    print( "  Adaptive LOD  ML Pipeline")
    print(f"{'#'*55}")
    print(f"  Stage selected : {args.stage}")

    check_data_dir()

    results = {}

    if args.stage in ["all", "merge"]:
        results["merge"] = run_stage(
            "1. Merge Training Data",
            merge_training_data.run
        )

    if args.stage in ["all", "label"]:
        results["label"] = run_stage(
            "2. Generate Oracle Labels",
            generate_oracle_labels.run,
            required_inputs=[TRAINING_MERGED]
        )

    if args.stage in ["all", "eval_compare"]:
        try:
            get_latest_eval_csv()
            eval_available = True
        except FileNotFoundError:
            eval_available = False

        if not eval_available:
            section("STAGE: 3. Eval Compare")
            print("  SKIP No eval_neural_*.csv in data/")
            print("  Copy Unity EvaluationLogger output into ml_pipeline/data/ and re-run.")
            results["eval_compare"] = False
        else:
            results["eval_compare"] = run_stage(
                "3. Compare Eval vs Training",
                lambda: compare_eval_vs_train.run(save_individual=args.separate_plots),
                required_inputs=[TRAINING_LABELED]
            )

    # ---- summary --------------------------------------------------------
    section("PIPELINE SUMMARY")
    all_ok = True
    for stage, ok in results.items():
        status = "YES" if ok else "NO"
        print(f"  {status}  {stage}")
        if not ok:
            all_ok = False

    if all_ok and results:
        print(f"\n  All stages passed.")
        if TRAINING_MERGED.exists():
            print(f"  Merged   -> {TRAINING_MERGED.name}")
        if TRAINING_LABELED.exists():
            print(f"  Labeled  -> {TRAINING_LABELED.name}")
        if EVAL_COMPARE_FIG.exists():
            print(f"  Figure   -> {EVAL_COMPARE_FIG.name}")
    else:
        print(f"\n  One or more stages failed or were skipped.")
        sys.exit(1)

if __name__ == "__main__":
    main()
