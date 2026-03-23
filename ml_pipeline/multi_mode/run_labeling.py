"""
run_labeling.py  CLI wrapper for the multi-mode data processing pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure we can find scripts/config.py
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

import data_processing
from config import DATA_DIR, TRAINING_LABELED

def main():
    parser = argparse.ArgumentParser(description="Multi-Mode Data Labeling Pipeline")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=str(DATA_DIR / "Train_Runs" / "Multi_Mode_Run"),
        help="Directory containing power-mode subfolders with CSV files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(TRAINING_LABELED),
        help="Path to save the final labeled CSV"
    )
    parser.add_argument(
        "--keep-rot-zero",
        action="store_true",
        help="Keep rows with rot_0.0 (default is to drop them for consistency)"
    )
    
    args = parser.parse_args()
    
    logging.info("Starting Multi-Mode Labeling Pipeline")
    logging.info(f"Input Directory: {args.input_dir}")
    logging.info(f"Output File: {args.output_file}")
    
    # Path setup
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1 & 2: Merge and Filter
    merged_df = data_processing.merge_multi_mode_data(
        args.input_dir, 
        drop_rot_zero=not args.keep_rot_zero
    )
    
    if merged_df is None or len(merged_df) == 0:
        logging.error("Pipeline failed: No data loaded.")
        sys.exit(1)
        
    # 3: Oracle Computation
    labeled_df = data_processing.compute_multi_mode_oracle(merged_df)
    
    # 5: Sanity Checks (runs on the dataframe with metadata still present)
    data_processing.run_sanity_checks(labeled_df)
    
    # 4: Finalize and Save
    data_processing.finalize_and_save(labeled_df, output_path)
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
