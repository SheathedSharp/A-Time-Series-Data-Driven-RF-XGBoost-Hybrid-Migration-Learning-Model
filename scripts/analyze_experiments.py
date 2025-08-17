"""
Analyze fault detection experiment results from log files.
Supports XGBoost, LightGBM, LSTM, MLP, and SVM models.
Extracts performance metrics and calculates statistical measures for different fault codes.
"""

import argparse
import sys

from utils.experiment_analyzer import (
    process_experiments,
    generate_statistics_summary,
    save_results,
    print_publication_table
)


def main():
    parser = argparse.ArgumentParser(description='Analyze fault detection experiment results')
    parser.add_argument('--model_type', type=str, 
                       choices=['xgboost', 'lightgbm', 'lstm', 'svm', 'mlp'], 
                       default='xgboost', 
                       help='Type of model to analyze (default: xgboost)')
    args = parser.parse_args()
    
    model_type = args.model_type
    
    # Process all experiments for the specified model type
    df = process_experiments(model_type)
    
    if df.empty:
        print(f"No data found for {model_type} model type.")
        sys.exit(1)
    
    # Generate statistical summary
    stats_df = generate_statistics_summary(df)
    
    # Save results
    save_results(df, stats_df, model_type)
    
    # Print publication-ready table
    print_publication_table(stats_df, model_type)


if __name__ == "__main__":
    main()