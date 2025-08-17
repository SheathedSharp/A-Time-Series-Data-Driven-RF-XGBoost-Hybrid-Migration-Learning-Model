"""
Analyze fault detection experiment results from log files.
Supports XGBoost, LightGBM, LSTM, MLP, and SVM models.
Extracts performance metrics and calculates statistical measures for different fault codes.
Includes visualization capabilities for confusion matrices and ROC curves.
"""

import argparse
import sys
import os
import pandas as pd
from tabulate import tabulate
from config import REPORT_DIR

from utils.experiment_analyzer import (
    process_experiments,
    generate_statistics_summary,
    save_results,
    print_publication_table,
    generate_confusion_matrices_from_metrics,
    create_confusion_matrix_table,
    create_combined_roc_curves,
    create_publication_confusion_matrices
)
from config import FAULT_DESCRIPTIONS, PIC_DIR


def print_confusion_matrices(confusion_matrices):
    """Print formatted confusion matrices for all fault types."""    
    for fault_code, cm_data in confusion_matrices.items():
        print(f"\nFault Code: {fault_code} ({FAULT_DESCRIPTIONS[int(fault_code)]})")
        print("-" * 60)
        
        table_data, headers, _ = create_confusion_matrix_table(fault_code, cm_data)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\nPerformance Metrics (Average across {cm_data['runs']} runs):")
        metrics_table = [
            ["Accuracy", f"{cm_data['avg_accuracy']:.4f}"],
            ["Precision", f"{cm_data['avg_precision']:.4f}"],
            ["Recall", f"{cm_data['avg_recall']:.4f}"],
            ["F1-Score", f"{cm_data['avg_f1']:.4f}"]
        ]
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))


def save_confusion_matrices_csv(confusion_matrices, output_file):
    """Save confusion matrices to CSV file."""
    data = []
    for fault_code, cm_data in confusion_matrices.items():
        cm_elements = cm_data['elements']
        data.append({
            'Fault_Code': fault_code,
            'Fault_Description': FAULT_DESCRIPTIONS[int(fault_code)],
            'True_Negative': cm_elements['TN'],
            'False_Positive': cm_elements['FP'],
            'False_Negative': cm_elements['FN'],
            'True_Positive': cm_elements['TP'],
            'Avg_Accuracy': cm_data['avg_accuracy'],
            'Avg_Precision': cm_data['avg_precision'],
            'Avg_Recall': cm_data['avg_recall'],
            'Avg_F1_Score': cm_data['avg_f1'],
            'Number_of_Runs': cm_data['runs']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"\nConfusion matrices saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze fault detection experiment results')
    parser.add_argument('--model_type', type=str, 
                       choices=['xgboost', 'lightgbm', 'lstm', 'svm', 'mlp'], 
                       default='xgboost', 
                       help='Type of model to analyze (default: xgboost)')
    parser.add_argument('--confusion_matrix', action='store_true',
                       help='Generate confusion matrices from metrics')
    parser.add_argument('--roc_curves', action='store_true',
                       help='Generate combined ROC curves for all faults')
    
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
    
    # Generate confusion matrices if requested
    if args.confusion_matrix:
        print("\nGenerating confusion matrices...")
        confusion_matrices = generate_confusion_matrices_from_metrics(df)
        print_confusion_matrices(confusion_matrices)
        
        # Save confusion matrices to CSV
        output_csv = os.path.join(REPORT_DIR, f'{model_type}_confusion_matrices.csv')
        save_confusion_matrices_csv(confusion_matrices, output_csv)
        
        # Create publication-quality confusion matrix visualizations
        print("Creating publication-quality confusion matrix visualizations...")
        create_publication_confusion_matrices(confusion_matrices, PIC_DIR)
        print(f"Confusion matrix visualizations saved to: {PIC_DIR}/confusion_matrices_all_faults.*")
    
    # Generate ROC curves if requested
    if args.roc_curves:
        print("\nGenerating publication-quality ROC curves...")
        curve_data = create_combined_roc_curves(df, PIC_DIR)
        
        # Save curve data summary
        curve_summary = []
        for fault_code, data in curve_data.items():
            curve_summary.append({
                'Fault_Code': fault_code,
                'Fault_Description': FAULT_DESCRIPTIONS[int(fault_code)],
                'ROC_AUC': data['auc']
            })
        
        if curve_summary:
            curve_df = pd.DataFrame(curve_summary)
            curve_csv = os.path.join(REPORT_DIR, f'{model_type}_roc_curve_summary.csv')
            curve_df.to_csv(curve_csv, index=False)
            print(f"ROC curve summary saved to: {curve_csv}")
            print(f"Publication-quality ROC curves saved to: {PIC_DIR}/combined_roc_curves_all_faults.*")


if __name__ == "__main__":
    main()