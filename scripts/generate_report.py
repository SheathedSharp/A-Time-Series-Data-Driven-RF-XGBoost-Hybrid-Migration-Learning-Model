"""
Generate comprehensive analysis report for fault detection experiments.
Supports XGBoost, LightGBM, LSTM, MLP, and SVM models.
Creates Markdown format analysis reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import argparse
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import REPORT_DIR

def calculate_confidence_interval(values, confidence=0.95):
    """Calculate confidence interval for a set of values."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    if n <= 1:
        return mean, mean
    
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_critical * (std / np.sqrt(n))
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return ci_lower, ci_upper

def generate_comprehensive_analysis(model_type='xgboost'):
    """Generate simplified Markdown analysis report.
    
    Args:
        model_type (str): Type of model ('xgboost', 'lightgbm', 'lstm', 'mlp', or 'svm')
    """
    
    # Load data from reports directory
    report_dir = Path(REPORT_DIR)
    
    detailed_df = pd.read_csv(report_dir / f'{model_type}_detailed_metrics_analysis.csv')
    
    # Convert fault codes to strings for consistent handling
    detailed_df['Fault_Code'] = detailed_df['Fault_Code'].astype(str)
    
    # Create Markdown report
    report = []
    
    # Title
    if model_type == 'lstm':
        model_name = 'LSTM'
    elif model_type == 'svm':
        model_name = 'SVM BASELINE'
    elif model_type == 'lightgbm':
        model_name = 'LIGHTGBM BASELINE'
    elif model_type == 'mlp':
        model_name = 'MLP BASELINE'
    else:
        model_name = model_type.upper()
    
    report.append(f"# COMPREHENSIVE {model_name} FAULT DETECTION PERFORMANCE ANALYSIS")
    report.append("")
    
    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    fault_codes = sorted([str(x) for x in detailed_df['Fault_Code'].unique()])
    total_experiments = len(detailed_df)
    report.append(f"- **Total experiments conducted**: {total_experiments}")
    report.append(f"- **Number of fault types analyzed**: {len(fault_codes)}")
    report.append(f"- **Experiments per fault type**: 20 (with different random seeds)")
    report.append(f"- **Fault codes investigated**: {', '.join(fault_codes)}")
    report.append("")
    
    # Overall Performance Statistics
    overall_accuracy = detailed_df['Accuracy'].mean()
    overall_precision = detailed_df['Precision'].mean()
    overall_recall = detailed_df['Recall'].mean()
    overall_f1 = detailed_df['F1_Score'].mean()
    
    report.append("## OVERALL PERFORMANCE ACROSS ALL FAULT TYPES")
    report.append(f"- **Mean Accuracy**: {overall_accuracy:.4f} ± {detailed_df['Accuracy'].std():.4f}")
    report.append(f"- **Mean Precision**: {overall_precision:.4f} ± {detailed_df['Precision'].std():.4f}")
    report.append(f"- **Mean Recall**: {overall_recall:.4f} ± {detailed_df['Recall'].std():.4f}")
    report.append(f"- **Mean F1-Score**: {overall_f1:.4f} ± {detailed_df['F1_Score'].std():.4f}")
    report.append("")
    
    # Detailed Analysis by Fault Code
    report.append("## DETAILED STATISTICAL ANALYSIS BY FAULT CODE")
    report.append("")
    
    for fault_code in fault_codes:
        fault_data = detailed_df[detailed_df['Fault_Code'] == fault_code]
        
        report.append(f"### Fault Code {fault_code}")
        report.append(f"**Sample Size**: n = {len(fault_data)}")
        report.append("")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for metric in metrics:
            values = fault_data[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            min_val = np.min(values)
            max_val = np.max(values)
            median_val = np.median(values)
            
            # Calculate 95% confidence interval
            ci_lower, ci_upper = calculate_confidence_interval(values)
            
            # Calculate coefficient of variation
            cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
            
            report.append(f"**{metric}**:")
            report.append(f"- Mean ± SD: {mean_val:.4f} ± {std_val:.4f}")
            report.append(f"- 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            report.append(f"- Range: [{min_val:.4f}, {max_val:.4f}]")
            report.append(f"- Median: {median_val:.4f}")
            report.append(f"- CV: {cv:.2f}%")
            report.append("")
        
        # Performance consistency analysis
        consistency_score = 1 - (fault_data['Accuracy'].std() / fault_data['Accuracy'].mean())
        report.append(f"**Performance Consistency Score**: {consistency_score:.4f}")
        report.append("")
    
    # Save comprehensive report as Markdown
    report_text = "\n".join(report)
    
    with open(report_dir / f'{model_type}_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"All files saved in: {report_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis report')
    parser.add_argument('--model_type', type=str, choices=['xgboost', 'lightgbm', 'lstm', 'svm', 'mlp'], 
                       default='xgboost', help='Type of model to analyze (default: xgboost)')
    args = parser.parse_args()
    
    generate_comprehensive_analysis(args.model_type)