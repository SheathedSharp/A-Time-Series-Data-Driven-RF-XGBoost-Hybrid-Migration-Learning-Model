"""
Experiment analysis utilities for fault detection performance evaluation.
Provides functions to extract metrics from log files and perform statistical analysis.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from config import RESULT_DIR, REPORT_DIR


def extract_metrics_from_log(log_file_path, model_type='xgboost'):
    """Extract performance metrics from a single log file.
    
    Args:
        log_file_path: Path to the log file
        model_type: Type of model ('xgboost', 'lightgbm', 'lstm', 'mlp', or 'svm')
    """
    with open(log_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract fault code from filename
    fault_code_match = re.search(r'_(\d{4})_', str(log_file_path))
    fault_code = fault_code_match.group(1) if fault_code_match else "Unknown"
    
    # Find all run sections
    runs = re.findall(r'Run (\d+): Random Seed (\d+).*?(?=Run \d+:|$)', content, re.DOTALL)
    
    metrics_list = []
    
    for run_num, seed in runs:
        # Extract metrics for this run
        run_start = content.find(f'Run {run_num}: Random Seed {seed}')
        if run_num == str(len(runs)):  # Last run
            run_content = content[run_start:]
        else:
            next_run_start = content.find(f'Run {int(run_num)+1}:', run_start + 1)
            if next_run_start == -1:
                run_content = content[run_start:]
            else:
                run_content = content[run_start:next_run_start]
        
        # Extract performance metrics using regex (different formats for different models)
        if model_type == 'lstm':
            # LSTM uses a different format
            accuracy_match = re.search(r'│\s+Accuracy\s+│\s+([\d.]+)\s+│', run_content)
            precision_match = re.search(r'│\s+Precision\s+│\s+([\d.]+)\s+│', run_content)
            recall_match = re.search(r'│\s+Recall\s+│\s+([\d.]+)\s+│', run_content)
            f1_match = re.search(r'│\s+F1 Score\s+│\s+([\d.]+)\s+│', run_content)
        elif model_type in ['svm', 'mlp']:
            # SVM and MLP use their own format
            accuracy_match = re.search(r'│\s+Accuracy\s+│\s+([\d.]+)\s+│', run_content)
            precision_match = re.search(r'│\s+Precision\s+│\s+([\d.]+)\s+│', run_content)
            recall_match = re.search(r'│\s+Recall\s+│\s+([\d.]+)\s+│', run_content)
            f1_match = re.search(r'│\s+F1 Score\s+│\s+([\d.]+)\s+│', run_content)
        else:
            # XGBoost and LightGBM format
            accuracy_match = re.search(r'│\s+Accuracy\s+│\s+([\d.]+)\s+│', run_content)
            precision_match = re.search(r'│\s+Precision\s+│\s+([\d.]+)\s+│', run_content)
            recall_match = re.search(r'│\s+Recall\s+│\s+([\d.]+)\s+│', run_content)
            f1_match = re.search(r'│\s+weighted avg - F1-Score\s+│\s+([\d.]+)\s+│', run_content)
        
        if accuracy_match and precision_match and recall_match and f1_match:
            metrics_list.append({
                'Fault_Code': fault_code,
                'Run': int(run_num),
                'Random_Seed': int(seed),
                'Accuracy': float(accuracy_match.group(1)),
                'Precision': float(precision_match.group(1)),
                'Recall': float(recall_match.group(1)),
                'F1_Score': float(f1_match.group(1))
            })
    
    return metrics_list


def calculate_statistics(df, fault_code):
    """Calculate statistical measures for a specific fault code.
    
    Args:
        df: DataFrame containing metrics data
        fault_code: Specific fault code to analyze
    """
    fault_data = df[df['Fault_Code'] == fault_code]
    
    stats = {}
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    for metric in metrics:
        values = fault_data[metric].values
        stats[f'{metric}_Mean'] = np.mean(values)
        stats[f'{metric}_Std'] = np.std(values, ddof=1)  # Sample standard deviation
        stats[f'{metric}_Min'] = np.min(values)
        stats[f'{metric}_Max'] = np.max(values)
        stats[f'{metric}_Median'] = np.median(values)
        stats[f'{metric}_95CI_Lower'] = np.percentile(values, 2.5)
        stats[f'{metric}_95CI_Upper'] = np.percentile(values, 97.5)
    
    stats['Fault_Code'] = fault_code
    stats['Sample_Size'] = len(values)
    
    return stats


def get_log_pattern(model_type):
    """Get the log file pattern for a specific model type.
    
    Args:
        model_type: Type of model
    """
    patterns = {
        'xgboost': 'xgboost_hybrid_batch_*.log',
        'lightgbm': 'lightgbm_baseline_batch_*.log',
        'svm': 'svm_baseline_batch_*.log',
        'mlp': 'mlp_baseline_batch_*.log',
        'lstm': 'lstm_batch_*.log'
    }
    return patterns.get(model_type, f'{model_type}_batch_*.log')


def find_log_files(model_type):
    """Find all log files for a specific model type.
    
    Args:
        model_type: Type of model to search for
    """
    log_pattern = get_log_pattern(model_type)
    result_dir = Path(RESULT_DIR)
    return sorted(list(result_dir.glob(log_pattern)))


def process_experiments(model_type):
    """Process all experiment log files for a model type.
    
    Args:
        model_type: Type of model to analyze
    """
    log_files = find_log_files(model_type)
    
    # Extract metrics from all log files
    all_metrics = []
    
    for log_file in log_files:
        try:
            metrics = extract_metrics_from_log(str(log_file), model_type)
            all_metrics.extend(metrics)
        except Exception as e:
            print(f"Error processing {log_file.name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    if df.empty:
        print("No metrics extracted. Please check log file format.")
    
    return df


def generate_statistics_summary(df):
    """Generate statistical summary for all fault codes.
    
    Args:
        df: DataFrame containing metrics data
    """
    fault_codes = sorted(df['Fault_Code'].unique())
    all_statistics = []
    
    for fault_code in fault_codes:
        stats = calculate_statistics(df, fault_code)
        all_statistics.append(stats)
    
    return pd.DataFrame(all_statistics)


def save_results(df, stats_df, model_type):
    """Save analysis results to CSV files.
    
    Args:
        df: DataFrame containing detailed metrics
        stats_df: DataFrame containing statistical summary
        model_type: Type of model (for filename)
    """
    report_dir = Path(REPORT_DIR)
    
    # Save detailed results
    df.to_csv(report_dir / f'{model_type}_detailed_metrics_analysis.csv', index=False)
    
    # Save statistical summary
    stats_df.to_csv(report_dir / f'{model_type}_statistical_summary.csv', index=False)


def print_publication_table(stats_df, model_type):
    """Print publication-ready results table.
    
    Args:
        stats_df: DataFrame containing statistical summary
        model_type: Type of model (for table title)
    """
    print(f"Table: {model_type.upper()} Performance Metrics by Fault Code")
    print("-" * 80)
    print(f"{'Fault Code':<12} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 80)
    
    fault_codes = sorted(stats_df['Fault_Code'].unique())
    
    for fault_code in fault_codes:
        stats = stats_df[stats_df['Fault_Code'] == fault_code].iloc[0]
        
        acc_str = f"{stats['Accuracy_Mean']:.3f}±{stats['Accuracy_Std']:.3f}"
        prec_str = f"{stats['Precision_Mean']:.3f}±{stats['Precision_Std']:.3f}"
        rec_str = f"{stats['Recall_Mean']:.3f}±{stats['Recall_Std']:.3f}"
        f1_str = f"{stats['F1_Score_Mean']:.3f}±{stats['F1_Score_Std']:.3f}"
        
        print(f"{fault_code:<12} {acc_str:<15} {prec_str:<15} {rec_str:<15} {f1_str:<15}")
    
    print("-" * 80)
    print("Note: Values shown as Mean ± Standard Deviation")