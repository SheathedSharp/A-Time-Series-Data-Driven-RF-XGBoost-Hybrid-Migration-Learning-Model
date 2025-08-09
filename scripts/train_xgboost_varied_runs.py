#!/usr/bin/env python3
"""
Script to train 20 XGBoost models with truly different random seeds for journal requirements.
This version ensures each run uses different feature selection and model parameters.
"""

import subprocess
import sys
import os
import shutil
from datetime import datetime
from config import get_feature_path

def run_xgboost_experiments_with_variance(production_line=1, fault_code=1001):
    """
    Run 20 XGBoost training experiments with different random seeds.
    Ensures true randomness by temporarily backing up pre-selected features.
    """
    
    # Generate 20 different random seeds, ensuring 42 is included
    random_seeds = [42]  # Start with 42 as requested
    
    # Add 19 more diverse seeds
    additional_seeds = [1, 7, 13, 21, 33, 55, 89, 144, 233, 377, 
                       610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657]
    random_seeds.extend(additional_seeds)
    
    print(f"Starting VARIED XGBoost experiments for Production Line {production_line}, Fault Code {fault_code}")
    print(f"Running {len(random_seeds)} experiments with random seeds: {random_seeds}")
    print("Note: Pre-selected features will be temporarily moved to ensure variance")
    print("=" * 80)
    
    # Backup pre-selected features if they exist
    features_path = get_feature_path(fault_code)
    backup_path = None
    if os.path.exists(features_path):
        backup_path = features_path + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(features_path, backup_path)
        print(f"Backed up pre-selected features to: {backup_path}")
    
    # Create timestamp for this experiment batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log = f"experiments/result/xgboost_varied_batch_{production_line}_{fault_code}_{timestamp}.log"
    
    successful_runs = 0
    failed_runs = 0
    
    try:
        with open(experiment_log, 'w') as log_file:
            log_file.write(f"XGBoost VARIED Batch Experiment Log\n")
            log_file.write(f"Production Line: {production_line}\n")
            log_file.write(f"Fault Code: {fault_code}\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"Random Seeds: {random_seeds}\n")
            log_file.write(f"Features backup: {backup_path if backup_path else 'None'}\n")
            log_file.write("Note: Each run performs fresh feature selection\n")
            log_file.write("=" * 80 + "\n\n")
            
            for i, seed in enumerate(random_seeds, 1):
                print(f"Run {i}/20: Training with random seed {seed} (fresh feature selection)...")
                
                # Construct the command with parameter optimization for more variance
                cmd = [
                    'uv', 'run', 'python', 'scripts/train_xgboost.py',
                    '--production_line', str(production_line),
                    '--fault_code', str(fault_code),
                    '--random-state', str(seed),
                    '--parameter-opt'  # Enable parameter optimization
                ]
                
                log_file.write(f"Run {i}: Random Seed {seed}\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.flush()
                
                try:
                    # Run the training command
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=os.getcwd()
                    )
                    
                    print(f"  ✓ Run {i} completed successfully")
                    log_file.write(f"Status: SUCCESS\n")
                    log_file.write(f"STDOUT:\n{result.stdout}\n")
                    if result.stderr:
                        log_file.write(f"STDERR:\n{result.stderr}\n")
                    successful_runs += 1
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Run {i} failed with return code {e.returncode}")
                    log_file.write(f"Status: FAILED (return code {e.returncode})\n")
                    log_file.write(f"STDOUT:\n{e.stdout}\n")
                    log_file.write(f"STDERR:\n{e.stderr}\n")
                    failed_runs += 1
                
                except Exception as e:
                    print(f"  ✗ Run {i} failed with error: {e}")
                    log_file.write(f"Status: ERROR - {str(e)}\n")
                    failed_runs += 1
                
                log_file.write("-" * 40 + "\n\n")
                log_file.flush()
                
                # Clean up any generated feature file after each run to ensure fresh selection
                if os.path.exists(features_path):
                    os.remove(features_path)
    
    finally:
        # Restore backup if it exists
        if backup_path and os.path.exists(backup_path):
            shutil.move(backup_path, features_path)
            print(f"Restored pre-selected features from backup")
    
    # Summary
    print("=" * 80)
    print("VARIED EXPERIMENT BATCH SUMMARY")
    print(f"Total runs: {len(random_seeds)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Success rate: {successful_runs/len(random_seeds)*100:.1f}%")
    print(f"Log file: {experiment_log}")
    print(f"Results saved to: experiments/results/")
    print(f"Confusion matrices saved to: experiments/pic/")
    print("Note: Each run used fresh feature selection for true variance")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run multiple XGBoost training experiments with varied feature selection and parameters'
    )
    parser.add_argument(
        '--production_line', 
        type=int, 
        default=1,
        help='Production line code (default: 1)'
    )
    parser.add_argument(
        '--fault_code', 
        type=int, 
        default=1001,
        help='Fault code (default: 1001)'
    )
    
    args = parser.parse_args()
    
    # Ensure experiments directory exists
    os.makedirs('experiments/results', exist_ok=True)
    os.makedirs('experiments/pic', exist_ok=True)
    
    run_xgboost_experiments_with_variance(args.production_line, args.fault_code)