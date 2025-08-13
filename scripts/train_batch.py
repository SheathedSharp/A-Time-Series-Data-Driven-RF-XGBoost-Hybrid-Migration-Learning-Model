import subprocess
import os
from datetime import datetime
from config import MODEL_CONFIGS

def run_model_experiments_with_variance(model_type, production_line=1, fault_code=1001):
    """
    Run 20 model training experiments with different random seeds.
    Supports both hybrid RF-XGBoost method and baseline models for comprehensive comparison.
    
    Args:
        model_type (str): Type of model ('xgboost', 'lightgbm', 'svm', 'mlp', 'lstm')
        production_line (int): Production line identifier
        fault_code (int): Fault code to predict
    """
    
    if model_type not in MODEL_CONFIGS:
        print(f"Error: Unsupported model type '{model_type}'")
        print(f"Supported models: {', '.join(MODEL_CONFIGS.keys())}")
        return
    
    config = MODEL_CONFIGS[model_type]
    
    # Generate 20 different random seeds, ensuring 42 is included
    random_seeds = [42]  # Start with 42 as requested
    
    # Add 19 more diverse seeds
    additional_seeds = [1, 7, 13, 21, 33, 55, 89, 144, 233, 377, 
                       610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657]
    random_seeds.extend(additional_seeds)
    
    print(f"Starting {config['name']} experiments for Production Line {production_line}, Fault Code {fault_code}")
    print(f"Running {len(random_seeds)} experiments with random seeds: {random_seeds}")
    
    if model_type == 'xgboost':
        print("HYBRID APPROACH:")
        print("  - Temporal feature engineering")
        print("  - CBSS balanced sampling") 
        print("  - RF feature selection")
        print(f"  - {config['description']}")
    else:
        print("BASELINE APPROACH:")
        print("  - Raw data only (no temporal features)")
        print("  - No CBSS balanced sampling")
        print("  - No RF feature selection")
        print(f"  - {config['description']}")
    print("=" * 80)
    
    # Create timestamp for this experiment batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_type = "hybrid" if model_type == 'xgboost' else "baseline"
    experiment_log = f"experiments/results/{model_type}_{batch_type}_batch_{production_line}_{fault_code}_{timestamp}.log"
    
    successful_runs = 0
    failed_runs = 0
    
    try:
        with open(experiment_log, 'w') as log_file:
            log_file.write(f"{config['name']} Batch Experiment Log\n")
            log_file.write(f"Production Line: {production_line}\n")
            log_file.write(f"Fault Code: {fault_code}\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"Random Seeds: {random_seeds}\n")
            if model_type == 'xgboost':
                log_file.write(f"HYBRID APPROACH:\n")
                log_file.write(f"  - Temporal feature engineering\n")
                log_file.write(f"  - CBSS balanced sampling\n") 
                log_file.write(f"  - RF feature selection\n")
                log_file.write(f"  - {config['description']}\n")
            else:
                log_file.write(f"BASELINE APPROACH:\n")
                log_file.write(f"  - Raw data only (no temporal features)\n")
                log_file.write(f"  - No CBSS balanced sampling\n")
                log_file.write(f"  - No RF feature selection\n")
                log_file.write(f"  - {config['description']}\n")
            log_file.write("=" * 80 + "\n\n")
            
            for i, seed in enumerate(random_seeds, 1):
                model_desc = "hybrid" if model_type == 'xgboost' else "baseline"
                print(f"Run {i}/20: Training {model_type.upper()} {model_desc} with random seed {seed}...")
                
                # Construct the command for baseline training
                cmd = [
                    'uv', 'run', 'python', config['script'],
                    '--production_line', str(production_line),
                    '--fault_code', str(fault_code),
                    '--random-state', str(seed)
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
    
    except Exception as e:
        print(f"Error during experiment batch: {e}")
        if 'log_file' in locals():
            log_file.write(f"BATCH ERROR: {str(e)}\n")
    
    # Summary
    print("=" * 80)
    print(f"{config['name']} EXPERIMENT BATCH SUMMARY")
    print(f"Total runs: {len(random_seeds)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Success rate: {successful_runs/len(random_seeds)*100:.1f}%")
    print(f"Log file: {experiment_log}")
    print(f"Models saved to: trained_models/")
    
    if model_type == 'xgboost':
        print("HYBRID CONFIGURATION:")
        print("  - Temporal feature engineering applied")
        print("  - CBSS balanced sampling applied")
        print("  - RF feature selection applied")
        print(f"  - {config['description']}")
        print("  - Complete preprocessing pipeline")
    else:
        print("BASELINE CONFIGURATION:")
        print("  - Used raw data only (no temporal features)")
        print("  - No CBSS balanced sampling applied")
        print("  - No RF feature selection applied")
        print(f"  - {config['description']}")
        print("  - Fair baseline for comparison with hybrid RF-XGBoost method")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run multiple model training experiments for statistical validation'
    )
    parser.add_argument(
        'model_type',
        choices=list(MODEL_CONFIGS.keys()),
        help='Type of model to train'
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
    
    run_model_experiments_with_variance(args.model_type, args.production_line, args.fault_code)