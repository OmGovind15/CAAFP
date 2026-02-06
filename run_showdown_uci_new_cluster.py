#
# FILENAME: run_showdown_uci.py
#
"""
UCI HAR Showdown Runner (Comparison @ ~70% Sparsity)
Runs all UCI-adapted baselines with multiple random seeds for reproducibility.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from datetime import datetime
import pickle

# --- Configuration ---
SEEDS = [42, 123, 456, 789, 1024]
NUM_ROUNDS = 50 
CLIENTS_PER_ROUND = 10
EPOCHS_PER_ROUND = 3
FINE_TUNE_EPOCHS = 25
# Map method names to their file (for reference/logging)
METHODS = {
    'caafp_uci': 'main_caafp_cnn_uci.py',
    'clusterfl_uci': 'main_clusterfl_cnn_uci.py',
    'caafp_uci_cluster': 'main_caafp_cnn_uci_cluster.py',
    'efficient_fl_uci': 'main_efficient_fl_cnn_uci.py',
    'fedsnip_uci': 'main_fedsnip_cnn_uci.py',
    'fedchar_uci': 'main_fedchar_cnn_uci.py',
    
    
}

def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set all seeds to {seed}")

def extract_tracker_metrics(raw_metrics):
    """Safe extraction of metrics from FLMetricsTracker output"""
    if not raw_metrics:
        return {}
        
    # Handle case where metrics might be a list (history) or dict
    data = raw_metrics
    if isinstance(data, list):
        data = data[-1]
    elif isinstance(data, dict):
        # If dict keys are rounds (integers), get the last round
        if any(isinstance(k, int) for k in data.keys()):
            last_round = max(data.keys())
            data = data[last_round]
            
    return {
        'total_comm_mb': data.get('total_comm_mb', 0),
        'total_gflops': data.get('total_gflops', 0),
        'wall_time': data.get('total_training_time', 0)
    }

def run_single_experiment(method, seed, run_id):
    """
    Run a single experiment with specified method and seed.
    Returns a standardized dictionary of results.
    """
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} with seed {seed}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    set_all_seeds(seed)

    # ---------------------------------------------------------
    # 1. FedCHAR (UCI)
    # ---------------------------------------------------------
    if method == 'fedchar_uci':
        from main_fedchar_cnn_uci import run_fedchar
        # FedCHAR UCI script returns a 'rich_metrics' dict directly
        rich_metrics = run_fedchar(
            num_rounds=NUM_ROUNDS, 
            initial_rounds=10, 
            alpha=0.1,
            epochs_per_round=EPOCHS_PER_ROUND,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            seed=seed
        )
        # Ensure ID consistency
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return rich_metrics

    # ---------------------------------------------------------
    # 2. ClusterFL (UCI)
    # ---------------------------------------------------------
    elif method == 'clusterfl_uci':
        from main_clusterfl_cnn_uci import run_clusterfl
        
        # Unpack the 4 return values explicitly
        server, clients, results, raw_metrics = run_clusterfl(
            num_rounds=NUM_ROUNDS,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            alpha=0.1,
            seed=seed
        )
        
        # 1. Calculate Accuracy manually from 'results'
        accuracies = [r['accuracy'] for r in results.values()]
        avg_acc = np.mean(accuracies) if accuracies else 0.0
        
        # 2. Prepare the return dictionary
        # Start with the system metrics (comm, time, etc.)
        rich_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        
        # Fallback if raw_metrics is empty/weird
        if not rich_metrics and hasattr(raw_metrics, 'get_results'):
             rich_metrics = raw_metrics.get_results()
        
        # 3. Inject the missing keys
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        rich_metrics['avg_accuracy'] = avg_acc  # <--- CRITICAL FIX
        rich_metrics['all_accuracies'] = accuracies
        
        # Optional: Add sparsity if missing (ClusterFL is dense, so 0.0)
        if 'avg_sparsity' not in rich_metrics:
            rich_metrics['avg_sparsity'] = 0.0
            
        return rich_metrics
    # ---------------------------------------------------------
    # 3. EfficientFL (UCI)
    # ---------------------------------------------------------
    elif method == 'efficient_fl_uci':
        from main_efficient_fl_cnn_uci import run_efficient_fl
        
        # 1. Capture the single result dictionary
        result_metrics = run_efficient_fl(
            num_rounds=NUM_ROUNDS, 
            target_sparsity=0.7, 
            alpha=0.1,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            seed=seed
        )
        
        # 2. Copy to rich_metrics
        rich_metrics = result_metrics.copy()
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        
        # 3. Safety defaults (Prevent crashes & Ensure non-zero comm cost if available)
        if 'avg_sparsity' not in rich_metrics:
            rich_metrics['avg_sparsity'] = 0.7
        if 'avg_accuracy' not in rich_metrics:
            rich_metrics['avg_accuracy'] = 0.0
        
        # Critical: If total_comm_mb is missing, default to 0.0
        if 'total_comm_mb' not in rich_metrics:
             rich_metrics['total_comm_mb'] = 0.0

        return rich_metrics
    # ---------------------------------------------------------
    # 4. FedSNIP (UCI)
    # ---------------------------------------------------------
    elif method == 'fedsnip_uci':
        from main_fedsnip_cnn_uci import run_fedsnip
        # Returns: server, clients, results, rich_metrics
        _, _, _, rich_metrics = run_fedsnip(
            num_rounds=NUM_ROUNDS, 
            target_sparsity=0.7, 
            alpha=0.1,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            seed=seed
        )
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return rich_metrics
    elif method == 'caafp_uci_cluster':
        from main_caafp_cnn_uci_cluster import run_caafp
        
        # Define the 4 combinations summing to 50 rounds
        # Format: (initial_rounds, post_cluster_warmup, pruning_rounds)
        combinations = [
           # (10, 10, 30),  # Combo 1: Balanced Warmup
           # (15, 10, 25),  # Combo 2: High Global Warmup
           # (10, 15, 25),  # Combo 3: High Cluster Warmup
            (15, 15, 20)   # Combo 4: Maximum Stability (Short Pruning)
        ]
        
        # We need to run ALL combinations, so we loop here.
        # Note: This changes the return type to a LIST of results, which
        # run_all_experiments handles (it appends them to the main list).
        
        combo_results = []
        
        for i, (init_r, post_r, prune_r) in enumerate(combinations):
            # Verify Total Rounds
            total_check = init_r + post_r + prune_r
            if total_check != 50:
                print(f"Skipping Invalid Combo {i+1}: Sums to {total_check}, not 50")
                continue

            # Construct a unique ID for this specific combo run
            combo_run_id = f"{run_id}_combo{i+1}_w{init_r}_p{post_r}_pr{prune_r}"
            
            print(f"\n--- Running Combo {i+1}: Global Warmup={init_r}, Post-Warmup={post_r}, Pruning={prune_r} ---")
            
            # For 'clustering_training_rounds', we pass the rounds remaining AFTER global warmup
            # This covers both post-cluster warmup AND pruning phases.
            # Example: 50 total - 10 init = 40. 
            # Inside run_caafp, it splits 40 into 10 (post) + 30 (pruning).
            rounds_after_global = 50 - init_r
            
            server, clients, results, final_metrics = run_caafp(
                num_clients=30,
                num_rounds=50,  # Total rounds is always 50
                
                initial_rounds=init_r,             
                post_cluster_warmup=post_r,        
                clustering_training_rounds=rounds_after_global, 
                
                clients_per_round=CLIENTS_PER_ROUND,
                epochs_per_round=EPOCHS_PER_ROUND,
                fine_tune_epochs=FINE_TUNE_EPOCHS,
                prune_rate=0.05, 
                start_sparsity=0.7, 
                alpha=0.1,
                seed=seed
            )
            
            accuracies = [r['accuracy'] for r in results.values()]
            track_stats = extract_tracker_metrics(final_metrics)
            
            # Calculate final sparsity safely
            from models_cnn_uci import get_model_sparsity
            sparsities = []
            if hasattr(server, 'final_pruned_models'):
                 sparsities = [get_model_sparsity(m) for m in server.final_pruned_models.values()]
            avg_sparsity = np.mean(sparsities) if sparsities else 0.0
            
            res_dict = {
                'method': f"caafp_combo_{i+1}", # Unique method name for CSV
                'seed': seed,
                'run_id': combo_run_id,
                'config': f"init{init_r}_post{post_r}_prune{prune_r}", # Store config details
                'avg_accuracy': np.mean(accuracies),
                'std_dev': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'avg_sparsity': avg_sparsity,
                
                'total_comm_mb': track_stats['total_comm_mb'],
                'total_gflops': track_stats['total_gflops'],
                'wall_time': track_stats['wall_time'],
                
                'acc_per_mb': np.mean(accuracies) / track_stats['total_comm_mb'] if track_stats['total_comm_mb'] > 0 else 0
            }
            combo_results.append(res_dict)
            
        # If run_single_experiment expects a single dict, we have a small issue.
        # However, looking at your runner loop: "all_results.append(result)"
        # You should modify the runner loop slightly to handle a list, 
        # OR just return the last one and save the others manually here.
        
        # BETTER FIX: Return the list. You must update 'run_all_experiments' loop 
        # to check: "if isinstance(result, list): all_results.extend(result)"
        return combo_results
    # ---------------------------------------------------------
    # 5. CA-AFP (UCI)
    # ---------------------------------------------------------
    elif method == 'caafp_uci':
        from main_caafp_cnn_uci import run_caafp
        # Returns: server, clients, results, final_metrics
        
        # Note: CA-AFP usually benefits from a longer schedule, but for 
        # fair "Showdown" we try to keep rounds consistent or slightly adjusted 
        # if the method requires phases.
        # We'll use 50 rounds total to match others (Warmup is implicit or 0).
        server, clients, results, final_metrics = run_caafp(
            num_clients=30,
            num_rounds=NUM_ROUNDS, 
            initial_rounds=0,  
            clustering_training_rounds=NUM_ROUNDS, # Run full pruning schedule
            clients_per_round=CLIENTS_PER_ROUND,
            epochs_per_round=EPOCHS_PER_ROUND,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            prune_rate=0.05, 
            start_sparsity=0.7, # Start high for UCI
            alpha=0.1,
            
            seed=seed
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        track_stats = extract_tracker_metrics(final_metrics)
        
        # Calculate final sparsity
        from models_cnn_uci import get_model_sparsity
        sparsities = [get_model_sparsity(m) for m in server.final_pruned_models.values()]
        avg_sparsity = np.mean(sparsities) if sparsities else 0.0
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': avg_sparsity,
            
            'total_comm_mb': track_stats['total_comm_mb'],
            'total_gflops': track_stats['total_gflops'],
            'wall_time': track_stats['wall_time'],
            
            'acc_per_mb': np.mean(accuracies) / track_stats['total_comm_mb'] if track_stats['total_comm_mb'] > 0 else 0,
            'all_accuracies': accuracies
        }

def save_results(results, filename):
    """Save results to JSON and pickle"""
    # JSON safe save
    with open(filename + '.json', 'w') as f:
        results_json = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                results_json[k] = float(v)
            elif isinstance(v, list):
                results_json[k] = v
            else:
                results_json[k] = str(v)
        json.dump(results_json, f, indent=2)
    
    # Pickle save
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(results, f)

def run_all_experiments():
    """Run all methods with all seeds (Seed-Major Order)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results_uci_showdown', exist_ok=True)
    
    all_results = []
    
    # --- CHANGED: Outer loop is SEEDS, Inner loop is METHODS ---
    for seed in SEEDS:
        print(f"\n\n{'#'*80}")
        print(f"STARTING BATCH FOR SEED: {seed}")
        print(f"{'#'*80}")
        
        for method in METHODS.keys():
            run_id = f"{method}_{seed}_{timestamp}"
            
            try:
                # result can be a dict (standard) or a list of dicts (combo)
                result = run_single_experiment(method, seed, run_id)
                
                if isinstance(result, list):
                    # It's a list of results (from the 4 combos)
                    all_results.extend(result)
                    
                    # Save the batch immediately
                    save_results({'combos': result}, f'results_uci_showdown/{run_id}')
                    
                    print(f"\n✓ Completed Combo Batch: {method} with seed {seed}")
                    for res in result:
                         print(f"    Config {res['config']}: Avg Acc {res['avg_accuracy']:.4f}")

                else:
                    # It's a single dictionary result
                    all_results.append(result)
                    
                    # Save individual result immediately
                    save_results(result, f'results_uci_showdown/{run_id}')
                    
                    print(f"\n✓ Completed: {method} with seed {seed}")
                    print(f"  Avg Accuracy: {result['avg_accuracy']:.4f}")
                    if 'avg_sparsity' in result:
                        print(f"  Avg Sparsity: {result['avg_sparsity']:.2%}")
                
            except Exception as e:
                print(f"\n✗ Failed: {method} with seed {seed}")
                import traceback
                traceback.print_exc()
                continue
    
    # --- Final CSV Summary ---
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(all_results)
    
    csv_path = f'results_uci_showdown/final_summary_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(f"UCI SHOWDOWN COMPLETE")
    print(f"Summary saved to: {csv_path}")
    print("="*70)
    
    # Print Pivot Table
    if not df.empty:
        # Group by 'method' AND 'config' if available to distinguish combos
        cols_to_group = ['method']
        if 'config' in df.columns:
            cols_to_group.append('config')
            
        pivot = df.groupby(cols_to_group)[['avg_accuracy', 'total_comm_mb', 'acc_per_mb']].mean()
        print("\nAverage Performance across seeds:")
        print(pivot)
if __name__ == "__main__":
    run_all_experiments()