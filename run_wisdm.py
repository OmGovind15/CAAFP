#
# FILENAME: run_showdown_wisdm.py
#
"""
WISDM Showdown Runner (Comparison on Human Activity Recognition)
Runs all WISDM-adapted baselines with multiple random seeds.
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
from data_loader import load_wisdm_dataset, create_natural_user_split, create_tf_datasets
# --- Configuration ---
SEEDS = [42, 123, 456, 789, 1024]
NUM_ROUNDS = 50 
CLIENTS_PER_ROUND = 10
EPOCHS_PER_ROUND = 3
FINE_TUNE_EPOCHS = 3
# Map method names to their file (for reference/logging)
# CHANGED: Updated filenames to _wisdm versions
METHODS = {
    'caafp_wisdm_cluster': 'main_caafp_cnn_cluster.py',
    #'fedchar_wisdm': 'main_fedchar_cnn.py',
    #'clusterfl_wisdm': 'main_clusterfl_cnn.py',
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
        
    data = raw_metrics
    if isinstance(data, list):
        data = data[-1]
    elif isinstance(data, dict):
        if any(isinstance(k, int) for k in data.keys()):
            last_round = max(data.keys())
            data = data[last_round]
            
    return {
        'total_comm_mb': data.get('total_comm_mb', 0),
        'total_gflops': data.get('total_gflops', 0),
        'wall_time': data.get('total_training_time', 0)
    }

def run_single_experiment(method, seed, run_id):
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} with seed {seed}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    set_all_seeds(seed)

    # --- HELPER: Load Data for Baselines (FedCHAR / ClusterFL) ---
    # CA-AFP loads data internally, so we skip this for it.
    if method in ['fedchar_wisdm', 'clusterfl_wisdm']:
        print(f"[INFO] Loading WISDM data for {method}...")
        from data_loader import load_wisdm_dataset, create_natural_user_split, create_tf_datasets
        
        X_data, y_data, user_ids = load_wisdm_dataset()
        client_data = create_natural_user_split(X_data, y_data, user_ids)
        num_clients = len(client_data)
        train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)

    # ---------------------------------------------------------
    # 1. FedCHAR (WISDM)
    # ---------------------------------------------------------
    if method == 'fedchar_wisdm':
        from main_fedchar_cnn import run_fedchar
        rich_metrics = run_fedchar(
            train_datasets,    # <--- PASSED POSITIONAL ARG
            test_datasets,     # <--- PASSED POSITIONAL ARG
            num_clients,       # <--- PASSED POSITIONAL ARG
            num_rounds=NUM_ROUNDS, 
            initial_rounds=10, 
            alpha=0.1,
            epochs_per_round=EPOCHS_PER_ROUND,
            seed=seed,
            fine_tune_epochs=FINE_TUNE_EPOCHS 
        )
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return [rich_metrics]

    # ---------------------------------------------------------
    # 2. ClusterFL (WISDM)
    # ---------------------------------------------------------
    elif method == 'clusterfl_wisdm':
        from main_clusterfl_cnn import run_clusterfl
        server, clients, results, raw_metrics = run_clusterfl(
            train_datasets,    # <--- PASSED POSITIONAL ARG
            test_datasets,     # <--- PASSED POSITIONAL ARG
            num_clients,       # <--- PASSED POSITIONAL ARG
            num_rounds=NUM_ROUNDS,
            alpha=0.1,
            seed=seed,
            fine_tune_epochs=FINE_TUNE_EPOCHS 
        )
        
        # Standardize Output
        accuracies = [r['accuracy'] for r in results.values()]
        rich_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        
        if not rich_metrics and hasattr(raw_metrics, 'get_results'):
             rich_metrics = raw_metrics.get_results()
        
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        rich_metrics['avg_accuracy'] = np.mean(accuracies) if accuracies else 0.0
        rich_metrics['std_accuracy'] = np.std(accuracies) if accuracies else 0.0
        rich_metrics['all_accuracies'] = accuracies
        if 'avg_sparsity' not in rich_metrics: rich_metrics['avg_sparsity'] = 0.0
            
        return [rich_metrics]

    # ---------------------------------------------------------
    # 3. CA-AFP (WISDM) - CLUSTER COMBINATIONS
    # ---------------------------------------------------------
    elif method == 'caafp_wisdm_cluster':
        from main_caafp_cnn_cluster import run_caafp
        
        # The 4 Combinations (Total 50 Rounds)
        combinations = [
           # (10, 10, 30),  # Combo 1: Balanced
           # (15, 10, 25),  # Combo 2: High Global Warmup
           # (10, 15, 25),  # Combo 3: High Cluster Warmup
            (15, 15, 20)   # Combo 4: Max Stability
        ]
        
        combo_results = []
        
        for i, (init_r, post_r, prune_r) in enumerate(combinations):
            combo_run_id = f"{run_id}_combo{i+1}_w{init_r}_p{post_r}_pr{prune_r}"
            print(f"\n--- Running Combo {i+1}: G-Warm={init_r}, Cl-Warm={post_r}, Prune={prune_r} ---")
            
            # Calculate remaining rounds for the pruning phase
            rounds_after_global = 50 - init_r
            
            # NOTE: run_caafp loads data INTERNALLY, so we don't pass datasets here
            server, clients, results, final_metrics = run_caafp(
                num_clients=30,
                num_rounds=50, 
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
            
            # Standardize Output
            accuracies = [r['accuracy'] for r in results.values()]
            track_stats = extract_tracker_metrics(final_metrics)
            
            from models_cnn import get_model_sparsity
            sparsities = []
            if hasattr(server, 'final_pruned_models'):
                 sparsities = [get_model_sparsity(m) for m in server.final_pruned_models.values()]
            avg_sparsity = np.mean(sparsities) if sparsities else 0.0
            
            res_dict = {
                'method': f"caafp_combo_{i+1}",
                'seed': seed,
                'run_id': combo_run_id,
                'config': f"init{init_r}_post{post_r}_prune{prune_r}",
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies) if accuracies else 0.0,
                'avg_sparsity': avg_sparsity,
                'total_comm_mb': track_stats.get('total_comm_mb', 0),
                'acc_per_mb': np.mean(accuracies) / track_stats.get('total_comm_mb', 1e-9) if track_stats.get('total_comm_mb', 0) > 0 else 0
            }
            combo_results.append(res_dict)
            
            # Partial Save
            save_results({'partial_combos': combo_results}, f'results_wisdm_showdown/{run_id}_partial')
            
        return combo_results
def save_results(results, filename):
    """Save results to JSON and pickle"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
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
    
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(results, f)

def run_all_experiments():
    """Run all methods with all seeds"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # CHANGED: New output directory
    output_dir = 'results_wisdm_showdown'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for method in METHODS.keys():
        for seed in SEEDS:
            run_id = f"{method}_{seed}_{timestamp}"
            
            try:
                # CHANGED BLOCK STARTS HERE
                result_list = run_single_experiment(method, seed, run_id)
                all_results.extend(result_list) # Use .extend() to flatten the list
                
                # Loop through the list to save each result individually
                for result in result_list:
                    # Use the specific run_id if it exists (for combos), else the main one
                    specific_id = result.get('run_id', run_id)
                    save_results(result, f'{output_dir}/{specific_id}')
                    
                    print(f"\n✓ Completed: {result.get('method', method)} with seed {seed}")
                    print(f"  Avg Accuracy: {result.get('avg_accuracy', 0):.4f}")
                    print(f"  Std Accuracy: {result.get('std_accuracy', 0):.4f}") # Print Std Dev
                # CHANGED BLOCK ENDS HERE
                
            except Exception as e:
                    print(f"\n✗ Failed: {method} with seed {seed}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # --- Final CSV Summary ---
    df = pd.DataFrame(all_results)
    csv_path = f'{output_dir}/final_summary_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(f"WISDM SHOWDOWN COMPLETE")
    print(f"Summary saved to: {csv_path}")
    print("="*70)
    
    if not df.empty:
        pivot = df.groupby('method')[['avg_accuracy', 'total_comm_mb', 'acc_per_mb']].mean()
        print("\nAverage Performance across seeds:")
        print(pivot)

if __name__ == "__main__":
    run_all_experiments()
