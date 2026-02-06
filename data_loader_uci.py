import numpy as np
import tensorflow as tf
import os
import sys
from collections import defaultdict

def load_uci_har_dataset(root_path='./UCI HAR Dataset'):
    """
    Load the UCI HAR dataset.
    Input Shape: (N, 128, 9) -> 9 channels
    """
    SIGNALS = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    def _load_signals(subset):
        signals_data = []
        for signal in SIGNALS:
            filename = f'{root_path}/{subset}/Inertial Signals/{signal}_{subset}.txt'
            try:
                with open(filename, 'r') as f:
                    data = [line.strip().split() for line in f.readlines()]
                signals_data.append(np.array(data, dtype=np.float32))
            except FileNotFoundError:
                print(f"Error: Could not find {filename}. Make sure 'UCI HAR Dataset' is unzipped.")
                sys.exit(1)
        
        return np.transpose(np.array(signals_data), (1, 2, 0))

    def _load_y(subset):
        filename = f'{root_path}/{subset}/y_{subset}.txt'
        with open(filename, 'r') as f:
            y = np.array([line.strip() for line in f.readlines()], dtype=np.int32)
        return y - 1

    def _load_subject(subset):
        filename = f'{root_path}/{subset}/subject_{subset}.txt'
        with open(filename, 'r') as f:
            return np.array([line.strip() for line in f.readlines()], dtype=np.int32)

    print("Loading UCI HAR Train data...")
    X_train = _load_signals('train')
    y_train = _load_y('train')
    sub_train = _load_subject('train')

    print("Loading UCI HAR Test data...")
    X_test = _load_signals('test')
    y_test = _load_y('test')
    sub_test = _load_subject('test')

    X_data = np.concatenate([X_train, X_test])
    y_data = np.concatenate([y_train, y_test])
    user_ids = np.concatenate([sub_train, sub_test])

    print(f"UCI HAR Loaded: {X_data.shape}")
    return X_data, y_data, user_ids

def create_natural_user_split(X_data, y_data, user_ids):
    """
    RECOMMENDED FOR CA-AFP
    Creates a Natural Split where 1 Real User = 1 FL Client.
    This preserves biometric heterogeneity (gait/style), which is 
    critical for testing pruning/clustering algorithms.
    """
    # Get unique users (Expected: 30 users for UCI HAR)
    unique_users = np.unique(user_ids)
    print(f"\nCreating NATURAL split (Users as Clients). Found {len(unique_users)} users.")
    
    client_data = {}
    
    # Map original User ID (e.g., 1, 2, 5) to a Client ID (0, 1, 2...)
    for client_idx, original_user_id in enumerate(unique_users):
        # Create a mask for this specific user
        user_mask = (user_ids == original_user_id)
        
        # Extract this user's data
        X_user = X_data[user_mask]
        y_user = y_data[user_mask]
        
        client_data[client_idx] = {
            'X': X_user,
            'y': y_user,
            'original_id': original_user_id  # Metadata for tracking
        }
        
    return client_data

def create_non_iid_data_split(X_data, y_data, user_ids, num_clients=30, alpha=0.5, seed=42):
    """
    LEGACY / BASELINE ONLY.
    Creates a FORCED Non-IID data split (Dirichlet) mixing multiple users.
    Use this only if you need to compare against a synthetic baseline.
    """
    print(f"\nCreating SYNTHETIC NON-IID split (Alpha={alpha})...")
    np.random.seed(seed)
    
    cluster_map = {
        0: [0, 1],  # Walking, Walking_Upstairs
        1: [2, 3],  # Walking_Downstairs, Sitting
        2: [4, 5]   # Standing, Laying
    }
    clients_per_cluster = num_clients // 3
    client_data = defaultdict(lambda: {'X': [], 'y': [], 'user_ids': []})
    
    sample_counts = np.random.lognormal(mean=4.5, sigma=0.8, size=num_clients)
    sample_counts = np.floor(sample_counts).astype(int) + 50
    
    for cluster_id, activities in cluster_map.items():
        cluster_mask = np.isin(y_data, activities)
        cluster_X = X_data[cluster_mask]
        cluster_y = y_data[cluster_mask]
        
        start_client = cluster_id * clients_per_cluster
        end_client = (cluster_id + 1) * clients_per_cluster
        
        for client_idx in range(start_client, end_client):
            total_samples = min(sample_counts[client_idx], len(cluster_X))
            proportions = np.random.dirichlet(np.array([alpha] * 2))
            n_act1 = int(proportions[0] * total_samples)
            n_act2 = total_samples - n_act1
            
            idx_act1 = np.where(cluster_y == activities[0])[0]
            idx_act2 = np.where(cluster_y == activities[1])[0]
            
            if len(idx_act1) > 0 and len(idx_act2) > 0:
                n_act1 = min(n_act1, len(idx_act1))
                n_act2 = min(n_act2, len(idx_act2))
                chosen_1 = np.random.choice(idx_act1, n_act1, replace=(n_act1 > len(idx_act1)))
                chosen_2 = np.random.choice(idx_act2, n_act2, replace=(n_act2 > len(idx_act2)))
                
                all_indices = np.concatenate([chosen_1, chosen_2])
                all_indices.sort() # Preserve time order
                
                client_data[client_idx]['X'] = cluster_X[all_indices]
                client_data[client_idx]['y'] = cluster_y[all_indices]
                
    return client_data

def create_tf_datasets(client_data, batch_size=32, test_split=0.2):
    """
    Creates TF Datasets using STRATIFIED splitting.
    
    CRITICAL FIX: 
    This ensures that if a class exists in a client's data, 
    it appears in BOTH Train and Test sets (preserving time order).
    This prevents the "Train on Walk / Test on Sit" bug.
    """
    train_datasets = {}
    test_datasets = {}
    
    for client_id, data in client_data.items():
        X = data['X']
        y = data['y']
        
        if len(X) == 0: continue

        train_indices = []
        test_indices = []
        
        # 1. Identify all unique classes this client possesses
        unique_classes = np.unique(y)
        
        for cls in unique_classes:
            # Get indices corresponding to this specific class
            cls_indices = np.where(y == cls)[0]
            
            # Calculate split point for THIS class
            n_total_cls = len(cls_indices)
            
            # Safety: If < 2 samples, we can't split effectively. Put in train.
            if n_total_cls < 2:
                train_indices.extend(cls_indices)
                continue
                
            n_test_cls = int(n_total_cls * test_split)
            n_train_cls = n_total_cls - n_test_cls
            
            # Ensure at least 1 test sample if we have enough data
            if n_test_cls == 0 and n_total_cls > 5:
                n_test_cls = 1
                n_train_cls = n_total_cls - 1

            # Sequential split WITHIN the class
            # (Assumes indices are sorted by time from previous steps)
            train_indices.extend(cls_indices[:n_train_cls])
            test_indices.extend(cls_indices[n_train_cls:])
            
        # 2. Sort indices again to maintain relative time order in the final batches
        train_indices.sort()
        test_indices.sort()
        
        # 3. Create arrays
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # 4. Create TF Datasets
        # Shuffle TRAIN only
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Do NOT shuffle TEST (easier for sequential analysis if needed)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_datasets[client_id] = train_ds
        test_datasets[client_id] = test_ds
    
    return train_datasets, test_datasets