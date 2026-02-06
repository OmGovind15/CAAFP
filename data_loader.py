import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. LOAD WISDM (Raw Text -> Windows) - NO SCALING HERE
# =====================================================

def load_wisdm_dataset(filepath='WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', 
                       window_size=200, step_size=100):
    """
    Loads raw WISDM text file, parses it, cleans it, and segments it into windows.
    Returns RAW data (unscaled) to prevent leakage.
    """
    print(f"[INFO] Loading WISDM dataset from {filepath}...")
    
    if not os.path.exists(filepath):
        # Fallback for common path variations if immediate file not found
        if os.path.exists('WISDM_ar_v1.1_raw.txt'):
            filepath = 'WISDM_ar_v1.1_raw.txt'
        else:
            raise FileNotFoundError(f"WISDM file not found at {filepath}")

    # 1. Parse the Raw Text File
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                # Clean line (remove trailing semi-colons, whitespace)
                line = line.strip().rstrip(';').rstrip(',')
                if not line: continue
                parts = line.split(',')
                if len(parts) != 6: continue
                
                # Validation
                if any(p == '' for p in parts): continue

                user = int(parts[0])
                activity = parts[1]
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5].replace(';', '')) # Handle potential lingering semicolon
                
                data.append([user, activity, x, y, z])
            except ValueError:
                continue

    df = pd.DataFrame(data, columns=['user', 'activity', 'x', 'y', 'z'])
    
    # 2. Encode Labels
    activity_map = {
        'Walking': 0, 'Jogging': 1, 'Upstairs': 2, 
        'Downstairs': 3, 'Sitting': 4, 'Standing': 5
    }
    df['label'] = df['activity'].map(activity_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # 3. Segment into Windows
    X_list, y_list, u_list = [], [], []
    unique_users = df['user'].unique()
    
    for uid in unique_users:
        user_df = df[df['user'] == uid]
        
        # Window per activity to ensure label consistency
        for act in user_df['label'].unique():
            act_df = user_df[user_df['label'] == act]
            values = act_df[['x', 'y', 'z']].values
            count = len(values)
            
            if count < window_size: continue
            
            for i in range(0, count - window_size + 1, step_size):
                window = values[i : i + window_size]
                X_list.append(window)
                y_list.append(act)
                u_list.append(uid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    users = np.array(u_list, dtype=np.int32)
    
    print(f"[INFO] WISDM Loaded (Raw). Shape: {X.shape}, Users: {len(np.unique(users))}")
    
    # NOTE: No normalization here. Normalization happens per-client in create_tf_datasets.
    return X, y, users

# =====================================================
# 2. CREATE CLIENT DATA (Natural User Split)
# =====================================================

def create_natural_user_split(X, y, users):
    """
    Splits the dataset based on the natural user IDs.
    """
    unique_users = np.unique(users)
    client_data = {}
    
    for new_id, original_uid in enumerate(unique_users):
        mask = (users == original_uid)
        client_data[new_id] = {
            'X': X[mask],
            'y': y[mask]
        }
        
    print(f"[INFO] Created Natural Split with {len(client_data)} clients.")
    return client_data

# =====================================================
# 3. CREATE TF DATASETS (Stratified + Safe Scaling)
# =====================================================

def create_tf_datasets(client_data, batch_size=32, test_split=0.2):
    """
    1. Splits data (Stratified + Sequential).
    2. Fits Scaler on TRAIN only.
    3. Transforms TRAIN and TEST.
    4. Creates TF Datasets.
    """
    train_datasets = {}
    test_datasets = {}
    
    for client_id, data in client_data.items():
        X = data['X']
        y = data['y']
        
        if len(X) < 10: continue 

        train_indices = []
        test_indices = []
        
        # --- Stratified Split (Sequential Time-Order) ---
        unique_classes = np.unique(y)
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            n_total = len(cls_indices)
            
            if n_total < 2:
                train_indices.extend(cls_indices)
                continue
            
            n_test = int(n_total * test_split)
            if n_test == 0 and n_total > 5: n_test = 1
            n_train = n_total - n_test
            
            # First 80% -> Train, Last 20% -> Test (Preserves local time dependency)
            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])
            
        train_indices.sort()
        test_indices.sort()
        
        if len(train_indices) == 0: continue

        X_train_raw = X[train_indices]
        y_train = y[train_indices]
        
        X_test_raw = X[test_indices]
        y_test = y[test_indices]

        # --- LEAK-PROOF NORMALIZATION ---
        # 1. Setup Scaler
        scaler = StandardScaler()
        
        # 2. Reshape for scaler (N, T, F) -> (N*T, F)
        N_train, T, F = X_train_raw.shape
        X_train_flat = X_train_raw.reshape(-1, F)
        
        # 3. Fit ONLY on Train
        scaler.fit(X_train_flat)
        
        # 4. Transform Train
        X_train_scaled = scaler.transform(X_train_flat).reshape(N_train, T, F)
        
        # 5. Transform Test (using Train statistics)
        if len(X_test_raw) > 0:
            N_test = X_test_raw.shape[0]
            X_test_flat = X_test_raw.reshape(-1, F)
            X_test_scaled = scaler.transform(X_test_flat).reshape(N_test, T, F)
        else:
            X_test_scaled = X_test_raw

        # --- Create TF Datasets ---
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))
        train_ds = train_ds.shuffle(len(X_train_scaled)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test))
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_datasets[client_id] = train_ds
        test_datasets[client_id] = test_ds
    
    return train_datasets, test_datasets