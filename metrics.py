#
# FILENAME: metrics.py
#
import numpy as np
import tensorflow as tf
import time

class FLMetricsTracker:
    def __init__(self, model_template, train_datasets, epochs_per_round=3):
        """
        Initializes the tracker with model architecture and real data stats.
        
        Args:
            model_template: A Keras model instance to analyze params/flops.
            train_datasets: Dictionary of {client_id: tf.data.Dataset}.
            epochs_per_round: Number of local epochs clients run.
        """
        self.start_time = 0
        self.epochs_per_round = epochs_per_round
        
        # Accumulators
        self.total_training_time = 0.0
        self.total_comm_mb = 0.0
        self.total_flops = 0.0
        
        # 1. Count Parameters (Float32 = 4 bytes)
        self.total_params = self._count_params(model_template)
        
        # 2. Estimate Base FLOPs per inference (Forward pass)
        self.base_flops_per_sample = self._estimate_cnn_flops(model_template)
        
        # 3. Calculate Average Samples per Client (Real Data)
        self.avg_samples = self._calculate_avg_samples(train_datasets)
        
        print(f"\n--- [Metrics Tracker Initialized] ---")
        print(f"  Total Params:       {self.total_params:,}")
        print(f"  Model Size (Dense): {self.total_params * 4 / (1024**2):.2f} MB")
        print(f"  Base FLOPs/sample:  {self.base_flops_per_sample:,}")
        print(f"  Avg Samples/Client: {self.avg_samples:.1f}")
        print(f"-------------------------------------\n")

    def _count_params(self, model):
        return np.sum([np.prod(v.shape) for v in model.trainable_variables])

    def _calculate_avg_samples(self, train_datasets):
        """
        Iterates through datasets to count real samples.
        This is done ONCE at initialization to avoid runtime overhead.
        """
        print("  Calculating dataset statistics (this takes a moment)...")
        counts = []
        # Check all clients to be accurate
        for cid, dataset in train_datasets.items():
            # tf.data.Dataset cardinality can be -1 or -2, so we iterate
            c = 0
            for _ in dataset: c += 1
            # Multiply by batch size (32) since dataset yields batches
            counts.append(c * 32) 
        return np.mean(counts)

    def _estimate_cnn_flops(self, model):
        """
        Estimates theoretical FLOPs for one forward pass of the CNN.
        Formula: 2 * Cin * Cout * K * L_out (Approx)
        """
        flops = 0.0
        # Standard input length for WISDM is usually 200 time steps
        # We track the shape as it flows through the network
        current_seq_len = 200 
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv1D):
                weights = layer.get_weights()
                if not weights: continue
                kernel = weights[0] # Shape: (K, Cin, Cout)
                k_size, c_in, c_out = kernel.shape
                
                # Update seq len based on padding (assuming 'same' or 'valid')
                # If stride=1 and padding='same', length stays.
                # If padding='valid', length -= (k-1)
                if layer.padding == 'valid':
                    current_seq_len -= (k_size - 1)
                if layer.strides[0] > 1:
                    current_seq_len //= layer.strides[0]
                
                # FLOPs = 2 * (MACs) * Output_Size
                layer_flops = 2 * (k_size * c_in) * c_out * current_seq_len
                flops += layer_flops
                
            elif isinstance(layer, (tf.keras.layers.MaxPooling1D, tf.keras.layers.AveragePooling1D)):
                if layer.pool_size:
                    current_seq_len //= layer.pool_size[0]
            
            elif isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if not weights: continue
                w = weights[0] # (Input, Output)
                flops += 2 * w.shape[0] * w.shape[1] # Matrix Mul
        
        # Add slight overhead for activations/bias
        return flops * 1.05 

    def start_round(self):
        self.start_time = time.time()

    def end_round(self, num_clients, sparsity=0.0):
        """
        Updates accumulated metrics for one round.
        Args:
            num_clients: How many clients participated this round.
            sparsity: 0.0 to 1.0 (fraction of zeros). 0.0 = Dense.
        """
        duration = time.time() - self.start_time
        self.total_training_time += duration
        
        density = 1.0 - sparsity
        
        # 1. Communication Cost (MB)
        # Size = Params * 4 bytes * 2 (Upload + Download) * Clients * Density
        # Note: We assume compressed communication for sparse models (sending only non-zeros)
        # Ideally, sparse format adds overhead (indices), so efficiency isn't perfect 1:1.
        # But density * params is the standard "theoretical" lower bound for papers.
        round_mb = (self.total_params * 4 / (1024*1024)) * density * 2 * num_clients
        self.total_comm_mb += round_mb
        
        # 2. Computation Cost (FLOPs)
        # FLOPs = Base_FLOPs * Density * Samples * Epochs * Clients * 3 (Forward+Backward+Update)
        # Backward pass is roughly 2x forward pass.
        round_flops = self.base_flops_per_sample * density * self.avg_samples * self.epochs_per_round * num_clients * 3
        self.total_flops += round_flops

    def get_results(self):
        return {
            'total_comm_mb': self.total_comm_mb,
            'total_training_time': self.total_training_time,
            'total_flops': self.total_flops,
            'total_gflops': self.total_flops / 1e9,
            'avg_samples_per_client': self.avg_samples
        }