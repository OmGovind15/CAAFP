#
# FILENAME: models_cnn.py
#
import tensorflow as tf
import numpy as np 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape=(200, 3), num_classes=6):
    """
    Creates a standard 1D-CNN model for HAR.
    Architecture: 2x Conv1D Blocks -> Flatten -> Dense -> Output
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # 1st Conv Block
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='conv_1'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # 2nd Conv Block
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='conv_2'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Flatten
        Flatten(),
        
        # Dense Block
        Dense(32, activation='relu', name='dense_1'),
        Dropout(0.2),
        
        # Output
        Dense(num_classes, activation='softmax', name='output')
    ], name='CNN_HAR_Model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_sparsity(model):
    """
    Calculate sparsity (percentage of zeros).
    Note: Checks kernels only (weights), ignores biases, as per standard pruning literature.
    """
    total_params = 0
    zero_params = 0
    
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy()
            total_params += weights.size
            zero_params += (weights == 0).sum()
            
    if total_params == 0: return 0.0
    return zero_params / total_params