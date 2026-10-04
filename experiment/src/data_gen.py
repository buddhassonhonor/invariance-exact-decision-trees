import numpy as np
import pandas as pd

def generate_spurious_data(n_samples, env_spurious_corr, core_accuracy=0.85, seed=42):
    np.random.seed(seed)
    # Core features: X1, X2.
    X1 = np.random.randint(0, 2, n_samples)
    X2 = np.random.randint(0, 2, n_samples)
    
    # Ground truth Y_clean
    Y_clean = X1 ^ X2
    
    # Y follows Y_clean with probability core_accuracy
    # This simulates label noise or inherent uncertainty in core mechanism
    Y = np.array(Y_clean)
    flip_mask = np.random.random(n_samples) > core_accuracy
    Y[flip_mask] = 1 - Y[flip_mask]
    
    # Spurious feature: X3. Correlated with Y (the observed Y) based on env_spurious_corr
    X3 = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if np.random.random() < env_spurious_corr:
            X3[i] = Y[i]
        else:
            X3[i] = 1 - Y[i]
            
    # Add some noise features
    X_noise = np.random.randint(0, 2, (n_samples, 2))
    
    # Stack features
    X = np.column_stack([X1, X2, X3, X_noise])
    feature_names = ['X1', 'X2', 'X_spurious', 'N1', 'N2']
    
    return X, Y, feature_names
