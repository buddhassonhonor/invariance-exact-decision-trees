import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from data_gen import generate_spurious_data
from ciedt import CIEDTClassifier

def run_scalability():
    print("Running Scalability Experiment...")
    os.makedirs('experiment/results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    results = []
    
    # Settings
    depths = [2, 3, 4, 5]
    sizes = [100, 500, 1000]
    seed = 42
    
    # Vary Depth (fixed size 500)
    for d in depths:
        print(f"Testing Depth {d}...")
        X1, y1, _ = generate_spurious_data(500, 0.9, seed=seed)
        X2, y2, _ = generate_spurious_data(500, 0.8, seed=seed+1)
        
        start = time.time()
        clf = CIEDTClassifier(max_depth=d, lambda_inv=1.0)
        clf.fit([X1, X2], [y1, y2])
        duration = time.time() - start
        
        results.append({'type': 'depth', 'value': d, 'time': duration})

    # Vary Size (fixed depth 3)
    for s in sizes:
        print(f"Testing Size {s}...")
        X1, y1, _ = generate_spurious_data(s, 0.9, seed=seed)
        X2, y2, _ = generate_spurious_data(s, 0.8, seed=seed+1)
        
        start = time.time()
        clf = CIEDTClassifier(max_depth=3, lambda_inv=1.0)
        clf.fit([X1, X2], [y1, y2])
        duration = time.time() - start
        
        results.append({'type': 'size', 'value': s, 'time': duration})

    df = pd.DataFrame(results)
    df.to_csv('experiment/results/scalability.csv', index=False)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df_depth = df[df['type'] == 'depth']
    plt.plot(df_depth['value'], df_depth['time'], marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('Time (s)')
    plt.title('Scalability vs Depth')
    
    plt.subplot(1, 2, 2)
    df_size = df[df['type'] == 'size']
    plt.plot(df_size['value'], df_size['time'], marker='o')
    plt.xlabel('Sample Size (per env)')
    plt.ylabel('Time (s)')
    plt.title('Scalability vs Size')
    
    plt.savefig('figures/scalability.png')
    print("Scalability finished.")

if __name__ == "__main__":
    run_scalability()
