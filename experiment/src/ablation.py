import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from data_gen import generate_spurious_data
from ciedt import CIEDTClassifier
from experiment_utils import accuracy_score

def run_ablation():
    print("Running Ablation Experiment...")
    os.makedirs('experiment/results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    lambdas = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
    seeds = [42, 43, 44]
    results = []
    
    CORE_ACC = 0.85
    
    for lam in lambdas:
        print(f"Testing Lambda {lam}...")
        train_accs = []
        test_accs = []
        
        for seed in seeds:
            X1, y1, _ = generate_spurious_data(500, 0.95, core_accuracy=CORE_ACC, seed=seed)
            X2, y2, _ = generate_spurious_data(500, 0.80, core_accuracy=CORE_ACC, seed=seed+100)
            X_test, y_test, _ = generate_spurious_data(500, 0.1, core_accuracy=CORE_ACC, seed=seed+200)
            
            clf = CIEDTClassifier(max_depth=3, lambda_inv=lam)
            clf.fit([X1, X2], [y1, y2])
            
            # Predict
            X_train = np.vstack([X1, X2])
            y_train = np.concatenate([y1, y2])
            
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
        results.append({
            'lambda': lam,
            'train_acc': np.mean(train_accs),
            'test_acc': np.mean(test_accs),
            'test_std': np.std(test_accs)
        })

    df = pd.DataFrame(results)
    df.to_csv('experiment/results/ablation.csv', index=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df['lambda'], df['train_acc'], marker='o', label='Train Acc')
    plt.plot(df['lambda'], df['test_acc'], marker='s', label='Test Acc (OOD)')
    plt.xlabel('Lambda (Invariance Penalty)')
    plt.ylabel('Accuracy')
    plt.title('Ablation: Invariance Strength')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/ablation.png')
    print("Ablation finished.")

if __name__ == "__main__":
    run_ablation()
