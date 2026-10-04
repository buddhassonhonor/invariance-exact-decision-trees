import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from data_gen import generate_spurious_data
from ciedt import CIEDTClassifier

def run_experiment():
    # Settings
    seeds = [42, 43, 44, 45, 46]
    results = []
    
    # Create results directory
    os.makedirs('experiment/results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    print("Starting Experiments...")
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        
        # Data Generation
        # Core accuracy 0.85 across all environments
        CORE_ACC = 0.85
        
        # Env 1: Train, Spurious Corr 0.95
        X1, y1, feats = generate_spurious_data(500, 0.95, core_accuracy=CORE_ACC, seed=seed)
        # Env 2: Train, Spurious Corr 0.80
        X2, y2, _ = generate_spurious_data(500, 0.80, core_accuracy=CORE_ACC, seed=seed+100)
        # Env Test: Test, Spurious Corr 0.1 (Anti-correlated)
        X_test, y_test, _ = generate_spurious_data(500, 0.1, core_accuracy=CORE_ACC, seed=seed+200)
        
        # 1. CART (Pooled)
        X_train_pooled = np.vstack([X1, X2])
        y_train_pooled = np.concatenate([y1, y2])
        
        start_time = time.time()
        cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
        cart.fit(X_train_pooled, y_train_pooled)
        cart_time = time.time() - start_time
        
        cart_train_acc = accuracy_score(y_train_pooled, cart.predict(X_train_pooled))
        cart_test_acc = accuracy_score(y_test, cart.predict(X_test))
        
        # 2. CIEDT (Exact Invariant)
        # lambda_inv needs to be strong enough to overcome the 2.5% accuracy gap (0.875 vs 0.85)
        # Let's try lambda = 20.0 to force invariance.
        
        start_time = time.time()
        ciedt = CIEDTClassifier(max_depth=3, lambda_inv=20.0)
        ciedt.fit([X1, X2], [y1, y2])
        ciedt_time = time.time() - start_time
        
        if seed == 42:
            print("CIEDT Tree Structure (Seed 42):")
            ciedt.print_tree()
        
        ciedt_train_pred = ciedt.predict(X_train_pooled)
        ciedt_test_pred = ciedt.predict(X_test)
        
        ciedt_train_acc = accuracy_score(y_train_pooled, ciedt_train_pred)
        ciedt_test_acc = accuracy_score(y_test, ciedt_test_pred)
        
        # 3. ODT (Exact non-invariant)
        start_time = time.time()
        odt = CIEDTClassifier(max_depth=3, lambda_inv=0.0)
        odt.fit([X1, X2], [y1, y2])
        odt_time = time.time() - start_time
        
        odt_train_acc = accuracy_score(y_train_pooled, odt.predict(X_train_pooled))
        odt_test_acc = accuracy_score(y_test, odt.predict(X_test))

        results.append({
            'seed': seed,
            'method': 'CART',
            'train_acc': cart_train_acc,
            'test_acc': cart_test_acc,
            'time': cart_time
        })
        results.append({
            'seed': seed,
            'method': 'CIEDT',
            'train_acc': ciedt_train_acc,
            'test_acc': ciedt_test_acc,
            'time': ciedt_time
        })
        results.append({
            'seed': seed,
            'method': 'ODT',
            'train_acc': odt_train_acc,
            'test_acc': odt_test_acc,
            'time': odt_time
        })

    df = pd.DataFrame(results)
    df.to_csv('experiment/results/summary.csv', index=False)
    print("Experiment finished. Results saved.")
    
    summary = df.groupby('method').mean()
    print(summary)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    # Boxplot
    data_to_plot = [df[df['method'] == m]['test_acc'] for m in ['CART', 'ODT', 'CIEDT']]
    plt.boxplot(data_to_plot, labels=['CART', 'ODT', 'CIEDT'])
    plt.ylabel('Test Accuracy (OOD)')
    plt.title('OOD Generalization Comparison')
    plt.savefig('figures/ood_comparison.png')
    print("Figure saved to figures/ood_comparison.png")

if __name__ == "__main__":
    run_experiment()
