import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_cmnist import generate_cmnist_tabular
from ciedt import CIEDTClassifier

def run_cmnist():
    print("Running CMNIST Experiment...")
    seeds = [42, 43, 44, 45, 46]
    results = []
    
    for seed in seeds:
        print(f"Seed {seed}...")
        datasets = generate_cmnist_tabular(n_samples_per_env=500, seeds=[seed, seed+1, seed+2])
        X1, y1 = datasets[0]
        X2, y2 = datasets[1]
        X_test, y_test = datasets[2]
        
        X_train_pooled = np.vstack([X1, X2])
        y_train_pooled = np.concatenate([y1, y2])
        
        # 1. CART
        start = time.time()
        cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
        cart.fit(X_train_pooled, y_train_pooled)
        cart_acc = accuracy_score(y_test, cart.predict(X_test))
        results.append({'seed': seed, 'method': 'CART', 'acc': cart_acc})
        
        # 2. Logistic Regression (Baseline)
        lr = LogisticRegression(random_state=seed, max_iter=1000)
        lr.fit(X_train_pooled, y_train_pooled)
        lr_acc = accuracy_score(y_test, lr.predict(X_test))
        results.append({'seed': seed, 'method': 'LR', 'acc': lr_acc})
        
        # 3. CIEDT
        print("  Fitting CIEDT...")
        start = time.time()
        ciedt = CIEDTClassifier(max_depth=3, lambda_inv=10.0)
        ciedt.fit([X1, X2], [y1, y2])
        print(f"  CIEDT Done in {time.time() - start:.2f}s")
        ciedt_acc = accuracy_score(y_test, ciedt.predict(X_test))
        results.append({'seed': seed, 'method': 'CIEDT', 'acc': ciedt_acc})
        
        print(f"  CART: {cart_acc:.3f}, LR: {lr_acc:.3f}, CIEDT: {ciedt_acc:.3f}")

    df = pd.DataFrame(results)
    print(df.groupby('method').mean())
    df.to_csv('experiment/results/cmnist_summary.csv', index=False)

if __name__ == "__main__":
    run_cmnist()
