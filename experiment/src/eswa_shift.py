"""Evaluate frozen fitted models at new synthetic shortcut interventions."""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from data_gen import generate_spurious_data
from eswa_models import predict_tree
from eswa_suite import OUT
rows=[]
for seed in [42,43,44,45,46]:
    for method in ['CART','ODT','Era','RobustOCT','CIEDT']:
        suffix='_warm' if method=='RobustOCT' else ''
        model=json.loads((OUT/'model_structures'/f'primary_synthetic_{seed}_{method}{suffix}.json').read_text(encoding='utf-8'))
        for i,corr in enumerate([.1,.2,.3,.5,.7,.8,.95]):
            X,y,_=generate_spurious_data(2000,corr,seed=seed+100001+101*i)
            rows.append(dict(seed=seed,method=method,target_match_probability=corr,n_test=2000,accuracy=accuracy_score(y,predict_tree(model['tree'],X))))
pd.DataFrame(rows).to_csv(OUT/'shift_runs.csv',index=False)
print('Saved 175 fixed-model intervention measurements.')
