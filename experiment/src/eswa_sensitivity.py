"""Predetermined synthetic sensitivity studies, separate from main selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from data_gen import generate_spurious_data
from eswa_models import ExactInvariantTree
from eswa_suite import OUT
def envs(seed,n,m):return [generate_spurious_data(n,c,seed=seed+101*(e+1))[:2] for e,c in enumerate([.95,.8,.7,.6,.5][:m])]
def targets(seed):return [generate_spurious_data(1000,c,seed=seed+50001+101*e)[:2] for e,c in enumerate([.2,.1])]
def scores(model,test):
    a=[accuracy_score(y,model.predict(X)) for X,y in test];return dict(mean_accuracy=np.mean(a),worst_accuracy=np.min(a))
def empty(tree):
    if tree['type']=='split':return empty(tree['left'])+empty(tree['right'])
    n=np.array(tree['counts']).sum(axis=1);return [float(np.mean(n==0))]
rows=[]
for seed in [42,43,44,45,46]:
    test=targets(seed)
    for m in [2,3,4,5]:
        train=envs(seed,500,m)
        for method in ['ODT','CIEDT']:
            best=None
            for d in [2,3]:
                for lam in ([0.] if method=='ODT' else [1.,5.,10.,20.,40.]):
                    val=[]
                    for v in range(m):
                        model=ExactInvariantTree(d,lam).fit([tr for e,tr in enumerate(train) if e!=v]);val.append(accuracy_score(train[v][1],model.predict(train[v][0])))
                    key=(min(val),np.mean(val),-d,-lam)
                    if best is None or key>best[0]:best=(key,d,lam)
            model=ExactInvariantTree(best[1],best[2]).fit(train)
            rows.append(dict(study='environment_count',seed=seed,m=m,n_per_env=500,depth=best[1],lam=best[2],alpha=1.,method=method,**scores(model,test)))
    for n in [25,50,100,500]:
        for alpha in [0.,.5,1.,2.]:
            model=ExactInvariantTree(3,10.,alpha).fit(envs(seed,n,3))
            rows.append(dict(study='smoothing',seed=seed,m=3,n_per_env=n,depth=3,lam=10.,alpha=alpha,method='CIEDT',empty_pair_fraction=np.mean(empty(model.tree)),**scores(model,test)))
    train=envs(seed,500,3)
    for d in [2,3]:
        for lam in [0.,1.,5.,10.,20.,40.]:
            model=ExactInvariantTree(d,lam).fit(train)
            rows.append(dict(study='depth_lambda',seed=seed,m=3,n_per_env=500,depth=d,lam=lam,alpha=1.,method='CIEDT',**scores(model,test)))
pd.DataFrame(rows).to_csv(OUT/'sensitivity_runs.csv',index=False)
print('Saved',len(rows),'synthetic sensitivity fits; target grid diagnostic only.')
