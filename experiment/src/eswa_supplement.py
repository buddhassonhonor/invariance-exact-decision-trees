"""Saved-data supplements: scaling, cache ablation and same-objective lookahead."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from eswa_models import ExactInvariantTree,tree_stats
from eswa_suite import OUT,dump,prepare,configs,native


def synthetic(seed,n,p,m=3):
    from data_gen import generate_spurious_data
    envs=[]
    for e,c in enumerate([.95,.8,.7][:m]):
        X,y,_=generate_spurious_data(n,c,seed=seed+101*(e+1))
        if p>5:X=np.column_stack([X,np.random.default_rng(seed+e+191).integers(0,2,(n,p-5))])
        envs.append((X,y))
    return envs


def scaling():
    rows=[]
    points=[('depth',d,d,8,250) for d in [1,2,3,4,5]]
    points += [('features',p,3,p,250) for p in [5,8,12,16,20]]
    points += [('samples',n,3,8,n) for n in [100,250,500,1000]]
    for axis,value,d,p,n in points:
        for seed in [42,43,44,45,46]:
            envs=synthetic(seed,n,p)
            start=time.perf_counter();model=ExactInvariantTree(d,10.).fit(envs);duration=time.perf_counter()-start
            rows.append(dict(axis=axis,value=value,depth=d,p=p,n_per_env=n,m=3,seed=seed,runtime_s=duration,states=model.calls,cache_hits=model.cache_info['hits'],objective=model.tree['cost']))
    pd.DataFrame(rows).to_csv(OUT/'scalability_runs.csv',index=False)
    ab=[]
    for d in [2,3,4]:
        for seed in [42,43,44,45,46]:
            envs=synthetic(seed,250,8);objectives=[]
            for memo in [False,True]:
                start=time.perf_counter();model=ExactInvariantTree(d,10.,memo=memo).fit(envs);duration=time.perf_counter()-start
                objectives.append(model.tree['cost'])
                ab.append(dict(depth=d,seed=seed,memo=memo,runtime_s=duration,states=model.calls,objective=model.tree['cost']))
            assert np.isclose(*objectives,atol=1e-10)
    pd.DataFrame(ab).to_csv(OUT/'cache_ablation_runs.csv',index=False)
    print('Saved 70 scaling measurements and 30 cache comparisons.',flush=True)


def mechanisms():
    rows=[]
    for jobfile in sorted((OUT/'jobs').glob('primary_*.json')):
        job=json.loads(jobfile.read_text(encoding='utf-8'));r=job['rows'][0]
        dataset,seed,target=r['dataset'],r['seed'] or 42,r['target_site']
        train,test,names,folds,manifests,final=prepare(dataset,seed,target)
        scorefn=balanced_accuracy_score if dataset=='heart' else accuracy_score
        # Fixed depth 3, lambda selected once using exact source validation.
        candidates=[c for c in configs('CIEDT') if c['depth']==3];best=None
        for ci,c in enumerate(candidates):
            scores=[]
            for tr,val in folds:
                model=ExactInvariantTree(3,c['lam']).fit(tr)
                scores.append(float(scorefn(val[1],model.predict(val[0]))))
            ranking=(min(scores),np.mean(scores),-ci)
            if best is None or ranking>best[0]:best=ranking,c['lam']
        for method,lam,greedy,alpha,weighted in [('Exact',best[1],False,1.,True),('GreedySameObjective',best[1],True,1.,True),('NoPenalty',0.,False,1.,True),('NoSmoothing',best[1],False,0.,True),('Alpha2',best[1],False,2.,True),('Unweighted',best[1],False,1.,False)]:
            start=time.perf_counter();model=ExactInvariantTree(3,lam,alpha,greedy,support_weight=weighted).fit(train);duration=time.perf_counter()-start
            pred=[model.predict(X) for X,y in test]
            acc=[float(accuracy_score(y,p)) for (X,y),p in zip(test,pred)]
            ba=[float(balanced_accuracy_score(y,p)) for (X,y),p in zip(test,pred)]
            row=dict(dataset=dataset,seed=r['seed'],target_site=target,method=method,depth=3,lam=lam,alpha=alpha,support_weight=weighted,objective=model.tree['cost'],avg_accuracy=float(np.mean(acc)),worst_accuracy=float(np.min(acc)),avg_balanced_accuracy=float(np.mean(ba)),runtime_s=duration,
                     **tree_stats(model.tree,train,None if dataset=='heart' else 2 if dataset=='synthetic' else len(names)-1))
            rows.append(row)
            dump(OUT/'model_structures'/f'{job["key"]}_mechanism_{method}.json',dict(tree=model.tree,feature_names=names,config=row))
        print('Mechanism',job['key'],flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'mechanism_runs.csv',index=False)
    for (ds,seed,target),group in frame.groupby(['dataset','seed','target_site'],dropna=False):
        exact=group[group.method=='Exact'].objective.iloc[0];greedy=group[group.method=='GreedySameObjective'].objective.iloc[0]
        assert exact<=greedy+1e-9


def legacy_trees():
    from ciedt import CIEDTClassifier
    from sklearn.tree import DecisionTreeClassifier
    from eswa_models import sklearn_tree
    old=pd.read_csv(Path(__file__).resolve().parents[2]/'experiment/results/revision_runs.csv')
    for ds,oldname in [('synthetic','synthetic_xor'),('digits','cmnist_tabular'),('breast','breast_cancer'),('wine','wine_binary')]:
        train,test,names,folds,manifests,final=prepare(ds,42,legacy=True)
        if ds=='synthetic':names=['core x1','core x2','shortcut','noise 1','noise 2']
        for method in ['CART','ODT','CIEDT']:
            row=old[(old.dataset==oldname)&(old.seed==42)&(old.method==method)].iloc[0]
            if method=='CART':
                m=DecisionTreeClassifier(max_depth=3,random_state=42).fit(np.vstack([X for X,y in train]),np.concatenate([y for X,y in train]));tree=sklearn_tree(m)
            else:
                m=CIEDTClassifier(int(row.selected_depth),float(row.selected_lambda)).fit([X for X,y in train],[y for X,y in train]);tree=m.tree
            score=np.mean([accuracy_score(y,m.predict(X)) for X,y in test])
            assert np.isclose(score,row.avg_ood_acc)
            dump(OUT/'model_structures'/f'legacy_{ds}_42_{method}.json',dict(tree=tree,feature_names=names,accuracy=score))
    print('Reproduced fixed-seed historical trees without altering historical results.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['scaling','mechanisms','legacy','all'],default='all');args=ap.parse_args()
    if args.part in ['scaling','all']:scaling()
    if args.part in ['mechanisms','all']:mechanisms()
    if args.part in ['legacy','all']:legacy_trees()
