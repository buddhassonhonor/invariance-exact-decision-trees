"""Nested source-only experiments. Run with --workers 2; resumable job files."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import hashlib
import itertools
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from eswa_models import ExactInvariantTree, EraTree, sklearn_tree, robust_tree, tree_stats

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'exp/eswa'
HEART_NAMES=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
HEART_CATS={2:[1,2,3,4],6:[0,1,2],10:[1,2,3],12:[3,6,7]}
METHODS=['CART','TunedCART','ODT','Era','DirectionalEra','RobustOCT','CIEDT','GreedyCIEDT','LR']


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=native),encoding='utf-8')
    tmp.replace(path)


def pool(envs):
    return np.vstack([x for x,y in envs]),np.concatenate([y for x,y in envs])


class BinaryTransform:
    def __init__(self,kind):self.kind=kind
    def fit(self,rows):
        X=np.vstack([r['X'] for r in rows])
        self.fit_ids=[i for r in rows for i in r['ids']]
        self.fit_sites=[r['name'] for r in rows]
        self.specs=[]
        if self.kind=='heart':
            for j,name in enumerate(HEART_NAMES):
                col=X[:,j];obs=col[np.isfinite(col)]
                if len(obs)<.5*len(col):continue
                if j in HEART_CATS:
                    values,counts=np.unique(obs,return_counts=True)
                    fill=float(values[np.argmax(counts)])
                    self.specs += [(j,'category',float(cat),fill,f'{name}={cat}') for cat in HEART_CATS[j]]
                elif j in [1,5,8]:
                    values,counts=np.unique(obs,return_counts=True)
                    fill=float(values[np.argmax(counts)])
                    self.specs.append((j,'binary',.5,fill,name))
                else:
                    med=float(np.median(obs))
                    self.specs.append((j,'threshold',med,med,f'{name}>{med:g}'))
        elif self.kind=='digits':
            binary=(X>8).astype(int)
            chosen=np.sort(np.argsort(-binary.var(axis=0),kind='stable')[:12])
            self.specs=[(int(j),'threshold',8.,0.,f'pixel({j//8},{j%8})>8') for j in chosen]
        else:
            med=np.median(X,axis=0)
            binary=(X>med).astype(int)
            chosen=np.sort(np.argsort(-binary.var(axis=0),kind='stable')[:12])
            self.specs=[(int(j),'threshold',float(med[j]),float(med[j]),f'{rows[0]["feature_names"][j]}>{med[j]:.4g}') for j in chosen]
        self.names=[s[-1] for s in self.specs]
        if self.kind!='heart':self.names.append('injected shortcut')
        return self
    def transform(self,row):
        columns=[]
        for j,mode,threshold,fill,name in self.specs:
            v=np.where(np.isfinite(row['X'][:,j]),row['X'][:,j],fill)
            columns.append((v==threshold) if mode=='category' else (v>threshold))
        X=np.column_stack(columns).astype(int)
        if self.kind!='heart':
            rng=np.random.default_rng(row['noise_seed'])
            rate=.1 if self.kind=='digits' else .03
            X ^= (rng.random(X.shape)<rate).astype(int)
            shortcut=row['y'].copy()
            flips=rng.random(len(shortcut))>=row['corr']
            shortcut[flips]=1-shortcut[flips]
            X=np.column_stack([X,shortcut])
        return X,row['y']
    def state(self):return dict(fit_ids=self.fit_ids,fit_sites=self.fit_sites,specs=self.specs,names=self.names)


def rows_for(dataset,seed=42):
    if dataset=='heart':
        rows=[]
        for site,file in [('Cleveland','cleveland'),('Hungarian','hungarian'),('Switzerland','switzerland'),('VA','va')]:
            d=pd.read_csv(Path(os.environ.get('CIEDT_HEART_DATA',str(ROOT/'data/heart_disease/raw')))/f'processed.{file}.data',header=None,na_values='?')
            assert d.shape[1]==14 and d.iloc[:,-1].isin([0,1,2,3,4]).all()
            rows.append(dict(name=site,X=d.iloc[:,:13].to_numpy(float),y=(d.iloc[:,-1].to_numpy()>0).astype(int),ids=[f'{site}:{i}' for i in range(len(d))]))
        return rows
    if dataset=='synthetic':
        from revision_suite import make_synthetic_envs
        train,test,meta=make_synthetic_envs(seed)
        return train,test,['core x1','core x2','injected shortcut','noise 1','noise 2']
    ds=load_digits() if dataset=='digits' else load_breast_cancer() if dataset=='breast' else load_wine()
    X,y=ds.data,ds.target
    if dataset=='digits':
        ix=np.flatnonzero(np.isin(y,[0,1]));X,y=X[ix],y[ix]
    else:
        ix=np.arange(len(y));y=(y==0).astype(int) if dataset=='wine' else y.astype(int)
    remaining=np.arange(len(y));chunks=[]
    # Three disjoint sources containing 70%, and two disjoint targets 30%.
    for k,fraction in enumerate([7/30,7/23,7/16,.5]):
        remaining,part=train_test_split(remaining,test_size=fraction,stratify=y[remaining],random_state=seed+17*k)
        chunks.append(part)
    chunks.append(remaining)
    rows=[]
    for e,part in enumerate(chunks):
        rows.append(dict(name=f'env{e}',X=X[part],y=y[part],ids=[f'{dataset}:{i}' for i in ix[part]],
                         feature_names=list(ds.feature_names) if dataset!='digits' else [],
                         corr=[.95,.8,.7,.1,.3][e],noise_seed=seed+3571*(e+1)))
    assert len(set(i for r in rows for i in r['ids']))==len(y)
    return rows


def prepare(dataset,seed,target=None,legacy=False):
    if dataset=='synthetic' or legacy:
        if dataset=='synthetic':train,test,names=rows_for(dataset,seed)
        else:
            from data_cmnist import generate_cmnist_tabular
            from data_tabular_shift import generate_shifted_tabular_envs
            if dataset=='digits':
                envs,meta=generate_cmnist_tabular(seed=seed,top_k_pixels=12);train,test=envs[:3],envs[3:]
            else:
                train,test,meta=generate_shifted_tabular_envs('breast_cancer' if dataset=='breast' else 'wine_binary',seed=seed,n_samples_per_env=350,top_k_features=12)
            names=[f'binary feature {j}' for j in range(train[0][0].shape[1]-1)]+['injected shortcut']
        folds=[([train[e] for e in range(3) if e!=v],train[v]) for v in range(3)]
        manifests=[dict(validation_site=f'env{v}',fit_sites=[f'env{e}' for e in range(3) if e!=v],preprocessing='fixed synthetic / historical full-pool preprocessing') for v in range(3)]
        return train,test,names,folds,manifests,{}
    rows=rows_for(dataset,seed)
    if dataset=='heart':sources=[r for r in rows if r['name']!=target];targets=[r for r in rows if r['name']==target]
    else:sources,targets=rows[:3],rows[3:]
    target_ids=set(i for r in targets for i in r['ids'])
    folds,manifests=[],[]
    for v,val in enumerate(sources):
        tr=[r for e,r in enumerate(sources) if e!=v]
        transform=BinaryTransform(dataset).fit(tr)
        assert not target_ids.intersection(transform.fit_ids)
        assert not set(val['ids']).intersection(transform.fit_ids)
        folds.append(([transform.transform(r) for r in tr],transform.transform(val)))
        manifests.append(dict(validation_site=val['name'],target_site=target,validation_ids=val['ids'],target_ids=sorted(target_ids),**transform.state()))
    transform=BinaryTransform(dataset).fit(sources)
    assert not target_ids.intersection(transform.fit_ids)
    train=[transform.transform(r) for r in sources];test=[transform.transform(r) for r in targets]
    final=dict(target_ids=sorted(target_ids),source_sizes={r['name']:len(r['y']) for r in sources},**transform.state())
    return train,test,transform.names,folds,manifests,final


def configs(method):
    if method=='CART':return [dict(depth=3)]
    if method=='TunedCART':return [dict(depth=d,min_leaf=l,ccp=c,class_weight=w) for d,l,c,w in itertools.product([2,3],[1,5,10],[0.,.005],[None,'balanced'])]
    if method=='ODT':return [dict(depth=d,lam=0.) for d in [2,3]]
    if method in ['CIEDT','GreedyCIEDT']:return [dict(depth=d,lam=l) for d,l in itertools.product([2,3],[1.,5.,10.,20.,40.])]
    if method=='Era':return [dict(depth=d,boltzmann=a) for d,a in itertools.product([2,3],[0.,-1.,-10.])]
    if method=='DirectionalEra':return [dict(depth=d,boltzmann=0.) for d in [2,3]]
    if method=='RobustOCT':return [dict(depth=d,budget_fraction=b) for d,b in itertools.product([2,3],[.01,.05])]
    if method=='LR':return [dict(C=c) for c in [.1,1.,10.]]
    raise ValueError(method)


def fit_model(method,config,envs,log_path):
    X,y=pool(envs);start=time.perf_counter();info={}
    with log_path.open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        if method in ['CIEDT','ODT','GreedyCIEDT']:
            model=ExactInvariantTree(config['depth'],config['lam'],greedy=method=='GreedyCIEDT').fit(envs)
            tree=model.tree;info.update(objective=tree['cost'],solver_status='exact_enumeration' if method!='GreedyCIEDT' else 'greedy',states=model.calls,cache=model.cache_info)
        elif method in ['Era','DirectionalEra']:
            model=EraTree(config['depth'],config['boltzmann'],method=='DirectionalEra').fit(envs);tree=model.tree
            info['solver_status']='greedy_paper_criterion'
        elif method in ['CART','TunedCART']:
            model=DecisionTreeClassifier(max_depth=config['depth'],min_samples_leaf=config.get('min_leaf',1),ccp_alpha=config.get('ccp',0.),class_weight=config.get('class_weight'),random_state=0).fit(X,y)
            tree=sklearn_tree(model);info['solver_status']='greedy'
        elif method=='LR':
            model=LogisticRegression(C=config['C'],max_iter=1000,random_state=0).fit(X,y);tree=None
            info['solver_status']='lbfgs'
        else:
            from odtlearn.robust_oct import RobustOCT
            model=RobustOCT('Gurobi',depth=config['depth'],time_limit=20,num_threads=2,verbose=False)
            gm=model._solver.model
            gm.setParam('OutputFlag',0);gm.setParam('MIPGap',0.);gm.setParam('Seed',0)
            gm.setParam('LogFile',str(log_path.with_suffix('.gurobi.log')))
            budget=float(np.floor(config['budget_fraction']*len(y)))
            model.fit(X,y,costs=np.ones_like(X,dtype=float),budget=budget)
            if gm.SolCount==0:raise RuntimeError('No feasible RobustOCT incumbent')
            tree=robust_tree(model)
            info.update(solver_status='optimal' if gm.Status==2 else 'time_limited_incumbent',gurobi_status=gm.Status,gap=gm.MIPGap,objective=gm.ObjVal,bound=gm.ObjBound,budget=budget,variables=gm.NumVars,solver_runtime_s=gm.Runtime)
    info['t_fit_s']=time.perf_counter()-start
    info['rss_mb']=psutil.Process().memory_info().rss/2**20
    return model,tree,info


def run_job(job):
    dataset,seed,target,legacy=job
    key=f'{"legacy" if legacy else "primary"}_{dataset}_{target or seed}'
    destination=OUT/'jobs'/f'{key}.json'
    if destination.exists():return key,'cached'
    t0=time.perf_counter()
    train,test,names,folds,manifests,final=prepare(dataset,seed,target,legacy)
    prep=time.perf_counter()-t0
    logs=OUT/'solver_logs'/key;logs.mkdir(parents=True,exist_ok=True)
    score_fn=balanced_accuracy_score if dataset=='heart' else accuracy_score
    rows,selection,models=[],[],{}
    methods=['Era','DirectionalEra','RobustOCT','GreedyCIEDT'] if legacy else METHODS
    for method in methods:
        ts=time.perf_counter();best=None
        for ci,cfg in enumerate(configs(method)):
            fold_scores=[];states=[]
            # Fixed CART is not selected from validation scores.
            if method=='CART':fold_scores=[0.]*len(folds)
            else:
                for fi,(tr,val) in enumerate(folds):
                    model,tree,info=fit_model(method,cfg,tr,logs/f'{method}_c{ci}_f{fi}.log')
                    fold_scores.append(float(score_fn(val[1],model.predict(val[0]))));states.append(info)
            ranking=(min(fold_scores),np.mean(fold_scores),-cfg.get('depth',0),-ci)
            selection.append(dict(method=method,config=cfg,scores=fold_scores,fit_info=states))
            if best is None or ranking>best[0]:best=ranking,cfg
        select_time=time.perf_counter()-ts
        model,tree,info=fit_model(method,best[1],train,logs/f'{method}_final.log')
        # Save fitted model/config/preprocessing before evaluating target labels.
        before=dict(config=best[1],tree=tree,feature_names=names,fit_info=info,preprocessing=final)
        dump(OUT/'model_structures'/f'{key}_{method}.json',before)
        tp=time.perf_counter();predictions=[model.predict(X) for X,y in test]
        predict_time=time.perf_counter()-tp
        scores=[float(accuracy_score(y,p)) for (X,y),p in zip(test,predictions)]
        balanced=[float(balanced_accuracy_score(y,p)) for (X,y),p in zip(test,predictions)]
        stats=tree_stats(tree,train,None if dataset=='heart' else (2 if dataset=='synthetic' else len(names)-1)) if tree is not None else {}
        row=dict(protocol='controlled_resampled_legacy' if legacy else 'heart_nested_v1' if dataset=='heart' else 'controlled_disjoint_v1',dataset=dataset,seed=None if dataset=='heart' else seed,target_site=target,method=method,config=best[1],
                 n_train=sum(len(y) for X,y in train),n_test=[len(y) for X,y in test],p=len(names),avg_ood_acc=float(np.mean(scores)),worst_ood_acc=float(np.min(scores)),avg_balanced_acc=float(np.mean(balanced)),worst_balanced_acc=float(np.min(balanced)),site_scores=scores,balanced_scores=balanced,
                 t_preprocess_s=prep,t_select_s=select_time,t_predict_s=predict_time,t_total_s=prep+select_time+info['t_fit_s']+predict_time,feature_names=names,**stats,**info)
        rows.append(row);models[method]=tree
        dump(OUT/'partial'/f'{key}_{method}.json',row)
        dump(OUT/'predictions'/f'{key}_{method}.json',[dict(env=i,y=y.tolist(),prediction=p.tolist()) for i,((X,y),p) in enumerate(zip(test,predictions))])
        print(key,method,f'acc={row["avg_ood_acc"]:.4f}',flush=True)
    dump(destination,dict(key=key,rows=rows,selection=selection,folds=manifests,final_transform=final,wall_time_s=time.perf_counter()-t0))
    return key,'done'


def aggregate():
    jobs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'jobs').glob('*.json'))]
    rows=[r for j in jobs for r in j['rows']]
    frame=pd.DataFrame(rows)
    for col in frame:
        if any(isinstance(v,(list,dict)) for v in frame[col]):frame[col]=frame[col].apply(lambda v:json.dumps(v,default=native) if isinstance(v,(dict,list)) else v)
    frame.to_csv(OUT/'runs.csv',index=False)
    metrics=['avg_ood_acc','worst_ood_acc','avg_balanced_acc','t_fit_s','t_select_s','t_total_s','actual_depth','leaves','spurious_fraction','train_invariance']
    frame.groupby(['protocol','dataset','method'])[metrics].agg(['mean','std','count']).to_csv(OUT/'summary.csv')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--panel',choices=['primary','legacy','heart','all'],default='primary');ap.add_argument('--datasets',nargs='*');ap.add_argument('--seeds',nargs='*',type=int,default=[42,43,44,45,46]);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,created='2026-10-10',seeds=[42,43,44,45,46],depths=[2,3],heart_selection='min balanced accuracy, mean balanced accuracy, smaller depth, fixed config order',controlled_selection='min accuracy, mean accuracy, smaller depth, fixed config order',robust_time_limit_s=20,robust_threads=2,robust_gap=0.,robust_budget_fractions=[.01,.05],robust_feature_cost=1.,heart_min_observed_fraction=.5,heart_categorical_codes=HEART_CATS,methods=METHODS)
    protocol_file=OUT/'protocol.json'
    if protocol_file.exists():assert json.loads(protocol_file.read_text())==json.loads(json.dumps(protocol))
    else:dump(protocol_file,protocol)
    dump(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),ram_gb=psutil.virtual_memory().total/2**30,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,gpu_used=False,solver='Gurobi 13.0.1',odtlearn='1.1.0',workers=args.workers))
    jobs=[];datasets=args.datasets or ['synthetic','digits','breast','wine']
    if args.panel in ['primary','all']:jobs += [(d,s,None,False) for d in datasets for s in args.seeds]
    if args.panel in ['legacy','all']:jobs += [(d,s,None,True) for d in datasets for s in args.seeds]
    if args.panel in ['heart','all']:jobs += [('heart',42,t,False) for t in ['Cleveland','Hungarian','Switzerland','VA']]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(run_job,j):j for j in jobs}
        for future in as_completed(futures):
            print('JOB',future.result(),flush=True)
    aggregate()
