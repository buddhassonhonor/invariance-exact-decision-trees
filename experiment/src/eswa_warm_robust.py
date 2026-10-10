"""Official RobustOCT with a source-only nominal ODT feasible MIP start.

    The uncertainty set, objective, callback and optimization model are unchanged.
"""
import contextlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from eswa_models import ExactInvariantTree, robust_tree, tree_stats
from eswa_suite import OUT, dump, native, pool, prepare, configs


def warm_fit(config,envs,log_path):
    from odtlearn.robust_oct import RobustOCT
    X,y=pool(envs);t0=time.perf_counter()
    nominal=ExactInvariantTree(config['depth'],0.).fit(envs)
    model=RobustOCT('Gurobi',depth=config['depth'],time_limit=20,num_threads=2,verbose=False)
    gm=model._solver.model
    gm.setParam('OutputFlag',0);gm.setParam('MIPGap',0.);gm.setParam('Seed',0)
    budget=float(np.floor(config['budget_fraction']*len(y)))
    original=model._create_main_problem
    def create_with_start():
        original()
        for key,var in model._b.items():var.Start=0
        for (n,k),var in model._w.items():var.Start=int(k==0)
        paths=[]
        def fill(tree,n,path):
            if tree['type']=='leaf':
                for k in model._labels:model._w[n,k].Start=int(k==tree['prediction'])
                paths.append((tree['prediction'],path));return
            for k in model._labels:model._w[n,k].Start=0
            feature=model._X_col_labels[tree['feature']]
            model._b[n,feature,0].Start=1
            fill(tree['left'],2*n,path+[(tree['feature'],0)])
            fill(tree['right'],2*n+1,path+[(tree['feature'],1)])
        fill(nominal.tree,1,[])
        correct=np.flatnonzero(nominal.predict(X)==y)
        attacks=[]
        for i in correct:
            distances=[sum(X[i,j]!=value for j,value in path) for label,path in paths if label!=y[i]]
            attacks.append((min(distances) if distances else budget+1,int(i)))
        spent=0.;attacked=set()
        for cost,i in sorted(attacks):
            if spent+cost>budget:break
            spent+=cost;attacked.add(i)
        correct=set(map(int,correct))
        for i,var in model._t.items():var.Start=int(i in correct and i not in attacked)
    model._create_main_problem=create_with_start
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        model.fit(X,y,costs=np.ones_like(X,dtype=float),budget=budget)
    if gm.SolCount==0:raise RuntimeError('No RobustOCT feasible incumbent')
    tree=robust_tree(model)
    info=dict(warm_start='source-only nominal ODT',t_fit_s=time.perf_counter()-t0,
              solver_status='optimal' if gm.Status==2 else 'time_limited_incumbent',
              gurobi_status=gm.Status,gap=float(gm.MIPGap),objective=float(gm.ObjVal),
              bound=float(gm.ObjBound),budget=budget,variables=gm.NumVars,solver_runtime_s=gm.Runtime)
    return model,tree,info


def strengthen(jobfile):
    job=json.loads(jobfile.read_text(encoding='utf-8'))
    existing=next(r for r in job['rows'] if r['method']=='RobustOCT')
    if existing.get('warm_start'):return job['key'],'cached'
    dataset=existing['dataset'];seed=existing['seed'] or 42;target=existing['target_site']
    train,test,names,folds,manifests,final=prepare(dataset,seed,target)
    score_fn=balanced_accuracy_score if dataset=='heart' else accuracy_score
    logdir=OUT/'solver_logs'/f'{job["key"]}_warm';logdir.mkdir(parents=True,exist_ok=True)
    ts=time.perf_counter();best=None;selection=[]
    for ci,cfg in enumerate(configs('RobustOCT')):
        scores=[];info_rows=[]
        for fi,(tr,val) in enumerate(folds):
            cache=logdir/f'c{ci}_f{fi}.json'
            if cache.exists():result=json.loads(cache.read_text())
            else:
                model,tree,info=warm_fit(cfg,tr,logdir/f'c{ci}_f{fi}.log')
                result=dict(score=float(score_fn(val[1],model.predict(val[0]))),fit_info=info)
                dump(cache,result)
            scores.append(result['score']);info_rows.append(result['fit_info'])
        key=(min(scores),np.mean(scores),-cfg['depth'],-ci)
        selection.append(dict(method='RobustOCT',config=cfg,scores=scores,fit_info=info_rows,warm_start=True))
        if best is None or key>best[0]:best=(key,cfg)
    tselect=time.perf_counter()-ts
    model,tree,info=warm_fit(best[1],train,logdir/'final.log')
    dump(OUT/'model_structures'/f'{job["key"]}_RobustOCT_warm.json',dict(config=best[1],tree=tree,feature_names=names,fit_info=info,preprocessing=final))
    tp=time.perf_counter();pred=[model.predict(X) for X,y in test];t_predict=time.perf_counter()-tp
    accuracy=[float(accuracy_score(y,p)) for (X,y),p in zip(test,pred)]
    balanced=[float(balanced_accuracy_score(y,p)) for (X,y),p in zip(test,pred)]
    row=dict(existing,config=best[1],avg_ood_acc=float(np.mean(accuracy)),worst_ood_acc=float(np.min(accuracy)),avg_balanced_acc=float(np.mean(balanced)),worst_balanced_acc=float(np.min(balanced)),site_scores=accuracy,balanced_scores=balanced,t_select_s=tselect,t_predict_s=t_predict,
             **tree_stats(tree,train,None if dataset=='heart' else 2 if dataset=='synthetic' else len(names)-1),**info)
    row['t_total_s']=row['t_preprocess_s']+tselect+info['t_fit_s']+t_predict
    cold=OUT/'cold_start_jobs'/jobfile.name
    if not cold.exists():dump(cold,job)
    job['rows']=[row if r['method']=='RobustOCT' else r for r in job['rows']]
    job['selection']=[r for r in job['selection'] if r['method']!='RobustOCT']+selection
    dump(jobfile,job)
    dump(OUT/'predictions'/f'{job["key"]}_RobustOCT.json',[dict(env=i,y=y.tolist(),prediction=p.tolist()) for i,((X,y),p) in enumerate(zip(test,pred))])
    print(job['key'],'warm robust',row['avg_ood_acc'],'gap',row['gap'],flush=True)
    return job['key'],'done'


if __name__=='__main__':
    import argparse
    from concurrent.futures import ProcessPoolExecutor,as_completed
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--key');args=ap.parse_args()
    jobs=list((OUT/'jobs').glob((args.key or 'primary_*')+'.json'))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(strengthen,p) for p in jobs]
        for f in as_completed(futures):print(f.result(),flush=True)
    from eswa_suite import aggregate
    aggregate()
