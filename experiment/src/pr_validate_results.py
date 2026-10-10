"""Validate stored predictions, selection, preprocessing, models."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from eswa_suite import OUT,ROOT,prepare,configs,METHODS
from eswa_models import predict_tree,tree_stats
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    report={};jobs=[read(p) for p in sorted((OUT/'jobs').glob('primary_*.json'))]
    assert len(jobs)==24
    checked=0;treechecked=0;selectionchecked=0;foldchecked=0
    for job in jobs:
        assert set(r['method'] for r in job['rows'])==set(METHODS)
        first=job['rows'][0];train,test,names,folds,manifests,final=prepare(first['dataset'],first['seed'] or 42,first['target_site'])
        for fold in job['folds']:
            if 'fit_ids' in fold:
                assert not set(fold['fit_ids'])&set(fold['validation_ids'])
                assert not set(fold['fit_ids'])&set(fold['target_ids'])
                foldchecked+=1
        if final:assert set(final['fit_ids']).isdisjoint(final['target_ids'])
        for row in job['rows']:
            method=row['method'];assert row['config'] in configs(method)
            saved=read(OUT/'predictions'/f'{job["key"]}_{method}.json')
            acc=[];ba=[]
            for pr,(X,y) in zip(saved,test):
                assert np.array_equal(pr['y'],y)
                acc.append(accuracy_score(y,pr['prediction']));ba.append(balanced_accuracy_score(y,pr['prediction']))
            assert np.allclose(acc,row['site_scores']) and np.allclose(ba,row['balanced_scores'])
            assert np.isclose(np.mean(acc),row['avg_ood_acc']) and np.isclose(np.min(acc),row['worst_ood_acc'])
            checked+=1
            if method!='CART':
                sel=[r for r in job['selection'] if r['method']==method]
                assert len(sel)==len(configs(method))
                best=max(enumerate(sel),key=lambda pair:(min(pair[1]['scores']),np.mean(pair[1]['scores']),-pair[1]['config'].get('depth',0),-pair[0]))[1]
                assert best['config']==row['config'];selectionchecked+=1
            if method!='LR':
                suffix='_warm' if method=='RobustOCT' else ''
                model=read(OUT/'model_structures'/f'{job["key"]}_{method}{suffix}.json')
                assert model['feature_names']==names
                for (X,y),pr in zip(test,saved):assert np.array_equal(predict_tree(model['tree'],X),pr['prediction'])
                stats=tree_stats(model['tree'],train,None if first['dataset']=='heart' else 2 if first['dataset']=='synthetic' else len(names)-1)
                assert stats['actual_depth']==row['actual_depth'] and stats['leaves']==row['leaves']
                assert np.isclose(stats['train_invariance'],row['train_invariance'])
                if method in ['CIEDT','ODT','GreedyCIEDT']:
                    def total(n):
                        if n['type']=='split':return total(n['left'])+total(n['right'])
                        counts=np.array(n['counts']);err=counts.sum(axis=0).min();p=(counts[:,1]+1)/(counts.sum(axis=1)+2)
                        return err+row['config']['lam']*counts.sum()*np.var(p)
                    assert np.isclose(total(model['tree']),row['objective'])
                treechecked+=1
            if method=='RobustOCT':assert row.get('warm_start') and row['gap']>=0
    report.update(primary_jobs=24,prediction_sets_checked=checked,trees_checked=treechecked,source_selections_checked=selectionchecked,inner_ID_manifests_checked=foldchecked)
    report['verification']=read(OUT/'model_verification.json') | {'tests':'see model_verification.json'}
    report['counts']={n:len(pd.read_csv(OUT/n)) for n in ['mechanism_runs.csv','scalability_runs.csv','cache_ablation_runs.csv','shift_runs.csv','sensitivity_runs.csv']}
    report['passed']=True
    (OUT/'pr_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
