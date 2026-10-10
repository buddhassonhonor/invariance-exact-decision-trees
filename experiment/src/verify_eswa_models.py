"""Independent small-tree enumeration and historical solver equivalence."""
from pathlib import Path
import json
import numpy as np
from ciedt import CIEDTClassifier
from eswa_models import ExactInvariantTree, era_split_score


def enumerate_costs(envs,depth,lam):
    # Enumerate every pair of feasible child costs, rather than Bellman minima.
    y=np.concatenate([y for X,y in envs]);counts=np.bincount(y,minlength=2)
    probs=np.array([(yy.sum()+1)/(len(yy)+2) for X,yy in envs])
    costs=[float(counts.min()+lam*len(y)*probs.var())]
    if depth:
        for j in range(envs[0][0].shape[1]):
            left=[(X[X[:,j]==0],yy[X[:,j]==0]) for X,yy in envs]
            right=[(X[X[:,j]==1],yy[X[:,j]==1]) for X,yy in envs]
            if not sum(len(yy) for X,yy in left) or not sum(len(yy) for X,yy in right):continue
            a,b=enumerate_costs(left,depth-1,lam),enumerate_costs(right,depth-1,lam)
            costs.extend(x+z for x in a for z in b)
    return costs


if __name__=='__main__':
    tests=[]
    for seed in range(5):
        rng=np.random.default_rng(seed)
        envs=[(rng.integers(0,2,(n,3)),rng.integers(0,2,n)) for n in [11,13,7]]
        for depth in [1,2,3]:
            for lam in [0.,1.,10.,40.]:
                old=CIEDTClassifier(depth,lam).fit([X for X,y in envs],[y for X,y in envs])
                new=ExactInvariantTree(depth,lam).fit(envs)
                assert np.isclose(old.tree['cost'],new.tree['cost'],atol=1e-10)
                assert np.array_equal(old.predict(np.vstack([X for X,y in envs])),new.predict(np.vstack([X for X,y in envs])))
                if depth<=2:
                    all_costs=enumerate_costs(envs,depth,lam)
                    assert np.isclose(new.tree['cost'],min(all_costs),atol=1e-10)
                tests.append(dict(seed=seed,depth=depth,lam=lam,cost=new.tree['cost']))
    # Literal transcription of the saved author ERAMSE proxy, with stable exp.
    envs=[(np.array([[0],[0],[1],[1]]),np.array([0,0,1,1])),(np.array([[0],[1],[1]]),np.array([0,0,1]))]
    proxy=[]
    for X,y in envs:
        left,right=y[X[:,0]==0],y[X[:,0]==1]
        proxy.append(left.sum()**2/(len(left)+1)+right.sum()**2/(len(right)+1)-y.sum()**2/(len(y)+1))
    proxy=np.array(proxy)
    for alpha in [0.,-1.,-10.]:
        w=np.exp(.5*alpha*proxy-(.5*alpha*proxy).max())
        assert np.isclose(era_split_score(envs,0,alpha)[0],.5*np.dot(proxy,w)/w.sum())
    out=Path(__file__).resolve().parents[2]/'exp/eswa/model_verification.json'
    out.write_text(json.dumps(dict(historical_equivalence_cases=len(tests),exhaustive_cases=40,era_author_proxy_equivalence=True,tests=tests),indent=2))
    print('Verified 60 historical cost/prediction cases, 40 exhaustive cases, Era proxy identity.')
