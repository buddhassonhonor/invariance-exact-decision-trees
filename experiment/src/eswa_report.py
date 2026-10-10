"""Generate manuscript numbers, tables and plots solely from saved results."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel,rankdata
import itertools
from eswa_suite import OUT,ROOT,METHODS,aggregate
T=OUT/'tables';T.mkdir(exist_ok=True)
NAMES={'synthetic':'Synthetic XOR','digits':'Digits 0/1','breast':'Breast Cancer','wine':'Wine binary'}
SHORT={'DirectionalEra':'Dir. Era','GreedyCIEDT':'Greedy CI','TunedCART':'Tuned CART','RobustOCT':'RobustOCT*'}
def name(m):return SHORT.get(m,m)
def esc(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('>',r'$>$')
def save(n,s): (T/(n+'.tex')).write_text(s+'\n',encoding='utf-8')
def table(caption,label,headers,rows,spec=None):
    spec=spec or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+spec+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule',*[(' & '.join(map(str,row))+r'\\') for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def pm(a):return f'{np.mean(a):.3f} $\\pm$ {np.std(a,ddof=1):.3f}'
def exact_signed_rank(a):
    a=np.round(a,12);a=a[a!=0]
    if not len(a):return 1.
    ranks=rankdata(np.abs(a));observed=ranks[a>0].sum()
    values=np.array([np.dot(sign,ranks) for sign in itertools.product([0,1],repeat=len(a))])
    return float(min(1.,2*min(np.mean(values<=observed+1e-12),np.mean(values>=observed-1e-12))))
def savefig(fig,n):
    fig.savefig(ROOT/'figures'/f'eswa_{n}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'eswa_{n}.png',dpi=100,bbox_inches='tight');plt.close(fig)
def paths(tree,names,path=()):
    if tree['type']=='leaf':return [(path,tree['prediction'],tree.get('samples','--'))]
    f=names[tree['feature']]
    return paths(tree['left'],names,path+('('+f+') = 0',))+paths(tree['right'],names,path+('('+f+') = 1',))
def main():
    jobs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'jobs').glob('primary_*.json'))]
    assert len(jobs)==24,f'Only {len(jobs)}/24 jobs complete'
    assert all(next(r for r in j['rows'] if r['method']=='RobustOCT').get('warm_start') for j in jobs),'Warm baseline incomplete'
    aggregate();df=pd.DataFrame([r for j in jobs for r in j['rows']]);controlled=df[df.dataset!='heart'];heart=df[df.dataset=='heart']
    assert len(df)==216 and controlled.groupby(['dataset','method']).size().eq(5).all()
    macro=controlled.groupby('method')[['avg_ood_acc','worst_ood_acc']].mean()
    syn=controlled[controlled.dataset=='synthetic'].groupby('method').avg_ood_acc.mean()
    hm=heart.groupby('method').avg_balanced_acc.mean()
    breast=controlled[controlled.dataset=='breast'].groupby('method').avg_ood_acc.mean()
    headline=(f'CIEDT raises mean target accuracy from {syn["ODT"]:.3f} to {syn["CIEDT"]:.3f} on Synthetic XOR and from {breast["ODT"]:.3f} to {breast["CIEDT"]:.3f} on Breast Cancer relative to pooled-error exact trees, while the remaining controlled tasks and natural sites show task-dependent comparative performance.')
    save('headlines',r'\newcommand{\AbstractResults}{'+headline+'}\n'+r'\newcommand{\ContributionResults}{'+f'CIEDT attains {syn["CIEDT"]:.3f} mean target accuracy on Synthetic XOR versus {syn["ODT"]:.3f} for the pooled-error optimal decision tree (ODT), and {breast["CIEDT"]:.3f} versus {breast["ODT"]:.3f} on Breast Cancer, while natural-site macro balanced accuracy is {hm["CIEDT"]:.3f} versus {hm["ODT"]:.3f}, delimiting the regularizer\'s empirical scope.'+'}\n'+r'\newcommand{\ConclusionResults}{'+f'The controlled-task macro mean target accuracy is {macro.loc["CIEDT","avg_ood_acc"]:.3f} for CIEDT versus {macro.loc["ODT","avg_ood_acc"]:.3f} for ODT, with gains concentrated in XOR and Breast Cancer; the natural-site macro balanced accuracies are {hm["CIEDT"]:.3f} and {hm["ODT"]:.3f}, respectively.'+'}')
    dsrows=[]
    for ds in NAMES:
        r=controlled[(controlled.dataset==ds)&(controlled.method=='CIEDT')].iloc[0]
        dsrows.append([NAMES[ds],r.n_train,'/'.join(map(str,r.n_test)),r.p,'3 / 2'])
    dsrows.append(['Heart Disease','617--797','123--303','fold-specific','3 / 1'])
    save('datasets',table('Primary dataset sizes. Controlled counts refer to seed 42; real-task rounding is deterministic. Heart source size depends on the held-out site.','tab:datasets',['Task','Source $N$','Target sizes','$p$','Sources / targets'],dsrows))
    blocks=[]
    for metric,cap,label in [('avg_ood_acc','Mean target-environment accuracy','tab:controlled-mean'),('worst_ood_acc','Worst target-environment accuracy','tab:controlled-worst')]:
        rows=[[name(m)]+[pm(controlled[(controlled.dataset==ds)&(controlled.method==m)][metric]) for ds in NAMES] for m in METHODS]
        blocks.append(table(cap+' (mean $\\pm$ sample SD over five seeds). RobustOCT* denotes the official solver with source-only warm starts and a fixed time limit.',''+label,['Method','XOR','Digits','Breast','Wine'],rows))
    save('controlled','\n'.join(blocks))
    comparisons=[]
    for ds in NAMES:
        g=controlled[controlled.dataset==ds].groupby('method').avg_ood_acc.mean();best=g.idxmax()
        comparisons.append(f'{NAMES[ds]}: CIEDT {g["CIEDT"]:.3f}, ODT {g["ODT"]:.3f}, and the best observed method {name(best).replace("*", "")} {g[best]:.3f}')
    save('controlled_text',r'Tables~\ref{tab:controlled-mean} and~\ref{tab:controlled-worst} distinguish average transfer from the weaker target environment, while Figure~\ref{fig:controlled} displays seed dispersion. '+f'The four-task macro mean accuracy is {macro.loc["CIEDT","avg_ood_acc"]:.3f} for CIEDT and {macro.loc["ODT","avg_ood_acc"]:.3f} for ODT; the corresponding worst-environment macro scores are {macro.loc["CIEDT","worst_ood_acc"]:.3f} and {macro.loc["ODT","worst_ood_acc"]:.3f}. '+ '; '.join(comparisons)+'. The gains are concentrated in XOR and Breast Cancer, with a small negative mean difference on Digits and equal mean performance on Wine.')
    robust=df[df.method=='RobustOCT'];timed=robust[robust.solver_status!='optimal']
    save('baseline_text',f'Across the controlled tasks, macro mean accuracy is {macro.loc["Era","avg_ood_acc"]:.3f} for Era, {macro.loc["DirectionalEra","avg_ood_acc"]:.3f} for directional Era, {macro.loc["RobustOCT","avg_ood_acc"]:.3f} for RobustOCT, and {macro.loc["CIEDT","avg_ood_acc"]:.3f} for CIEDT. Of 24 final RobustOCT fits, {len(timed)} terminate with a nonzero or unfinished optimality gap; final gaps range from {robust.gap.min():.3f} to {robust.gap.max():.3f}, with median {robust.gap.median():.3f}. These solver states qualify the comparison; they do not establish superiority over a fully solved robust objective.')
    time_rows=[]
    for m in METHODS:
        g=df[df.method==m];time_rows.append([name(m),f'{g.t_fit_s.median():.3f}',f'{g.t_select_s.median():.2f}',f'{g.t_total_s.median():.2f}'])
    save('runtime',table('Median final fit, complete selection, and total wall times across the 24 primary tasks (seconds). Total includes preprocessing and prediction.','tab:runtime',['Method','Final fit','Selection','Total'],time_rows))
    structure=[]
    for m in METHODS[:-1]:
        g=controlled[controlled.method==m]
        structure.append([name(m),f'{g.actual_depth.mean():.2f}',f'{g.leaves.mean():.2f}',f'{(g.train_invariance/g.n_train).mean():.4f}',f'{100*g.spurious_root.mean():.1f}',f'{100*g.spurious_fraction.mean():.1f}'])
    save('structure',table('Controlled-task structural means over 20 fits per tree method. Variance is normalized by source support; shortcut-root and shortcut-split percentages inspect the complete tree.','tab:structure',['Method','Depth','Leaves','$I/N$','Root (\\%)','Splits (\\%)'],structure))
    sites=['Cleveland','Hungarian','Switzerland','VA'];hr=[]
    for m in METHODS:
        g=heart[heart.method==m].set_index('target_site')
        hr.append([name(m)]+[f'{g.loc[s,"avg_ood_acc"]:.3f}/{g.loc[s,"avg_balanced_acc"]:.3f}' for s in sites]+[f'{g.avg_balanced_acc.mean():.3f}'])
    save('heart',table('Held-out Heart sites: each cell is accuracy / balanced accuracy. The final column averages balanced accuracy equally across sites. There is one fixed outer fit per method/site.','tab:heart',['Method','Cleveland','Hungarian','Swiss','VA','Macro BA'],hr))
    hg=heart[heart.method=='CIEDT'].set_index('target_site');best=hm.idxmax()
    save('heart_text',f'CIEDT has four-site macro balanced accuracy {hm["CIEDT"]:.3f}, compared with {hm["ODT"]:.3f} for ODT, {hm["Era"]:.3f} for Era, and {hm["RobustOCT"]:.3f} for RobustOCT. The highest observed macro balanced accuracy is {hm[best]:.3f} ({name(best).replace("*", "")}). CIEDT site balanced accuracies range from {hg.avg_balanced_acc.min():.3f} to {hg.avg_balanced_acc.max():.3f}. These results are descriptive across four fixed sites and do not support a claim of consistent improvement at every hospital.')
    mechanism=pd.read_csv(OUT/'mechanism_runs.csv');assert len(mechanism)==144
    mr=[]
    for m in mechanism.method.unique():
        mr.append([esc(m)]+[f'{mechanism[(mechanism.method==m)&(mechanism.dataset==ds)].avg_accuracy.mean():.3f}' for ds in NAMES]+[f'{mechanism[(mechanism.method==m)&(mechanism.dataset=="heart")].avg_balanced_accuracy.mean():.3f}'])
    save('mechanisms',table('Coupled depth-three mechanism controls. Controlled columns average target accuracy over five seeds; Heart averages balanced accuracy over four sites.','tab:mechanisms',['Variant','XOR','Digits','Breast','Wine','Heart BA'],mr))
    pv=mechanism.pivot(index=['dataset','seed','target_site'],columns='method',values='objective');delta=pv.GreedySameObjective-pv.Exact
    assert (delta>=-1e-9).all()
    av=mechanism.pivot(index=['dataset','seed','target_site'],columns='method',values='avg_accuracy')
    save('mechanism_text',f'Exact search has an objective no greater than the coupled greedy tree in all 24 tasks, with a strict improvement larger than $10^{{-9}}$ in {(delta>1e-9).sum()}. Target accuracy improves in {((av.Exact-av.GreedySameObjective)>1e-12).sum()} of the 24 pairs, illustrating that the objective improvement and target improvement are distinct. Smoothing and support-weight changes have task-dependent effects, as shown in Table~\\ref{{tab:mechanisms}}; their scores are not compared as instances of an identical objective.')
    stats=[];rng=np.random.default_rng(20261010)
    for ds in NAMES:
        g=controlled[controlled.dataset==ds].pivot(index='seed',columns='method',values='avg_ood_acc');a=(g.CIEDT-g.ODT).to_numpy();boot=np.mean(rng.choice(a,(10000,5),replace=True),axis=1);lo,hi=np.quantile(boot,[.025,.975]);wp=exact_signed_rank(a);tp=float(ttest_rel(g.CIEDT,g.ODT).pvalue) if np.std(a)>0 else 1.
        stats.append(dict(dataset=ds,delta=float(a.mean()),bootstrap_low=float(lo),bootstrap_high=float(hi),paired_t_p=tp,wilcoxon_p=wp,seeds=5))
    ordered=sorted(range(4),key=lambda i:stats[i]['wilcoxon_p']);prev=0
    for rank,i in enumerate(ordered):prev=max(prev,min(1.,(4-rank)*stats[i]['wilcoxon_p']));stats[i]['holm_p']=prev
    (OUT/'paired_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    save('statistics',table('Exploratory paired CIEDT--ODT differences (five seeds). Bootstrap intervals resample pairs 10,000 times. Signed-rank values enumerate all signs of nonzero pairs; Holm correction covers four tasks.','tab:statistics',['Task','Mean delta','Bootstrap 95\\% CI','Signed-rank $p$','Holm $p$'],[[NAMES[s['dataset']],f'{s["delta"]:+.3f}',f'[{s["bootstrap_low"]:+.3f}, {s["bootstrap_high"]:+.3f}]',f'{s["wilcoxon_p"]:.3f}',f'{s["holm_p"]:.3f}'] for s in stats]))
    scaling=pd.read_csv(OUT/'scalability_runs.csv');cache=pd.read_csv(OUT/'cache_ablation_runs.csv')
    cs=cache.pivot(index=['depth','seed'],columns='memo',values='runtime_s');speed=cs[False]/cs[True]
    sdepth=scaling[scaling.axis=='depth'].groupby('value').runtime_s.median();sp=scaling[scaling.axis=='features'].groupby('value').runtime_s.median()
    save('scaling_text',f'Median fit time rises from {sdepth.loc[1]:.4f} seconds at depth one to {sdepth.loc[5]:.3f} seconds at depth five, and from {sp.loc[5]:.4f} seconds at five features to {sp.loc[20]:.3f} seconds at 20 features. The 15 matched cache comparisons (depths two through four, five seeds each) retain equal objective values and show median speedup {speed.median():.2f} times. Aggregation reduces repeated sample work but does not eliminate the depth/feature growth. Table~\\ref{{tab:statistics}} reports paired uncertainty rather than inferring significance from positive means alone.\n'+r'\input{exp/eswa/tables/runtime}')
    cfgrows=[['CART',r'$d=3$'],['Tuned CART',r'$d\in\{2,3\}$, leaf size $\{1,5,10\}$, pruning $\{0,.005\}$, weights none/balanced'],['ODT',r'$d\in\{2,3\}$, $\lambda=0$'],['CIEDT / greedy',r'$d\in\{2,3\}$, $\lambda\in\{1,5,10,20,40\}$, $\alpha=1$'],['Era',r'$d\in\{2,3\}$, Boltzmann $a\in\{0,-1,-10\}$'],['Directional Era',r'$d\in\{2,3\}$; mean-gain tie-break'],['RobustOCT',r'$d\in\{2,3\}$, $b\in\{.01,.05\}$, 20 s / fit, 2 threads'],['LR',r'$C\in\{.1,1,10\}$, 1000 iteration cap']]
    save('configurations',table('Frozen configuration grids. Source-only worst-score, mean-score, depth, and fixed-order tie-breaking apply as specified in the protocol.','tab:configurations',['Method','Grid'],cfgrows,'l p{0.70\\textwidth}'))
    caseblocks=[]
    for key in ['primary_synthetic_42','primary_digits_42','primary_breast_42','primary_wine_42','primary_heart_Cleveland']:
        model=json.loads((OUT/'model_structures'/f'{key}_CIEDT.json').read_text(encoding='utf-8'))
        dataset=key.split('_')[1]
        heading='Heart Disease (Cleveland held out)' if dataset=='heart' else NAMES[dataset]+' (seed 42)'
        caseblocks.append(r'\paragraph{'+heading+'}'+f' Depth bound {model["config"]["depth"]}, $\\lambda={model["config"]["lam"]:g}$.')
        caseblocks.append(r'\par\noindent\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrr}\toprule Path (binary transformed features) & Class & Source support\\\midrule')
        for path,pred,n in paths(model['tree'],model['feature_names']):caseblocks.append(' $\\land$ '.join(esc(p) for p in path or ['root leaf'])+f' & {pred} & {n}'+r'\\')
        caseblocks.append(r'\bottomrule\end{tabularx}\par\medskip')
    save('cases','\n'.join(caseblocks))
    old=pd.read_csv(ROOT/'experiment/results/revision_runs.csv');leg=[]
    for m in ['CART','TunedCART','ODT','CIEDT','LR']:
        leg.append([name(m)]+[f'{old[(old.dataset==ds)&(old.method==m)].avg_ood_acc.mean():.3f}' for ds in ['synthetic_xor','cmnist_tabular','breast_cancer','wine_binary']])
    save('legacy',table('Preserved historical mean target accuracies (five seeds). Real-task sources and targets resample a shared original pool; these are supplementary shortcut-intervention results.','tab:legacy',['Method','XOR','Digits','Breast','Wine'],leg))
    # All visualization inputs are the saved frames above; no fitted model is changed.
    plt.rcParams.update({'font.size':18,'axes.labelsize':18,'axes.titlesize':18,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,10),layout='constrained')
    colors=['#aeb9c4' if m!='CIEDT' else '#1b7a78' for m in METHODS]
    for ax,ds in zip(axes.flat,NAMES):
        g=controlled[controlled.dataset==ds].groupby('method').avg_ood_acc.agg(['mean','std']).reindex(METHODS)
        ax.barh([name(m) for m in METHODS],g['mean'],xerr=g['std'],color=colors,capsize=3);ax.invert_yaxis();ax.set_xlim(0,1.02);ax.set_title(NAMES[ds]);ax.set_xlabel('Target accuracy');ax.grid(axis='x',alpha=.2)
    savefig(fig,'controlled')
    shift=pd.read_csv(OUT/'shift_runs.csv');fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    for m,g in shift.groupby('method',sort=False):
        gg=g.groupby('target_match_probability').accuracy.agg(['mean','std']);x=gg.index.to_numpy();ax.plot(x,gg['mean'],marker='o',label=name(m));ax.fill_between(x,gg['mean']-gg['std'],gg['mean']+gg['std'],alpha=.12)
    ax.set(xlabel='Target shortcut match probability',ylabel='Accuracy',ylim=(0,1));ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3);ax.grid(alpha=.2);savefig(fig,'shift')
    fig,axes=plt.subplots(3,1,figsize=(8,10.5),layout='constrained')
    for ax,axis,xlabel in zip(axes,['depth','features','samples'],['Maximum depth','Binary features','Samples per source']):
        g=scaling[scaling.axis==axis].groupby('value').runtime_s;med=g.median();ax.plot(med.index,med,marker='o',color='#1b7a78');ax.fill_between(med.index,g.quantile(.25),g.quantile(.75),alpha=.2,color='#1b7a78');ax.set(xlabel=xlabel,ylabel='Fit time (s)',yscale='log');ax.grid(alpha=.2)
    savefig(fig,'scaling')
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained');deltas=[]
    for ds in NAMES:
        g=controlled[controlled.dataset==ds].pivot(index='seed',columns='method',values='avg_ood_acc');deltas.append((g.CIEDT-g.ODT).to_numpy())
    ax.boxplot(deltas,tick_labels=list(NAMES.values()),showfliers=False)
    for i,a in enumerate(deltas):ax.scatter(i+1+np.linspace(-.12,.12,5),a,color='#1b7a78',zorder=3)
    ax.axhline(0,color='#647382',linestyle='--');ax.set_ylabel('Paired target accuracy difference');ax.grid(axis='y',alpha=.2);savefig(fig,'stability')
    sensitivity=pd.read_csv(OUT/'sensitivity_runs.csv');assert len(sensitivity)==180
    env=sensitivity[sensitivity.study=='environment_count'];smooth=sensitivity[sensitivity.study=='smoothing'];grid=sensitivity[sensitivity.study=='depth_lambda']
    fig,axes=plt.subplots(3,1,figsize=(8,10.5),layout='constrained')
    for method,g in env.groupby('method'):
        gg=g.groupby('m').mean_accuracy.agg(['mean','std']);axes[0].errorbar(gg.index,gg['mean'],yerr=gg['std'],marker='o',capsize=3,label=method)
    axes[0].set(xlabel='Source environments',ylabel='Target accuracy',ylim=(0,1));axes[0].legend();axes[0].grid(alpha=.2)
    for alpha,g in smooth.groupby('alpha'):
        gg=g.groupby('n_per_env').mean_accuracy.mean();axes[1].plot(gg.index,gg,marker='o',label=f'alpha={alpha:g}')
    axes[1].set(xlabel='Samples per source',ylabel='Target accuracy',xscale='log',ylim=(0,1));axes[1].legend(fontsize=16);axes[1].grid(alpha=.2)
    for depth,g in grid.groupby('depth'):
        gg=g.groupby('lam').mean_accuracy.agg(['mean','std']);axes[2].errorbar(gg.index,gg['mean'],yerr=gg['std'],marker='o',capsize=3,label=f'depth {depth}')
    axes[2].set(xlabel='Regularization lambda',ylabel='Target accuracy',ylim=(0,1));axes[2].legend();axes[2].grid(alpha=.2)
    savefig(fig,'sensitivity')
    save('sensitivity_text',f'The additional predetermined synthetic studies contain 40 source-count fits with source-only depth/regularization selection, 80 fixed-configuration smoothing fits, and 60 depth--regularization grid fits. With 500 examples per source, selected CIEDT mean target accuracies for two, three, four, and five sources are '+', '.join(f'{env[(env.method=="CIEDT")&(env.m==m)].mean_accuracy.mean():.3f}' for m in [2,3,4,5])+'. The smoothing study varies $\\alpha\\in\\{0,.5,1,2\\}$ and source size $25$--$500$ at depth three and $\\lambda=10$; unsupported-pair fractions are saved. The depth--regularization target curve is diagnostic only: it does not feed back into any primary configuration choice. These synthetic panels should not be interpreted as a universal benefit of more environments or larger regularization.')
    from eswa_clean_presentation import main as refresh_presentation
    refresh_presentation()
    manifest=dict(primary_jobs=24,primary_rows=216,controlled_seeds=5,heart_sites=4,mechanism_rows=144,scaling_rows=len(scaling),cache_rows=len(cache),shift_rows=len(shift),sensitivity_rows=len(sensitivity),macro_controlled=macro.to_dict(),heart_macro_balanced=hm.to_dict(),robust_incumbents=len(timed),robust_gap_max=float(robust.gap.max()),figures=['controlled','shift','scaling','primary_trees','sensitivity','stability'])
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
