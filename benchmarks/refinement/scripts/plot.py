from pathlib import Path
import os,json,math,sys
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).parent
name=sys.argv[1] if len(sys.argv)>1 else 'full'
variant=sys.argv[2] if len(sys.argv)>2 else 'monotone'
s=json.loads((root/(name+'-summary.json')).read_text())[variant];cases=s['cases']
baseline_label=sys.argv[3] if len(sys.argv)>3 else 'Baseline f937dab'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def panel(ax, group, metric='time'):
 pairs=[]
 for c in group:
  pair=[]
  for tag in ['b','c']:
   val=c[('baseline_time' if tag=='b' else 'candidate_time') if metric=='time' else 'iters_'+tag]
   pair.append(max(val or 1,1e-30) if c['solved_'+tag] else math.inf)
  pairs.append(pair)
 a=np.array(pairs,dtype=float);ref=a.min(axis=1);ratios=np.divide(a,ref[:,None],out=np.full_like(a,np.inf),where=np.isfinite(a)&np.isfinite(ref[:,None]))
 x=np.unique(np.r_[1,ratios[np.isfinite(ratios)&(ratios<=8)],8])
 for j,label,color in [(0,baseline_label,'#343434'),(1,'Monotone refinement','#1769aa')]:
  y=np.array([(ratios[:,j]<=v).mean() for v in x]);ax.step(x,y,where='post',label=label,color=color,lw=1.6)
 ax.set_xscale('log',base=2);ax.set_xlim(1,8);ax.set_ylim(0,1.015);ax.set_xticks([1,1.25,1.5,2,4,8],['1','1.25','1.5','2','4','8']);ax.grid(alpha=.2);ax.set_xlabel('Performance ratio');ax.set_ylabel('Fraction of all problems solved')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,metric,title in zip(axes,['time','iters'],['Runtime (median of repeated solves)','Outer iterations']):panel(ax,cases,metric);ax.set_title(title)
axes[0].legend(loc='lower right');fig.suptitle(f'Refinement stopping rule: {len(cases)} local problems · presolve off · unchanged termination')
fig.savefig(root/'profiles.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
for ax,ds in zip(axes.flat,['all']+sorted({c['problem'].split('/')[0] for c in cases})):
 group=[c for c in cases if ds=='all' or c['problem'].startswith(ds+'/')];panel(ax,group);ax.set_title(f'{ds} (n={len(group)})')
axes.flat[0].legend(loc='lower right');fig.suptitle('Runtime Dolan–Moré profiles by collection')
fig.savefig(root/'profiles-by-dataset.png',dpi=150)
