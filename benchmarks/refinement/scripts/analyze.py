from pathlib import Path
import json,statistics,math,sys,collections
out=Path(__file__).parent
name=sys.argv[1] if len(sys.argv)>1 else 'screen'
records=[json.loads(l) for l in (out/(name+'.jsonl')).read_text().splitlines()]
by=collections.defaultdict(list)
for r in records:by[(r['problem'],r['variant'])].append(r['result'])
baseline_variant=sys.argv[2] if len(sys.argv)>2 else 'baseline'
base={p for p,v in by if v==baseline_variant}
variants=sorted({v for p,v in by}-{baseline_variant});result={}
for v in variants:
 ids=sorted(base&{p for p,w in by if w==v});cases=[]
 for p in ids:
  bs=by[p,baseline_variant];cs=by[p,v]
  b=bs[0];c=cs[0];bt=statistics.median(r.get('elapsed',math.inf) for r in bs);ct=statistics.median(r.get('elapsed',math.inf) for r in cs)
  solved_b=all(r['status']=='solved' for r in bs);solved_c=all(r['status']=='solved' for r in cs)
  cases.append({'problem':p,'baseline_status':b['status'],'candidate_status':c['status'],'baseline_time':bt,'candidate_time':ct,'ratio':ct/bt,'solved_b':solved_b,'solved_c':solved_c,'iters_b':b.get('iters'),'iters_c':c.get('iters'),'refinement_b':b.get('refinement_solves'),'refinement_c':c.get('refinement_solves'),'same_vectors':b.get('vectors_sha256')==c.get('vectors_sha256')})
 groups={}
 for ds in ['all']+sorted({p.split('/')[0] for p in ids}):
  group=[c for c in cases if ds=='all' or c['problem'].startswith(ds+'/')];common=[c for c in group if c['solved_b'] and c['solved_c']];profiles={}
  for metric in ['time','iters']:
   pairs=[(max(c['baseline_time' if metric=='time' else 'iters_b'] or 1,1e-30) if c['solved_b'] else math.inf,max(c['candidate_time' if metric=='time' else 'iters_c'] or 1,1e-30) if c['solved_c'] else math.inf) for c in group]
   profiles[metric]={str(tau):[sum(math.isfinite(x[j]) and x[j]<=tau*min(x) for x in pairs)/len(pairs) for j in [0,1]] for tau in [1,1.01,1.05,1.1,1.25,1.5,2,4,8]}
  groups[ds]={'n':len(group),'solved_b':sum(c['solved_b'] for c in group),'solved_c':sum(c['solved_c'] for c in group),'geomean_ratio_common':math.exp(statistics.mean(math.log(c['ratio']) for c in common)) if common else None,'same_vectors':sum(c['same_vectors'] for c in group),'profiles':profiles}
 result[v]={'groups':groups,'cases':cases}
 print(v,groups['all'])
 print('LOSSES',[(c['problem'],c['baseline_status'],c['candidate_status']) for c in cases if c['solved_b'] and not c['solved_c']])
 print('GAINS',[(c['problem'],c['baseline_status'],c['candidate_status']) for c in cases if not c['solved_b'] and c['solved_c']])
 print('SLOW',[(c['problem'],round(c['ratio'],2),c['iters_b'],c['iters_c']) for c in sorted(cases,key=lambda c:c['ratio'],reverse=True) if c['baseline_time']>.01][:12])
(out/(name+'-summary.json')).write_text(json.dumps(result,indent=2))
