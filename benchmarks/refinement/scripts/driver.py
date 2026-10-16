from pathlib import Path
import argparse,concurrent.futures,gzip,hashlib,json,os,random,selectors,subprocess,sys,threading,time
ROOT=Path(os.environ.get('CPO3_REPO', '/Users/matteosantamaria/cpo3-workspace/cpo3'));OUT=Path(__file__).parent;HARNESS=Path(os.environ.get('CPO3_BENCH_HARNESS', str(ROOT.parent/'solver_benchmarks')))
def run(args):
 cfg=json.loads((OUT/'suite.json').read_text());specs=cfg['problems']; variants=args.variants.split(',')
 if args.time_limit is not None:cfg['settings']['time_limit']=args.time_limit
 if args.jobs==1:
  for spec in specs:spec['batch']=max(1,min(300,int(.03/max(spec['historical']['elapsed'],.00001))))
 if args.suite=='screen':specs=[s for s in specs if s['id'] in cfg['screen_ids']]
 if args.ids:specs=[s for s in specs if s['id'] in args.ids.split(',')]
 random.Random(813).shuffle(specs)
 dest=OUT/(args.name+'.jsonl');done=set()
 if dest.exists():done={(r['problem'],r['variant'],r['repeat']) for r in map(json.loads,dest.read_text().splitlines())}
 tls=threading.local();procs=[];lock=threading.Lock()
 env={**os.environ,**{k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}
 def start(variant):
  log=(OUT/(args.name+'-'+variant+'-stderr.log')).open('a')
  p=subprocess.Popen([sys.executable,str(OUT/'worker.py'),'worker','--package',str(OUT/(variant+'-package')),'--harness',str(HARNESS)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
  with lock:procs.append((p,log))
  return p
 def task(item):
  index,spec=item
  if not hasattr(tls,'workers'):tls.workers={}
  records=[]
  for repeat in range(-args.warmups,args.repeats):
   order=variants if (index+repeat)%2==0 else variants[::-1]
   if args.balanced_order:
    offset=(index+repeat)%len(variants);order=variants[offset:]+variants[:offset]
    if ((index+repeat)//len(variants))%2:order=order[::-1]
   for v in order:
    if (spec['id'],v,repeat) in done:continue
    p=tls.workers.get(v)
    if p is None or p.poll() is not None:p=start(v);tls.workers[v]=p
    p.stdin.write(json.dumps({'problem':spec,'settings':cfg['settings'],'batch':spec['batch']})+'\n');p.stdin.flush()
    with selectors.DefaultSelector() as sel:
     sel.register(p.stdout,selectors.EVENT_READ);ready=sel.select(max(120,cfg['settings']['time_limit']+60))
    if ready:
     line=p.stdout.readline();res=json.loads(line) if line else {'status':'worker_failed'}
    else:p.kill();res={'status':'external_timeout'}
    if repeat<0:continue
    record={'problem':spec['id'],'dataset':spec['dataset'],'variant':v,'repeat':repeat,'batch':spec['batch'],'result':res};records.append(record)
    with lock:
     with dest.open('a') as f:f.write(json.dumps(record)+'\n')
  return records
 started=time.time()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
   for i,future in enumerate(concurrent.futures.as_completed([pool.submit(task,item) for item in enumerate(specs)]),1):
    records=future.result()
    if i%10==0 or i==len(specs):print(args.name,i,'/',len(specs),'seconds',round(time.time()-started),flush=True)
 finally:
  for p,log in procs:
   if p.poll() is None:p.terminate()
   p.wait();log.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--setup',action='store_true');ap.add_argument('--suite',default='screen');ap.add_argument('--name',default='screen');ap.add_argument('--variants',default='baseline,monotone,best_effort');ap.add_argument('--repeats',type=int,default=2);ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--ids');ap.add_argument('--time-limit',type=float);ap.add_argument('--warmups',type=int,default=0);ap.add_argument('--balanced-order',action='store_true');args=ap.parse_args()
 if args.setup:raise SystemExit('Use the archived suite.json manifest')
 else:run(args)
