#!/usr/bin/env python3
import argparse,json,platform,random,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('data/manifest.json'));p.add_argument('--benchmark',type=Path,default=Path('./maxpool_benchmark'));p.add_argument('--output',type=Path,default=Path('results/raw.jsonl'));p.add_argument('--trials',type=int,default=10);p.add_argument('--repeats',type=int,default=100);p.add_argument('--warmup',type=int,default=5);p.add_argument('--cpu',type=int,default=0);p.add_argument('--seed',type=int,default=20260811);a=p.parse_args()
 if platform.system()!='Linux':raise SystemExit('Linux required')
 conditions=json.loads(a.manifest.read_text());rng=random.Random(a.seed);jobs=[]
 for trial in range(a.trials):
  cs=conditions[:];rng.shuffle(cs)
  for c in cs:
   modes=['branched','control'];rng.shuffle(modes)
   for mode in modes:jobs.append((trial,c,mode))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w') as f:
  for j,(trial,c,mode) in enumerate(jobs,1):
   cmd=[str(a.benchmark.resolve()),'--input',c['input_path'],'--mode',mode,'--repeats',str(a.repeats),'--warmup',str(a.warmup)]
   if a.cpu>=0:cmd=['taskset','-c',str(a.cpu),*cmd]
   run=subprocess.run(cmd,check=True,text=True,capture_output=True);measured=json.loads(run.stdout.strip().splitlines()[-1])
   row={'condition_id':c['condition_id'],'trial':trial,'cpu':a.cpu,'host':platform.node(),**measured};f.write(json.dumps(row)+'\n');f.flush();print(f'[{j}/{len(jobs)}]',c['condition_id'],trial,mode)
if __name__=='__main__':main()
