#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
PRED=('entropy_rate_bits','last_outcome_error_rate','bayes_first_order_error_rate','two_bit_counter_error_rate','empirical_two_bit_trace_error_rate','position_markov_entropy_rate_bits','position_markov_bayes_error_rate','position_only_entropy_rate_bits','position_only_bayes_error_rate')
def corr(x,y):return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else float('nan')
def rank(x):
 o=np.argsort(x);r=np.empty(len(x),float)
 for v in np.unique(x):r[x==v]=np.mean(np.flatnonzero(np.sort(x)==v))
 return r
def fit(x,y):
 X=np.c_[np.ones(len(x)),x];coef=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@coef;ss=np.sum((y-y.mean())**2);press=0
 for i in range(len(y)):
  keep=np.arange(len(y))!=i;c=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0];press+=(y[i]-X[i]@c)**2
 return {'coefficients':coef.tolist(),'r_squared':float(1-np.sum((y-pred)**2)/ss),'loocv_r_squared':float(1-press/ss),'rmse':float(np.sqrt(np.mean((y-pred)**2))),'loocv_rmse':float(np.sqrt(press/len(y)))}
def write(path,rows):
 keys=sorted({k for r in rows for k in r});f=path.open('w',newline='');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows);f.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('data/manifest.json'));p.add_argument('--measurements',type=Path,default=Path('results/raw.jsonl'));p.add_argument('--output-dir',type=Path,default=Path('results/analysis'));a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest={x['condition_id']:x for x in json.loads(a.manifest.read_text())};pairs=defaultdict(dict)
 for line in a.measurements.read_text().splitlines():
  r=json.loads(line);pairs[(r['condition_id'],r['trial'])][r['mode']]=r
 paired=[]
 for (cid,t),m in sorted(pairs.items()):
  if set(m)!={'branched','control'}:raise RuntimeError('incomplete pair')
  b,c=m['branched'],m['control'];n=b['target_comparisons'];paired.append({'condition_id':cid,'trial':t,'target_comparisons':n,'excess_branches':b['branches']-c['branches'],'excess_branch_misses':b['branch_misses']-c['branch_misses'],'target_branch_count_ratio':(b['branches']-c['branches'])/n,'pmu_excess_misses_per_comparison':(b['branch_misses']-c['branch_misses'])/n})
 write(a.output_dir/'paired_trials.csv',paired);group=defaultdict(list)
 for r in paired:group[r['condition_id']].append(r)
 summary=[]
 for cid,rows in sorted(group.items()):
  rates=np.array([r['pmu_excess_misses_per_comparison'] for r in rows]);ratios=np.array([r['target_branch_count_ratio'] for r in rows]);summary.append({'condition_id':cid,**manifest[cid]['markov'],'pmu_excess_miss_rate_mean':rates.mean(),'pmu_excess_miss_rate_std':rates.std(ddof=1),'target_branch_count_ratio_mean':ratios.mean(),'target_branch_count_ratio_std':ratios.std(ddof=1)})
 write(a.output_dir/'condition_summary.csv',summary);y=np.array([r['pmu_excess_miss_rate_mean'] for r in summary]);evaluation={}
 for name in PRED:
  x=np.array([r[name] for r in summary]);evaluation[name]={'pearson_r':corr(x,y),'spearman_r':corr(rank(x),rank(y)),'uncalibrated_mae':float(np.mean(abs(x-y))),'uncalibrated_rmse':float(np.sqrt(np.mean((x-y)**2))),'calibration':fit(x[:,None],y)}
 report={'conditions':len(summary),'target_branch_count_ratio_mean':float(np.mean([r['target_branch_count_ratio_mean'] for r in summary])),'validity_rule':'ratio must be close to 1.0','evaluations':evaluation};(a.output_dir/'model_fit.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');print('wrote',a.output_dir)
if __name__=='__main__':main()
