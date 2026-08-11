#!/usr/bin/env python3
"""Generate finite FP32 windows with controlled record-outcome Markov chains."""
import argparse,json
from pathlib import Path
import numpy as np
from markov import *

DEFAULT=((.05,.95),(.2,.8),(.35,.65),(.5,.5),(.65,.35),(.8,.2),(.95,.05),(.1,.1),(.25,.25),(.75,.75),(.9,.9))

def generate_chain(n,alpha,beta,rng):
    b=np.empty(n,np.uint8); b[0]=rng.random()<alpha/(alpha+beta); u=rng.random(n-1)
    for i in range(1,n): b[i]=u[i-1]<alpha if b[i-1]==0 else not(u[i-1]<beta)
    return b

def outcomes_to_windows(outcomes,rng,gap_low=.05,gap_high=.5):
    b=np.asarray(outcomes,np.uint8).reshape(-1,3); x=np.empty((len(b),4),np.float32)
    x[:,0]=rng.uniform(1,2,len(b)); m=x[:,0].copy()
    for j in range(3):
        gap=rng.uniform(gap_low,gap_high,len(b)).astype(np.float32); take=b[:,j].astype(bool)
        x[:,j+1]=np.where(take,m+gap,m-gap); m=np.where(take,x[:,j+1],m)
    return x

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=Path('data'))
    p.add_argument('--windows',type=int,default=200_000); p.add_argument('--seed',type=int,default=20260811)
    p.add_argument('--condition',action='append',default=[]); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    conditions=DEFAULT if not a.condition else tuple(tuple(map(float,s.split(','))) for s in a.condition); manifest=[]
    for k,(alpha,beta) in enumerate(conditions):
        rng=np.random.default_rng(a.seed+k*10007); requested=generate_chain(a.windows*3,alpha,beta,rng)
        windows=outcomes_to_windows(requested,rng)
        if not np.array_equal(record_outcomes(windows),requested): raise RuntimeError('construction mismatch')
        executed=comparison_outcomes(windows); pos=analyze_positioned_outcomes(executed)
        metrics=analyze_outcomes(executed).to_dict(); metrics['empirical_two_bit_trace_error_rate']=empirical_two_bit_error(executed)
        metrics.update({k:v for k,v in pos.items() if isinstance(v,(int,float))})
        cid=f'a{alpha:.2f}_b{beta:.2f}'.replace('.','p'); binary=a.output_dir/f'{cid}.f32'; windows.tofile(binary)
        item={'condition_id':cid,'input_path':str(binary.resolve()),'windows':a.windows,'requested_alpha':alpha,
          'requested_beta':beta,'markov':metrics,'position_aware':pos,'record_only_markov':analyze_outcomes(requested).to_dict()}
        (a.output_dir/f'{cid}.json').write_text(json.dumps(item,indent=2)+'\n'); manifest.append(item); print('generated',cid)
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
