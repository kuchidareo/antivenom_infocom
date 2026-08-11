#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from markov import *
def nchw_to_windows(x):
 x=np.asarray(x,np.float32)
 if x.ndim!=4 or not np.isfinite(x).all():raise ValueError('finite [N,C,H,W] required')
 h=x.shape[-2]//2*2;w=x.shape[-1]//2*2;x=np.ascontiguousarray(x[...,:h,:w]);return x.reshape(*x.shape[:2],h//2,2,w//2,2).transpose(0,1,2,4,3,5).reshape(-1,4).copy()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path('data'));p.add_argument('--condition-id',required=True);a=p.parse_args();x=np.load(a.input,mmap_mode='r');windows=nchw_to_windows(x);b=comparison_outcomes(windows);pos=analyze_positioned_outcomes(b);metrics=analyze_outcomes(b).to_dict();metrics['empirical_two_bit_trace_error_rate']=empirical_two_bit_error(b);metrics.update({k:v for k,v in pos.items() if isinstance(v,(int,float))});a.output_dir.mkdir(parents=True,exist_ok=True);binary=a.output_dir/f'{a.condition_id}.f32';windows.tofile(binary);item={'condition_id':a.condition_id,'input_path':str(binary.resolve()),'source_shape':list(x.shape),'windows':len(windows),'markov':metrics,'position_aware':pos};(a.output_dir/f'{a.condition_id}.json').write_text(json.dumps(item,indent=2)+'\n');print(json.dumps(item,indent=2))
if __name__=='__main__':main()
