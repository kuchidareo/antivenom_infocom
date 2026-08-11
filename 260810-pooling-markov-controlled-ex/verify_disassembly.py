#!/usr/bin/env python3
import argparse,re,subprocess
p=argparse.ArgumentParser();p.add_argument('binary',nargs='?',default='./maxpool_benchmark');a=p.parse_args()
s=subprocess.run(['objdump','-d','-C','--disassemble=branched_maxpool4',a.binary],check=True,text=True,capture_output=True).stdout;print(s)
cmp=bool(re.search(r'\b(?:u?comiss|vcomiss)\b',s)); jumps=re.findall(r'\bj(?!mp)[a-z]+\b',s)
if not cmp or not jumps:raise SystemExit('FAIL: floating compare + conditional jump not found')
print('PASS candidates:',len(jumps),'- manually verify one jump consumes comparison flags and loop is not unrolled')
