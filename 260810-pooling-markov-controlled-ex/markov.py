"""Markov models for PyTorch-style scalar 2x2 MaxPool comparison traces."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import numpy as np

@dataclass(frozen=True)
class MarkovMetrics:
    n_outcomes: int
    count_00: int; count_01: int; count_10: int; count_11: int
    p_1_given_0: float; p_0_given_1: float
    stationary_p0: float; stationary_p1: float
    entropy_rate_bits: float
    last_outcome_error_rate: float
    bayes_first_order_error_rate: float
    two_bit_counter_error_rate: float
    def to_dict(self): return asdict(self)

def _check_windows(windows):
    x=np.asarray(windows,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=4: raise ValueError("windows must be [W,4]")
    if not np.isfinite(x).all(): raise ValueError("finite values required")
    return x

def record_outcomes(windows):
    """Three data-dependent outcomes after initializing max from x1."""
    x=_check_windows(windows); m=x[:,0].copy(); out=np.empty((len(x),3),np.uint8)
    for j in range(1,4):
        take=x[:,j]>m; out[:,j-1]=take; m=np.where(take,x[:,j],m)
    return out.reshape(-1)

def comparison_outcomes(windows):
    """Four outcomes executed by PyTorch NCHW CPU: max=-inf, then x1..x4."""
    r=record_outcomes(windows).reshape(-1,3)
    return np.c_[np.ones(len(r),dtype=np.uint8),r].reshape(-1)

def _h(p):
    return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)

def transition_matrix(outcomes,laplace=.5):
    b=np.asarray(outcomes,np.uint8).reshape(-1)
    if len(b)<2 or np.any(b>1): raise ValueError("binary trace length >=2 required")
    counts=np.zeros((2,2),np.int64); np.add.at(counts,(b[:-1],b[1:]),1)
    q=counts.astype(float)+laplace; q/=q.sum(1,keepdims=True)
    return counts,q

def stationary(q):
    a=np.vstack((q.T-np.eye(2),np.ones(2))); y=np.array([0.,0.,1.])
    p=np.linalg.lstsq(a,y,rcond=None)[0]; p=np.clip(p,0,None); return p/p.sum()

def two_bit_stationary_error(q):
    # Joint state: previous outcome (2) x saturating-counter state (4).
    Q=np.zeros((8,8))
    ix=lambda prev,counter: prev*4+counter
    for prev in (0,1):
      for counter in range(4):
       for nxt in (0,1):
        nc=min(3,counter+1) if nxt else max(0,counter-1)
        Q[ix(prev,counter),ix(nxt,nc)]+=q[prev,nxt]
    a=np.vstack((Q.T-np.eye(8),np.ones(8))); y=np.r_[np.zeros(8),1.]
    pi=np.linalg.lstsq(a,y,rcond=None)[0]; pi=np.clip(pi,0,None); pi/=pi.sum()
    err=0.
    for prev in (0,1):
      for c in range(4): err+=pi[ix(prev,c)]*q[prev,1-int(c>=2)]
    return float(err)

def analyze_outcomes(outcomes,laplace=.5):
    b=np.asarray(outcomes,np.uint8).reshape(-1); counts,q=transition_matrix(b,laplace)
    pi=stationary(q); alpha=float(q[0,1]); beta=float(q[1,0])
    return MarkovMetrics(len(b),int(counts[0,0]),int(counts[0,1]),int(counts[1,0]),int(counts[1,1]),
      alpha,beta,float(pi[0]),float(pi[1]),float(pi[0]*_h(alpha)+pi[1]*_h(beta)),
      float(pi[0]*alpha+pi[1]*beta),float(pi[0]*min(alpha,1-alpha)+pi[1]*min(beta,1-beta)),
      two_bit_stationary_error(q))

def empirical_two_bit_error(outcomes,initial=1):
    c=initial; misses=0; b=np.asarray(outcomes,np.uint8).reshape(-1)
    for bit in b:
        misses+=int(c>=2)!=int(bit); c=min(3,c+1) if bit else max(0,c-1)
    return misses/max(1,len(b))

def analyze_positioned_outcomes(outcomes,laplace=.5):
    """Position-conditioned model for B1,B2,B3,B4 and B4->next B1."""
    b=np.asarray(outcomes,np.uint8).reshape(-1,4); matrices=[]; ent=[]; bayes=[]
    pos_ent=[]; pos_bayes=[]
    for pos in range(4):
        prev,cur=(b[:-1,3],b[1:,0]) if pos==0 else (b[:,pos-1],b[:,pos])
        cnt=np.zeros((2,2),np.int64); np.add.at(cnt,(prev,cur),1)
        q=cnt.astype(float)+laplace; q/=q.sum(1,keepdims=True)
        pp=np.bincount(prev,minlength=2).astype(float); pp/=pp.sum()
        ent.append(sum(pp[s]*_h(float(q[s,1])) for s in (0,1)))
        bayes.append(sum(pp[s]*min(q[s]) for s in (0,1)))
        p1=float(cur.mean()); pos_ent.append(_h(p1)); pos_bayes.append(min(p1,1-p1))
        matrices.append({"position":pos+1,"counts":cnt.tolist(),"probabilities":q.tolist()})
    return {"position_p1":b.mean(0).tolist(),"transition_by_position":matrices,
      "position_markov_entropy_rate_bits":float(np.mean(ent)),
      "position_markov_bayes_error_rate":float(np.mean(bayes)),
      "position_only_entropy_rate_bits":float(np.mean(pos_ent)),
      "position_only_bayes_error_rate":float(np.mean(pos_bayes))}
