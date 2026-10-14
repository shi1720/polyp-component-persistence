"""Compute per-image sufficient statistics, preserving all grid points."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from components import component_masks
from scipy import ndimage as ndi
from scipy.special import expit,logit

METHODS=['raw','lcc','prefix','anchor','opening']

def process(args):
    i,scores,mask,thresholds,conn=args
    stats=np.zeros((5,len(thresholds),3),np.float64)
    prev=None; nonnested=False; risk_increase=False
    area_y=mask.sum(); V=mask.size
    structure=ndi.generate_binary_structure(2,conn)
    anchor=np.unravel_index(np.argmax(scores),scores.shape)
    history=np.zeros_like(mask)
    for j,threshold in enumerate(thresholds):
        raw=scores>=threshold; labels,n=ndi.label(raw,structure)
        counts=np.bincount(labels.ravel(),minlength=n+1); counts[0]=0
        true_counts=np.bincount(labels[mask],minlength=n+1); true_counts[0]=0
        winner=int(np.argmax(counts)) if n else 0
        largest=(labels==winner) if n else np.zeros_like(raw)
        history |= largest
        anchor_label=labels[anchor]
        keep=counts>=64
        areas=[counts.sum(),counts[winner],history.sum(),counts[anchor_label],counts[keep].sum()]
        tps=[true_counts.sum(),true_counts[winner],np.count_nonzero(history&mask),true_counts[anchor_label],true_counts[keep].sum()]
        areas=np.array(areas);tps=np.array(tps)
        stats[:,j,0]=1-tps/area_y
        stats[:,j,1]=areas/V
        stats[:,j,2]=2*tps/(areas+area_y)
        if prev is not None and np.any(prev & ~largest): nonnested=True
        if j and stats[1,j,0]>stats[1,j-1,0]+1e-12: risk_increase=True
        prev=largest
    return i,stats,nonnested,risk_increase


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--grid',type=int,default=257); p.add_argument('--grid-scale',choices=['uniform','t10'],default='t10'); p.add_argument('--connectivity',type=int,default=2); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    scores=np.load(a.cache/'scores.npy',mmap_mode='r'); masks=np.load(a.cache/'masks.npy',mmap_mode='r')
    u=np.linspace(1,0,a.grid)
    thresholds=u if a.grid_scale=='uniform' else expit(10*logit(u))
    N=len(scores)
    stats=np.zeros((N,5,a.grid,3)); nonnested=np.zeros(N,bool); increases=np.zeros(N,bool); start=time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i,z,nn,ri in ex.map(process,((i,scores[i],masks[i],thresholds,a.connectivity) for i in range(N))):
            stats[i]=z; nonnested[i]=nn; increases[i]=ri
            if i%50==0: print(i,N,round(time.time()-start,1),flush=True)
    np.savez_compressed(a.out/f'paths_g{a.grid}_c{a.connectivity}_{a.grid_scale}.npz',stats=stats,thresholds=thresholds,methods=METHODS,nonnested=nonnested,risk_increase=increases)
    print('complete',time.time()-start,'nonnested',nonnested.sum(),'risk increase',increases.sum(),flush=True)
if __name__=='__main__':main()
