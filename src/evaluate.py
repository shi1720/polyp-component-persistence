"""Paired finite-pool random-split evaluation; split spread is not a population CI."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from components import crc_index,suffix_envelope

DEFS={'Raw CRC':('raw','raw'),'Raw CRC + LCC':('raw','lcc'),'LCC recalibration':('lcc','lcc'),'Prefix union CRC':('prefix','prefix'),'LCC envelope CRC':('envelope','lcc'),'Anchor CRC':('anchor','anchor'),'Area opening CRC':('opening','opening')}

def run(z,ids,alphas,repeats=500,seed=20406):
    stats=z['stats']; methods=list(z['methods']); thresholds=z['thresholds']; losses={k:stats[:,j,:,0] for j,k in enumerate(methods)}; losses['envelope']=suffix_envelope(losses['lcc'])
    rng=np.random.default_rng(seed); rows=[]
    for rep in range(repeats):
        perm=rng.permutation(ids); n=len(perm)//2; cal,test=perm[:n],perm[n:]
        for alpha in alphas:
            indices={k:crc_index(v[cal],alpha) for k,v in losses.items()}
            for name,(calibration,output) in DEFS.items():
                j=indices[calibration]; k=methods.index(output); vals=stats[test,k,j]
                cleanup=stats[test,1,j,0]-stats[test,0,j,0]
                rows.append(dict(rep=rep,alpha=alpha,method=name,index=j,threshold=thresholds[j],risk=vals[:,0].mean(),area=vals[:,1].mean(),dice=vals[:,2].mean(),cleanup_penalty=cleanup.mean(),cleanup_gt1pct=(cleanup>.01).mean(),cal_n=n,test_n=len(test)))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--paths',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); z=np.load(a.paths); manifest=json.loads(a.manifest.read_text()); N=len(z['stats']);
    df=run(z,np.arange(N),[.05,.1,.2]); df.to_csv(a.out/'splits.csv',index=False)
    summary=df.groupby(['alpha','method'],sort=False).agg(risk=('risk','mean'),risk_sd=('risk','std'),risk_p025=('risk',lambda x:x.quantile(.025)),risk_p975=('risk',lambda x:x.quantile(.975)),area=('area','mean'),dice=('dice','mean'),threshold=('threshold','mean'),cleanup_penalty=('cleanup_penalty','mean'),cleanup_gt1pct=('cleanup_gt1pct','mean')).reset_index();summary.to_csv(a.out/'summary.csv',index=False)
    diagnostic=[]
    for dataset in sorted(set(r['dataset'] for r in manifest['images'])):
        ids=np.array([r['id'] for r in manifest['images'] if r['dataset']==dataset])
        diagnostic.append(dict(dataset=dataset,n=len(ids),nonnested=int(z['nonnested'][ids].sum()),risk_increase=int(z['risk_increase'][ids].sum())))
    pd.DataFrame(diagnostic).to_csv(a.out/'diagnostics.csv',index=False)
    # Fixed full-pool calibration is descriptive, not held-out performance.
    losses=z['stats'][:,0,:,0]; j=crc_index(losses,.1); delta=z['stats'][:,1,j,0]-losses[:,j]; rng=np.random.default_rng(809)
    boots=[delta[rng.integers(0,N,N)].mean() for _ in range(5000)]
    details={'N':N,'nonnested':int(z['nonnested'].sum()),'risk_increase':int(z['risk_increase'].sum()),'fixed_threshold':float(z['thresholds'][j]),'fixed_threshold_cleanup_penalty':float(delta.mean()),'image_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'fixed_threshold_cleanup_gt1pct':int((delta>.01).sum()),'fixed_threshold_cleanup_max':float(delta.max())}
    (a.out/'details.json').write_text(json.dumps(details,indent=2))
    # Source-held-out stress: calibrate on every other source, evaluate whole source.
    methods=list(z['methods']); L={k:z['stats'][:,i,:,0] for i,k in enumerate(methods)};L['envelope']=suffix_envelope(L['lcc']);stress=[]
    for dataset in sorted(set(r['dataset'] for r in manifest['images'])):
        test=np.array([r['id'] for r in manifest['images'] if r['dataset']==dataset]); cal=np.setdiff1d(np.arange(N),test)
        for name,(c,o) in DEFS.items():
            j=crc_index(L[c][cal],.1); vals=z['stats'][test,methods.index(o),j];stress.append(dict(dataset=dataset,method=name,n=len(test),threshold=z['thresholds'][j],risk=vals[:,0].mean(),area=vals[:,1].mean(),dice=vals[:,2].mean()))
    pd.DataFrame(stress).to_csv(a.out/'source_stress.csv',index=False)
    print(summary.to_string(index=False));print(details)
if __name__=='__main__':main()
