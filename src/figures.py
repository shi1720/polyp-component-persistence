"""Generate publication figures from recorded results; no invented observations."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from components import component_masks,crc_index

COLORS={'Raw CRC':'#2468a2','Raw CRC + LCC':'#c64f43','LCC recalibration':'#9a7a31','Prefix union CRC':'#008574','LCC envelope CRC':'#7255a0','Anchor CRC':'#6b7379','Area opening CRC':'#b75f8c'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();out=a.root/'figures';out.mkdir(exist_ok=True)
    z=np.load(a.root/'results/paths_g257_c2_t10.npz');stats=z['stats'];thresholds=z['thresholds'];summary=pd.read_csv(a.root/'results/summary.csv');splits=pd.read_csv(a.root/'results/splits.csv')
    # Exact synthetic mechanism: disconnected regions of sizes two and three.
    s=np.zeros((5,9));s[1,1:3]=.9;s[3,5:8]=.8;y=np.zeros_like(s,bool);y[1,1:3]=True
    path=list(component_masks(s,[1,.9,.8,0]));fig,axs=plt.subplots(2,3,figsize=(7.4,3.5))
    for row,j in enumerate([1,2]):
        for col,key in enumerate(['raw','lcc','prefix']):
            mask=path[j][key];rgb=np.ones((*s.shape,3))*.94;rgb[mask]=[.24,.55,.78];rgb[mask&y]=[.05,.55,.38];rgb[y&~mask]=[.82,.24,.2]
            ax=axs[row,col];ax.imshow(rgb,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);loss=1-(mask&y).sum()/y.sum();ax.set_title(f'{["Raw mask","Largest component","Prefix union"][col]} | missed {loss:.0%}')
            if col==0:ax.set_ylabel('Threshold '+str([.9,.8][row]))
    fig.text(.5,.01,'Synthetic counterexample. Green: included annotation; red: deleted annotation; blue: other foreground.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);save(fig,out,'mechanism')
    # Primary risk/area comparison: all recorded splits, no SE across overlapping splits.
    primary=summary[np.isclose(summary.alpha,.1)].set_index('method');names=list(COLORS)
    fig,axs=plt.subplots(1,2,figsize=(8,4.0),sharey=True)
    yy=np.arange(len(names)); risks=np.array([primary.loc[n,'risk']*100 for n in names]);areas=np.array([primary.loc[n,'area']*100 for n in names])
    lo=np.array([primary.loc[n,'risk_p025']*100 for n in names]);hi=np.array([primary.loc[n,'risk_p975']*100 for n in names])
    for i,n in enumerate(names):
        axs[0].plot([lo[i],hi[i]],[i,i],color=COLORS[n],lw=2);axs[0].scatter(risks[i],i,color=COLORS[n],s=38,zorder=3)
        axs[1].barh(i,areas[i],color=COLORS[n],height=.6)
    axs[0].set_yticks(yy,names);axs[0].invert_yaxis();axs[0].axvline(10,color='#333333',linestyle='--',lw=1,label='Target');axs[0].set_xlabel('Mean missed annotated area (%)');axs[0].legend(frameon=False,fontsize=8)
    axs[1].set_xlabel('Mean predicted frame area (%)');axs[1].tick_params(left=False);axs[0].set_title('A  Recall loss');axs[1].set_title('B  Output area')
    fig.tight_layout();save(fig,out,'primary')
    # Sensitivity targets.
    fig,ax=plt.subplots(figsize=(6.8,3.5))
    for n in names:
        q=summary[summary.method==n].sort_values('alpha');ax.plot(q.alpha*100,q.risk*100,'o-',color=COLORS[n],label=n,ms=4)
    ax.plot([5,20],[5,20],'k--',lw=1,label='Target');ax.set_xlabel('Target missed area (%)');ax.set_ylabel('Evaluation missed area (%)');ax.legend(frameon=False,ncol=2,fontsize=7);fig.tight_layout();save(fig,out,'sensitivity')
    # Two examples chosen by greatest full-pool cleanup penalty from distinct sources.
    j=crc_index(stats[:,0,:,0],.1);delta=stats[:,1,j,0]-stats[:,0,j,0];manifest=json.loads((a.cache/'manifest.json').read_text());meta=manifest['images'];chosen=[];sources=set()
    for i in np.argsort(delta)[::-1]:
        if meta[i]['dataset']!='Kvasir-SEG' and meta[i]['dataset'] not in sources:chosen.append(int(i));sources.add(meta[i]['dataset'])
        if len(chosen)==2:break
    scores=np.load(a.cache/'scores.npy',mmap_mode='r');masks=np.load(a.cache/'masks.npy',mmap_mode='r');fig,axs=plt.subplots(2,4,figsize=(9.2,5.1))
    for row,i in enumerate(chosen):
        m=meta[i];im=Image.open(a.data/'test'/m['dataset']/'images'/m['file']).convert('RGB').resize((352,352));y=masks[i];path=list(component_masks(scores[i],thresholds[:j+1]));preds=path[-1]
        for col,key in enumerate(['annotation','raw','lcc','prefix']):
            ax=axs[row,col];ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
            if key=='annotation':
                ax.contour(y,levels=[.5],colors=['#36ef77'],linewidths=.9);title=m['dataset']+'\nAnnotation'
            else:
                mask=preds[key];overlay=np.zeros((352,352,4));overlay[mask&y]=[.1,.85,.4,.5];overlay[mask&~y]=[.1,.45,1,.5];overlay[~mask&y]=[1,.15,.15,.6];ax.imshow(overlay);loss=1-(mask&y).sum()/y.sum();title={'raw':'Raw','lcc':'Largest component','prefix':'Prefix union'}[key]+f'\nMissed {loss:.1%}'
            ax.set_title(title,fontsize=9)
    fig.subplots_adjust(wspace=.05,hspace=.20,top=.90,bottom=.03);save(fig,out,'examples');(out/'example_selection.json').write_text(json.dumps({'threshold':float(thresholds[j]),'image_ids':chosen,'files':[meta[i] for i in chosen],'selection':'largest full-pool cleanup penalty per source, top two distinct sources excluding Kvasir to avoid acquisition text in display'},indent=2))
    # Upload-ready versions, fitting within 800x800 and 500KB.
    dest=a.root/'submission'
    for name in ['mechanism','primary','examples']:
        im=Image.open(out/(name+'.png')).convert('RGB');im.thumbnail((790,790));im.save(dest/(name+'.jpg'),quality=90,optimize=True)
    print('figures written')
if __name__=='__main__':main()
