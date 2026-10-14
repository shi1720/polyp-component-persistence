"""Fixed PraNet inference and exact decoded-image audit; no model fitting."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF


def digest(path):
    im = Image.open(path).convert('RGB')
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', choices=['pranet','pvt'], default='pranet')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.upstream.resolve()))
    torch.set_num_threads(2)
    if a.model=='pranet':
        from polyps.PraNet_Res2Net import PraNet
        model=PraNet()
    else:
        from lib.pvt import PolypPVT
        model=PolypPVT()
    torch.set_num_threads(2)
    if a.model=='pvt':
        # MPS lacks non-divisible adaptive pooling; CPU evaluates the same operator.
        import torch.nn.functional as F
        original_pool=F.adaptive_avg_pool2d
        def compatible_pool(x,size):
            if x.device.type=='mps':
                return original_pool(x.cpu(),size).to(x.device)
            return original_pool(x,size)
        F.adaptive_avg_pool2d=compatible_pool
    def predict(x):
        out=model(x)
        return out[-1] if a.model=='pranet' else out[0]+out[1]
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True)); model.eval()
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    train={digest(f) for f in sorted((a.data/'train').glob('*/images/*.png'))}
    seen=set(); rows=[]; excluded=[]; paths=[]
    for f in sorted((a.data/'test').glob('*/images/*.png')):
        h=digest(f)
        if h in train or h in seen:
            excluded.append({'path':str(f.relative_to(a.data)),'sha256':h,'reason':'training duplicate' if h in train else 'test duplicate'})
            continue
        seen.add(h); paths.append(f)
        rows.append({'id':len(rows),'dataset':f.parts[-3],'file':f.name,'sha256':h,'width':Image.open(f).width,'height':Image.open(f).height})
    (a.out/'manifest.json').write_text(json.dumps({'train_unique':len(train),'images':rows,'excluded':excluded,'device':device},indent=2))
    preds=np.lib.format.open_memmap(a.out/'scores.npy',mode='w+',dtype=np.float32,shape=(len(rows),352,352))
    masks=np.lib.format.open_memmap(a.out/'masks.npy',mode='w+',dtype=bool,shape=(len(rows),352,352))
    start=time.time(); checks=[]
    with torch.inference_mode():
        for i,f in enumerate(paths):
            im=Image.open(f).convert('RGB').resize((352,352),Image.Resampling.BILINEAR)
            x=TF.normalize(TF.to_tensor(im),[.485,.456,.406],[.229,.224,.225])[None]
            logits=predict(x.to(device))
            score=logits.sigmoid().cpu().numpy()[0,0]
            if i<3 and device=='mps':
                model.cpu(); reference=predict(x).sigmoid().numpy()[0,0]
                checks.append(float(np.max(np.abs(reference-score)))); model.to(device)
            preds[i]=score
            mask=Image.open(f.parent.parent/'masks'/f.name).convert('L').resize((352,352),Image.Resampling.NEAREST)
            masks[i]=np.asarray(mask)>=128
            assert masks[i].sum()>0
            if i%50==0: print(i,len(paths),round(time.time()-start,1),flush=True)
    preds.flush(); masks.flush()
    (a.out/'inference.json').write_text(json.dumps({'seconds':time.time()-start,'cpu_mps_max_abs_error_first3':checks,'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__': main()
