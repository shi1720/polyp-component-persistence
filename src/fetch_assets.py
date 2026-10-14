"""Download the upstream research assets, verifying hashes and pinning source."""
import hashlib, urllib.request, zipfile
from pathlib import Path

DATA_SHA='554b9f23ba697083334bc166ebc98273512bc1669ebfe4025153121b7cba0b59'
REV='b9b2276e3e1ccf26ab74f2c750e51e091aa409bd'
ROOT=Path('work'); ROOT.mkdir(exist_ok=True)

def download(url,path,sha=None):
    if not path.exists():
        print('Downloading',path,flush=True);urllib.request.urlretrieve(url,path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha and actual!=sha:raise ValueError(f'Checksum mismatch for {path}')
    print(path,actual,flush=True)

if __name__=='__main__':
    download('https://github.com/chrisyeh96/conformal-risk-training/releases/download/v1.0.0/polyps_data.zip',ROOT/'polyps_data.zip',DATA_SHA)
    download('https://github.com/chrisyeh96/conformal-risk-training/releases/download/v1.0.0/PraNet-19.pth',ROOT/'PraNet-19.pth','1e178247113e4b870d2ce1de6b36ef5dcb042a283703d455487d870635c8edf2')
    target=ROOT/'upstream/polyps';target.mkdir(parents=True,exist_ok=True)
    for name in ['PraNet_Res2Net.py','Res2Net_v1b.py']:
        download(f'https://raw.githubusercontent.com/chrisyeh96/conformal-risk-training/{REV}/polyps/{name}',target/name)
    # Release names are relative to extraction directory; reject traversal.
    with zipfile.ZipFile(ROOT/'polyps_data.zip') as z:
        dest=(ROOT/'polyps_data').resolve()
        for name in z.namelist():
            if not (dest/name).resolve().is_relative_to(dest):raise ValueError('Unsafe archive path')
        z.extractall(dest)
    print('Assets obtained for research use. Check upstream terms before other uses.')
    # Published transformer replication; the checkpoint includes the full backbone.
    pvt_rev='cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4'
    pvt=ROOT/'upstream_pvt/lib';pvt.mkdir(parents=True,exist_ok=True)
    for name in ['pvt.py','pvtv2.py']:
        download(f'https://raw.githubusercontent.com/DengPingFan/Polyp-PVT/{pvt_rev}/lib/{name}',pvt/name)
    source=pvt/'pvt.py';s=source.read_text()
    marker="        path = './pretrained_pth/pvt_v2_b2.pth'"
    if marker in s:
        start=s.index(marker)
        end=s.index('\n',s.index('self.backbone.load_state_dict(model_dict)',start))
        source.write_text(s[:start]+s[end:])
    import gdown
    checkpoint=ROOT/'PolypPVT.pth'
    if not checkpoint.exists():gdown.download(id='1l60Nzd2EQNZWbMHkvItDUUUp6KTlD0vt',output=str(checkpoint),quiet=False)
    download('',checkpoint,'aef5ac34f7795b85af0ecbb730502db251d416a73a4c6b04a74e30aa45554048')
