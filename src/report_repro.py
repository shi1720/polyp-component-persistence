from pathlib import Path
import json,hashlib,platform
root=Path(__file__).resolve().parents[1];a=json.load(open(root/'results/inference.json'));b=json.load(open(root/'results/pvt/inference.json'))
assets={'data_archive':{'url':'https://github.com/chrisyeh96/conformal-risk-training/releases/download/v1.0.0/polyps_data.zip','sha256':'554b9f23ba697083334bc166ebc98273512bc1669ebfe4025153121b7cba0b59'},'pranet_checkpoint':{'url':'https://github.com/chrisyeh96/conformal-risk-training/releases/download/v1.0.0/PraNet-19.pth','sha256':a['weights_sha256']},'pvt_checkpoint':{'url':'https://drive.google.com/uc?id=1l60Nzd2EQNZWbMHkvItDUUUp6KTlD0vt','sha256':b['weights_sha256']},'architecture_revisions':{'conformal-risk-training':'b9b2276e3e1ccf26ab74f2c750e51e091aa409bd','Polyp-PVT':'cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4'},'platform':platform.platform(),'python':platform.python_version()}
(root/'results/asset_manifest.json').write_text(json.dumps(assets,indent=2))
text=r'''The reference environment uses Python 3.12, PyTorch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, Pillow 12.3.0, and timm 1.0.30 on macOS/Apple Silicon. A complete dependency lock file is released. CPU computations use two PyTorch threads; path generation uses two workers. Model inference runs once per retained image. Subsequent splits reuse the same fixed paths and do not refit a model or postprocessor.

'''
text+=f"The maximum absolute CPU--MPS sigmoid differences over the first three images were {max(a['cpu_mps_max_abs_error_first3']):.3g} for PraNet and {max(b['cpu_mps_max_abs_error_first3']):.3g} for Polyp-PVT. Reported inference times, including reference checks but excluding data download and duplicate hashing, were {a['seconds']:.1f} and {b['seconds']:.1f} seconds, respectively. These are reference-run timings, not deployment latency benchmarks.\n\n"
text+=r'''\begin{table}[htbp]\centering\small
\caption{Reference artifact identifiers. Complete hashes and download URLs are in the machine-readable asset manifest.}
\begin{tabular}{ll}\toprule
Asset & SHA-256 prefix / Git revision prefix\\\midrule
Processed data archive & \texttt{554b9f23ba697083}\\
PraNet checkpoint & \texttt{1e178247113e4b87}\\
Polyp-PVT checkpoint & \texttt{aef5ac34f7795b85}\\
Data/architecture source revision & \texttt{b9b2276e3e1ccf26}\\
Polyp-PVT source revision & \texttt{cd71fbf9fcd8c32c}\\\bottomrule
\end{tabular}\end{table}

The artifact separates original code from third-party assets. The download script verifies data and checkpoint hashes, obtains pinned architecture files, and documents the initialization-only patch for Polyp-PVT. A released image manifest fixes ordering and duplicate retention. For each image, method, and grid point, the path files store missed fraction, frame-area fraction, and Dice. These sufficient statistics reproduce every split-level comparison without re-running a neural network. Figure regeneration additionally requires the upstream images and the inference cache. The uniform-probability-grid pilot remains available, including its full-frame outcome, to document the exploratory amendment.
'''
(root/'paper/reproducibility.tex').write_text(text)
print(text[:700])
