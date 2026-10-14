from pathlib import Path
import json,numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];p=r/'paper';s=pd.read_csv(r/'results/summary.csv');d=json.loads((r/'results/details.json').read_text());diag=pd.read_csv(r/'results/diagnostics.csv')
q=s[np.isclose(s.alpha,.1)].set_index('method')
def pct(x):return f'{100*x:.2f}'
def tex(s):return s.replace('%',r'\%').replace('_',r'\_')
original={'CVC-ClinicDB':62,'CVC-ColonDB':380,'ETIS-LaribPolypDB':196,'Kvasir-SEG':100}
table=r'''\begin{table}[htbp]
\centering\small
\caption{Released test images and retained distinct images. Path diagnostics use the primary eight-connected, transformed 257-point threshold grid.}
\label{tab:data}
\begin{tabular}{lrrrr}\toprule
Source & Released & Retained & Nonnested LCC & Loss increase\\\midrule
'''
for row in diag.itertuples():table+=f'{row.dataset} & {original[row.dataset]} & {row.n} & {row.nonnested} & {row.risk_increase} \\\\\n'
table+=f'\\midrule Total & 738 & 701 & {d["nonnested"]} & {d["risk_increase"]} \\\\\n'+r'\bottomrule\end{tabular}\end{table}'
(p/'data_table.tex').write_text(table)
table=r'''\begin{table}[htbp]\centering\small
\caption{PraNet primary evaluation at target 10\% missed annotated area. Values average 500 shared splits, with 350 calibration and 351 evaluation images. The range is the 2.5th--97.5th percentile of split risks, not a population confidence interval. All numerical columns except Dice are percentages.}
\label{tab:primary}
\begin{tabular}{lrrrr}\toprule
Procedure & Missed area & Split range & Frame area & Dice\\\midrule
'''
for name,row in q.iterrows():table+=f'{name} & {pct(row.risk)} & {pct(row.risk_p025)}--{pct(row.risk_p975)} & {pct(row.area)} & {row.dice:.3f} \\\\\n'
table+=r'\bottomrule\end{tabular}\end{table}'
(p/'primary_table.tex').write_text(table)
raw=q.loc['Raw CRC'];lcc=q.loc['Raw CRC + LCC'];pref=q.loc['Prefix union CRC'];env=q.loc['LCC envelope CRC'];naive=q.loc['LCC recalibration'];anchor=q.loc['Anchor CRC'];opening=q.loc['Area opening CRC']
text=f'''The primary transformed grid revealed nonnested LCC paths in {d['nonnested']}/701 images ({100*d['nonnested']/701:.1f}\\%) and adjacent increases in annotated-area loss in {d['risk_increase']}/701 ({100*d['risk_increase']/701:.1f}\\%; Table~\\ref{{tab:data}}). The two quantities differ because deleted pixels need not intersect the reference annotation. Figure~\\ref{{fig:mechanism}} isolates the mechanism in an exact synthetic example.

\\begin{{figure}}[htbp]\\centering\\includegraphics[width=.94\\linewidth]{{../figures/mechanism.pdf}}\\caption{{A synthetic switch between two disconnected regions. Lowering the threshold adds pixels to the raw mask but replaces its largest component. The two-pixel reference annotation is then lost. The running union retains it. Colors distinguish annotated foreground from other selected pixels.}}\\label{{fig:mechanism}}\\end{{figure}}

\\subsection{{Cleanup after raw calibration}}
Across the shared splits, raw CRC attained mean missed area {pct(raw.risk)}\\%, whereas subsequent LCC cleanup increased it to {pct(lcc.risk)}\\% (Table~\\ref{{tab:primary}}). The paired increase was {100*(lcc.risk-raw.risk):.2f} percentage points. Cleanup removed more than 1\\% of annotated area in a mean {pct(raw.cleanup_gt1pct)}\\% of evaluation images. The predicted frame area changed from {pct(raw.area)}\\% to {pct(lcc.area)}\\%, a decrease in displayed area accompanied by a substantial recall cost. These are properties of the sampled images, not estimates of clinical polyp miss rate.

At the descriptive full-pool calibrated threshold $t={d['fixed_threshold']:.3g}$, the mean per-image cleanup penalty was {100*d['fixed_threshold_cleanup_penalty']:.2f} percentage points (image-bootstrap 95\\% interval {100*d['image_bootstrap_95'][0]:.2f}--{100*d['image_bootstrap_95'][1]:.2f}). In {d['fixed_threshold_cleanup_gt1pct']} images the penalty exceeded one percentage point. This threshold uses the entire pool and is not a held-out operating-point estimate.

\\input{{primary_table.tex}}
\\begin{{figure}}[htbp]\\centering\\includegraphics[width=\\linewidth]{{../figures/primary.pdf}}\\caption{{PraNet evaluation at a 10\\% target. Points are split-mean risks; horizontal lines show the central 95\\% of risks across overlapping random splits. Bars report mean predicted frame area. The dashed line marks the target.}}\\label{{fig:primary}}\\end{{figure}}

\\subsection{{Recalibration and output constraints}}
Persistence CRC attained mean missed area {pct(pref.risk)}\\% with {pct(pref.area)}\\% frame area. The established loss-envelope method attained {pct(env.risk)}\\% with {pct(env.area)}\\% frame area while returning an LCC mask. Naive LCC recalibration attained {pct(naive.risk)}\\%; its empirical performance does not supply a general guarantee for nonmonotone paths. The anchor and area-opening controls attained {pct(anchor.risk)}\\% and {pct(opening.risk)}\\%, respectively. Figure~\\ref{{fig:examples}} shows two deliberately selected high-penalty cases, chosen by the largest cleanup penalty in distinct source datasets after excluding Kvasir images to avoid acquisition text in the display, rather than representative random images.

\\begin{{figure}}[htbp]\\centering\\includegraphics[width=\\linewidth]{{../figures/examples.pdf}}\\caption{{Two high-penalty PraNet examples at the same descriptive full-pool threshold. Green denotes included annotated area, red missed annotated area, and blue selected background. The prefix union is evaluated at this same threshold for mechanism illustration; its independently calibrated operating point differs. These examples are selected by outcome and do not estimate effect frequency.}}\\label{{fig:examples}}\\end{{figure}}

\\input{{secondary_results.tex}}
'''
(p/'results_text.tex').write_text(text)
v=pd.read_csv(r/'results/pvt/summary.csv');vq=v[np.isclose(v.alpha,.1)].set_index('method')
abstract=f'''\\begin{{abstract}}
Recall calibration for polyp segmentation can be invalidated by a subsequent mask-cleanup operation. We audit largest-connected-component (LCC) filtering, separating deletion of true-positive pixels at a fixed calibrated threshold from nonmonotonicity across thresholds. A running-union component-persistence completion is the smallest nested family containing the LCC outputs on a fixed grid; an established conformal loss-envelope baseline instead retains a single-component output. We evaluate fixed published PraNet and Polyp-PVT models on 701 distinct images from four public polyp datasets after removing 37 exact test repeats. In 500 shared calibration/evaluation splits, PraNet mean missed annotated area increased from {pct(raw.risk)}\\% to {pct(lcc.risk)}\\% when LCC cleanup followed raw-mask calibration at a 10\\% target. Persistence and loss-envelope calibration attained {pct(pref.risk)}\\% and {pct(env.risk)}\\%, with mean output areas {pct(pref.area)}\\% and {pct(env.area)}\\% of the frame. For Polyp-PVT, cleanup increased missed area from {pct(vq.loc['Raw CRC','risk'])}\\% to {pct(vq.loc['Raw CRC + LCC','risk'])}\\%; persistence calibration attained {pct(vq.loc['Prefix union CRC','risk'])}\\%. A classical anchor-component control attained {pct(anchor.risk)}\\% missed area with {pct(anchor.area)}\\% frame area for PraNet. The study releases code, complete threshold-path statistics, and sensitivity analyses. The results support calibrating the final returned mask and measuring cleanup-induced deletion; the guarantees concern expected image-level missed area under exchangeability, rather than clinical detection or resection outcomes.
\\end{{abstract}}
'''
(p/'abstract.tex').write_text(abstract)
print(q.to_string());print(d)
