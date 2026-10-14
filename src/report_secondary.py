from pathlib import Path
import pandas as pd,numpy as np,json
r=Path(__file__).resolve().parents[1];p=r/'paper';s=pd.read_csv(r/'results/summary.csv');v=pd.read_csv(r/'results/pvt/summary.csv');d=json.loads((r/'results/pvt/details.json').read_text());base=s[np.isclose(s.alpha,.1)].set_index('method');q=v[np.isclose(v.alpha,.1)].set_index('method')
def pct(x):return f'{100*x:.2f}'
text='\\subsection{Transformer replication and sensitivity analyses}\n'
text+=f"On Polyp-PVT, the same intervention changed mean missed area from {pct(q.loc['Raw CRC','risk'])}\\% to {pct(q.loc['Raw CRC + LCC','risk'])}\\%. Persistence, envelope, and anchor calibration attained {pct(q.loc['Prefix union CRC','risk'])}\\%, {pct(q.loc['LCC envelope CRC','risk'])}\\%, and {pct(q.loc['Anchor CRC','risk'])}\\%, respectively. The transformer had {d['nonnested']}/701 nonnested LCC paths and {d['risk_increase']}/701 paths with an annotated-loss increase. The model comparison changes the magnitude of the effect; neither model is selected on these results.\n\n"
text+=r'''\begin{table}[htbp]\centering\small
\caption{Polyp-PVT replication at the 10\% target, with the same images and 500 paired splits. All numerical columns except Dice are percentages; split ranges are partition sensitivity.}
\label{tab:pvt}\begin{tabular}{lrrrr}\toprule
Procedure & Missed area & Split range & Frame area & Dice\\\midrule
'''
for name,row in q.iterrows():text+=f'{name} & {pct(row.risk)} & {pct(row.risk_p025)}--{pct(row.risk_p975)} & {pct(row.area)} & {row.dice:.3f} \\\\\n'
text+=r'\bottomrule\end{tabular}\end{table}'+'\n\n'
# Explicitly report when heuristic recalibration coincides with the valid envelope.
for label,path in [('PraNet',r/'results/splits.csv'),('Polyp-PVT',r/'results/pvt/splits.csv')]:
 z=pd.read_csv(path);z=z[np.isclose(z.alpha,.1)].pivot(index='rep',columns='method',values='index');same=(z['LCC recalibration']==z['LCC envelope CRC']).sum()
 text+=f'At the primary target, naive LCC recalibration and envelope calibration selected the same grid index in {same}/500 {label} splits. '
text+='Naive recalibration had mean Polyp-PVT risk 10.11\\%, slightly above the 10\\% target; the split variability and finite Monte Carlo experiment do not establish a systematic target violation. Consequently, the experiment demonstrates failure of transferring raw-mask calibration through cleanup, but does not establish that naive final-mask recalibration fails empirically at this target.\n\n'
for label,path in [('four-connectivity',r/'results/connectivity4/summary.csv'),('513-point grid',r/'results/grid513/summary.csv')]:
 z=pd.read_csv(path);z=z[np.isclose(z.alpha,.1)].set_index('method');text+=f"For PraNet with {label}, raw/cleaned mean missed areas were {pct(z.loc['Raw CRC','risk'])}\\%/{pct(z.loc['Raw CRC + LCC','risk'])}\\%; persistence/envelope/anchor risks were {pct(z.loc['Prefix union CRC','risk'])}\\%/{pct(z.loc['LCC envelope CRC','risk'])}\\%/{pct(z.loc['Anchor CRC','risk'])}\\%. "
text+='The target sensitivity results and source-held-out stress results are included in Appendix~\\ref{app:additional}.\n\n'
text+=f"The classical anchor control used less mean frame area than persistence in the primary PraNet comparison ({pct(base.loc['Anchor CRC','area'])}\\% versus {pct(base.loc['Prefix union CRC','area'])}\\%) and attained a higher mean Dice ({base.loc['Anchor CRC','dice']:.3f} versus {base.loc['Prefix union CRC','dice']:.3f}). Thus the running-union construction is useful for preserving a display-history constraint, but the data do not support preferring it over established increasing filters solely for segmentation utility.\n"
(p/'secondary_results.tex').write_text(text)
# Target sweep, both models, compact table.
a=r'''\section{Additional benchmark results}\label{app:additional}
\begin{table}[htbp]\centering\small
\caption{Mean missed annotated area (\%) across shared splits at three targets. P denotes PraNet and V denotes Polyp-PVT. No entry is a per-patient result.}
\begin{tabular}{lrrrrrr}\toprule
 & P: 5\% & P: 10\% & P: 20\% & V: 5\% & V: 10\% & V: 20\%\\\midrule
'''
for name in base.index:
 vals=[float(z[(z.method==name)&np.isclose(z.alpha,alpha)].iloc[0].risk) for z in [s,v] for alpha in [.05,.1,.2]]
 a+=name+' & '+' & '.join(pct(x) for x in vals)+r' \\'+'\n'
a+=r'\bottomrule\end{tabular}\end{table}'+'\n'
for model,folder in [('PraNet','results'),('Polyp-PVT','results/pvt')]:
 z=pd.read_csv(r/folder/'source_stress.csv')
 a+=r'\begin{table}[htbp]\centering\small'+f'\\caption{{{model} source-held-out stress: missed annotated area (\\%) when calibrating at 10\\% on the other sources. Source shift is outside the exchangeability guarantee.}}\n'+r'\begin{tabular}{lrrrr}\toprule'+'\nProcedure & ClinicDB & ColonDB & ETIS & Kvasir\\\\\\midrule\n'
 for name in base.index:
  vals=[float(z[(z.method==name)&(z.dataset==ds)].iloc[0].risk) for ds in ['CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','Kvasir-SEG']]
  a+=name+' & '+' & '.join(pct(x) for x in vals)+r' \\'+'\n'
 a+=r'\bottomrule\end{tabular}\end{table}'+'\n'
a+='Source-dependent risks can exceed the nominal target even for mathematically valid procedures when calibration and evaluation distributions differ. These stress results do not extend the finite-pool guarantee to clinical populations.\n'
(p/'additional_results.tex').write_text(a)
print(q.to_string());print(d)
