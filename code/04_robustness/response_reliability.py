#!/usr/bin/env python3
"""Reliability of EEG-computational study participant-level response contrasts.

Primary: 11 pre-existing ERP/post-response-TF response-monitoring features.
Secondary: global theta/beta fractions. Uses repeated random split-halves plus a
deterministic early-vs-late trial-order sensitivity; no assumed experimental blocks.
"""
from __future__ import annotations
import argparse, importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
PRIMARY=[
'resp_ern_fz_50_150_mean_uv','resp_ern_fz_50_150_min_uv','resp_ern_fz_50_150_max_uv',
'resp_pe_fz_200_400_mean_uv','resp_pe_fz_200_400_max_uv','resp_pe_cz_200_400_mean_uv','resp_pe_cz_200_400_max_uv',
'resp_theta_fz_0_200','resp_theta_fz_0_400','resp_theta_cz_0_400','resp_beta_cz_200_500']
SECONDARY=['resp_global_theta_fraction','resp_global_beta_fraction']

def imp(path,name):
 spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def sb(r): return np.nan if not np.isfinite(r) or np.isclose(1+r,0) else float(2*r/(1+r))

def temporal(d,features,thresholds):
 rows=[]
 for f in features:
  for th in thresholds:
   a=[];b=[]
   for sid,g in d.sort_values('trial_index_numeric').groupby('subject_id'):
    halves=[]
    for cond in ['nogo_fa','go_hit']:
     z=g[g.outcome.eq(cond)].copy(); z[f]=pd.to_numeric(z[f],errors='coerce'); z=z.dropna(subset=[f])
     if len(z)<th or len(z)<4: halves=[]; break
     k=len(z)//2; halves.append((z.iloc[:k][f].mean(),z.iloc[k:][f].mean()))
    if len(halves)==2: a.append(halves[0][0]-halves[1][0]); b.append(halves[0][1]-halves[1][1])
   if len(a)>=8 and np.std(a)>0 and np.std(b)>0:
    rows.append({'feature':f,'min_trials_each_condition':th,'n_subjects':len(a),'sb_pearson_early_late':sb(pearsonr(a,b).statistic),'sb_spearman_early_late':sb(spearmanr(a,b).statistic)})
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--erp-features',default=None); ap.add_argument('--tf-features',default=None); ap.add_argument('--entropy-features',default=None); ap.add_argument('--out',default=None); ap.add_argument('--reps',type=int,default=2000); ap.add_argument('--seed',type=int,default=2026)
 a=ap.parse_args(); root=Path(a.project_root).resolve(); sys.path.insert(0,str(root/'shared'))
 paths=[]
 for val,default in [(a.erp_features,root/'controlled_inputs/eeg_features/erp_trial_features.csv'),(a.tf_features,root/'controlled_inputs/eeg_features/time_frequency_trial_features.csv')]:
  p=Path(val).resolve() if val else default
  if not p.exists(): raise FileNotFoundError(p)
  paths.append(p)
 ep=Path(a.entropy_features).resolve() if a.entropy_features else root/'controlled_inputs/eeg_features/entropy_trial_features.csv'
 if ep.exists(): paths.append(ep)
 out=Path(a.out).resolve() if a.out else root/'reproduced/controlled_rerun/robustness/response_reliability'; out.mkdir(parents=True,exist_ok=True)
 mod=imp(root/'shared/eeg/reliability.py','rel24'); d=mod.merge(paths)
 features=[f for f in PRIMARY+SECONDARY if f in d.columns]; miss=[f for f in PRIMARY if f not in d.columns]
 if miss: raise ValueError('Missing primary reliability features: '+', '.join(miss))
 raw,summ=mod.run(d,features,[5,10,15,20],a.reps,a.seed); raw.to_csv(out/'split_half_repetitions.csv.gz',index=False,compression='gzip'); summ.to_csv(out/'split_half_summary.csv',index=False)
 temp=temporal(d,features,[5,10,15,20]); temp.to_csv(out/'early_late_temporal_stability.csv',index=False)
 summ.assign(family=lambda x:np.where(x.feature.isin(PRIMARY),'primary_erp_tf','secondary_global_fraction')).to_csv(out/'reliability_summary_labeled.csv',index=False)
 (out/'analysis_settings.json').write_text(json.dumps({'primary_features':PRIMARY,'secondary_features':SECONDARY,'thresholds':[5,10,15,20],'reps':a.reps,'seed':a.seed,'temporal_sensitivity':'within-condition trial-order halves; no assumed blocks'},indent=2),encoding='utf-8')
 print('Wrote',out)
if __name__=='__main__': main()
