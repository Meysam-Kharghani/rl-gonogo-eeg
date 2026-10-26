#!/usr/bin/env python3
"""Adjusted within-subject trial-level RL -> response-EEG encoding using cross-fitted M6 latents.

Crucial provenance rule: embedded RL columns in EEG-derived
tables are discarded. p_press/actor_qdiff/control are merged ONLY from the canonical
nested/cross-fitted M6 trial-latent table before adjusted inference.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, sys
from pathlib import Path
import numpy as np, pandas as pd
PRIMARY_TARGETS=[
'resp_ern_fz_50_150_mean_uv','resp_ern_fz_50_150_min_uv','resp_ern_fz_50_150_max_uv',
'resp_pe_fz_200_400_mean_uv','resp_pe_fz_200_400_max_uv','resp_pe_cz_200_400_mean_uv','resp_pe_cz_200_400_max_uv',
'resp_theta_fz_0_200','resp_theta_fz_0_400','resp_theta_cz_0_400','resp_beta_cz_200_500']

def imp(p,n):
 s=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--trial-features',default=None); ap.add_argument('--crossfitted-latents',default=None); ap.add_argument('--out',default=None); ap.add_argument('--n-perm',type=int,default=5000); ap.add_argument('--k',type=int,default=5); ap.add_argument('--seed',type=int,default=2026)
 a=ap.parse_args(); root=Path(a.project_root).resolve(); sys.path.insert(0,str(root/'shared')); feat=Path(a.trial_features).resolve() if a.trial_features else root/'controlled_inputs/eeg_features/entropy_trial_features.csv'; latp=Path(a.crossfitted_latents).resolve() if a.crossfitted_latents else root/'controlled_inputs/crossfitted_rl/crossfitted_trial_latents.csv'; out=Path(a.out).resolve() if a.out else root/'reproduced/controlled_rerun/robustness/trial_level_crossfitted_rl_eeg'; out.mkdir(parents=True,exist_ok=True)
 for p in [feat,latp]:
  if not p.exists(): raise FileNotFoundError(p)
 mod=imp(root/'shared/eeg/adjusted_trial_level_encoding.py','m22'); d=mod.load_table(feat)
 # purge potentially embedded/full-data computational columns
 purge=[c for c in d.columns if c in {'p_press','control_current','actor_qdiff_current','belief_nogo_current','associability_current'} or c.startswith('latent_')]
 d=d.drop(columns=purge,errors='ignore'); lat=pd.read_csv(latp); lat['subject_id']=lat['subject_id'].astype(str); lat['trial_index_numeric']=pd.to_numeric(lat['trial_index'],errors='raise').astype(int)+1  # EEG tables historically map EEG index - 1 = canonical RL index
 keep=['subject_id','trial_index_numeric','p_press','control_current','actor_qdiff_current','nested_rl_fold','nested_rl_seed','nested_rl_variant','nested_rl_source']; lat=lat[keep].copy(); d['subject_id']=d['subject_id'].astype(str); merged=d.merge(lat,on=['subject_id','trial_index_numeric'],how='left',validate='one_to_one')
 if merged[['p_press','control_current','actor_qdiff_current']].isna().any().any(): raise ValueError('Cross-fitted latent merge incomplete; inspect trial-index convention before proceeding.')
 merged=mod.prepare(merged,block_size=0); targets=[t for t in PRIMARY_TARGETS if t in merged.columns]; miss=[t for t in PRIMARY_TARGETS if t not in merged.columns]
 if miss: raise ValueError('Missing primary response targets: '+', '.join(miss))
 merged.to_csv(out/'crossfitted_trial_analysis_table.csv.gz',index=False,compression='gzip'); inf=mod.run_inference(merged,targets,a.n_perm,a.seed); inf.to_csv(out/'crossfitted_trial_level_inference.csv',index=False); pred=mod.run_grouped_prediction(merged,targets,a.k,a.seed); pred.to_csv(out/'crossfitted_grouped_prediction_folds.csv',index=False); pred.groupby(['target','model'],as_index=False).agg(n_folds=('fold','nunique'),mean_rmse=('rmse','mean'),mean_r2=('r2','mean')).to_csv(out/'crossfitted_grouped_prediction_summary.csv',index=False)
 (out/'analysis_settings.json').write_text(json.dumps({'rl_source':str(latp),'rl_provenance':'canonical nested/cross-fitted M6 only; embedded EEG RL columns purged','targets':PRIMARY_TARGETS,'n_perm':a.n_perm,'k':a.k,'block_size':0,'interpretation':'within-subject adjusted association; not causal'},indent=2),encoding='utf-8'); print('Wrote',out)
if __name__=='__main__': main()
