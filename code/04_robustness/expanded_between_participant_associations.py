#!/usr/bin/env python3
"""Selection-robust EEG-computational study between-participant RL–EEG association analysis: 11 ERP/TF features x 3 RL predictors.

The neural universe is fixed by pre-existing response-monitoring feature definitions,
not by significance. Primary inference is group-adjusted Freedman-Lane with BH over
all 33 tests. Global theta/beta fractions are not in the primary family.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, sys
from pathlib import Path
import numpy as np, pandas as pd
FEATURES=[
'resp_ern_fz_50_150_mean_uv','resp_ern_fz_50_150_min_uv','resp_ern_fz_50_150_max_uv',
'resp_pe_fz_200_400_mean_uv','resp_pe_fz_200_400_max_uv','resp_pe_cz_200_400_mean_uv','resp_pe_cz_200_400_max_uv',
'resp_theta_fz_0_200','resp_theta_fz_0_400','resp_theta_cz_0_400','resp_beta_cz_200_500']
PREDS=['mean_p_press','mean_actor_qdiff_current','mean_control_current']

def imp(path,name):
 spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def norm_subject(s): return s.astype(str).str.strip()

def load_merge(erp,tf):
 e=pd.read_csv(erp); t=pd.read_csv(tf)
 for d in [e,t]:
  if 'subject_id' not in d.columns and 'subject_id_norm' in d.columns: d.rename(columns={'subject_id_norm':'subject_id'},inplace=True)
  d['subject_id']=norm_subject(d['subject_id'])
  if 'trial_index_numeric' not in d.columns:
   c='trial_index' if 'trial_index' in d.columns else 'trial_ordinal'; d['trial_index_numeric']=pd.to_numeric(d[c],errors='coerce')
 keys=['subject_id','trial_index_numeric']; keep=keys+[c for c in t.columns if c not in e.columns]
 return e.merge(t[keep],on=keys,how='left',validate='one_to_one')

def infer_outcome(d):
 if 'outcome' in d.columns:return d.outcome.astype(str)
 trial=d.get('trial_class','').astype(str).str.lower(); press=pd.to_numeric(d.get('response_present',0),errors='coerce').fillna(0).astype(int)
 out=np.where((trial=='nogo')&(press==1),'nogo_fa',np.where((trial=='nogo')&(press==0),'nogo_cr',np.where((trial=='go')&(press==1),'go_hit','go_miss')))
 return pd.Series(out,index=d.index)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--erp-features',default=None); ap.add_argument('--tf-features',default=None); ap.add_argument('--rl-subject-summary',default=None); ap.add_argument('--out',default=None); ap.add_argument('--min-trials',type=int,default=10); ap.add_argument('--n-perm',type=int,default=10000); ap.add_argument('--n-boot',type=int,default=5000); ap.add_argument('--seed',type=int,default=2026)
 a=ap.parse_args(); root=Path(a.project_root).resolve()
 erp=Path(a.erp_features).resolve() if a.erp_features else root/'controlled_inputs/eeg_features/erp_trial_features.csv'; tf=Path(a.tf_features).resolve() if a.tf_features else root/'controlled_inputs/eeg_features/time_frequency_trial_features.csv'; rlp=Path(a.rl_subject_summary).resolve() if a.rl_subject_summary else root/'controlled_inputs/crossfitted_rl/participant_rl_behavior_summary.csv'
 for p in [erp,tf,rlp]:
  if not p.exists(): raise FileNotFoundError(p)
 out=Path(a.out).resolve() if a.out else root/'reproduced/controlled_rerun/robustness/expanded_between_participant_associations'; out.mkdir(parents=True,exist_ok=True)
 d=load_merge(erp,tf); d['outcome']=infer_outcome(d); miss=[f for f in FEATURES if f not in d.columns]
 if miss: raise ValueError('Missing frozen features: '+', '.join(miss))
 rows=[]
 for sid,g in d.groupby('subject_id'):
  fa=g[g.outcome.eq('nogo_fa')]; hi=g[g.outcome.eq('go_hit')]
  if len(fa)<a.min_trials or len(hi)<a.min_trials: continue
  r={'subject_id':str(sid),'n_fa':len(fa),'n_hit':len(hi)}
  for f in FEATURES:r[f]=pd.to_numeric(fa[f],errors='coerce').mean()-pd.to_numeric(hi[f],errors='coerce').mean()
  rows.append(r)
 contrasts=pd.DataFrame(rows); rl=pd.read_csv(rlp); rl['subject_id']=norm_subject(rl['subject_id']); dat=rl.merge(contrasts,on='subject_id',how='inner',validate='one_to_one'); dat.to_csv(out/'broad_association_analysis_table.csv',index=False)
 m37=imp(root/'code/03_primary_response/between_participant_rl_eeg_associations.py','m37')
 results=[]
 for fi,f in enumerate(FEATURES):
  for pi,p in enumerate(PREDS):
   x,y,X,Z,j,nuis=m37.prepare_design(dat,f,p,['group_high'],winsor=False); obs=m37.ols_from_arrays(y,X,j); fl,nperm_valid=m37.freedman_lane_p(x,y,X,Z,j,obs['t'],a.n_perm,np.random.default_rng(a.seed+fi*100+pi),True); lo,hi,nboot_valid=m37.bootstrap_ci(dat,f,p,['group_high'],False,a.n_boot,np.random.default_rng(a.seed+10000+fi*100+pi)); loo=m37.loo_summary(dat,f,p,['group_high'],False); lv=loo.pop('_loo_vals',np.array([])); loo['loo_same_direction_fraction']=float(np.mean(np.sign(lv)==np.sign(obs['beta']))) if len(lv) and obs['beta']!=0 else np.nan
   results.append({'feature':f,'predictor':p,'n_subjects':len(x),'standardized_beta':obs['beta'],'se':obs['se'],'t_value':obs['t'],'freedman_lane_p':fl,'n_permutations_valid':nperm_valid,'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'n_bootstrap_valid':nboot_valid,'predictor_vif':m37.predictor_vif(X,j),'design_condition_number':m37.condition_number(X),**loo})
 res=pd.DataFrame(results); res['q_fdr_bh_33']=m37.fdr_bh(res.freedman_lane_p); res=res.sort_values(['q_fdr_bh_33','freedman_lane_p']); res.to_csv(out/'broad_33_test_association_family.csv',index=False)
 (out/'analysis_settings.json').write_text(json.dumps({'features':FEATURES,'predictors':PREDS,'n_tests':33,'nuisance':['group_high'],'permutation':'Freedman-Lane within group','n_perm':a.n_perm,'n_boot':a.n_boot,'BH_family':33,'min_trials':a.min_trials},indent=2),encoding='utf-8')
 print(res.to_string(index=False)); print('Wrote',out)
if __name__=='__main__': main()
