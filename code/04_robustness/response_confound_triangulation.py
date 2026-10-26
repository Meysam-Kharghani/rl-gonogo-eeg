#!/usr/bin/env python3
"""Confound-aware EEG-computational study response triangulation using pre-existing ERP/TF features.

Primary response family is fixed to 11 ERP/post-response-TF features. Matching is
within subject and cue and uses NO assumed 4x95 block structure. Results remain
performance-monitoring evidence; the task cannot identify a pure response-locked
error effect because FA is NoGo/error whereas Hit is Go/correct.
"""
from __future__ import annotations
import argparse, importlib.util, sys, json
from pathlib import Path
import pandas as pd, numpy as np
PRIMARY_RESPONSE=[
'resp_ern_fz_50_150_mean_uv','resp_ern_fz_50_150_min_uv','resp_ern_fz_50_150_max_uv',
'resp_pe_fz_200_400_mean_uv','resp_pe_fz_200_400_max_uv','resp_pe_cz_200_400_mean_uv','resp_pe_cz_200_400_max_uv',
'resp_theta_fz_0_200','resp_theta_fz_0_400','resp_theta_cz_0_400','resp_beta_cz_200_500']

def imp(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def first_existing(cands):
    for p in cands:
        if p.exists(): return p
    raise FileNotFoundError('None of these files exists:\n'+'\n'.join(map(str,cands)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--erp-features',default=None); ap.add_argument('--tf-features',default=None); ap.add_argument('--out',default=None); ap.add_argument('--n-perm',type=int,default=10000); ap.add_argument('--n-bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=2026)
    a=ap.parse_args(); root=Path(a.project_root).resolve(); sys.path.insert(0,str(root/'shared'))
    erp=Path(a.erp_features).resolve() if a.erp_features else first_existing([root/'controlled_inputs/eeg_features/erp_trial_features.csv'])
    tf=Path(a.tf_features).resolve() if a.tf_features else first_existing([root/'controlled_inputs/eeg_features/time_frequency_trial_features.csv'])
    out=Path(a.out).resolve() if a.out else root/'reproduced/controlled_rerun/robustness/response_confound_triangulation'; out.mkdir(parents=True,exist_ok=True)
    mod=imp(root/'shared/eeg/response_matching_and_factorial.py','m21')
    d=mod.load_merge(erp,tf); d=mod.add_trial_history(d,'subject_id','trial_index_numeric',block_size=0); d['cue_type']=d['cue_type'].astype(str).str.lower(); d['log_rt']=np.log(pd.to_numeric(d.get('rt_response_from_target_ms'),errors='coerce').where(lambda x:x>0))
    response=[c for c in PRIMARY_RESPONSE if c in d.columns]; missing=[c for c in PRIMARY_RESPONSE if c not in d.columns]
    if missing: raise ValueError('Missing frozen response features: '+', '.join(missing))
    target=[c for c in mod.DEFAULT_TARGET_FEATURES if c in d.columns]; cue=[c for c in mod.DEFAULT_CUE_FEATURES if c in d.columns]
    summaries=[]; balances=[]
    for cal in [0.25,0.50,0.75]:
        od=out/f'rt_caliper_{cal:.2f}sd'; od.mkdir(exist_ok=True)
        p,s,b=mod.matched_analysis(d[d.outcome.isin(['nogo_fa','go_hit'])],'nogo_fa','go_hit',response,
            ['log_rt','trial_progress','prev_error','prev_press'],['subject_id','cue_type'],'log_rt',cal,a.n_perm,a.n_bootstrap,a.seed+int(cal*100))
        p.to_csv(od/'matched_fa_hit_pairs.csv.gz',index=False,compression='gzip'); s.to_csv(od/'matched_fa_hit_effects.csv',index=False); b.to_csv(od/'matched_fa_hit_balance.csv',index=False)
        s['rt_caliper_sd']=cal; summaries.append(s); b['rt_caliper_sd']=cal; balances.append(b)
    crp,crs,crb=mod.matched_analysis(d[d.outcome.isin(['nogo_cr','nogo_fa'])],'nogo_cr','nogo_fa',target+cue,
        ['trial_progress','prev_error','prev_press'],['subject_id','cue_type'],None,0.5,a.n_perm,a.n_bootstrap,a.seed+101)
    crp.to_csv(out/'matched_nogo_cr_fa_pairs.csv.gz',index=False,compression='gzip'); crs.to_csv(out/'matched_nogo_cr_fa_effects.csv',index=False); crb.to_csv(out/'matched_nogo_cr_fa_balance.csv',index=False)
    fac=mod.factorial_models(d,target+cue); fac.to_csv(out/'target_cue_factorial_models.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(out/'response_matched_all_calipers.csv',index=False); pd.concat(balances,ignore_index=True).to_csv(out/'response_matching_balance_all_calipers.csv',index=False)
    (out/'analysis_settings.json').write_text(json.dumps({'primary_response_features':PRIMARY_RESPONSE,'rt_calipers_sd':[.25,.5,.75],'block_size':0,'exact_match':['subject_id','cue_type'],'n_perm':a.n_perm,'n_bootstrap':a.n_bootstrap,'interpretation':'performance monitoring; not pure error identification'},indent=2),encoding='utf-8')
    print('Wrote',out)
if __name__=='__main__': main()
