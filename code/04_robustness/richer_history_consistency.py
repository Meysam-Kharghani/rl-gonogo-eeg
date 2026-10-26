#!/usr/bin/env python3
"""EEG-computational study post-development robustness: richer-history consistency benchmark.

Re-runs the central cue-EEG action comparison using the strictly causal richer_lagged_history
behavioral baseline established in companion behavioral-computational study. Default is the frozen seed2026.
Optionally supply multiple seeds after inspecting seed2026.

Key questions:
1) Does combined cue EEG add beyond richer_lagged_history?
2) Does RL still add beyond richer_lagged_history + EEG?
3) Are actor/control vs policy conclusions materially changed by richer_lagged_history?
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def imp(path,name):
    path=Path(path).expanduser().resolve()
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod


def stable_seed(base,*parts):
    d=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int((base+int.from_bytes(d[:4],"little"))%(2**32-1))


def add_history(d):
    d=d.copy()
    trial="trial_index" if "trial_index" in d.columns else "trial_index_numeric"
    d=d.sort_values(["subj_id_ddm",trial]).reset_index(drop=True)
    action_col="action_press" if "action_press" in d.columns else "press"
    press=pd.to_numeric(d[action_col],errors="coerce").fillna(0).astype(float)
    err=pd.to_numeric(d["error"] if "error" in d.columns else 0,errors="coerce").fillna(0).astype(float)
    nogo=pd.to_numeric(d["is_nogo"],errors="coerce").fillna(0).astype(int)
    for k in [2,3,4,5]:
        d[f"press_lag{k}"]=press.groupby(d.subj_id_ddm).shift(k).fillna(0)
        d[f"error_lag{k}"]=err.groupby(d.subj_id_ddm).shift(k).fillna(0)
    for alpha in [.10,.30]:
        ep=np.zeros(len(d)); ef=np.zeros(len(d))
        for _,g in d.groupby("subj_id_ddm",sort=False):
            idx=g.index.to_numpy(); p=press.loc[idx].to_numpy(); e=err.loc[idx].to_numpy(); ng=nogo.loc[idx].to_numpy()
            mp,mf=.5,.2
            for j,row_idx in enumerate(idx):
                ep[row_idx]=mp; ef[row_idx]=mf
                mp=(1-alpha)*mp+alpha*p[j]
                if ng[j]==1: mf=(1-alpha)*mf+alpha*(1.0 if e[j]>0 else 0.0)
        d[f"press_ewma_a{int(alpha*100)}"]=ep
        d[f"fa_ewma_a{int(alpha*100)}"]=ef
    return d


def metrics(y,p):
    from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    lp=y*np.log(p)+(1-y)*np.log(1-p)
    return {"elpd":float(lp.sum()),"elpd_per_trial":float(lp.mean()),
            "logloss":float(log_loss(y,p,labels=[0,1])),
            "brier":float(brier_score_loss(y,p)),
            "auc":float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,
            "n_trials":len(y)}


def boot(x,n,seed):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    rng=np.random.default_rng(seed)
    b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)])
    return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out-root",default=None)
    ap.add_argument("--seeds",nargs="+",type=int,default=[2026])
    ap.add_argument("--max-rl-draws",type=int,default=300)
    ap.add_argument("--n-bootstrap",type=int,default=10000)
    a=ap.parse_args()

    root=Path(a.project_root).expanduser().resolve()
    outroot=Path(a.out_root).expanduser().resolve() if a.out_root else root/"reproduced/controlled_rerun/robustness/richer_history_consistency"
    outroot.mkdir(parents=True,exist_ok=True)

    nested=imp(root/"shared/rl/nested_crossvalidated_latent_generation.py","richer_history_nested")
    rl=imp(root/"shared/rl/actor_control_model_family.py","richer_history_rl")
    ddm=imp(root/"shared/ddm/one_boundary_diffusion.py","richer_history_ddm")
    eegmod=imp(root/"code/02_primary_cue/cue_action_prediction.py","richer_history_eeg")
    action=imp(root/"shared/action/action_occurrence_benchmark.py","richer_history_action")

    eeg,eegblocks=eegmod.load_eeg_table(root/"controlled_inputs/cue_eeg/canonical_cue_feature_table.csv")
    rlout=root/"controlled_inputs/rl_crossvalidated_posteriors"
    arr=rl.load_artifacts(rlout); meta=rl.get_meta(arr); all_idx=set(range(int(meta["n_subj"])))
    probs=action.cue_map(root/"configs/instructed_cue_probabilities.json")

    base=["is_nogo","cue_up","cue_bi","prev_error"]
    oldhist=["prev_press","prior_press_rate","prior_fa_rate"]
    rich=["press_lag2","press_lag3","press_lag4","press_lag5","error_lag2","error_lag3","error_lag4","error_lag5",
          "press_ewma_a10","press_ewma_a30","fa_ewma_a10","fa_ewma_a30"]
    H=base+oldhist+rich; P=["logit_p_press"]; AC=["actor_qdiff_current","control_current"]; EEG=eegblocks["combined"]
    specs={
        "R0":("richer_lagged_history",H),
        "R1":("richer_lagged_history_plus_policy",H+P),
        "R2":("richer_lagged_history_plus_actor_control",H+AC),
        "R6":("richer_lagged_history_plus_combined_EEG",H+EEG),
        "R7":("richer_lagged_history_plus_policy_plus_EEG",H+P+EEG),
        "R8":("richer_lagged_history_plus_actor_control_plus_EEG",H+AC+EEG),
        "R9":("richer_lagged_history_plus_full_RL_plus_EEG",H+P+AC+EEG),
        "R10":("richer_lagged_history_plus_full_RL",H+P+AC),
    }
    Cs=[.0003,.001,.003,.01,.03,.1,.3,1,3,10]
    all_contrasts=[]

    for seed in a.seeds:
        out=outroot/f"seed_{seed}"; out.mkdir(parents=True,exist_ok=True)
        cv=nested.read_json(nested.cv_summary_path(rlout,seed,"M6"))
        fold_rows=[]; subj_rows=[]; tune_rows=[]; align_rows=[]
        for fr in cv["folds"]:
            fi=int(fr["fold"]); test_idx=[int(x) for x in fr.get("test_subject_indices",[])]
            if not test_idx:
                fr2=nested.read_json(nested.fold_result_path(rlout,seed,"M6",fi),default={})
                test_idx=[int(x) for x in fr2.get("test_subject_indices",[])]
            train_idx=sorted(all_idx-set(test_idx))
            idata=nested.load_rl_fold_idata(rlout,seed,"M6",fi)
            atr,ate=rl.subset_subjects(arr,train_idx),rl.subset_subjects(arr,test_idx)
            ltr=nested.posterior_draws_mean_latents(rl,atr,"M6",idata,a.max_rl_draws,stable_seed(seed,"richer_history_tr",fi))
            lte=nested.posterior_draws_mean_latents(rl,ate,"M6",idata,a.max_rl_draws,stable_seed(seed,"richer_history_te",fi))
            rtr=nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,seed,"M6")
            rte=nested.arr_to_nested_latent_rows(rl,ate,lte,fi,seed,"M6")
            tr=eegmod.load_fold_frame(ddm,rtr,root/"controlled_inputs/behavioral_trial_data",700)
            te=eegmod.load_fold_frame(ddm,rte,root/"controlled_inputs/behavioral_trial_data",700)
            tr,qa=eegmod.merge_eeg(tr,eeg,EEG); te,qb=eegmod.merge_eeg(te,eeg,EEG)
            tr=eegmod.prepare_action_frame(tr,False); te=eegmod.prepare_action_frame(te,False)
            tr=add_history(tr); te=add_history(te)
            align_rows += [{"fold":fi,"split":"train",**qa},{"fold":fi,"split":"test",**qb}]
            ytr=pd.to_numeric(tr.action_press).astype(int); yte=pd.to_numeric(te.action_press).astype(int)

            for mk,(name,cols) in specs.items():
                C=float(action.tune_C(tr,cols,Cs,4,stable_seed(seed,"richer_history_C",fi,mk)))
                model=action.make_model(C,stable_seed(seed,"richer_history_fit",fi,mk))
                model.fit(tr[cols],ytr); p=model.predict_proba(te[cols])[:,1]
                fold_rows.append({"seed":seed,"fold":fi,"model":mk,"model_name":name,"C":C,"n_features":len(cols),**metrics(yte,p)})
                tune_rows.append({"seed":seed,"fold":fi,"model":mk,"C":C})
                tmp=pd.DataFrame({"subject_id":te.subj_id_ddm.astype(str),"y":yte.values,"p":p})
                for sid,g in tmp.groupby("subject_id",sort=False):
                    subj_rows.append({"seed":seed,"fold":fi,"subject_id":sid,"model":mk,**metrics(g.y,g.p)})

        f=pd.DataFrame(fold_rows); s=pd.DataFrame(subj_rows)
        f.to_csv(out/"richer_history_eeg_fold_scores.csv",index=False)
        s.to_csv(out/"richer_history_eeg_subject_scores.csv",index=False)
        pd.DataFrame(tune_rows).to_csv(out/"richer_history_eeg_tuning.csv",index=False)
        pd.DataFrame(align_rows).to_csv(out/"richer_history_eeg_alignment_qc.csv",index=False)

        planned=[
            ("R6","R0","combined cue EEG beyond richer lagged behavioral history"),
            ("R7","R1","combined cue EEG beyond richer lagged behavioral history+policy"),
            ("R8","R2","combined cue EEG beyond richer lagged behavioral history+actor/control"),
            ("R9","R6","total RL beyond richer lagged behavioral history + EEG"),
            ("R9","R7","actor/control beyond richer lagged behavioral history + policy + EEG"),
            ("R9","R8","policy beyond richer lagged behavioral history + actor/control + EEG"),
            ("R10","R0","total RL beyond richer lagged behavioral history"),
            ("R9","R10","combined cue EEG beyond richer lagged behavioral history+fullRL"),
        ]
        wide=s.pivot_table(index=["fold","subject_id"],columns="model",values="elpd")
        cr=[]
        for i,(m,r,label) in enumerate(planned):
            x=(wide[m]-wide[r]).dropna().to_numpy()
            mean,lo,hi=boot(x,a.n_bootstrap,stable_seed(seed,"richer_history_contrast",i))
            cr.append({"seed":seed,"model":m,"reference":r,"label":label,"n_subjects":len(x),
                       "mean_subject_delta_elpd":mean,"ci_low":lo,"ci_high":hi,
                       "positive_subjects":int((x>0).sum()),"total_delta_elpd":float(x.sum())})
        c=pd.DataFrame(cr); c.to_csv(out/"richer_history_eeg_planned_contrasts.csv",index=False)
        all_contrasts.append(c)

    C=pd.concat(all_contrasts,ignore_index=True)
    C.to_csv(outroot/"all_seed_contrasts.csv",index=False)
    if len(a.seeds)>1:
        summ=C.groupby(["model","reference","label"],as_index=False).agg(
            n_seeds=("seed","nunique"),mean_delta=("mean_subject_delta_elpd","mean"),
            min_delta=("mean_subject_delta_elpd","min"),max_delta=("mean_subject_delta_elpd","max"),
            positive_seeds=("mean_subject_delta_elpd",lambda x:int((x>0).sum()))
        )
        summ.to_csv(outroot/"seed_stability_summary.csv",index=False)

    (outroot/"analysis_settings.json").write_text(json.dumps({
        "status":"cross-paper consistency robustness analysis","seeds":a.seeds,"max_rl_draws":a.max_rl_draws,
        "history_features":rich,"EEG_features":EEG,"models":specs
    },indent=2),encoding="utf-8")
    print(C.to_string(index=False))
    print("Wrote",outroot)


if __name__=="__main__":
    main()
