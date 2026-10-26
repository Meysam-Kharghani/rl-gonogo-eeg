#!/usr/bin/env python3
"""Strict fold-aligned RL -> cue-EEG action benchmark for EEG-computational study."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def import_py(path, name):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(base, *parts):
    payload = "|".join(map(str, parts)).encode()
    digest = hashlib.sha256(payload).digest()
    return int((base + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def safe_logit(x, eps=1e-5):
    z = np.clip(np.asarray(x, float), eps, 1-eps)
    return np.log(z/(1-z))


def ensure_dir(path):
    p = Path(path).expanduser().resolve(); p.mkdir(parents=True, exist_ok=True); return p


def write_json(path, obj):
    def conv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, dict): return {str(k): conv(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        return x
    Path(path).write_text(json.dumps(conv(obj), indent=2), encoding="utf-8")


def load_eeg_table(path):
    eeg = pd.read_csv(Path(path).expanduser().resolve())
    req = {"subject_id","trial_index","cue_type","action_press"}
    miss = sorted(req - set(eeg.columns))
    if miss: raise ValueError(f"Canonical EEG table missing: {miss}")
    if eeg.duplicated(["subject_id","trial_index"]).any():
        raise ValueError("Duplicate subject/trial rows in EEG table")
    erp = sorted(c for c in eeg.columns if c.startswith("erp__"))
    tf = sorted(c for c in eeg.columns if c.startswith("tf__"))
    ent = sorted(c for c in eeg.columns if c.startswith("entropy__"))
    if not erp or not tf or not ent:
        raise ValueError(f"Empty EEG block: ERP={len(erp)}, TF={len(tf)}, entropy={len(ent)}")
    forbidden = ("resp_","response","target_","action_press","outcome","correct","error","fa","p_press","control_current","actor_qdiff_current","belief_nogo_current","associability_current","policy_entropy","action_surprise","trial_class_surprise")
    feats = erp + tf + ent
    flagged = [c for c in feats if any(t in c.lower() for t in forbidden)]
    if flagged: raise ValueError("Forbidden EEG predictor names: " + ", ".join(flagged))
    return eeg, {"erp":erp,"tf":tf,"entropy":ent,"combined":feats}


def load_fold_frame(ddm, rows, data_root, deadline_ms):
    with tempfile.TemporaryDirectory(prefix="cue_window_action_") as td:
        fp = Path(td)/"latents.csv"
        pd.DataFrame(rows).to_csv(fp, index=False)
        raw = ddm.load_latents(fp, deadline_ms=float(deadline_ms), data_root=Path(data_root))
    return raw.reset_index(drop=True)


def detect(df, candidates, what):
    for c in candidates:
        if c in df.columns: return c
    raise ValueError(f"Cannot detect {what} column")


def norm_text(x): return str(x).strip().lower()


def alignment_score(raw, eeg, off):
    sid = detect(raw,["subj_id_ddm","subject_id","subject_id_norm","subj_id"],"subject")
    tr = detect(raw,["trial_index","trial_index_numeric","trial_ordinal"],"trial")
    cue = next((c for c in ["cue_label_ddm","cue_type","cue_type_norm"] if c in raw.columns), None)
    act = next((c for c in ["action_press","response_present"] if c in raw.columns), None)
    x = raw[[sid,tr] + ([cue] if cue else []) + ([act] if act and act != cue else [])].copy()
    x["_sid"] = x[sid].astype(str); x["_trial"] = pd.to_numeric(x[tr]) + off
    r = eeg[["subject_id","trial_index","cue_type","action_press"]].copy()
    r["_sid"] = r["subject_id"].astype(str); r["_trial"] = pd.to_numeric(r["trial_index"])
    r = r.rename(columns={"cue_type":"_cue_ref","action_press":"_act_ref"})
    m = x.merge(r[["_sid","_trial","_cue_ref","_act_ref"]], on=["_sid","_trial"], how="left", validate="one_to_one")
    matched = float(m["_cue_ref"].notna().mean())
    cue_match = np.nan; act_match = np.nan
    if cue:
        ok = m["_cue_ref"].notna()
        cue_match = float((m.loc[ok,cue].map(norm_text)==m.loc[ok,"_cue_ref"].map(norm_text)).mean()) if ok.any() else np.nan
    if act:
        a = pd.to_numeric(m[act], errors="coerce"); b = pd.to_numeric(m["_act_ref"], errors="coerce"); ok = a.notna() & b.notna()
        act_match = float((a[ok].astype(int)==b[ok].astype(int)).mean()) if ok.any() else np.nan
    vals=[matched] + ([cue_match] if np.isfinite(cue_match) else []) + ([act_match] if np.isfinite(act_match) else [])
    return {"offset":off,"matched_fraction":matched,"cue_match_fraction":cue_match,"action_match_fraction":act_match,"score":float(np.mean(vals))}


def merge_eeg(raw, eeg, feats):
    scores = pd.DataFrame([alignment_score(raw,eeg,o) for o in range(-2,3)]).sort_values(["score","matched_fraction"], ascending=False).reset_index(drop=True)
    best = scores.iloc[0]
    if best["matched_fraction"] < .999999 or (np.isfinite(best["cue_match_fraction"]) and best["cue_match_fraction"] < .999999) or (np.isfinite(best["action_match_fraction"]) and best["action_match_fraction"] < .999999):
        raise ValueError("EEG alignment failed:\n" + scores.to_string(index=False))
    sid = detect(raw,["subj_id_ddm","subject_id","subject_id_norm","subj_id"],"subject")
    tr = detect(raw,["trial_index","trial_index_numeric","trial_ordinal"],"trial")
    d = raw.copy(); d["_sid"] = d[sid].astype(str); d["_trial"] = pd.to_numeric(d[tr]) + int(best["offset"])
    e = eeg[["subject_id","trial_index",*feats]].copy(); e["_sid"] = e["subject_id"].astype(str); e["_trial"] = pd.to_numeric(e["trial_index"]); e=e.drop(columns=["subject_id","trial_index"])
    d = d.merge(e,on=["_sid","_trial"],how="left",validate="one_to_one")
    if d[feats].isna().any().any():
        bad=d[feats].isna().mean(); bad=bad[bad>0]
        raise ValueError("Missing primary EEG after merge:\n"+bad.to_string())
    return d, best.to_dict()


def prepare_action_frame(d, include_group=False):
    d=d.copy()
    if "subj_id_ddm" not in d.columns:
        sid = detect(d,["subject_id","subject_id_norm","subj_id"],"subject")
        d["subj_id_ddm"] = d[sid].astype(str)
    d["logit_p_press"] = safe_logit(pd.to_numeric(d["p_press"], errors="coerce"))
    req=["is_nogo","cue_up","cue_bi","prev_error","prev_press","prior_press_rate","prior_fa_rate","actor_qdiff_current","control_current","action_press","subj_id_ddm"] + (["group_high"] if include_group else [])
    miss=[c for c in req if c not in d.columns]
    if miss: raise ValueError("Missing companion behavioral-computational action/history columns: "+", ".join(miss))
    return d


def model_specs(blocks, include_group=False):
    base=["is_nogo","cue_up","cue_bi","prev_error"] + (["group_high"] if include_group else [])
    hist=["prev_press","prior_press_rate","prior_fa_rate"]
    H=base+hist; P=["logit_p_press"]; AC=["actor_qdiff_current","control_current"]
    return {
      "E0":("history",H),
      "E1":("history_plus_policy",H+P),
      "E2":("history_plus_actor_control",H+AC),
      "E3":("history_plus_cue_erp",H+blocks["erp"]),
      "E4":("history_plus_cue_tf",H+blocks["tf"]),
      "E5":("history_plus_cue_entropy",H+blocks["entropy"]),
      "E6":("history_plus_combined_cue_eeg",H+blocks["combined"]),
      "E7":("history_plus_policy_plus_combined_cue_eeg",H+P+blocks["combined"]),
      "E8":("history_plus_actor_control_plus_combined_cue_eeg",H+AC+blocks["combined"]),
      "E9":("history_plus_policy_actor_control_plus_combined_cue_eeg",H+P+AC+blocks["combined"]),
    }


def metrics(y,p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    lp=y*np.log(p)+(1-y)*np.log(1-p)
    auc=float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan
    return {"n_trials":len(y),"elpd":float(lp.sum()),"elpd_per_trial":float(lp.mean()),"logloss":float(log_loss(y,p,labels=[0,1])),"brier":float(brier_score_loss(y,p)),"auc":auc}


def boot(vals,n,seed):
    x=np.asarray(vals,float); x=x[np.isfinite(x)]
    if not len(x): return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); bm=np.array([rng.choice(x,size=len(x),replace=True).mean() for _ in range(n)])
    return float(x.mean()),float(np.quantile(bm,.025)),float(np.quantile(bm,.975))


def contrasts(subj,n_boot,seed):
    planned=[("E3","E0","ERP beyond strictly lagged behavioral history"),("E4","E0","TF beyond strictly lagged behavioral history"),("E5","E0","entropy beyond strictly lagged behavioral history"),("E6","E0","combined cue EEG beyond strictly lagged behavioral history"),("E7","E1","combined cue EEG beyond history plus policy"),("E8","E2","combined cue EEG beyond history plus actor/control"),("E9","E7","actor/control beyond history + policy + cue EEG"),("E9","E8","policy beyond history + actor/control + cue EEG"),("E9","E6","total RL information beyond history + cue EEG"),("E9","E0","total RL + cue EEG beyond history")]
    avail=set(subj.model); out=[]
    for a,b,label in planned:
        if a not in avail or b not in avail: continue
        x=subj[subj.model==a][["subject_id","fold","elpd"]].rename(columns={"elpd":"a"})
        y=subj[subj.model==b][["subject_id","fold","elpd"]].rename(columns={"elpd":"b"})
        m=x.merge(y,on=["subject_id","fold"],validate="one_to_one"); m["d"]=m.a-m.b
        mean,lo,hi=boot(m.d,n_boot,stable_seed(seed,"contrast",a,b)); fd=m.groupby("fold").d.sum()
        out.append({"model":a,"reference":b,"label":label,"n_subjects":m.subject_id.nunique(),"mean_subject_delta_elpd":mean,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,"positive_subjects":int((m.d>0).sum()),"positive_folds":int((fd>0).sum()),"n_folds":fd.size,"total_delta_elpd":float(m.d.sum())})
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser()
    for x in ["nested-script","rl-script","ddm-script","action-script","rl-out","data-root","eeg-table","out"]: ap.add_argument("--"+x, required=True)
    ap.add_argument("--rl-variant",default="M6"); ap.add_argument("--seed",type=int,default=2026); ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--folds",nargs="+",type=int,default=None); ap.add_argument("--models",nargs="+",default=None); ap.add_argument("--max-rl-draws",type=int,default=300); ap.add_argument("--deadline-ms",type=float,default=700.0)
    ap.add_argument("--Cs",nargs="+",type=float,default=[0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0]); ap.add_argument("--inner-k",type=int,default=4); ap.add_argument("--include-group",action="store_true"); ap.add_argument("--n-bootstrap",type=int,default=5000)
    a=ap.parse_args(); out=ensure_dir(a.out)
    nested=import_py(a.nested_script,"nested_latent_generation"); rl=import_py(a.rl_script,"actor_control_model"); ddm=import_py(a.ddm_script,"first_passage_support"); action=import_py(a.action_script,"action_benchmark")
    if not hasattr(action,"tune_C") or not hasattr(action,"make_model"): raise AttributeError("Companion action benchmark must expose tune_C and make_model")
    eeg,blocks=load_eeg_table(a.eeg_table); specs=model_specs(blocks,a.include_group); models=list(specs) if a.models is None else a.models
    bad=[m for m in models if m not in specs]
    if bad: raise ValueError(f"Unknown models: {bad}")
    arr=rl.load_artifacts(Path(a.rl_out)); meta=rl.get_meta(arr); all_idx=set(range(int(meta["n_subj"])))
    cv=nested.read_json(nested.cv_summary_path(Path(a.rl_out),a.seed,a.rl_variant)); folds=cv.get("folds",[])
    if len(folds)!=a.k: raise ValueError(f"Expected {a.k} RL folds; found {len(folds)}")
    selected={int(fr["fold"]) for fr in folds} if a.folds is None else set(a.folds)
    fold_rows=[]; subj_rows=[]; fit_rows=[]; align_rows=[]
    for fr in folds:
        fi=int(fr["fold"])
        if fi not in selected: continue
        test_idx=[int(x) for x in fr.get("test_subject_indices",[])]
        if not test_idx:
            fr2=nested.read_json(nested.fold_result_path(Path(a.rl_out),a.seed,a.rl_variant,fi),default={}); test_idx=[int(x) for x in fr2.get("test_subject_indices",[])]
        train_idx=sorted(all_idx-set(test_idx)); print(f"[outer fold {fi}] train={len(train_idx)} test={len(test_idx)}", flush=True)
        idata=nested.load_rl_fold_idata(Path(a.rl_out),a.seed,a.rl_variant,fi); atr=rl.subset_subjects(arr,train_idx); ate=rl.subset_subjects(arr,test_idx)
        ltr=nested.posterior_draws_mean_latents(rl,atr,a.rl_variant,idata,a.max_rl_draws,stable_seed(a.seed,"p2train",fi)); lte=nested.posterior_draws_mean_latents(rl,ate,a.rl_variant,idata,a.max_rl_draws,stable_seed(a.seed,"p2test",fi))
        rtr=nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,a.seed,a.rl_variant); rte=nested.arr_to_nested_latent_rows(rl,ate,lte,fi,a.seed,a.rl_variant)
        tr=load_fold_frame(ddm,rtr,a.data_root,a.deadline_ms); te=load_fold_frame(ddm,rte,a.data_root,a.deadline_ms)
        tr,qa=merge_eeg(tr,eeg,blocks["combined"]); te,qb=merge_eeg(te,eeg,blocks["combined"]); tr=prepare_action_frame(tr,a.include_group); te=prepare_action_frame(te,a.include_group)
        align_rows += [{"fold":fi,"split":"train",**qa},{"fold":fi,"split":"test",**qb}]
        ytr=pd.to_numeric(tr.action_press).astype(int); yte=pd.to_numeric(te.action_press).astype(int)
        for mk in models:
            name,cols=specs[mk]; print(f"[outer fold {fi}] fit {mk}: {name} p={len(cols)}", flush=True)
            C=float(action.tune_C(tr,cols,[float(x) for x in a.Cs],a.inner_k,stable_seed(a.seed,"tune",fi,mk))); model=action.make_model(C,stable_seed(a.seed,"fit",fi,mk)); model.fit(tr[cols],ytr); p=model.predict_proba(te[cols])[:,1]
            fm=metrics(yte,p); fold_rows.append({"fold":fi,"model":mk,"model_name":name,"chosen_C":C,"n_features":len(cols),**fm}); fit_rows.append({"fold":fi,"model":mk,"model_name":name,"chosen_C":C,"n_features":len(cols),"train_n_trials":len(tr),"train_n_subjects":tr.subj_id_ddm.nunique(),"test_n_trials":len(te),"test_n_subjects":te.subj_id_ddm.nunique()})
            tmp=pd.DataFrame({"subject_id":te.subj_id_ddm.astype(str),"y":yte.values,"p":p})
            for sid,g in tmp.groupby("subject_id",sort=False): subj_rows.append({"fold":fi,"subject_id":sid,"model":mk,"model_name":name,**metrics(g.y.values,g.p.values)})
    f=pd.DataFrame(fold_rows); s=pd.DataFrame(subj_rows); ft=pd.DataFrame(fit_rows); q=pd.DataFrame(align_rows)
    f.to_csv(out/"cue_action_fold_scores.csv",index=False); s.to_csv(out/"cue_action_subject_metrics.csv",index=False); ft.to_csv(out/"cue_action_fit_summary.csv",index=False); q.to_csv(out/"cue_action_alignment_qc.csv",index=False)
    summary=f.groupby(["model","model_name"],as_index=False).agg(n_folds=("fold","nunique"),total_elpd=("elpd","sum"),mean_elpd_per_trial=("elpd_per_trial","mean"),mean_logloss=("logloss","mean"),mean_brier=("brier","mean"),mean_auc=("auc","mean"),mean_n_features=("n_features","mean")).sort_values("total_elpd",ascending=False); summary.to_csv(out/"cue_action_model_summary.csv",index=False)
    c=contrasts(s,a.n_bootstrap,a.seed); c.to_csv(out/"cue_action_planned_contrasts.csv",index=False)
    write_json(out/"analysis_settings.json",vars(a))
    (out/"CUE_ACTION_PREDICTION_REPORT.md").write_text("# Cue-window EEG action-prediction benchmark\n\n## Model summary\n\n"+summary.to_markdown(index=False)+"\n\n## Planned contrasts\n\n"+(c.to_markdown(index=False) if len(c) else "No contrasts available in this run.")+"\n",encoding="utf-8")
    print("\n=== Model summary ===\n",summary.to_string(index=False)); print("\n=== Planned contrasts ===\n",c.to_string(index=False) if len(c) else "None"); print("\nOutput:",out)

if __name__=="__main__": main()
