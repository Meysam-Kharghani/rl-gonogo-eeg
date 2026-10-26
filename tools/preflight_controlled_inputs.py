#!/usr/bin/env python3
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser(description='Check controlled inputs required for EEG-computational study reruns without reading participant data.')
 ap.add_argument('--scope',choices=['cue','response','all'],default='all')
 a=ap.parse_args(); errors=[]; checks=[]
 def req(rel,label,kind='file'):
  p=ROOT/rel; ok=p.is_file() if kind=='file' else (p.is_dir() and any(p.rglob('*')))
  checks.append((label,p,ok));
  if not ok: errors.append(f'missing or empty: {rel}')
 if a.scope in {'cue','all'}:
  req('controlled_inputs/behavioral_trial_data','behavioral trial-data directory','dir')
  req('controlled_inputs/rl_crossvalidated_posteriors','cross-validated RL posterior directory','dir')
  req('controlled_inputs/cue_eeg/canonical_cue_feature_table.csv','canonical cue EEG table')
 if a.scope in {'response','all'}:
  req('controlled_inputs/eeg_features/erp_trial_features.csv','ERP trial-feature table')
  req('controlled_inputs/eeg_features/time_frequency_trial_features.csv','time-frequency trial-feature table')
  req('controlled_inputs/eeg_features/entropy_trial_features.csv','entropy-derived trial-feature table')
  req('controlled_inputs/crossfitted_rl/crossfitted_trial_latents.csv','cross-fitted RL trial table')
  req('controlled_inputs/crossfitted_rl/participant_rl_behavior_summary.csv','participant-level RL/behavior summary')
 for label,p,ok in checks: print(('OK  ' if ok else 'MISS'),label,':',p.relative_to(ROOT))
 if errors:
  print('\nCONTROLLED-INPUT PREFLIGHT FAIL')
  for e in errors: print('ERROR:',e)
  return 1
 print('\nCONTROLLED-INPUT PREFLIGHT PASS')
 return 0
if __name__=='__main__': raise SystemExit(main())
