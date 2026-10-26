#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib
from pathlib import Path

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--generated',required=True)
    ap.add_argument('--reference',required=True)
    a=ap.parse_args()
    gen,ref=Path(a.generated),Path(a.reference)
    refs=sorted(p for p in ref.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.pdf','.svg','.md'})
    errors=[]
    if not refs: errors.append('no frozen reference figure assets found')
    for rp in refs:
        gp=gen/rp.name
        if not gp.exists(): errors.append(f'missing generated asset: {rp.name}'); continue
        if gp.stat().st_size!=rp.stat().st_size or sha256(gp)!=sha256(rp):
            errors.append(f'generated asset differs from frozen reference: {rp.name}')
    extras=sorted(p.name for p in gen.iterdir() if p.is_file() and p.name not in {r.name for r in refs})
    if extras: errors.append('unexpected generated assets: '+', '.join(extras))
    print(f'FIGURE ASSETS CHECKED: {len(refs)}')
    if errors:
        for e in errors: print('ERROR:',e)
        return 1
    print('FIGURE REPRODUCTION PASS (exact SHA-256 identity)')
    return 0
if __name__=='__main__': raise SystemExit(main())
