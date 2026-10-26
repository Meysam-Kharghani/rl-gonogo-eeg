#!/usr/bin/env python3
"""Validate the public EEG-computational reproducibility release."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import py_compile
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/release_manifest.tsv"
EXPECTED_VERSION = "1.0.1"
TEXT_EXT = {".md", ".txt", ".py", ".json", ".csv", ".tsv", ".yml", ".yaml", ".toml", ".sh", ".m"}

# Generic publication-facing checks. Specific legacy labels are audited externally
# during release preparation rather than embedded in the public repository.
DEVELOPMENT_VERSION_PATTERN = re.compile(r"(?i)\brl(?:core)?[_-]?v\d+(?:[_.-]\d+)*\b")
ARTICLE_ORDINAL_PATTERN = re.compile(r"(?i)\b" + "pa" + "per" + r"[ _-]?[12]\b")
NONPORTABLE_PATH_PATTERNS = [
    re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/"),
    re.compile("/" + "Users" + r"/[A-Za-z0-9_.-]+/"),
    re.compile("/" + "mnt" + "/" + "data" + r"(?:/|\b)"),
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile("sandbox" + r":/", re.I),
]
PLACEHOLDER_PATTERNS = [
    re.compile(r"\[(?:AUTHORS?|AFFILIATIONS?|NAME|EMAIL|POSTAL ADDRESS)\]"),
    re.compile(r"<repo(?:sitory)?[-_ ]?url>", re.I),
    re.compile(r"<release[-_ ]?version>", re.I),
    re.compile(r"\b(?:TO" + "DO|T" + "BD|FIX" + "ME)\b", re.I),
]
PARTICIPANT_KEYS = {
    "subject_id", "subject_id_norm", "subj_id", "participant_id", "participant",
    "subject", "subject_code", "subject_label", "left_out_subject",
    "test_subject_ids", "train_subject_ids",
}
ID_RE = re.compile(r"(?<![A-Za-z0-9])[HL](?:[1-9][0-9]{0,2})(?![A-Za-z0-9])")

EXPECTED_FIGURE_STEMS = [
    "figure_01_response_monitoring",
    "figure_02_between_participant_rl_eeg_associations",
    "supplementary_figure_01_cue_action_prediction",
    "supplementary_figure_02_cue_rl_state_encoding",
    "supplementary_figure_03_association_robustness",
]
EXPECTED_TABLE_DATA = [
    "main_table_01_response_monitoring_source_data.csv",
    "main_table_02_between_participant_rl_eeg_associations_source_data.csv",
    "supplementary_table_01_cue_action_source_data.csv",
    "supplementary_table_02_cue_encoding_source_data.csv",
    "supplementary_table_03_association_robustness_source_data.csv",
    "supplementary_table_04_marker_qc_source_data.csv",
    "supplementary_table_05_cue_feature_manifest_source_data.csv",
]

class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[str] = []
    def ok(self, msg: str) -> None: self.checks.append(msg)
    def err(self, msg: str) -> None: self.errors.append(msg)
    def warn(self, msg: str) -> None: self.warnings.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_release_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in {"reproduced", ".git"}:
            continue
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if rel.as_posix() == "provenance/release_manifest.tsv":
            continue
        yield path


def check_release_metadata(r: Report) -> None:
    version = ROOT / "VERSION"
    license_file = ROOT / "LICENSE"
    if not version.exists() or version.read_text(encoding="utf-8").strip() != EXPECTED_VERSION:
        r.err(f"VERSION must equal {EXPECTED_VERSION}")
    if not license_file.exists() or not license_file.read_text(encoding="utf-8", errors="replace").startswith("MIT License\n"):
        r.err("LICENSE is missing or is not the MIT License text")
    if not (ROOT / "CITATION.md").exists(): r.err("missing CITATION.md")
    cff = ROOT / "CITATION.cff"
    if not cff.exists(): r.err("missing CITATION.cff")
    else:
        cff_text = cff.read_text(encoding="utf-8")
        for author in ("Meysam Kharghani", "Abolfazl Jafarian", "Farhad Farkhondeh Tale Navi", "Leila Mehdizadeh Fanid"):
            parts = author.split()
            if parts[0] not in cff_text or parts[-1] not in cff_text:
                r.err(f"CITATION.cff missing verified author metadata for {author}")
    r.ok("release metadata, verified citation metadata, and MIT license checked")


def check_required(r: Report) -> None:
    required = [
        "README.md", "VERSION", "LICENSE", "CITATION.md", "CITATION.cff", "Makefile",
        ".github/workflows/validate.yml",
        "docs/analysis_lineage_and_decision_log.md", "docs/analysis_stage_registry.tsv",
        "docs/eeg_analysis_completeness.csv", "docs/eeg_version_lineage.csv",
        "docs/model_nomenclature.md", "docs/evidence_hierarchy.md",
        "docs/eeg_preprocessing_provenance.md", "docs/temporal_provenance.md",
        "docs/privacy_and_exclusion_policy.md", "docs/excluded_public_artifacts.tsv",
        "docs/controlled_input_schema.md", "docs/controlled_rerun_commands.md",
        "docs/data_availability.md", "docs/robustness_results_index.md",
        "docs/public_naming_policy.md", "docs/release_quality_report.md",
        "docs/quality_assurance.md", "docs/publication_metadata.md",
        "docs/manuscript_alignment.md",
        "code/00_model_development/README.md",
        "code/01_feature_generation/extract_eeg_neural_dynamics.py",
        "code/01_feature_generation/run_eeg_time_frequency_analysis.py",
        "code/01_feature_generation/run_eeg_entropy_feature_analysis.py",
        "code/02_primary_cue/build_cue_feature_table.py",
        "code/02_primary_cue/cue_action_prediction.py",
        "code/02_primary_cue/cue_rl_state_encoding.py",
        "code/03_primary_response/performance_monitoring_contrasts.py",
        "code/03_primary_response/between_participant_rl_eeg_associations.py",
        "code/04_robustness/repeated_partition_cue_benchmarks.py",
        "code/04_robustness/richer_history_consistency.py",
        "code/04_robustness/response_confound_triangulation.py",
        "code/04_robustness/response_reliability.py",
        "code/04_robustness/expanded_between_participant_associations.py",
        "code/04_robustness/trial_level_crossfitted_rl_eeg.py",
        "code/05_reporting/make_publication_figures.py",
        "shared/rl/actor_control_model_family.py",
        "shared/rl/nested_crossvalidated_latent_generation.py",
        "shared/eeg/response_matching_and_factorial.py",
        "shared/eeg/adjusted_trial_level_encoding.py",
        "requirements_reporting.txt", "requirements_analysis.txt", "environment/environment.yml",
        "provenance/README.md", "provenance/historical_source_hashes.tsv",
        "results/model_development", "results/cue_action", "results/cue_encoding",
        "results/cue_feature_spec", "results/marker_qc", "results/response_monitoring",
        "results/between_participant_rl_eeg_associations", "results/robustness", "results/reporting",
        "controlled_inputs/README.md",
    ]
    for item in required:
        if not (ROOT / item).exists(): r.err(f"missing required path: {item}")
    r.ok("required-path inventory checked")


def check_reporting_contract(r: Report) -> None:
    fig_dir = ROOT / "results/reporting/figures"
    expected = {f"{stem}.{ext}" for stem in EXPECTED_FIGURE_STEMS for ext in ("png", "pdf", "svg")}
    expected.add("FIGURE_INDEX.md")
    if not fig_dir.exists():
        r.err("missing results/reporting/figures")
    else:
        actual = {p.name for p in fig_dir.iterdir() if p.is_file()}
        if expected - actual: r.err("missing reporting figure asset(s): " + ", ".join(sorted(expected - actual)))
        if actual - expected: r.err("unexpected reporting figure asset(s): " + ", ".join(sorted(actual - expected)))
    tdir = ROOT / "results/reporting/table_data"
    for name in EXPECTED_TABLE_DATA:
        if not (tdir / name).exists(): r.err(f"missing reporting table-data file: {name}")
    smap = ROOT / "results/reporting/source_data_map.tsv"
    if not smap.exists():
        r.err("missing source_data_map.tsv")
    else:
        with smap.open(encoding="utf-8", newline="") as h:
            for row in csv.DictReader(h, delimiter="\t"):
                src = ROOT / row["source_data"]
                if not src.exists(): r.err(f"reporting source-data map points to missing file: {row['source_data']}")
    r.ok("publication-facing reporting contract checked")


def check_names_and_text(r: Report) -> None:
    for path in iter_release_files():
        rel = path.relative_to(ROOT).as_posix()
        for label, pattern in (("development-version label", DEVELOPMENT_VERSION_PATTERN), ("article-ordinal label", ARTICLE_ORDINAL_PATTERN)):
            if pattern.search(rel): r.err(f"{label} in public path: {rel}")
        if path.suffix.lower() not in TEXT_EXT or path.stat().st_size > 20_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in (("development-version label", DEVELOPMENT_VERSION_PATTERN), ("article-ordinal label", ARTICLE_ORDINAL_PATTERN)):
            if pattern.search(text): r.err(f"{label} in public content: {rel}")
        for pattern in NONPORTABLE_PATH_PATTERNS:
            if pattern.search(text): r.err(f"nonportable absolute path in content: {rel}")
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(text): r.err(f"unresolved publication/release placeholder in content: {rel}")
    r.ok("formal public naming, nonportable-path, and placeholder policy checked")


def _bad_header_fields(header: list[str]) -> list[str]:
    bad=[]
    for field in header:
        key=field.strip().lower()
        if key in PARTICIPANT_KEYS or re.search(r"(?:^|_)(?:subject|subj|participant)(?:_|-)?(?:id|ids|code|label)(?:_|$)", key): bad.append(field)
    return bad


def check_privacy(r: Report) -> None:
    for path in iter_release_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "docs/excluded_public_artifacts.tsv":
            continue
        if path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size < 30_000_000:
            sep="\t" if path.suffix.lower()==".tsv" else ","
            try:
                with path.open(encoding="utf-8", errors="replace", newline="") as h: header=next(csv.reader(h, delimiter=sep), [])
                bad=_bad_header_fields(header)
                if bad: r.err(f"participant identifier-like column(s) in public table {rel}: {bad}")
            except Exception as exc: r.err(f"cannot inspect table header {rel}: {exc}")
        if path.name.lower().endswith(".csv.gz") and path.stat().st_size < 80_000_000:
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as h: header=next(csv.reader(h), [])
                bad=_bad_header_fields(header)
                if bad: r.err(f"participant identifier-like column(s) in compressed public table {rel}: {bad}")
            except Exception as exc: r.err(f"cannot inspect compressed table header {rel}: {exc}")
        if path.suffix.lower()==".json" and path.stat().st_size < 30_000_000:
            try:
                obj=json.loads(path.read_text(encoding="utf-8"))
                def walk(v):
                    if isinstance(v,dict):
                        for k,c in v.items():
                            kl=str(k).lower()
                            if kl in PARTICIPANT_KEYS or re.search(r"(?:^|_)(?:subject|subj|participant)(?:_|-)?(?:id|ids|code|label)(?:_|$)", kl): return True
                            if walk(c): return True
                    elif isinstance(v,list): return any(walk(c) for c in v)
                    return False
                if walk(obj): r.err(f"participant identifier-like key in public JSON: {rel}")
            except json.JSONDecodeError: pass
        if path.suffix.lower() in TEXT_EXT and path.stat().st_size < 10_000_000:
            text=path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"(?i)\b[HL][0-9]{1,3}[^\n,]*\.set\b", text): r.err(f"participant-coded EEG filename in public content: {rel}")
            if rel.startswith("results/") and path.suffix.lower() in {".csv", ".tsv", ".json"} and ID_RE.search(text): r.err(f"participant-coded H/L identifier token in public result artifact: {rel}")
    r.ok("participant-level public-boundary checks completed")


def check_structured(r: Report) -> None:
    for path in iter_release_files():
        rel=path.relative_to(ROOT).as_posix()
        if path.suffix.lower()==".json":
            try: json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc: r.err(f"invalid JSON {rel}: {exc}")
        if path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size < 30_000_000:
            sep="\t" if path.suffix.lower()==".tsv" else ","
            try:
                with path.open(encoding="utf-8", errors="replace", newline="") as h:
                    if next(csv.reader(h, delimiter=sep), None) is None: r.err(f"empty table: {rel}")
            except Exception as exc: r.err(f"invalid delimited file {rel}: {exc}")
        if path.name.lower().endswith(".csv.gz") and path.stat().st_size < 80_000_000:
            try:
                with gzip.open(path,"rt",encoding="utf-8",errors="replace",newline="") as h:
                    if next(csv.reader(h),None) is None: r.err(f"empty compressed table: {rel}")
            except Exception as exc: r.err(f"invalid compressed table {rel}: {exc}")
    r.ok("JSON/CSV/TSV/compressed-table parse checks completed")


def check_python(r: Report) -> None:
    with tempfile.TemporaryDirectory(prefix="repository-python-compile-") as td:
        i=0
        for base in [ROOT/"code", ROOT/"shared", ROOT/"tools"]:
            for p in base.rglob("*.py"):
                try:
                    py_compile.compile(str(p), cfile=str(Path(td)/f"module_{i:04d}.pyc"), doraise=True); i+=1
                except Exception as exc: r.err(f"Python compile failure {p.relative_to(ROOT)}: {exc}")
    r.ok("Python syntax compilation completed")


def check_manifest(r: Report) -> None:
    if not MANIFEST.exists(): r.err("missing provenance/release_manifest.tsv"); return
    expected={}
    with MANIFEST.open(encoding="utf-8",newline="") as h:
        for row in csv.DictReader(h,delimiter="\t"): expected[row["path"]]=(row["sha256"],int(row["size_bytes"]))
    actual={p.relative_to(ROOT).as_posix():p for p in iter_release_files()}
    missing=sorted(set(expected)-set(actual)); extra=sorted(set(actual)-set(expected))
    if missing: r.err("manifest entries missing from release: "+", ".join(missing[:20]))
    if extra: r.err("files missing from manifest: "+", ".join(extra[:20]))
    for rel,p in actual.items():
        if rel not in expected: continue
        eh,es=expected[rel]
        if p.stat().st_size!=es: r.err(f"manifest size mismatch: {rel}")
        elif sha256(p)!=eh: r.err(f"manifest SHA-256 mismatch: {rel}")
    r.ok(f"release manifest checked ({len(actual)} files)")


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--checksums-only",action="store_true"); a=ap.parse_args()
    r=Report()
    if a.checksums_only: check_manifest(r)
    else:
        for fn in [check_release_metadata,check_required,check_reporting_contract,check_names_and_text,check_structured,check_python,check_privacy,check_manifest]: fn(r)
    for c in r.checks: print("PASS:",c)
    for w in r.warnings: print("WARNING:",w)
    for e in r.errors: print("ERROR:",e)
    print(f"SUMMARY: {len(r.errors)} error(s), {len(r.warnings)} warning(s)")
    return 1 if r.errors else 0

if __name__=="__main__": raise SystemExit(main())
