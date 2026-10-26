SHELL := /bin/bash

.PHONY: validate figures quality-check checksums controlled-preflight

validate:
	python tools/validate_repository.py

figures:
	rm -rf reproduced/figures
	mkdir -p reproduced/figures
	python code/05_reporting/make_publication_figures.py --results-root results --out reproduced/figures
	python tools/verify_reproduced_outputs.py --generated reproduced/figures --reference results/reporting/figures

quality-check: validate figures

checksums:
	python tools/validate_repository.py --checksums-only

controlled-preflight:
	python tools/preflight_controlled_inputs.py
