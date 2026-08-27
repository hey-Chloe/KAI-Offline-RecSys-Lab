PYTHON := .venv/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup boundary smoke test reproduce-small verify-public verify-esmm verify-amazon-v3 verify-playground verify-two-tower-v2 verify-cvr-esmm playground playground-data export-playground two-tower-v2-dev two-tower-v2-test amazon-v3-dev amazon-v3-test criteo-esmm-status criteo-esmm-run sequence-v2-dev sequence-v2-test ctr-scale-v2 cvr-esmm-v1 million-scale-v1 serve-local serve-smoke

setup:
	$(PYTHON) -m pip install -e .

boundary:
	$(PYTHON) scripts/verify_boundaries.py

smoke: boundary
	$(PYTHON) -m kai_recsys_lab.cli synthetic-smoke --output artifacts/synthetic-smoke.json

reproduce-small:
	$(PYTHON) scripts/reproduce_small.py

test: boundary
	$(PYTHON) -m pytest

verify-public: boundary
	$(PYTHON) scripts/verify_public_reports.py \
		reports/amazon-retrieval-v1-results.json \
		reports/amazon-sequence-v1-results.json \
		reports/criteo-ctr-v1-results.json \
		reports/position-bias-open-bandit-full-ope-v1.json \
		reports/position-bias-open-bandit-small-v1.json \
		reports/amazon-two-tower-v2-results.json \
		reports/criteo-ctr-scale-v2-results.json \
		reports/criteo-attribution-cvr-esmm-v1-results.json \
		reports/amazon-million-scale-v1-results.json \
		reports/amazon-sequence-v2-results.json
	$(PYTHON) scripts/verify_criteo_esmm_v1.py
	$(PYTHON) scripts/verify_amazon_end_to_end_v3.py

verify-esmm:
	$(PYTHON) scripts/verify_criteo_esmm_v1.py --require-artifact

verify-amazon-v3:
	$(PYTHON) scripts/verify_amazon_end_to_end_v3.py --require-artifacts

verify-two-tower-v2: boundary
	$(PYTHON) scripts/verify_amazon_two_tower_v2.py

verify-cvr-esmm: boundary
	$(PYTHON) scripts/verify_cvr_esmm.py --require-data

playground-data:
	$(PYTHON) scripts/build_playground_data.py

export-playground:
	$(PYTHON) scripts/export_playground.py

verify-playground:
	$(PYTHON) scripts/verify_playground.py

playground:
	@echo "Recommendation Algorithm Playground: http://127.0.0.1:4190/playground/"
	$(PYTHON) -m http.server 4190 --bind 127.0.0.1 --directory .

two-tower-v2-dev:
	$(PYTHON) scripts/run_amazon_two_tower_v2.py --phase dev-select

two-tower-v2-test:
	$(PYTHON) scripts/run_amazon_two_tower_v2.py --phase test-final

amazon-v3-dev:
	PYTHONPATH=src $(PYTHON) scripts/run_amazon_end_to_end_v3.py --phase dev-select

amazon-v3-test:
	PYTHONPATH=src $(PYTHON) scripts/run_amazon_end_to_end_v3.py --phase test-final

criteo-esmm-status:
	PYTHONPATH=src $(PYTHON) scripts/run_criteo_esmm_v1.py --status-only

criteo-esmm-run:
	PYTHONPATH=src $(PYTHON) scripts/run_criteo_esmm_v1.py

sequence-v2-dev:
	$(PYTHON) scripts/run_amazon_sequence_v2.py --phase dev-select

sequence-v2-test:
	$(PYTHON) scripts/run_amazon_sequence_v2.py --phase test-final

ctr-scale-v2:
	$(PYTHON) scripts/run_criteo_ctr_scale.py

cvr-esmm-v1:
	$(PYTHON) scripts/run_criteo_attribution_cvr_esmm.py

million-scale-v1:
	$(PYTHON) scripts/run_amazon_million_scale.py

serve-local:
	$(PYTHON) scripts/serve_local.py --config configs/serving-local.json

serve-smoke:
	$(PYTHON) scripts/serve_smoke.py --url http://127.0.0.1:4280
