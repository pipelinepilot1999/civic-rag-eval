PY ?= python3
SNAPSHOT_DATE ?= $(shell date +%F)

.PHONY: help ingest data sets eval eval-real test api docker clean

help:
	@echo "make ingest    pull a dated CIViC snapshot"
	@echo "make data      download the SEQC2 HCC1395 somatic truth set"
	@echo "make sets      build the evaluation sets"
	@echo "make eval      run every arm with the free baselines (no API key)"
	@echo "make eval-real run every arm against Claude (needs ANTHROPIC_API_KEY)"
	@echo "make test      run the test suite"
	@echo "make api       serve the three endpoints on :8000"

ingest:
	$(PY) -m src.ingest --date $(SNAPSHOT_DATE)

data:
	mkdir -p data/seqc2
	curl -sL -o data/seqc2/hc_snv.vcf.gz   "https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic_Mutation_WG/release/latest/high-confidence_sSNV_in_HC_regions_v1.2.1.vcf.gz"
	curl -sL -o data/seqc2/hc_indel.vcf.gz "https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic_Mutation_WG/release/latest/high-confidence_sINDEL_in_HC_regions_v1.2.1.vcf.gz"
	$(PY) scripts/annotate_seqc2.py

sets:
	$(PY) -m eval.build_sets

eval:
	$(PY) -m eval.run_eval --backend abstain --out results/metrics_abstain.json
	$(PY) -m eval.run_eval --backend echo    --out results/metrics_echo.json

eval-real:
	$(PY) -m eval.run_eval --backend anthropic --model $(or $(MODEL),claude-opus-5) \
		--out results/metrics.json

test:
	$(PY) -m pytest tests -q

api:
	$(PY) -m uvicorn src.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t civic-rag-eval:latest .

clean:
	rm -rf __pycache__ */__pycache__ results/*.log

eval-dense:
	$(PY) -m eval.run_eval --backend abstain --dense --out results/metrics_dense.json

