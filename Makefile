PYTHON ?= python
REPORT_DIR ?= reports
MODEL_BACKEND ?= xgboost
MODEL_OUTPUT ?= models/xgboost_lap_delta.json
ARTIFACT_ROOT ?= artifacts/models
ARTIFACT_ID ?=
EXTRA_ARGS ?=
TRAIN_LAPS ?= 28
TRAIN_SEEDS ?= 64
TRAIN_ROUNDS ?= 140
REPLAY_DATASET ?= examples/replay_telemetry.csv
REAL_DATA ?=
REAL_DATA_MULTI ?= \
  data/fastf1-2024-bahrain-r-ver.csv data/fastf1-2024-bahrain-r-HAM.csv data/fastf1-2024-bahrain-r-NOR.csv data/fastf1-2024-bahrain-r-LEC.csv data/fastf1-2023-bahrain-r-VER.csv \
  data/fastf1-2024-monaco-r-ver.csv  data/fastf1-2024-monaco-r-HAM.csv  data/fastf1-2024-monaco-r-NOR.csv  data/fastf1-2024-monaco-r-LEC.csv  data/fastf1-2023-monaco-r-VER.csv \
  data/fastf1-2024-monza-r-ver.csv   data/fastf1-2024-monza-r-HAM.csv   data/fastf1-2024-monza-r-NOR.csv   data/fastf1-2024-monza-r-LEC.csv   data/fastf1-2023-monza-r-VER.csv \
  data/fastf1-2024-silverstone-r-ver.csv data/fastf1-2024-silverstone-r-HAM.csv data/fastf1-2024-silverstone-r-NOR.csv data/fastf1-2024-silverstone-r-LEC.csv data/fastf1-2023-silverstone-r-VER.csv \
  data/fastf1-2024-singapore-r-ver.csv   data/fastf1-2024-singapore-r-HAM.csv   data/fastf1-2024-singapore-r-NOR.csv   data/fastf1-2024-singapore-r-LEC.csv   data/fastf1-2023-singapore-r-VER.csv
# Auto-discover all exported OpenF1 CSVs (populated by export-openf1-bulk)
REAL_DATA_OPENF1 ?= $(wildcard data/openf1-*.csv)
FASTF1_YEAR ?= 2024
FASTF1_EVENT ?= Bahrain
FASTF1_SESSION ?= R
FASTF1_DRIVER ?= VER
FASTF1_OUTPUT ?= data/fastf1_replay.csv
FASTF1_CACHE ?= data/fastf1-cache
# Space-separated Grand Prix names for bulk export (2024 season sample)
MULTI_SESSIONS ?= Bahrain Monaco Monza Silverstone Singapore
MULTI_DRIVER ?= VER
MULTI_YEAR ?= 2024

.PHONY: help install install-api install-persistence install-data install-ml compile test test-integration regression evaluate replay-evaluate replay-benchmark replay-manifest-check export-fastf1-replay export-multi export-openf1 export-openf1-bulk reports train train-real train-evaluate train-evaluate-real train-multi train-evaluate-multi train-openf1 train-evaluate-openf1 register-local-artifacts prune-artifacts promote-artifact rollback-candidate
.PHONY: ci docker-build clean-reports deploy deploy-prod certs export-parquet

help:
	@printf '%s\n' \
		'Targets:' \
		'  install              Install the package in editable mode' \
		'  install-api          Install API and observability extras' \
		'  install-ml           Install ML + persistence + tracking extras' \
		'  install-persistence  Install DuckDB persistence extra' \
		'  install-data         Install FastF1 replay export dependencies' \
		'  compile              Compile src and tests' \
		'  test                 Run unittest suite' \
		'  test-integration     Run live OpenF1 API integration tests' \
		'  regression           Run deterministic regression gates' \
		'  evaluate             Print Markdown evaluation report' \
		'  replay-evaluate      Run replay holdout gates against REPLAY_DATASET' \
		'  replay-benchmark     Run committed benchmark replay slice gates' \
		'  replay-manifest-check Check committed benchmark replay sidecar manifests' \
		'  export-fastf1-replay Export a FastF1 session to replay CSV format' \
		'  export-multi         Bulk-export MULTI_SESSIONS for MULTI_DRIVER (2024 race)' \
		'  reports              Write JSON and Markdown evaluation reports' \
		'  train                Train MODEL_BACKEND artifact to MODEL_OUTPUT' \
		'  train-real           Train with synthetic + real data (REAL_DATA=path/to.csv)' \
		'  train-evaluate       Train, evaluate, and register an artifact bundle' \
		'  train-evaluate-real  Train + evaluate with real data augmentation' \
		'  train-multi          Train with synthetic + all 5-circuit real data' \
		'  train-evaluate-multi Train + evaluate with all 5-circuit real data' \
		'  export-openf1-bulk   Bulk-export OpenF1 public API data (all seasons/drivers)' \
		'  train-openf1         Train on FastF1 + all exported OpenF1 CSVs' \
		'  train-evaluate-openf1 Train + evaluate on all OpenF1 + FastF1 data' \
		'  register-local-artifacts Register existing models/ files as artifact bundles' \
		'  prune-artifacts      Archive older candidate artifacts' \
		'  promote-artifact     Promote ARTIFACT_ID after registry gate checks' \
		'  rollback-candidate   Print latest promoted rollback candidate' \
		'  ci                   Run compile, test, regression, and reports' \
		'  docker-build         Build local API Docker image' \
		'  certs                Generate self-signed TLS certificate for nginx' \
		'  deploy               Full deploy via scripts/deploy.sh (cert + build + up)' \
		'  deploy-prod          Docker compose up with prod overlay (assumes certs exist)' \
		'  export-parquet       Export DuckDB tables to Parquet in data/exports/'

install:
	$(PYTHON) -m pip install -e .

install-api:
	$(PYTHON) -m pip install -e ".[api,observability,persistence]"

install-persistence:
	$(PYTHON) -m pip install -e ".[persistence]"

install-data:
	$(PYTHON) -m pip install -e ".[data]"

compile:
	$(PYTHON) -m compileall -q src tests

test:
	$(PYTHON) -m unittest discover -s tests

regression:
	$(PYTHON) -m f1_strategy.regression

evaluate:
	$(PYTHON) -m f1_strategy.evaluation --format markdown

replay-evaluate:
	$(PYTHON) -m f1_strategy.replay --dataset $(REPLAY_DATASET)

replay-benchmark:
	$(PYTHON) -m f1_strategy.replay --benchmark

replay-manifest-check:
	$(PYTHON) -m f1_strategy.replay --check-benchmark-manifests

export-openf1:
	$(PYTHON) -m f1_strategy.data_sources.openf1_export \
		--year $(FASTF1_YEAR) \
		--event "$(FASTF1_EVENT)" \
		--session "$(FASTF1_SESSION)" \
		--driver "$(FASTF1_DRIVER)" \
		--output "$(FASTF1_OUTPUT)"

# Bulk-export all OPENF1_YEARS seasons for OPENF1_DRIVERS (skips existing files)
OPENF1_YEARS ?= 2024 2023
OPENF1_DRIVERS ?= VER HAM NOR LEC
export-openf1-bulk:
	$(PYTHON) -m f1_strategy.data_sources.openf1_bulk_export \
		--years $(OPENF1_YEARS) \
		--drivers $(OPENF1_DRIVERS)

export-fastf1-replay:
	$(PYTHON) -m f1_strategy.data_sources.fastf1_export \
		--year $(FASTF1_YEAR) \
		--event "$(FASTF1_EVENT)" \
		--session "$(FASTF1_SESSION)" \
		--driver "$(FASTF1_DRIVER)" \
		--output "$(FASTF1_OUTPUT)" \
		--cache-dir "$(FASTF1_CACHE)"

export-multi:
	@mkdir -p data
	@for event in $(MULTI_SESSIONS); do \
		out="data/fastf1-$(MULTI_YEAR)-$$(echo $$event | tr '[:upper:]' '[:lower:]')-r-$(MULTI_DRIVER).csv"; \
		echo "Exporting $$event -> $$out"; \
		$(PYTHON) -m f1_strategy.data_sources.fastf1_export \
			--year $(MULTI_YEAR) \
			--event "$$event" \
			--session R \
			--driver "$(MULTI_DRIVER)" \
			--output "$$out" \
			--cache-dir "$(FASTF1_CACHE)" || echo "  WARN: $$event export failed, skipping"; \
	done

reports:
	mkdir -p $(REPORT_DIR)
	$(PYTHON) -m f1_strategy.evaluation --format json --output $(REPORT_DIR)/model-evaluation.json
	$(PYTHON) -m f1_strategy.evaluation --format markdown --output $(REPORT_DIR)/model-evaluation.md

train:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS)

train-real:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		$(if $(REAL_DATA),--real-data $(REAL_DATA),)

train-evaluate:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--artifact-root $(ARTIFACT_ROOT) \
		--replay-dataset $(REPLAY_DATASET)

train-evaluate-real:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--artifact-root $(ARTIFACT_ROOT) \
		--replay-dataset $(REPLAY_DATASET) \
		$(if $(REAL_DATA),--real-data $(REAL_DATA),)

train-openf1:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--real-data $(REAL_DATA_OPENF1) $(REAL_DATA_MULTI)

train-evaluate-openf1:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--artifact-root $(ARTIFACT_ROOT) \
		--replay-dataset $(REPLAY_DATASET) \
		--real-data $(REAL_DATA_OPENF1) $(REAL_DATA_MULTI)

train-multi:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--real-data $(REAL_DATA_MULTI)

train-evaluate-multi:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--artifact-root $(ARTIFACT_ROOT) \
		--replay-dataset $(REPLAY_DATASET) \
		--real-data $(REAL_DATA_MULTI)

register-local-artifacts:
	$(PYTHON) -m f1_strategy.artifacts register-local \
		--artifact-root $(ARTIFACT_ROOT) \
		--replay-dataset $(REPLAY_DATASET)

prune-artifacts:
	$(PYTHON) -m f1_strategy.artifacts prune \
		--artifact-root $(ARTIFACT_ROOT)

promote-artifact:
	$(PYTHON) -m f1_strategy.artifacts promote \
		--artifact-id $(ARTIFACT_ID) \
		--artifact-root $(ARTIFACT_ROOT) \
		$(EXTRA_ARGS)

rollback-candidate:
	$(PYTHON) -m f1_strategy.artifacts rollback-candidate \
		--backend $(MODEL_BACKEND) \
		--active-artifact-id $(ARTIFACT_ID) \
		--artifact-root $(ARTIFACT_ROOT)

ci: compile test regression replay-manifest-check replay-benchmark reports

docker-build:
	docker build \
		--build-arg F1_BUILD_SHA=local \
		--build-arg F1_BUILD_DATE=local \
		-t f1-tire-energy-strategy:local .

clean-reports:
	rm -f $(REPORT_DIR)/model-evaluation.json $(REPORT_DIR)/model-evaluation.md

install-ml:
	$(PYTHON) -m pip install -e ".[ml,catboost,persistence,tracking]"

test-integration:
	$(PYTHON) -m pytest tests/test_integration_openf1.py -m integration -v

certs:
	mkdir -p monitoring/nginx/certs
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout monitoring/nginx/certs/server.key \
		-out    monitoring/nginx/certs/server.crt \
		-subj   "/CN=167.233.125.215" \
		-addext "subjectAltName=IP:167.233.125.215" 2>/dev/null
	chmod 600 monitoring/nginx/certs/server.key
	@echo "Self-signed certificate written to monitoring/nginx/certs/"

deploy:
	bash scripts/deploy.sh

deploy-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --remove-orphans

export-parquet:
	f1-export-data --db data/f1_strategy.duckdb --output data/exports
