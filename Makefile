PYTHON ?= python
REPORT_DIR ?= reports
MODEL_BACKEND ?= xgboost
MODEL_OUTPUT ?= models/xgboost_lap_delta.json
ARTIFACT_ROOT ?= artifacts/models
ARTIFACT_ID ?=
TRAIN_LAPS ?= 28
TRAIN_SEEDS ?= 64
TRAIN_ROUNDS ?= 140

.PHONY: help install install-api install-persistence compile test regression evaluate reports train train-evaluate promote-artifact rollback-candidate
.PHONY: ci docker-build clean-reports

help:
	@printf '%s\n' \
		'Targets:' \
		'  install              Install the package in editable mode' \
		'  install-api          Install API and observability extras' \
		'  install-persistence  Install DuckDB persistence extra' \
		'  compile              Compile src and tests' \
		'  test                 Run unittest suite' \
		'  regression           Run deterministic regression gates' \
		'  evaluate             Print Markdown evaluation report' \
		'  reports              Write JSON and Markdown evaluation reports' \
		'  train                Train MODEL_BACKEND artifact to MODEL_OUTPUT' \
		'  train-evaluate       Train, evaluate, and register an artifact bundle' \
		'  promote-artifact     Promote ARTIFACT_ID after registry gate checks' \
		'  rollback-candidate   Print latest promoted rollback candidate' \
		'  ci                   Run compile, test, regression, and reports' \
		'  docker-build         Build local API Docker image'

install:
	$(PYTHON) -m pip install -e .

install-api:
	$(PYTHON) -m pip install -e ".[api,observability,persistence]"

install-persistence:
	$(PYTHON) -m pip install -e ".[persistence]"

compile:
	$(PYTHON) -m compileall -q src tests

test:
	$(PYTHON) -m unittest discover -s tests

regression:
	$(PYTHON) -m f1_strategy.regression

evaluate:
	$(PYTHON) -m f1_strategy.evaluation --format markdown

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

train-evaluate:
	$(PYTHON) -m f1_strategy.training \
		--backend $(MODEL_BACKEND) \
		--output $(MODEL_OUTPUT) \
		--laps $(TRAIN_LAPS) \
		--seeds $(TRAIN_SEEDS) \
		--rounds $(TRAIN_ROUNDS) \
		--artifact-root $(ARTIFACT_ROOT)

promote-artifact:
	$(PYTHON) -m f1_strategy.artifacts promote \
		--artifact-id $(ARTIFACT_ID) \
		--artifact-root $(ARTIFACT_ROOT)

rollback-candidate:
	$(PYTHON) -m f1_strategy.artifacts rollback-candidate \
		--backend $(MODEL_BACKEND) \
		--active-artifact-id $(ARTIFACT_ID) \
		--artifact-root $(ARTIFACT_ROOT)

ci: compile test regression reports

docker-build:
	docker build \
		--build-arg F1_BUILD_SHA=local \
		--build-arg F1_BUILD_DATE=local \
		-t f1-tire-energy-strategy:local .

clean-reports:
	rm -f $(REPORT_DIR)/model-evaluation.json $(REPORT_DIR)/model-evaluation.md
