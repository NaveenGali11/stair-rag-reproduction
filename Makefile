SHELL := /usr/bin/env bash

CORE_PYTHON ?= .venv/bin/python
NVEMBED_PYTHON ?= .venv-nvembed/bin/python
MIXTRAL_PYTHON ?= .venv-mixtral/bin/python
RETRIEVER_PYTHON ?= .venv-retriever/bin/python
JUPYTER ?= jupyter

.PHONY: help setup-core setup-nvembed setup-mixtral setup-retriever \
	derive-corpus validate-corpus validate-experiment validate-repository \
	generate-plan generate-questions validate-questions split-data \
	prepare-retriever evaluate-bm25 evaluate-nvembed train-dsi train-stair \
	evaluate-dsi evaluate-stair compare freeze-results report notebook \
	execute-notebook validate-code

help:
	@echo "Fast paths"
	@echo "  make setup-core           Create the CPU/reporting environment"
	@echo "  make validate-repository  Validate all tracked release artifacts"
	@echo "  make evaluate-bm25        Run the CPU BM25 benchmark"
	@echo "  make report               Rebuild figures from frozen results"
	@echo "  make notebook             Regenerate the clean results notebook"
	@echo
	@echo "Full reproduction paths"
	@echo "  make derive-corpus        Rebuild corpus data from the ignored PDF"
	@echo "  make generate-questions   Generate questions with local Mixtral"
	@echo "  make split-data           Rebuild and validate frozen splits"
	@echo "  make prepare-retriever    Build DSI and STAIR training records"
	@echo "  make train-dsi            Train the DSI LoRA adapter"
	@echo "  make train-stair          Train the STAIR LoRA adapter"
	@echo "  make evaluate-dsi         Evaluate the selected DSI adapter"
	@echo "  make evaluate-stair       Evaluate the selected STAIR adapter"
	@echo "  make compare              Run paired DSI/STAIR analysis"
	@echo "  make freeze-results       Validate and freeze result metadata"

setup-core:
	python3 -m venv .venv
	$(CORE_PYTHON) -m pip install --upgrade pip
	$(CORE_PYTHON) -m pip install -r requirements.txt

setup-nvembed:
	python3 -m venv --system-site-packages .venv-nvembed
	$(NVEMBED_PYTHON) -m pip install --upgrade pip
	$(NVEMBED_PYTHON) -m pip install -r requirements-nvembed.txt

setup-mixtral:
	python3 -m venv .venv-mixtral
	$(MIXTRAL_PYTHON) -m pip install --upgrade pip
	$(MIXTRAL_PYTHON) -m pip install -r requirements-mixtral.txt

setup-retriever:
	python3 -m venv --system-site-packages .venv-retriever
	$(RETRIEVER_PYTHON) -m pip install --upgrade pip
	$(RETRIEVER_PYTHON) -m pip install -r requirements-retriever.txt

derive-corpus:
	test -f data/raw/whole-child.pdf
	$(CORE_PYTHON) scripts/derive_leaves.py
	$(CORE_PYTHON) scripts/check_toc_heading_coverage.py
	$(CORE_PYTHON) scripts/extract_sections.py
	$(CORE_PYTHON) scripts/validate_sections.py
	$(CORE_PYTHON) scripts/check_toc_block_coverage.py
	$(CORE_PYTHON) scripts/derive_paragraphs.py
	$(CORE_PYTHON) scripts/validate_paragraphs.py
	$(CORE_PYTHON) scripts/derive_generation_units.py

validate-corpus:
	$(CORE_PYTHON) scripts/validate_sections.py
	$(CORE_PYTHON) scripts/validate_paragraphs.py

validate-experiment: validate-corpus
	$(CORE_PYTHON) scripts/validate_generated_questions.py
	$(CORE_PYTHON) scripts/validate_splits.py

validate-repository:
	python3 scripts/validate_repository.py

generate-plan:
	$(MIXTRAL_PYTHON) scripts/generate_questions.py --plan

generate-questions:
	$(MIXTRAL_PYTHON) scripts/generate_questions.py --batch-size 8

validate-questions:
	$(CORE_PYTHON) scripts/validate_generated_questions.py

split-data:
	$(CORE_PYTHON) scripts/split_generated_questions.py
	$(CORE_PYTHON) scripts/validate_splits.py

prepare-retriever:
	$(RETRIEVER_PYTHON) scripts/prepare_retriever_data.py

evaluate-bm25:
	$(CORE_PYTHON) scripts/evaluate_bm25.py

evaluate-nvembed:
	$(NVEMBED_PYTHON) scripts/evaluate_nvembed.py

train-dsi:
	$(RETRIEVER_PYTHON) scripts/train_retriever_lora.py --method dsi --fresh

train-stair:
	$(RETRIEVER_PYTHON) scripts/train_retriever_lora.py --method stair --fresh

evaluate-dsi:
	$(RETRIEVER_PYTHON) scripts/evaluate_retriever_lora.py --method dsi

evaluate-stair:
	$(RETRIEVER_PYTHON) scripts/evaluate_retriever_lora.py --method stair --batch-size 6

compare:
	$(RETRIEVER_PYTHON) scripts/compare_retriever_results.py

freeze-results:
	$(RETRIEVER_PYTHON) scripts/freeze_results.py

report:
	$(CORE_PYTHON) scripts/plot_results.py

notebook:
	$(CORE_PYTHON) scripts/build_results_notebook.py

execute-notebook: notebook
	$(JUPYTER) nbconvert --to notebook --execute --inplace \
		notebooks/whole_child_results.ipynb \
		--ExecutePreprocessor.timeout=180

validate-code:
	python3 -m compileall -q scripts
