# Usage:  make run                      # dev set (data/dataset) -> outputs/results.json + score
#         make run DATA=data/heldout    # any folder of scenes   -> outputs/results_heldout.json
#         make run GPU=1                # use the NVIDIA GPU for Ollama and the vision models
# Needs Docker with Compose v2. First run downloads Apertus (~5 GB) and the vision models.
.PHONY: run down clean

DATA ?= data/dataset
NAME := $(notdir $(patsubst %/,%,$(DATA)))
OUT  ?= outputs/results$(if $(filter dataset,$(NAME)),,_$(NAME)).json
COMPOSE := docker compose -f docker-compose.yml $(if $(GPU),-f docker-compose.gpu.yml)

run:
	$(COMPOSE) up -d --build --wait ollama
	$(COMPOSE) run --rm --build app sh -c '\
	  python src/run.py --data $(DATA) --out $(OUT) && \
	  if [ -f $(DATA)/ground_truth/ground_truth.json ]; then python src/starter/evaluate.py $(OUT) --data $(DATA); fi'

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
