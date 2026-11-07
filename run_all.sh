#!/usr/bin/env bash
# Full reproduction of the paper. Edit SEEDS if short on time (>=3 recommended).
set -e
SEEDS="0 1 2"
python -m nasae.experiments --dataset ecg5000 --seeds $SEEDS
python -m nasae.experiments --dataset kdd99   --seeds $SEEDS
python -m nasae.experiments --dataset cifar10 --seeds $SEEDS --ablation
python -m nasae.report --results results
