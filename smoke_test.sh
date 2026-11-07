#!/usr/bin/env bash
# 2-minute check that the code runs on your machine. Uses FAKE data, numbers are meaningless.
set -e
for d in ecg5000 kdd99 cifar10; do
  python -m nasae.experiments --dataset $d --synthetic --quick --ablation --out results_smoke
done
python -m nasae.report --results results_smoke
