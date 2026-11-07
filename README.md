# NAS-AE: Differentiable NAS for Task-Specific Autoencoder Bottleneck Design

Evaluated on CIFAR-10, ECG5000 and KDD Cup 99.

STATUS: work in progress. Code and results are under verification against the paper. Do not cite these numbers yet.

## Setup

    pip install -r requirements.txt

## Run

    python -m nasae.experiments --dataset ecg5000 --seeds 0 1 2 --out results
    python -m nasae.experiments --dataset kdd99 --seeds 0 1 2 --out results
    python -m nasae.experiments --dataset cifar10 --seeds 0 1 2 --ablation --out results
    python -m nasae.report --results results

## Results

Raw per-run JSON files are in results/<dataset>/seed<N>/.
CIFAR-10 is partial: seed 0 complete, seed 1 partial, seed 2 pending.