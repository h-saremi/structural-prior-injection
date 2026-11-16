# Do Structural Priors Still Help?

Code and data for the paper *"Do Structural Priors Still Help? Injecting Static Centrality into Temporal Graph Neural Networks for Influential-Node Ranking"*.

## Overview

This repository contains the full experimental pipeline:

- Temporal SIR simulation on time-respecting graphs
- Monte Carlo ground-truth spreading power estimation
- Static centrality feature extraction
- Baseline models (GradientBoosting, RandomForest, MLP)
- Temporal GNN with input / intermediate / output prior injection
- Degree-controlled excess-attention interpretability analysis

## Quick Start

Install dependencies:

    pip install -r requirements.txt

Then run the pipeline in order:

    python scripts/01_prepare_data.py
    python scripts/02_compute_ground_truth.py
    python scripts/03_extract_features.py
    python scripts/04_train_baselines.py
    python scripts/05_injection_experiments.py
    python scripts/06_interpretability.py

## Reproducibility

- Fixed random seed: 42
- Diffusion: beta = 0.2, mu = 0.1, I = 1,000 Monte Carlo runs
- Valid nodes filter: out-degree > 0 (1,350 of 1,899 nodes retained)
- Train/val/test split: 70/15/15, node-level, fixed index file

See configs/default.yaml for all parameters.

## Tests

Run the simulator validation tests:

    pytest tests/ -v

tests/test_simulator.py validates the SIR simulator against a hand-crafted 5-node graph with known expected outputs.

## Citation

If you use this code, please cite:

    [Author names]. Do Structural Priors Still Help? Injecting Static Centrality
    into Temporal Graph Neural Networks for Influential-Node Ranking.
    [Journal], [Year].

## License

MIT License (see LICENSE).
