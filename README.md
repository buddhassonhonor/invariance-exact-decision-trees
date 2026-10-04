# CIEDT: Invariance-Regularized Exact Decision Trees for OOD Generalization

This artifact package provides the official implementation, experimental benchmarks, processed evaluation results, and visualization scripts for **Causal Invariance-regularized Exact Decision Trees (CIEDT)**.

## Directory Structure
- `src/`: Core implementation and experimental scripts.
  - `ciedt.py`: Bellman-style recursive solver with leaf-level Laplace smoothing and invariance penalty.
  - `data_gen.py`, `data_cmnist.py`, `data_tabular_shift.py`: Multi-environment dataset generation and spurious shortcut injection.
  - `revision_suite.py`: Main multi-environment evaluation benchmark across 4 datasets and 5 seeds.
  - `scalability.py`: Empirical scalability analysis over tree depth and sample size.
  - `synthetic_supplement.py`: Comprehensive hyperparameter grid (depth and lambda) validation.
  - `ablation.py`, `sparse_leaf_smoothing.py`: Parameter sensitivity and smoothing stability studies.
- `results/`: Processed experimental logs, bootstrap paired confidence intervals, and metrics summaries.
- `figures/`: High-resolution figures generated from experimental runs.

## System Requirements
- Python >= 3.9
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

## Quick Reproduction
```bash
# 1. Run synthetic grid validation:
python src/synthetic_supplement.py

# 2. Run scalability benchmark:
python src/scalability.py

# 3. Run full multi-environment evaluation suite:
python src/revision_suite.py
```
