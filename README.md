# Distributionally Robust Causal Abstractions

This repository contains the implementation and evaluation framework for Distributionally Robust Causal Abstractions (DiRoCA) paper.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DiRoCA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Generate data** for experiments:
```bash
python generate_data.py --experiment slc
```
Creates synthetic low-level and high-level Structural Causal Models (SCMs) and prepares cross-validation folds for experiments.


2. **Run optimization** (choose one approach):
```bash
# Gaussian optimization
python gauss_optimization.py --experiment slc

# Empirical optimization  
python empirical_optimization.py --experiment slc
```
Learns abstraction mappings between low-level and high-level models using either a parametric Gaussian assumption or a fully non-parametric empirical approach.


3. **Run evaluation**:
```bash
# Gaussian evaluation
python run_evaluation.py --experiment slc

# Empirical evaluation
python run_empirical_evaluation.py --experiment slc
```
Computes abstraction errors across varying contamination levels and noise scales to assess robustness.


4. **Analyze results** using the provided notebooks:
```bash
jupyter notebook huber_analysis.ipynb
jupyter notebook contamination_analysis.ipynb
```
Provides visual and quantitative analysis of method performance under distributional shifts and structural misspecifications.


## Project Structure

```
DiRoCA_TBS/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── generate_data.py                   # Data generation script
├── gauss_optimization.py              # Gaussian optimization
├── empirical_optimization.py          # Empirical optimization
├── run_evaluation.py                  # Gaussian evaluation
├── run_empirical_evaluation.py        # Empirical evaluation
├── huber_analysis.ipynb               # Main analysis notebook
├── contamination_analysis.ipynb       # Contamination analysis
├── run_all.sh                         # Complete Gaussian evaluation
├── run_all_empirical.sh               # Complete empirical evaluation
├── configs/                           # Configuration files
│   ├── *_opt_config.yaml              # Gaussian optimization configs
│   └── *_opt_config_empirical.yaml    # Empirical optimization configs
├── data/                              # Generated data and results
│   └── {experiment}/
│       ├── cv_folds.pkl
│       ├── LLmodel.pkl
│       ├── HLmodel.pkl
│       ├── results/                   # Gaussian results
│       ├── results_empirical/         # Empirical results
│       └── evaluation_results/        # Evaluation outputs
├── plots/                             # Generated visualizations
└── src/                               # Source code
    ├── CBN.py                         # Causal Bayesian Networks
    ├── data_generator.py              # Data generation utilities
    ├── optimizer.py                   # Optimization algorithms
    ├── pipeline.py                    # Main pipeline
    └── examples/                      # Example models
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{diroca2025,
  title={Distributionally Robust Causal Abstractions},
  author={Felekis, Yorgos and Giampouras, Paris and Damoulas, Theodoros},
  journal={arXiv preprint},
  year={2025}
}
```