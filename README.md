# Distributionally Robust Causal Abstractions

This repository contains the implementation and evaluation framework for Distributionally Robust Causal Abstractions (DiRoCA) paper.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yfelekis/DiRoCA.git
cd DiRoCA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Generate data** for experiments:
```bash
python generate_data.py --experiment {experiment}
```
Creates synthetic low-level and high-level Structural Causal Models (SCMs) and prepares cross-validation folds for experiments.


2. **Run optimization** (choose one approach):
```bash
# Gaussian optimization
python gauss_optimization.py --experiment {experiment}

# Empirical optimization  
python empirical_optimization.py --experiment {experiment}
```
Learns abstraction mappings between low-level and high-level models using either a parametric Gaussian assumption or a fully non-parametric empirical approach.


3. **Run evaluation**:
```bash
# Gaussian evaluation
python run_evaluation.py --experiment {experiment}

# Empirical evaluation
python run_empirical_evaluation.py --experiment {experiment}
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
├── README.md                                    # This file
├── requirements.txt                             # Dependencies
├── generate_data.py                             # Data generation script
├── gauss_optimization.py                        # Gaussian optimization
├── empirical_optimization.py                    # Empirical optimization
├── run_evaluation.py                            # Gaussian evaluation
├── run_empirical_evaluation.py                  # Empirical evaluation
├── huber_analysis.ipynb                         # Main analysis notebook
├── contamination_analysis.ipynb                 # Contamination analysis
├── configs/                                     # Configuration files
│   ├── *_opt_config_{experiment}.yaml           # Gaussian optimization configs
│   └── *_opt_config_empirical_{experiment}.yaml # Empirical optimization configs
│   ├── lilucas_config.yaml                      # LiLucas experiment config
│   └── slc_config.yaml                          # SLC experiment config
├── data/                                        # Generated data and results
│   └── {experiment}/
│       ├── cv_folds.pkl
│       ├── LLmodel.pkl
│       ├── HLmodel.pkl
│       ├── results/                   # Gaussian results
│       ├── results_empirical/         # Empirical results
│       └── evaluation_results/        # Evaluation outputs
├── plots/                             # Generated visualizations
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{diroca2025,
  title={Distributionally Robust Causal Abstractions},
  author={Felekis, Yorgos and Damoulas, Theodoros and Giampouras, Paris},
  journal={arXiv preprint},
  year={2025}
}
```