# DiRoCA: Distributionally Robust Causal Abstractions

## Description
This repository contains the implementation of DiRoCA, focusing on causal abstraction learning across different datasets including SLC, LiLUCAS, EBM, and Colored MNIST. The project includes data generation, optimization/learning procedures, and evaluation scripts for both Gaussian and empirical cases.

## Workflow & Notebooks

### Synthetic Data Generation
1. **SLC Dataset**: 
   - Run `synth1.ipynb` to generate the SLC dataset

2. **LiLUCAS Dataset**:
   - Run `lucas6x3.ipynb` to generate the LiLUCAS dataset

### Optimization and Learning

#### Gaussian Case
1. Run `opt_ell.ipynb` for optimization/learning
2. Results will be automatically saved
3. Run `eval_ell.ipynb` for evaluation

#### Empirical Case
1. Run `opt_emp.ipynb` for optimization/learning
2. Results will be automatically saved
3. Run `eval_emp.ipynb` for evaluation

### EBM Dataset
Single notebook workflow in `battery.ipynb`:
- Data loading
- Preprocessing
- Optimization/learning
- Downstream task evaluation

For testing distribution shifts and misspecifications:
- Run `eval_battery.ipynb`

### Colored MNIST Dataset
Two-part workflow:
1. `colored_mnist.ipynb`:
   - Data generation/loading
   - Preprocessing
   - Optimization
2. `colored_mnist_eval.ipynb`:
   - Evaluation procedures

## Project Structure
```
.
├── requirements.txt    # Project dependencies
├── notebooks/
│   ├── synth1.ipynb           # SLC dataset generation
│   ├── lucas6x3.ipynb         # LiLUCAS dataset generation
│   ├── opt_ell.ipynb          # Gaussian case optimization
│   ├── opt_emp.ipynb          # Empirical case optimization
│   ├── eval_ell.ipynb         # Gaussian case evaluation
│   ├── eval_emp.ipynb         # Empirical case evaluation
│   ├── battery.ipynb          # EBM dataset pipeline
│   ├── eval_battery.ipynb     # EBM evaluation
│   ├── colored_mnist.ipynb    # Colored MNIST pipeline
│   └── colored_mnist_eval.ipynb # Colored MNIST evaluation
└── [other files/directories]
```

## Dependencies
This project requires the following main packages:
- PyTorch (>=2.1.0)
- NumPy (>=1.24.0)
- scikit-learn (>=1.3.0)
- NetworkX (>=3.2.0)

For a complete list of dependencies, see `requirements.txt`.