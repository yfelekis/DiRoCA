# DiRoCA: Distributionally Robust Causal Abstractions

## Description
This repository contains the implementation of DiRoCA, focusing on causal abstraction learning across different datasets including SLC, LiLUCAS, EBM, and Colored MNIST. The project includes data generation, optimization/learning procedures, and evaluation scripts for both Gaussian and empirical cases.

## New Modular Structure

The codebase has been modularized for better maintainability and experimentation. The new structure separates data generation, configuration, and pipeline management.

### Quick Start

#### Data Generation
```python
from src.pipeline import run_data_generation

# Generate data for synth1 experiment
data = run_data_generation('synth1')

# Generate data for lucas6x3 experiment  
data = run_data_generation('lucas6x3')
```

#### Using the Pipeline
```python
from src.pipeline import DiRoCAPipeline

# Create pipeline for an experiment
pipeline = DiRoCAPipeline('synth1')

# Check if data exists
if not pipeline.check_data_exists():
    # Generate data
    data = pipeline.generate_data()
else:
    # Load existing data
    data = pipeline.load_existing_data()

# Get experiment summary
summary = pipeline.get_data_summary()
```

#### Command Line Usage
```bash
# Generate data for synth1
python src/pipeline.py synth1

# Force regenerate data
python src/pipeline.py synth1 --force

# Test the modular system
python test_modular_system.py
```

### Legacy Workflow & Notebooks

#### Synthetic Data Generation
1. **SLC Dataset**: 
   - Run `synth1.ipynb` to generate the SLC dataset

2. **LiLUCAS Dataset**:
   - Run `lucas6x3.ipynb` to generate the LiLUCAS dataset

#### Optimization and Learning

##### Gaussian Case
1. Run `opt_ell.ipynb` for optimization/learning
2. Results will be automatically saved
3. Run `eval_ell.ipynb` for evaluation

##### Empirical Case
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
├── requirements.txt              # Project dependencies
├── src/                         # Modular source code
│   ├── __init__.py
│   ├── experiment_config.py     # Experiment configuration management
│   ├── data_generator.py        # Data generation for experiments
│   ├── pipeline.py              # Main pipeline orchestration
│   └── [other modules]
├── example_data_generation.ipynb # Example notebook for new system
├── test_modular_system.py       # Test script for modular system
├── synth1.ipynb                 # Legacy SLC dataset generation
├── lucas6x3.ipynb               # Legacy LiLUCAS dataset generation
├── opt_ell.ipynb                # Gaussian case optimization
├── opt_emp.ipynb                # Empirical case optimization
├── eval_ell.ipynb               # Gaussian case evaluation
├── eval_emp.ipynb               # Empirical case evaluation
└── [other files/directories]
```

## Dependencies
This project requires the following main packages:
- PyTorch (>=2.1.0)
- NumPy (>=1.24.0)
- scikit-learn (>=1.3.0)
- NetworkX (>=3.2.0)

For a complete list of dependencies, see `requirements.txt`.