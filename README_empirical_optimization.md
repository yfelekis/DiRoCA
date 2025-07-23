# Empirical Optimization Script

This script converts the `empirical_optimization.ipynb` Jupyter notebook into a command-line runnable Python script that performs empirical optimization experiments using DiRoCA, GRADCA, BARYCA, and Abs-LiNGAM methods.

## Features

- **Command-line interface** with flexible argument parsing
- **Comprehensive logging** to both console and file
- **Modular optimization** - can run individual methods or all four
- **Organized output** with results saved to structured directories
- **Empirical radius computation** with theoretical bounds
- **Cross-validation support** with multiple folds

## Usage

### Basic Usage

Run all empirical optimizations with default settings:
```bash
python empirical_optimization.py
```

### Advanced Usage

Run with custom experiment and output directory:
```bash
python empirical_optimization.py --experiment my_experiment --output-dir results/my_experiment
```

Run only specific optimization methods:
```bash
# Run only DiRoCA empirical optimization
python empirical_optimization.py --skip-gradca --skip-baryca --skip-abslingam

# Run only GRADCA and BARYCA
python empirical_optimization.py --skip-diroca --skip-abslingam
```

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | str | `lilucas` | Experiment name to run |
| `--output-dir` | str | `data/{experiment}/results_empirical` | Output directory for results |
| `--skip-diroca` | flag | False | Skip DiRoCA empirical optimization |
| `--skip-gradca` | flag | False | Skip GRADCA empirical optimization |
| `--skip-baryca` | flag | False | Skip BARYCA empirical optimization |
| `--skip-abslingam` | flag | False | Skip Abs-LiNGAM optimization |

## Output Structure

The script creates the following directory structure:

```
data/{experiment}/results_empirical/
├── diroca_cv_results_empirical.pkl      # DiRoCA empirical optimization results
├── gradca_cv_results_empirical.pkl      # GRADCA empirical optimization results
├── baryca_cv_results_empirical.pkl      # BARYCA empirical optimization results
├── abslingam_cv_results_empirical.pkl   # Abs-LiNGAM optimization results
└── empirical_optimization.log           # Log file with execution details
```

## Expected Results

### Optimization Results Files

Each `.pkl` file contains a dictionary with the following structure:

**DiRoCA Empirical Results:**
```python
{
    'fold_0': {
        'eps_0.321_delta_0.103': {
            'T_matrix': ...,
            'optimization_params': {...},
            'test_indices': [...]
        },
        'eps_1.0_delta_1.0': {...},
        # ... more hyperparameter combinations
    },
    'fold_1': {...}
}
```

**GRADCA/BARYCA Empirical Results:**
```python
{
    'fold_0': {
        'gradca_run': {  # or 'baryca_run'
            'T_matrix': ...,
            'optimization_params': {...},
            'test_indices': [...]
        }
    },
    'fold_1': {...}
}
```

**Abs-LiNGAM Results:**
```python
{
    'fold_0': {
        'Perfect': {
            'T_matrix': ...,
            'test_indices': [...]
        },
        'Noisy': {
            'T_matrix': ...,
            'test_indices': [...]
        }
    },
    'fold_1': {...}
}
```

## Key Differences from Gaussian Optimization

### Empirical vs Gaussian Methods

1. **Empirical Radius Computation**: Uses `ut.compute_empirical_radius()` instead of theoretical bounds
2. **Noise Data**: Utilizes `U_ll_hat` and `U_hl_hat` noise estimates from the data
3. **Empirical Parameters**: Uses `ut.assemble_empirical_parameters()` for parameter assembly
4. **Empirical Optimization Functions**: Calls `optools.run_empirical_erica_optimization()` and `optools.run_empirical_bary_optim()`
5. **Abs-LiNGAM**: Additional baseline method for comparison

### Hyperparameter Testing

The DiRoCA empirical optimization tests multiple (ε, δ) pairs:
- Theoretical bounds computed for each fold
- Fixed values: (1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0)

## Logging

The script provides comprehensive logging:
- **Console output**: Real-time progress updates
- **Log file**: Detailed execution log saved to `empirical_optimization.log`

## Example Commands

### Quick Test Run
```bash
# Run a quick test with just DiRoCA empirical optimization
python empirical_optimization.py --skip-gradca --skip-baryca --skip-abslingam
```

### Full Experiment
```bash
# Run complete empirical experiment with all methods
python empirical_optimization.py --experiment lilucas
```

### Custom Output Location
```bash
# Save results to a custom location
python empirical_optimization.py --output-dir /path/to/my/empirical_results
```

### Run Only Baseline Methods
```bash
# Run only Abs-LiNGAM for comparison
python empirical_optimization.py --skip-diroca --skip-gradca --skip-baryca
```

## Dependencies

Make sure you have all required dependencies installed:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- scipy
- matplotlib
- seaborn
- networkx
- pyyaml
- joblib
- torch (for empirical optimization)

## Configuration Files

The script uses empirical-specific configuration files:
- `configs/diroca_opt_config_empirical.yaml`
- `configs/gradca_opt_config_empirical.yaml`
- `configs/baryca_opt_config_empirical.yaml`

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all local modules (`utilities`, `opt_tools`) are in the Python path
2. **Configuration errors**: Check that empirical config files exist in the `configs/` directory
3. **Data errors**: Verify that experiment data exists in `data/{experiment}/` and contains noise estimates
4. **Memory issues**: Empirical optimization can be memory-intensive; consider running fewer folds

### Log Analysis

Check the log file for detailed error information:
```bash
tail -f empirical_optimization.log
```

## Performance Notes

- The script runs optimizations sequentially by default
- DiRoCA empirical optimization is the most computationally intensive due to multiple hyperparameter testing
- Abs-LiNGAM is typically the fastest method
- Consider using `--skip-*` flags to run only the methods you need
- Results are automatically saved after each optimization completes

## Comparison with Gaussian Optimization

| Feature | Gaussian | Empirical |
|---------|----------|-----------|
| Radius computation | Theoretical bounds | Empirical bounds |
| Data usage | Direct observations | Noise estimates + observations |
| Methods | DiRoCA, GRADCA, BARYCA | DiRoCA, GRADCA, BARYCA, Abs-LiNGAM |
| Hyperparameters | Fixed ε=δ values | Computed bounds + fixed values |
| Output directory | `results/` | `results_empirical/` | 