# Gaussian Optimization Script

This script converts the `gauss_optimization.ipynb` Jupyter notebook into a command-line runnable Python script that performs optimization experiments using DiRoCA, GRADCA, and BARYCA methods.

## Features

- **Command-line interface** with flexible argument parsing
- **Comprehensive logging** to both console and file
- **Modular optimization** - can run individual methods or all three
- **Organized output** with results saved to structured directories
- **Summary generation** with plots and metrics (optional)

## Usage

### Basic Usage

Run all optimizations with default settings:
```bash
python gauss_optimization.py
```

### Advanced Usage

Run with custom experiment and output directory:
```bash
python gauss_optimization.py --experiment my_experiment --output-dir results/my_experiment
```

Run only specific optimization methods:
```bash
# Run only DiRoCA
python gauss_optimization.py --skip-gradca --skip-baryca

# Run only GRADCA and BARYCA
python gauss_optimization.py --skip-diroca
```

Generate summary plots and analysis:
```bash
python gauss_optimization.py --generate-plots
```

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | str | `lilucas` | Experiment name to run |
| `--output-dir` | str | `data/{experiment}/results` | Output directory for results |
| `--skip-diroca` | flag | False | Skip DiRoCA optimization |
| `--skip-gradca` | flag | False | Skip GRADCA optimization |
| `--skip-baryca` | flag | False | Skip BARYCA optimization |
| `--generate-plots` | flag | False | Generate summary plots and analysis |

## Output Structure

The script creates the following directory structure:

```
data/{experiment}/results/
├── diroca_cv_results.pkl      # DiRoCA optimization results
├── gradca_cv_results.pkl      # GRADCA optimization results
├── baryca_cv_results.pkl      # BARYCA optimization results
├── optimization_summary.txt   # Text summary of results (if --generate-plots)
├── optimization_metrics.txt   # Numerical metrics (if --generate-plots)
├── plots/                     # Generated plots (if --generate-plots)
└── gauss_optimization.log     # Log file with execution details
```

## Expected Results

### Optimization Results Files

Each `.pkl` file contains a dictionary with the following structure:

**DiRoCA Results:**
```python
{
    'fold_0': {
        'eps_delta_4': {
            'T_matrix': ...,
            'optimization_params': {...},
            'test_indices': [...],
            'monitor': ...
        },
        'eps_delta_8': {...},
        # ... more hyperparameters
    },
    'fold_1': {...}
}
```

**GRADCA/BARYCA Results:**
```python
{
    'fold_0': {
        'gradca_run': {  # or 'baryca_run'
            'T_matrix': ...,
            'test_indices': [...]
        }
    },
    'fold_1': {...}
}
```

### Summary Files (with --generate-plots)

- **optimization_summary.txt**: Contains distribution summaries for both low-level and high-level models
- **optimization_metrics.txt**: Contains trajectory length and spread metrics
- **plots/**: Directory containing any generated visualizations

## Logging

The script provides comprehensive logging:
- **Console output**: Real-time progress updates
- **Log file**: Detailed execution log saved to `gauss_optimization.log`

## Example Commands

### Quick Test Run
```bash
# Run a quick test with just DiRoCA
python gauss_optimization.py --skip-gradca --skip-baryca --generate-plots
```

### Full Experiment
```bash
# Run complete experiment with all methods and analysis
python gauss_optimization.py --experiment lilucas --generate-plots
```

### Custom Output Location
```bash
# Save results to a custom location
python gauss_optimization.py --output-dir /path/to/my/results --generate-plots
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

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all local modules (`utilities`, `opt_tools_test`) are in the Python path
2. **Configuration errors**: Check that config files exist in the `configs/` directory
3. **Data errors**: Verify that experiment data exists in `data/{experiment}/`

### Log Analysis

Check the log file for detailed error information:
```bash
tail -f gauss_optimization.log
```

## Performance Notes

- The script runs optimizations sequentially by default
- DiRoCA optimization is the most computationally intensive
- Consider using `--skip-*` flags to run only the methods you need
- Results are automatically saved after each optimization completes 