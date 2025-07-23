# Optimization Scripts Overview

This repository contains two main optimization scripts converted from Jupyter notebooks:

1. **`gauss_optimization.py`** - Gaussian optimization using theoretical bounds
2. **`empirical_optimization.py`** - Empirical optimization using noise estimates

## Quick Start

### Gaussian Optimization
```bash
# Run all Gaussian optimizations
python gauss_optimization.py

# Run only DiRoCA with plots
python gauss_optimization.py --skip-gradca --skip-baryca --generate-plots
```

### Empirical Optimization
```bash
# Run all empirical optimizations
python empirical_optimization.py

# Run only DiRoCA empirical with custom experiment
python empirical_optimization.py --experiment lilucas --skip-gradca --skip-baryca --skip-abslingam
```

## Script Comparison

| Feature | Gaussian (`gauss_optimization.py`) | Empirical (`empirical_optimization.py`) |
|---------|-----------------------------------|----------------------------------------|
| **Methods** | DiRoCA, GRADCA, BARYCA | DiRoCA, GRADCA, BARYCA, Abs-LiNGAM |
| **Radius Computation** | Theoretical bounds | Empirical bounds |
| **Data Usage** | Direct observations | Noise estimates + observations |
| **Config Files** | `*_opt_config.yaml` | `*_opt_config_empirical.yaml` |
| **Output Directory** | `data/{exp}/results/` | `data/{exp}/results_empirical/` |
| **Hyperparameters** | Fixed ε=δ values | Computed bounds + fixed values |
| **Cross-validation** | 2 folds | 10 folds |

## File Structure

```
DiRoCA_TBS/
├── gauss_optimization.py              # Gaussian optimization script
├── empirical_optimization.py          # Empirical optimization script
├── test_gauss_optimization.py         # Test script for Gaussian
├── test_empirical_optimization.py     # Test script for Empirical
├── README_gauss_optimization.md       # Gaussian script documentation
├── README_empirical_optimization.md   # Empirical script documentation
├── operations.py                      # Operations module (placeholder)
├── configs/
│   ├── diroca_opt_config.yaml         # Gaussian DiRoCA config
│   ├── gradca_opt_config.yaml         # Gaussian GRADCA config
│   ├── baryca_opt_config.yaml         # Gaussian BARYCA config
│   ├── diroca_opt_config_empirical.yaml # Empirical DiRoCA config
│   ├── gradca_opt_config_empirical.yaml # Empirical GRADCA config
│   └── baryca_opt_config_empirical.yaml # Empirical BARYCA config
└── data/
    └── {experiment}/
        ├── results/                    # Gaussian results
        │   ├── diroca_cv_results.pkl
        │   ├── gradca_cv_results.pkl
        │   ├── baryca_cv_results.pkl
        │   └── gauss_optimization.log
        └── results_empirical/          # Empirical results
            ├── diroca_cv_results_empirical.pkl
            ├── gradca_cv_results_empirical.pkl
            ├── baryca_cv_results_empirical.pkl
            ├── abslingam_cv_results_empirical.pkl
            └── empirical_optimization.log
```

## Common Usage Patterns

### 1. Quick Testing
```bash
# Test Gaussian optimization
python gauss_optimization.py --skip-gradca --skip-baryca

# Test Empirical optimization
python empirical_optimization.py --skip-gradca --skip-baryca --skip-abslingam
```

### 2. Full Experiments
```bash
# Run complete Gaussian experiment
python gauss_optimization.py --experiment lilucas --generate-plots

# Run complete Empirical experiment
python empirical_optimization.py --experiment lilucas
```

### 3. Custom Output Locations
```bash
# Gaussian with custom output
python gauss_optimization.py --output-dir results/my_gaussian_experiment

# Empirical with custom output
python empirical_optimization.py --output-dir results/my_empirical_experiment
```

### 4. Method Comparison
```bash
# Run only DiRoCA on both approaches
python gauss_optimization.py --skip-gradca --skip-baryca
python empirical_optimization.py --skip-gradca --skip-baryca --skip-abslingam
```

## Testing Your Setup

Before running the full optimizations, test your setup:

```bash
# Test Gaussian setup
python test_gauss_optimization.py

# Test Empirical setup
python test_empirical_optimization.py
```

Both test scripts will verify:
- All required imports
- Configuration files
- Data availability
- Script functionality

## Expected Results

### Gaussian Results Structure
```python
{
    'fold_0': {
        'eps_delta_4': {
            'T_matrix': ...,
            'optimization_params': {...},
            'test_indices': [...],
            'monitor': ...
        }
    }
}
```

### Empirical Results Structure
```python
{
    'fold_0': {
        'eps_0.321_delta_0.103': {
            'T_matrix': ...,
            'optimization_params': {...},
            'test_indices': [...]
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing Configuration Files**
   - Check that all config files exist in `configs/`
   - Ensure empirical configs are present for empirical optimization

3. **Data Issues**
   - Verify experiment data exists in `data/{experiment}/`
   - For empirical optimization, ensure noise data is available

4. **Memory Issues**
   - Empirical optimization can be memory-intensive
   - Consider running fewer folds or individual methods

### Log Analysis
```bash
# Check Gaussian optimization logs
tail -f gauss_optimization.log

# Check Empirical optimization logs
tail -f empirical_optimization.log
```

## Performance Notes

- **Gaussian optimization**: Faster, uses theoretical bounds
- **Empirical optimization**: More computationally intensive, uses noise estimates
- **DiRoCA**: Most intensive method in both cases
- **Abs-LiNGAM**: Fastest method (empirical only)
- Results are automatically saved after each optimization

## Dependencies

Required packages (see `requirements.txt`):
- numpy, scipy, matplotlib, seaborn
- networkx, pyyaml, joblib
- torch (for empirical optimization)
- plotly (for visualizations)

## Next Steps

After running the optimizations, you can:

1. **Analyze results** using the saved `.pkl` files
2. **Compare methods** across different approaches
3. **Generate visualizations** using the saved data
4. **Evaluate performance** using cross-validation results

## Support

For issues or questions:
1. Check the individual README files for detailed documentation
2. Run the test scripts to verify your setup
3. Check the log files for detailed error information
4. Ensure all dependencies are properly installed 