# Empirical Evaluation Script Documentation

## Overview

The `run_empirical_evaluation.py` script provides a command-line interface to run empirical evaluations from the `empirical_evaluation.ipynb` notebook. It calculates empirical errors (mean squared error) between transformed low-level data and high-level data, rather than using the Wasserstein distance approach of the gaussian evaluation.

## Key Differences from Gaussian Evaluation

- **Error Metric**: Uses empirical mean squared error instead of Wasserstein-2 distance
- **Default Parameters**: Smaller default values (5 alpha steps, 5 noise steps, 2 trials) for faster execution
- **Output Filename**: Prefixed with "empirical_evaluation_" to distinguish from gaussian evaluations

## Usage

### Basic Usage

```bash
# Run with default parameters
python run_empirical_evaluation.py

# Run with custom parameters
python run_empirical_evaluation.py --experiment slc --alpha_min 0.1 --alpha_max 0.9 --alpha_steps 5 
                                  --noise_min 0.5 --noise_max 3.0 --noise_steps 8 --trials 5
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | str | 'slc' | Experiment name |
| `--alpha_min` | float | 0.0 | Minimum alpha value |
| `--alpha_max` | float | 1.0 | Maximum alpha value |
| `--alpha_steps` | int | 5 | Number of alpha steps |
| `--noise_min` | float | 0.0 | Minimum noise level |
| `--noise_max` | float | 10.0 | Maximum noise level |
| `--noise_steps` | int | 5 | Number of noise steps |
| `--trials` | int | 2 | Number of trials |
| `--zero_mean` | str | 'True' | Use zero mean for contamination |
| `--shift_type` | str | 'additive' | Type of shift ('additive' or 'multiplicative') |
| `--distribution` | str | 'gaussian' | Distribution type ('gaussian', 'exponential', 'student-t') |
| `--output` | str | None | Output filename (optional) |

### Examples

```bash
# Quick test run
python run_empirical_evaluation.py --alpha_steps 2 --noise_steps 2 --trials 1

# Full evaluation with custom parameters
python run_empirical_evaluation.py --experiment slc --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 
                                  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 10

# Different distribution
python run_empirical_evaluation.py --distribution exponential --shift_type multiplicative

# Non-zero mean contamination
python run_empirical_evaluation.py --zero_mean False
```

## Output

### File Location
Results are automatically saved to:
```
data/{experiment}/evaluation_results/
```

### Filename Format
```
empirical_evaluation_{shift_type}_{distribution}_alpha{steps}-{min}-{max}_noise{steps}-{min}-{max}_trials{num}_zero_mean{bool}_{timestamp}.csv
```

Example:
```
empirical_evaluation_additive_gaussian_alpha5-0.0-1.0_noise5-0.0-10.0_trials2_zero_meanTrue_20250725_134645.csv
```

### Output Format
The CSV file contains the following columns:
- `method`: Method name (e.g., "DIROCA (run_1)", "GradCA", "BARYCA")
- `alpha`: Contamination level (0 to 1)
- `noise_scale`: Noise scale level
- `trial`: Trial number
- `fold`: Cross-validation fold number
- `error`: Empirical error (mean squared error)

## Loading Results

### Using the Loading Helper

```python
from load_results import load_empirical_latest, load_empirical_additive_gaussian

# Load latest empirical results
df = load_empirical_latest('slc')

# Load specific empirical results
df = load_empirical_additive_gaussian('slc')
```

### Direct Loading

```python
import pandas as pd

# Load by filename
df = pd.read_csv("data/slc/evaluation_results/empirical_evaluation_additive_gaussian_alpha5-0.0-1.0_noise5-0.0-10.0_trials2_zero_meanTrue_20250725_134645.csv")
```

## Comparison with Gaussian Evaluation

### Loading Both Types

```python
from load_results import load_gaussian_latest, load_empirical_latest

# Load both types
gaussian_df = load_gaussian_latest('slc')
empirical_df = load_empirical_latest('slc')

# Compare results
print(f"Gaussian mean error: {gaussian_df['error'].mean():.4f}")
print(f"Empirical mean error: {empirical_df['error'].mean():.4f}")
```

### Running Comparison

Use the `example_empirical_usage.py` script to run both evaluations and compare results:

```bash
python example_empirical_usage.py
```

## Data Requirements

The script requires the same data structure as the gaussian evaluation:

```
data/{experiment}/
├── cv_folds.pkl
└── results/
    ├── diroca_cv_results.pkl
    ├── gradca_cv_results.pkl
    └── baryca_cv_results.pkl
```

## Error Handling

The script includes comprehensive error handling for:
- Missing experiment data
- Missing result files
- Invalid parameter combinations
- File I/O errors

## Performance Considerations

- Empirical evaluation is generally faster than gaussian evaluation
- Default parameters are optimized for quick testing
- For production runs, consider increasing `alpha_steps`, `noise_steps`, and `trials`
- Memory usage scales with the number of configurations and data size

## Troubleshooting

### Common Issues

1. **"Experiment data not found"**: Ensure the experiment directory exists in `data/`
2. **"No result files found"**: Check that optimization results exist in `data/{experiment}/results/`
3. **"Could not compute error"**: May indicate numerical issues with the data

### Debug Mode

For debugging, you can modify the script to add more verbose output or check intermediate results.

## Integration with Analysis Workflow

The empirical evaluation results can be seamlessly integrated into your analysis workflow:

1. Run empirical evaluation: `python run_empirical_evaluation.py`
2. Load results: `df = load_empirical_latest('slc')`
3. Analyze and visualize results
4. Compare with gaussian evaluation results

This provides a complete pipeline for both theoretical (gaussian) and empirical evaluation approaches. 