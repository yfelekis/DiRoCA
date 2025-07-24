# DiRoCA TBS Evaluation Script

This directory contains a Python script that allows you to run the evaluation block from the `gauss_evaluation.ipynb` notebook from the command line with configurable parameters.

## Files

- `run_evaluation.py`: Main evaluation script
- `example_evaluation_usage.py`: Example script showing how to use the evaluation programmatically
- `README_evaluation_script.md`: This documentation file

## Quick Start

### Command Line Usage

Run the evaluation with default parameters:
```bash
python run_evaluation.py
```

Run with custom parameters:
```bash
python run_evaluation.py --experiment slc --alpha_min 0.1 --alpha_max 0.9 --alpha_steps 5 --noise_min 0.5 --noise_max 3.0 --noise_steps 8 --trials 5 --output results.csv
```

### Programmatic Usage

```python
from run_evaluation import run_evaluation
import numpy as np

# Define parameters
alpha_values = np.linspace(0, 1.0, 10)
noise_levels = np.linspace(0, 5.0, 10)

# Run evaluation
results_df = run_evaluation(
    experiment='slc',
    alpha_values=alpha_values,
    noise_levels=noise_levels,
    num_trials=3,
    zero_mean=True,
    shift_type='additive',
    distribution='gaussian',
    output_file='my_results.csv'  # Will be saved to data/slc/evaluation_results/my_results.csv
)

# Analyze results
print(f"Total records: {len(results_df)}")
print(f"Mean error: {results_df['error'].mean():.6f}")
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment` | str | 'slc' | Experiment name |
| `--alpha_min` | float | 0.0 | Minimum alpha value |
| `--alpha_max` | float | 1.0 | Maximum alpha value |
| `--alpha_steps` | int | 10 | Number of alpha steps |
| `--noise_min` | float | 0.0 | Minimum noise level |
| `--noise_max` | float | 5.0 | Maximum noise level |
| `--noise_steps` | int | 10 | Number of noise steps |
| `--trials` | int | 3 | Number of trials |
| `--zero_mean` | str | 'True' | Use zero mean for contamination |
| `--shift_type` | str | 'additive' | Type of shift ('additive' or 'multiplicative') |
| `--distribution` | str | 'gaussian' | Distribution type ('gaussian' or 'uniform') |
| `--output` | str | None | Output filename (saved to data/{experiment}/evaluation_results/) |

## Examples

### 1. Quick Test Run
```bash
python run_evaluation.py --alpha_steps 3 --noise_steps 4 --trials 2
```

### 2. Full Evaluation
```bash
python run_evaluation.py --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 20 --noise_min 0.0 --noise_max 10.0 --noise_steps 20 --trials 10 --output full_evaluation.csv
```

### 3. Custom Shift Configuration
```bash
python run_evaluation.py --shift_type multiplicative --distribution gaussian --zero_mean False
```

### 4. Different Experiment
```bash
python run_evaluation.py --experiment lilucas --alpha_steps 5 --noise_steps 5 --trials 3
```

## Output

The script outputs:
1. **Console output**: Progress bars and summary statistics
2. **CSV file**: Raw evaluation results saved to `data/{experiment}/evaluation_results/` with columns:
   - `method`: Method name (DIROCA, GradCA, BARYCA, etc.)
   - `alpha`: Contamination level
   - `noise_scale`: Noise scale level
   - `trial`: Trial number
   - `fold`: Cross-validation fold
   - `error`: Abstraction error

## Data Requirements

The script expects the following directory structure:
```
data/
└── {experiment}/
    ├── cv_folds.pkl
    ├── LLmodel.pkl
    ├── HLmodel.pkl
    ├── results/
    │   ├── diroca_cv_results.pkl
    │   ├── gradca_cv_results.pkl
    │   └── baryca_cv_results.pkl
    └── evaluation_results/  # Created automatically
        └── evaluation_*.csv
```

## Analysis Example

After running the evaluation, you can analyze the results:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_df = pd.read_csv('results.csv')

# Basic statistics
print(f"Total records: {len(results_df)}")
print(f"Methods: {results_df['method'].unique()}")

# Error by method
method_stats = results_df.groupby('method')['error'].agg(['mean', 'std'])
print(method_stats)

# Create visualization
plt.figure(figsize=(10, 6))
for method in results_df['method'].unique():
    method_data = results_df[results_df['method'] == method]
    alpha_means = method_data.groupby('alpha')['error'].mean()
    plt.plot(alpha_means.index, alpha_means.values, label=method, marker='o')

plt.xlabel('Alpha (Contamination Level)')
plt.ylabel('Mean Error')
plt.title('Error vs Alpha by Method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Performance Considerations

- **Total configurations** = alpha_steps × noise_steps × trials × folds × methods
- For a full evaluation (20×20×10×5×3 = 60,000 configurations), expect several hours of computation
- Use fewer steps for quick testing
- The script includes progress bars to monitor completion

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that the experiment data exists in `data/{experiment}/`
2. **ImportError**: Ensure all required packages are installed (`numpy`, `pandas`, `joblib`, `tqdm`)
3. **MemoryError**: Reduce the number of configurations or use a machine with more RAM

### Dependencies

Required packages:
```bash
pip install numpy pandas joblib tqdm matplotlib seaborn scipy scikit-learn
```

## Advanced Usage

### Custom Analysis Script

```python
from run_evaluation import run_evaluation
import numpy as np
import pandas as pd

# Run multiple evaluations with different parameters
results_list = []

for experiment in ['slc', 'lilucas']:
    for shift_type in ['additive', 'multiplicative']:
        results_df = run_evaluation(
            experiment=experiment,
            alpha_values=np.linspace(0, 1.0, 5),
            noise_levels=np.linspace(0, 3.0, 5),
            num_trials=3,
            shift_type=shift_type
        )
        results_df['experiment'] = experiment
        results_df['shift_type'] = shift_type
        results_list.append(results_df)

# Combine results
all_results = pd.concat(results_list, ignore_index=True)
all_results.to_csv('combined_results.csv', index=False)
```

This evaluation script provides a flexible and efficient way to run the DiRoCA TBS evaluation with configurable parameters, making it easy to experiment with different settings and analyze the results systematically. 