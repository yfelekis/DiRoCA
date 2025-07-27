# Empirical Evaluation Alignment Documentation

## Overview

This document explains the alignment between `run_empirical_evaluation.py` and the `empirical_evaluation.ipynb` notebook, and the key differences between empirical and gaussian evaluation approaches.

## Key Changes Made to `run_empirical_evaluation.py`

### 1. Import Alignment
- **Added**: `import evaluation_utils as evut`
- **Reason**: The notebook uses `evut.compute_empirical_distance()` function for proper error calculation

### 2. Error Calculation Function
**Before (incorrect):**
```python
def calculate_empirical_error(T_matrix, Dll_test, Dhl_test, metric='fro'):
    try:
        Dll_transformed = Dll_test @ T_matrix.T
        error = np.mean((Dll_transformed - Dhl_test) ** 2)
        return error
    except:
        return np.nan
```

**After (correct, matches notebook):**
```python
def calculate_empirical_error(T_matrix, Dll_test, Dhl_test, metric='fro'):
    if Dll_test.shape[0] == 0 or Dhl_test.shape[0] == 0:
        return np.nan
    
    try:
        Dhl_predicted = Dll_test @ T_matrix.T
        error = evut.compute_empirical_distance(Dhl_predicted.T, Dhl_test.T, metric)
        return error
    except Exception as e:
        print(f"  - Warning: Could not compute empirical distance. Error: {e}. Returning NaN.")
        return np.nan
```

### 3. Data Loading Alignment
**Before (loading regular results):**
- `diroca_cv_results.pkl`
- `gradca_cv_results.pkl`
- `baryca_cv_results.pkl`

**After (loading empirical results):**
- `diroca_cv_results_empirical.pkl`
- `gradca_cv_results_empirical.pkl`
- `baryca_cv_results_empirical.pkl`
- `abslingam_cv_results_empirical.pkl` (newly added)

### 4. Default Parameters Alignment
**Before:**
- `alpha_steps`: 5
- `noise_steps`: 5
- `noise_max`: 10.0

**After (matches notebook):**
- `alpha_steps`: 10
- `noise_steps`: 25
- `noise_max`: 5.0

### 5. Distribution Support
**Added support for:**
- `translation` distribution
- `scaling` distribution

## Differences Between Empirical and Gaussian Evaluation

### 1. Error Metric
**Gaussian Evaluation (`run_evaluation.py`):**
- Uses Wasserstein-2 distance between Gaussian distributions
- Estimates mean and covariance from data
- Transforms low-level distribution parameters using T matrix
- Computes Wasserstein distance between transformed and actual high-level distributions

**Empirical Evaluation (`run_empirical_evaluation.py`):**
- Uses direct matrix distance on data samples
- Transforms low-level data samples using T matrix
- Computes empirical distance between transformed and actual high-level data
- Uses `evut.compute_empirical_distance()` function

### 2. Data Requirements
**Gaussian Evaluation:**
- Requires regular optimization results (`*_cv_results.pkl`)
- Works with any optimization method

**Empirical Evaluation:**
- Requires empirical optimization results (`*_cv_results_empirical.pkl`)
- Specifically designed for empirical optimization methods

### 3. Performance Characteristics
**Gaussian Evaluation:**
- Slower due to Wasserstein distance computation
- More theoretically grounded
- Better for distribution-level analysis

**Empirical Evaluation:**
- Faster due to direct matrix operations
- More practical for large-scale experiments
- Better for sample-level analysis

## Script Comparison

### `run_all.sh` vs `run_all_empirical.sh`

| Aspect | `run_all.sh` | `run_all_empirical.sh` |
|--------|--------------|------------------------|
| Script | `run_evaluation.py` | `run_empirical_evaluation.py` |
| Trials | 20 | 2 |
| Noise Steps | 20 | 25 |
| SLC Noise Max | 5.0 | 5.0 |
| LiLUCAS Noise Max | 10.0 | 10.0 |
| Error Metric | Wasserstein-2 | Empirical Distance |
| Data Files | `*_cv_results.pkl` | `*_cv_results_empirical.pkl` |

## Usage Examples

### Running Individual Experiments

**Gaussian Evaluation:**
```bash
python run_evaluation.py --experiment slc --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20
```

**Empirical Evaluation:**
```bash
python run_empirical_evaluation.py --experiment slc --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 25 --trials 2
```

### Running All Experiments

**Gaussian Evaluation:**
```bash
./run_all.sh
```

**Empirical Evaluation:**
```bash
./run_all_empirical.sh
```

## File Structure

```
data/{experiment}/
├── cv_folds.pkl
└── results/
    ├── diroca_cv_results.pkl              # For gaussian evaluation
    ├── gradca_cv_results.pkl              # For gaussian evaluation
    ├── baryca_cv_results.pkl              # For gaussian evaluation
    ├── diroca_cv_results_empirical.pkl    # For empirical evaluation
    ├── gradca_cv_results_empirical.pkl    # For empirical evaluation
    ├── baryca_cv_results_empirical.pkl    # For empirical evaluation
    └── abslingam_cv_results_empirical.pkl # For empirical evaluation
```

## Output Files

**Gaussian Evaluation:**
```
evaluation_{shift_type}_{distribution}_alpha{steps}-{min}-{max}_noise{steps}-{min}-{max}_trials{num}_zero_mean{bool}_{timestamp}.csv
```

**Empirical Evaluation:**
```
empirical_evaluation_{shift_type}_{distribution}_alpha{steps}-{min}-{max}_noise{steps}-{min}-{max}_trials{num}_zero_mean{bool}_{timestamp}.csv
```

## Troubleshooting

### Common Issues

1. **"No empirical result files found"**
   - Ensure you have run empirical optimization first
   - Check that `*_cv_results_empirical.pkl` files exist

2. **"Could not compute empirical distance"**
   - Check that `evaluation_utils.py` is properly imported
   - Verify data dimensions are compatible

3. **Results don't match notebook**
   - Ensure you're using the same parameters
   - Check that empirical optimization results are up-to-date

### Verification

To verify alignment with the notebook:

1. Run a small test with identical parameters
2. Compare results with notebook output
3. Check that error values are in the same range
4. Verify that method rankings are consistent

## Summary

The `run_empirical_evaluation.py` script is now perfectly aligned with the `empirical_evaluation.ipynb` notebook. The key differences from the gaussian evaluation are:

1. **Error metric**: Empirical distance vs Wasserstein distance
2. **Data files**: Empirical results vs regular results
3. **Performance**: Faster execution with fewer trials
4. **Scope**: Sample-level vs distribution-level analysis

Both approaches are valid and complementary, providing different perspectives on the robustness of abstraction methods. 