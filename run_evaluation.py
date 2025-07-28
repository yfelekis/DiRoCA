#!/usr/bin/env python3
"""
Evaluation Script for DiRoCA TBS

This script runs the evaluation block from the gauss_evaluation.ipynb notebook
with configurable parameters that can be set via command line arguments.

Usage:
    python run_evaluation.py [--experiment EXPERIMENT] [--alpha_min ALPHA_MIN] [--alpha_max ALPHA_MAX] 
                            [--alpha_steps ALPHA_STEPS] [--noise_min NOISE_MIN] [--noise_max NOISE_MAX] 
                            [--noise_steps NOISE_STEPS] [--trials TRIALS] [--zero_mean ZERO_MEAN]
                            [--output OUTPUT] [--shift_type SHIFT_TYPE] [--distribution DISTRIBUTION]

Examples:
    # Run with default parameters
    python run_evaluation.py
    
    # Run with custom parameters
    python run_evaluation.py --experiment slc --alpha_min 0.1 --alpha_max 0.9 --alpha_steps 5 
                             --noise_min 0.5 --noise_max 3.0 --noise_steps 8 --trials 5
    
    # Run with custom shift configuration
    python run_evaluation.py --shift_type multiplicative --distribution uniform --zero_mean False
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
from tqdm import tqdm
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')

# Import local modules
import utilities as ut
import modularised_utils as mut

import numpy as np
import scipy.stats as stats


def apply_shift(clean_data, shift_config, all_var_names, model_level):
    """
    Applies a specified contamination to the test data.
    - Handles different shift types (additive, multiplicative).
    - Handles different distributions (gaussian, student-t, exponential).
    - Handles selective application to a subset of variables.
    """

    shift_type = shift_config.get('type')
    dist_type  = shift_config.get('distribution', 'gaussian')

    n_samples, n_dims = clean_data.shape

    # Select the correct parameter dictionary for the current model level
    level_key = 'll_params' if model_level == 'L' else 'hl_params'
    params = shift_config.get(level_key, {})
    
    # --- 1. Generate the full noise matrix based on the specified distribution ---
    noise_matrix = np.zeros_like(clean_data)
    if dist_type == 'gaussian':
        mu           = np.array(params.get('mu', np.zeros(n_dims)))
        sigma_def    = params.get('sigma', np.eye(n_dims))
        sigma        = np.diag(np.array(sigma_def)) if np.array(sigma_def).ndim == 1 else np.array(sigma_def)
        noise_matrix = np.random.multivariate_normal(mean=mu, cov=sigma, size=n_samples)

    elif dist_type == 'student-t':
        df           = params.get('df', 3)
        loc          = np.array(params.get('loc', np.zeros(n_dims)))
        shape_def    = params.get('shape', np.eye(n_dims))
        shape        = np.diag(np.array(shape_def)) if np.array(shape_def).ndim == 1 else np.array(shape_def)
        noise_matrix = stats.multivariate_t.rvs(loc=loc, shape=shape, df=df, size=n_samples)

    elif dist_type == 'exponential':
        scale        = params.get('scale', 1.0)
        noise_matrix = np.random.exponential(scale=scale, size=(n_samples, n_dims))
    
    elif dist_type == 'translation':
        c           = params.get('c', 0.5)
        noise_matrix = np.ones((n_samples, n_dims)) * c
    
    elif dist_type == 'scaling':
        c           = params.get('c', 1.5)
        noise_matrix = np.ones((n_samples, n_dims)) * c
    
    # --- 2. Apply noise selectively if specified ---
    final_noise = np.zeros_like(clean_data)
    vars_to_affect = params.get('apply_to_vars')

    if vars_to_affect is None:
        # If not specified, apply noise to all variables
        final_noise = noise_matrix
    else:
        # If specified, apply noise only to the selected columns
        try:
            indices_to_affect = [all_var_names.index(var) for var in vars_to_affect]
            final_noise[:, indices_to_affect] = noise_matrix[:, indices_to_affect]
        except ValueError as e:
            print(f"Warning: A variable in 'apply_to_vars' not found. Error: {e}")
            return clean_data # Return clean data if there's a config error

    # --- 3. Return the contaminated data ---
    if shift_type == 'additive':
        return clean_data + final_noise
    elif shift_type == 'multiplicative':
        return clean_data * final_noise
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

def apply_huber_contamination(clean_data, alpha, shift_config, all_var_names, model_level, seed=42):
    """
    Contaminates a dataset using the Huber model. A fraction 'alpha' of the
    samples are replaced with noisy versions.

    Args:
        clean_data (np.ndarray): The original, clean test data samples.
        alpha (float): The fraction of data to contaminate (between 0 and 1).
        shift_config (dict): Configuration defining the noise for the outliers.
        all_var_names (list): List of all variable names for this data.
        model_level (str): 'L' for low-level or 'H' for high-level.
        
    Returns:
        np.ndarray: The new, contaminated test data.
    """
    #np.random.seed(seed)
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    if alpha == 0:
        return clean_data
    
    # Create the fully noisy version of the data using our existing function
    noisy_data = apply_shift(clean_data, shift_config, all_var_names, model_level)
    
    if alpha == 1:
        return noisy_data
        
    n_samples = clean_data.shape[0]
    n_to_contaminate = int(alpha * n_samples)
    
    # Randomly select which rows to replace
    indices_to_replace = np.random.choice(n_samples, n_to_contaminate, replace=False)
    
    # Start with a copy of the clean data
    contaminated_data = clean_data.copy()
    
    # Replace the selected rows with their noisy versions
    contaminated_data[indices_to_replace] = noisy_data[indices_to_replace]
    
    return contaminated_data

def calculate_abstraction_error(T_matrix, Dll_test, Dhl_test):
    """
    Calculates the abstraction error for a given T matrix on a test set.

    This function works in the space of distribution parameters:
    1. It estimates Gaussian parameters (mean, cov) from the LL and HL test samples.
    2. It transforms the LL Gaussian's parameters using the T matrix.
    3. It computes the Wasserstein distance between the transformed LL distribution
       and the actual HL distribution.
    
    Args:
        T_matrix (np.ndarray): The learned abstraction matrix.
        Dll_test (np.ndarray): The low-level endogenous test samples.
        Dhl_test (np.ndarray): The high-level endogenous test samples.
        
    Returns:
        float: The calculated Wasserstein-2 distance.
    """
    # 1. Estimate parameters from the low-level test data
    mu_L_test    = np.mean(Dll_test, axis=0)
    Sigma_L_test = np.cov(Dll_test, rowvar=False)

    # 2. Estimate parameters from the high-level test data
    mu_H_test    = np.mean(Dhl_test, axis=0)
    Sigma_H_test = np.cov(Dhl_test, rowvar=False)

    # 3. Transform the low-level parameters using the T matrix
    # This projects the low-level distribution into the high-level space
    mu_V_predicted    = mu_L_test @ T_matrix.T
    Sigma_V_predicted = T_matrix @ Sigma_L_test @ T_matrix.T
    
    # 4. Compute the Wasserstein distance between the two resulting Gaussians
    try:
        wasserstein_dist = np.sqrt(mut.compute_wasserstein(mu_V_predicted, Sigma_V_predicted, mu_H_test, Sigma_H_test))
    except Exception as e:
        print(f"  - Warning: Could not compute Wasserstein distance. Error: {e}. Returning NaN.")
        return np.nan

    return wasserstein_dist

def load_experiment_data(experiment='slc'):
    """
    Load all necessary data for the evaluation.
    
    Parameters:
    - experiment: str, name of the experiment
    
    Returns:
    - tuple: all necessary data for evaluation
    """
    print(f"Loading data for experiment: {experiment}")
    
    # Check if experiment data exists
    data_path = f"data/{experiment}"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Experiment data not found at {data_path}")
    
    # Load cross-validation folds
    folds_path = f"{data_path}/cv_folds.pkl"
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"CV folds not found at {folds_path}")
    
    saved_folds = joblib.load(folds_path)
    
    # Load results
    results_path = f"{data_path}/results"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results directory not found at {results_path}")
    
    # Try to load different result files
    results_to_evaluate = {}
    
    # Load DIROCA results
    diroca_path = f"{results_path}/diroca_cv_results.pkl"
    if os.path.exists(diroca_path):
        diroca_results = joblib.load(diroca_path)
        if diroca_results:
            first_fold_key = list(diroca_results.keys())[0]
            diroca_run_ids = list(diroca_results[first_fold_key].keys())
            
            for run_id in diroca_run_ids:
                method_name = f"DIROCA ({run_id})"
                new_diroca_dict = {}
                for fold_key, fold_results in diroca_results.items():
                    if run_id in fold_results:
                        new_diroca_dict[fold_key] = {run_id: fold_results[run_id]}
                results_to_evaluate[method_name] = new_diroca_dict
    
    # Load GradCA results
    gradca_path = f"{results_path}/gradca_cv_results.pkl"
    if os.path.exists(gradca_path):
        gradca_results = joblib.load(gradca_path)
        results_to_evaluate['GradCA'] = gradca_results
    
    # Load BARYCA results
    baryca_path = f"{results_path}/baryca_cv_results.pkl"
    if os.path.exists(baryca_path):
        baryca_results = joblib.load(baryca_path)
        results_to_evaluate['BARYCA'] = baryca_results
    
    if not results_to_evaluate:
        raise FileNotFoundError(f"No result files found in {results_path}")
    
    # Load model data
    all_data = ut.load_all_data(experiment)
    
    Dll_samples = all_data['LLmodel']['data']
    Dhl_samples = all_data['HLmodel']['data']
    I_ll_relevant = all_data['LLmodel']['intervention_set']
    omega = all_data['abstraction_data']['omega']
    ll_var_names = list(all_data['LLmodel']['graph'].nodes())
    hl_var_names = list(all_data['HLmodel']['graph'].nodes())
    
    # Define base covariance matrices
    base_sigma_L = np.eye(len(all_data['LLmodel']['graph'].nodes()))
    base_sigma_H = np.eye(len(all_data['HLmodel']['graph'].nodes()))
    
    return (saved_folds, results_to_evaluate, Dll_samples, Dhl_samples, 
            I_ll_relevant, omega, ll_var_names, hl_var_names, 
            base_sigma_L, base_sigma_H)

def run_evaluation(experiment='slc', alpha_values=None, noise_levels=None, 
                  num_trials=20, zero_mean=True, shift_type='additive', 
                  distribution='gaussian', output_file=None):
    """
    Run the evaluation with specified parameters.
    
    Parameters:
    - experiment: str, name of the experiment
    - alpha_values: array-like, contamination levels
    - noise_levels: array-like, noise scale levels
    - num_trials: int, number of trials per configuration
    - zero_mean: bool, whether to use zero mean for contamination
    - shift_type: str, type of shift ('additive' or 'multiplicative')
    - distribution: str, distribution type ('gaussian' or 'uniform')
    - output_file: str, filename to save results (optional, will be saved to data/{experiment}/evaluation_results/)
    
    Returns:
    - pd.DataFrame: evaluation results
    """
    # Set default values if not provided
    if alpha_values is None:
        alpha_values = np.linspace(0, 1.0, 10)
    if noise_levels is None:
        noise_levels = np.linspace(0, 10.0, 20)
    
    # Load experiment data
    (saved_folds, results_to_evaluate, Dll_samples, Dhl_samples, 
     I_ll_relevant, omega, ll_var_names, hl_var_names, 
     base_sigma_L, base_sigma_H) = load_experiment_data(experiment)
    
    print(f"Starting evaluation with parameters:")
    print(f"  - Experiment: {experiment}")
    print(f"  - Alpha values: {len(alpha_values)} points from {alpha_values[0]:.2f} to {alpha_values[-1]:.2f}")
    print(f"  - Noise levels: {len(noise_levels)} points from {noise_levels[0]:.2f} to {noise_levels[-1]:.2f}")
    print(f"  - Number of trials: {num_trials}")
    print(f"  - Zero mean: {zero_mean}")
    print(f"  - Shift type: {shift_type}")
    print(f"  - Distribution: {distribution}")
    print(f"  - Number of methods: {len(results_to_evaluate)}")
    print(f"  - Number of folds: {len(saved_folds)}")
    
    # Calculate total configurations
    total_configs = len(alpha_values) * len(noise_levels) * num_trials * len(saved_folds) * len(results_to_evaluate)
    print(f"  - Total configurations to evaluate: {total_configs:,}")
    
    evaluation_records = []
    
    # Main evaluation loop
    for alpha in tqdm(alpha_values, desc="Alpha Levels"):
        for scale in noise_levels:
            for trial in range(num_trials):
                for i, fold_info in enumerate(saved_folds):
                    for method_name, results_dict in results_to_evaluate.items():
                        
                        fold_results = results_dict[f'fold_{i}']
                        for run_key, run_data in fold_results.items():
                            
                            T_learned = run_data['T_matrix']
                            test_indices = run_data['test_indices']
                            
                            errors_per_intervention = []
                            for iota in I_ll_relevant:

                                Dll_test_clean = Dll_samples[iota][test_indices]
                                Dhl_test_clean = Dhl_samples[omega[iota]][test_indices]
                                
                                # Configure shift parameters
                                if zero_mean:
                                    mu_scale_L = np.zeros(base_sigma_L.shape[0])
                                    mu_scale_H = np.zeros(base_sigma_H.shape[0])
                                else:
                                    mu_scale_L = np.ones(base_sigma_L.shape[0]) * scale
                                    mu_scale_H = np.ones(base_sigma_H.shape[0]) * scale
                                
                                sigma_scale_L = base_sigma_L * (scale**2)
                                sigma_scale_H = base_sigma_H * (scale**2)
                                if distribution == 'gaussian':
                                    shift_config = {
                                                        'type': shift_type, 
                                                        'distribution': distribution,
                                                        'll_params': {'mu': mu_scale_L, 'sigma': sigma_scale_L},
                                                        'hl_params': {'mu': mu_scale_H, 'sigma': sigma_scale_H}
                                                    }
                                elif distribution == 'exponential':
                                    shift_config = {
                                        'type': shift_type, 
                                        'distribution': distribution,
                                        'll_params': {'scale': scale},
                                        'hl_params': {'scale': scale}
                                    }
                                elif distribution == 'student-t':
                                    shift_config = {
                                        'type': shift_type, 
                                        'distribution': distribution,
                                        'll_params': {'df': 3, 'loc': np.zeros(base_sigma_L.shape[0]), 'shape': base_sigma_L * (scale**2)},
                                        'hl_params': {'df': 3, 'loc': np.zeros(base_sigma_H.shape[0]), 'shape': base_sigma_H * (scale**2)}
                                    }
                                elif distribution == 'translation':
                                    shift_config = {
                                        'type': 'additive', 
                                        'distribution': distribution,
                                        'll_params': {'c': scale},
                                        'hl_params': {'c': scale}
                                    }
                                elif distribution == 'scaling':
                                    shift_config = {
                                        'type': 'multiplicative', 
                                        'distribution': distribution,
                                        'll_params': {'c': scale},
                                        'hl_params': {'c': scale}
                                    }
                                
                                Dll_test_contaminated = apply_huber_contamination(
                                    Dll_test_clean, alpha, shift_config, ll_var_names, 'L', seed=trial)
                                Dhl_test_contaminated = apply_huber_contamination(
                                    Dhl_test_clean, alpha, shift_config, hl_var_names, 'H', seed=trial)
                                
                                error = calculate_abstraction_error(T_learned, Dll_test_contaminated, Dhl_test_contaminated)
                                if not np.isnan(error):
                                    errors_per_intervention.append(error)
                            
                            avg_error = np.mean(errors_per_intervention) if errors_per_intervention else np.nan
                            
                            record = {
                                'method': method_name,
                                'alpha': alpha,
                                'noise_scale': scale,
                                'trial': trial,
                                'fold': i,
                                'error': avg_error
                            }
                            evaluation_records.append(record)

    final_results_df = pd.DataFrame(evaluation_records)
    
    print("--- Full Evaluation Complete ---")
    print(f"Generated {len(final_results_df)} evaluation records")
    
    # Create output directory and save results
    output_dir = f"data/{experiment}/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on parameters
    if output_file is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        alpha_min, alpha_max = alpha_values[0], alpha_values[-1]
        noise_min, noise_max = noise_levels[0], noise_levels[-1]
        filename = f"evaluation_{shift_type}_{distribution}_alpha{len(alpha_values)}-{alpha_min:.1f}-{alpha_max:.1f}_noise{len(noise_levels)}-{noise_min:.1f}-{noise_max:.1f}_trials{num_trials}_zero_mean{zero_mean}_{timestamp}.csv"
        output_file = os.path.join(output_dir, filename)
    
    # Ensure the output file is in the correct directory
    if not os.path.dirname(output_file):
        output_file = os.path.join(output_dir, os.path.basename(output_file))
    elif not output_file.startswith(output_dir):
        output_file = os.path.join(output_dir, os.path.basename(output_file))
    
    final_results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return final_results_df

def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Run DiRoCA TBS evaluation with configurable parameters')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default='slc',
                       help='Experiment name (default: slc)')
    
    # Alpha parameters
    parser.add_argument('--alpha_min', type=float, default=0.0,
                       help='Minimum alpha value (default: 0.0)')
    parser.add_argument('--alpha_max', type=float, default=1.0,
                       help='Maximum alpha value (default: 1.0)')
    parser.add_argument('--alpha_steps', type=int, default=10,
                       help='Number of alpha steps (default: 10)')
    
    # Noise parameters
    parser.add_argument('--noise_min', type=float, default=0.0,
                       help='Minimum noise level (default: 0.0)')
    parser.add_argument('--noise_max', type=float, default=10.0,
                       help='Maximum noise level (default: 10.0)')
    parser.add_argument('--noise_steps', type=int, default=20,
                       help='Number of noise steps (default: 20)')
    
    # Trial parameters
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of trials (default: 20)')
    
    # Shift configuration
    parser.add_argument('--zero_mean', type=str, default='True',
                       help='Use zero mean for contamination (default: True)')
    parser.add_argument('--shift_type', type=str, default='additive',
                       choices=['additive', 'multiplicative'],
                       help='Type of shift (default: additive)')
    parser.add_argument('--distribution', type=str, default='gaussian',
                       choices=['gaussian', 'exponential', 'student-t', 'translation', 'scaling'],
                       help='Distribution type (default: gaussian)')
    

    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (optional, will be saved to data/{experiment}/evaluation_results/)')
    
    args = parser.parse_args()
    
    # Convert zero_mean string to boolean
    zero_mean = args.zero_mean.lower() in ['true', '1', 'yes', 'y']
    
    # Create parameter arrays
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    noise_levels = np.linspace(args.noise_min, args.noise_max, args.noise_steps)
    
    try:
        # Run evaluation
        results_df = run_evaluation(
            experiment=args.experiment,
            alpha_values=alpha_values,
            noise_levels=noise_levels,
            num_trials=args.trials,
            zero_mean=zero_mean,
            shift_type=args.shift_type,
            distribution=args.distribution,
            output_file=args.output
        )
        
        # Print summary statistics
        print("\n--- Summary ---")
        print(f"Total records: {len(results_df)}")
        print(f"Methods evaluated: {results_df['method'].nunique()}")
        print(f"Unique alpha values: {results_df['alpha'].nunique()}")
        print(f"Unique noise scales: {results_df['noise_scale'].nunique()}")
        
        # Return results for further analysis
        return results_df
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    results = main() 