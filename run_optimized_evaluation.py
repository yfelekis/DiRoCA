#!/usr/bin/env python3
"""
Optimized Evaluation Script for DiRoCA TBS

This script demonstrates how to use the optimized evaluation functions
to significantly speed up the computation compared to the original notebook.

Usage:
    python run_optimized_evaluation.py

The script provides three levels of optimization:
1. Basic optimizations (sequential)
2. Parallel processing
3. Ultra-fast mode with simplified metrics
"""

import numpy as np
import pandas as pd
import joblib
import time
from tqdm import tqdm

# Import the optimized functions
from optimized_evaluation import (
    run_optimized_evaluation, 
    run_fast_evaluation,
    calculate_abstraction_error_optimized
)

def load_experiment_data(experiment='slc'):
    """Load all necessary data for the evaluation."""
    print(f"Loading data for experiment: {experiment}")
    
    # Load cross-validation folds
    saved_folds = joblib.load(f"data/{experiment}/cv_folds.pkl")
    
    # Load results
    path = f"data/{experiment}/results"
    diroca_results = joblib.load(f"{path}/diroca_cv_results.pkl")
    gradca_results = joblib.load(f"{path}/gradca_cv_results.pkl")
    baryca_results = joblib.load(f"{path}/baryca_cv_results.pkl")
    
    # Prepare results dictionary
    results_to_evaluate = {}
    results_to_evaluate['GradCA'] = gradca_results
    results_to_evaluate['BARYCA'] = baryca_results
    
    # Unpack DIROCA runs
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
    
    # Load model data
    import utilities as ut
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

def run_comparison():
    """Run a comparison between different optimization levels."""
    
    # Load data
    (saved_folds, results_to_evaluate, Dll_samples, Dhl_samples, 
     I_ll_relevant, omega, ll_var_names, hl_var_names, 
     base_sigma_L, base_sigma_H) = load_experiment_data()
    
    # Define evaluation parameters (reduced for demonstration)
    alpha_values = np.linspace(0, 1.0, 2)  # Reduced from 3
    noise_levels = np.linspace(0, 10.0, 5)  # Reduced from 10
    num_trials = 3  # Reduced from 10
    
    print(f"Evaluation parameters:")
    print(f"  - Alpha values: {alpha_values}")
    print(f"  - Noise levels: {noise_levels}")
    print(f"  - Number of trials: {num_trials}")
    print(f"  - Number of methods: {len(results_to_evaluate)}")
    print(f"  - Number of folds: {len(saved_folds)}")
    
    # Calculate total configurations
    total_configs = len(alpha_values) * len(noise_levels) * num_trials * len(saved_folds)
    total_methods = sum(len(results_dict[f'fold_0']) for results_dict in results_to_evaluate.values())
    total_evaluations = total_configs * total_methods * len(I_ll_relevant)
    
    print(f"  - Total evaluations: {total_evaluations:,}")
    print()
    
    # Test 1: Sequential optimized evaluation
    print("=" * 60)
    print("TEST 1: Sequential Optimized Evaluation")
    print("=" * 60)
    
    start_time = time.time()
    results_seq = run_optimized_evaluation(
        alpha_values, noise_levels, num_trials, saved_folds,
        results_to_evaluate, I_ll_relevant, Dll_samples, Dhl_samples,
        omega, ll_var_names, hl_var_names, base_sigma_L, base_sigma_H,
        use_parallel=False
    )
    seq_time = time.time() - start_time
    
    print(f"Sequential evaluation completed in {seq_time:.2f} seconds")
    print(f"Results shape: {results_seq.shape}")
    print()
    
    # Test 2: Parallel evaluation
    print("=" * 60)
    print("TEST 2: Parallel Evaluation")
    print("=" * 60)
    
    start_time = time.time()
    results_parallel = run_optimized_evaluation(
        alpha_values, noise_levels, num_trials, saved_folds,
        results_to_evaluate, I_ll_relevant, Dll_samples, Dhl_samples,
        omega, ll_var_names, hl_var_names, base_sigma_L, base_sigma_H,
        use_parallel=True, n_jobs=4  # Use 4 cores
    )
    parallel_time = time.time() - start_time
    
    print(f"Parallel evaluation completed in {parallel_time:.2f} seconds")
    print(f"Results shape: {results_parallel.shape}")
    print(f"Speedup: {seq_time/parallel_time:.2f}x")
    print()
    
    # Test 3: Ultra-fast evaluation
    print("=" * 60)
    print("TEST 3: Ultra-Fast Evaluation (Simplified Metrics)")
    print("=" * 60)
    
    start_time = time.time()
    results_fast = run_fast_evaluation(
        alpha_values, noise_levels, num_trials, saved_folds,
        results_to_evaluate, I_ll_relevant, Dll_samples, Dhl_samples,
        omega, ll_var_names, hl_var_names, base_sigma_L, base_sigma_H
    )
    fast_time = time.time() - start_time
    
    print(f"Ultra-fast evaluation completed in {fast_time:.2f} seconds")
    print(f"Results shape: {results_fast.shape}")
    print(f"Speedup vs sequential: {seq_time/fast_time:.2f}x")
    print(f"Speedup vs parallel: {parallel_time/fast_time:.2f}x")
    print()
    
    # Summary
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Sequential optimized: {seq_time:.2f}s")
    print(f"Parallel (4 cores):   {parallel_time:.2f}s ({(seq_time/parallel_time):.1f}x faster)")
    print(f"Ultra-fast:           {fast_time:.2f}s ({(seq_time/fast_time):.1f}x faster)")
    print()
    
    # Verify results are similar
    print("=" * 60)
    print("RESULT COMPARISON")
    print("=" * 60)
    
    # Compare mean errors for each method
    methods = results_seq['method'].unique()
    for method in methods:
        seq_mean = results_seq[results_seq['method'] == method]['error'].mean()
        par_mean = results_parallel[results_parallel['method'] == method]['error'].mean()
        fast_mean = results_fast[results_fast['method'] == method]['error'].mean()
        
        print(f"{method:20s}: Seq={seq_mean:.4f}, Par={par_mean:.4f}, Fast={fast_mean:.4f}")
    
    return results_seq, results_parallel, results_fast

def run_full_optimized_evaluation():
    """Run the full evaluation with optimizations."""
    
    # Load data
    (saved_folds, results_to_evaluate, Dll_samples, Dhl_samples, 
     I_ll_relevant, omega, ll_var_names, hl_var_names, 
     base_sigma_L, base_sigma_H) = load_experiment_data()
    
    # Full evaluation parameters
    alpha_values = np.linspace(0, 1.0, 3)
    noise_levels = np.linspace(0, 10.0, 10)
    num_trials = 10
    
    print("Running full optimized evaluation...")
    print(f"  - Alpha values: {len(alpha_values)}")
    print(f"  - Noise levels: {len(noise_levels)}")
    print(f"  - Number of trials: {num_trials}")
    
    start_time = time.time()
    
    # Use parallel processing for full evaluation
    final_results_df = run_optimized_evaluation(
        alpha_values, noise_levels, num_trials, saved_folds,
        results_to_evaluate, I_ll_relevant, Dll_samples, Dhl_samples,
        omega, ll_var_names, hl_var_names, base_sigma_L, base_sigma_H,
        use_parallel=True, n_jobs=None  # Auto-detect cores
    )
    
    total_time = time.time() - start_time
    
    print(f"\nFull evaluation completed in {total_time:.2f} seconds")
    print(f"Results shape: {final_results_df.shape}")
    
    # Save results
    final_results_df.to_csv('optimized_evaluation_results.csv', index=False)
    print("Results saved to 'optimized_evaluation_results.csv'")
    
    return final_results_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Run full evaluation
        results = run_full_optimized_evaluation()
    else:
        # Run comparison
        print("Running performance comparison...")
        print("Use 'python run_optimized_evaluation.py full' for full evaluation")
        print()
        results = run_comparison() 