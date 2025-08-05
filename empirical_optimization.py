#!/usr/bin/env python3
"""
Empirical Optimization Script
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml
import utilities as ut 
import opt_tools as optools
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('empirical_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(experiment_name):
    """Setup the experiment configuration and data."""
    logger.info(f"Setting up experiment: {experiment_name}")
    
    # Load configuration files
    config_files = {
        'hyperparams_diroca': 'configs/diroca_opt_config_empirical.yaml',
        'hyperparams_gradca': 'configs/gradca_opt_config_empirical.yaml',
        'hyperparams_baryca': 'configs/baryca_opt_config_empirical.yaml'
    }
    configs = ut.load_configs(config_files)
    
    # Load data
    all_data = ut.load_all_data(experiment_name)
    all_data['experiment_name'] = experiment_name
    
    # Prepare cross-validation folds
    Dll_obs = all_data['LLmodel']['data'][None]
    Dhl_obs = all_data['HLmodel']['data'][None]
    folds_path = f"data/{experiment_name}/cv_folds.pkl"
    saved_folds = ut.prepare_cv_folds(Dll_obs, configs['hyperparams_diroca']['k_folds'], 
                                    configs['hyperparams_diroca']['seed'], folds_path)
    
    # Load noise data
    U_ll_hat = all_data['LLmodel']['noise'][None]
    U_hl_hat = all_data['HLmodel']['noise'][None]
    
    return configs, all_data, saved_folds, U_ll_hat, U_hl_hat

def run_diroca_empirical_optimization(all_data, saved_folds, U_ll_hat, U_hl_hat, hyperparams_diroca, output_dir):
    """Run DiRoCA empirical optimization across all folds and hyperparameters."""
    logger.info("Starting DiRoCA empirical optimization...")
    
    diroca_cv_results_empirical = {}
    
    # Iterate through each cross-validation fold
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Starting Empirical Optimization for Fold {i+1}/{len(saved_folds)}")
        fold_key = f'fold_{i}'
        diroca_cv_results_empirical[fold_key] = {}
        
        # Calculate the theoretical bounds for THIS FOLD'S training data
        train_n = len(fold_info['train'])

        l = U_ll_hat.shape[1]
        h = U_hl_hat.shape[1]
        
        ll_bound = round(ut.compute_empirical_radius(N=train_n, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=l), 3)
        hl_bound = round(ut.compute_empirical_radius(N=train_n, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=h), 3)
        
        # Define the list of (epsilon, delta) pairs to search over
        radius_pairs_to_test = [
            (ll_bound, hl_bound), # The theoretical lower bound
            (1.0, 1.0),
            (2.0, 2.0),
            (4.0, 4.0),
            (8.0, 8.0)
        ]

        # Get the train split for the noise data for this fold
        U_ll_train = U_ll_hat[fold_info['train']]
        U_hl_train = U_hl_hat[fold_info['train']]

        # Iterate through each (epsilon, delta) pair
        for epsilon, delta in radius_pairs_to_test:
            logger.info(f"Training for ε = {epsilon}, δ = {delta}")
            
            # Assemble the parameters using the helper function
            params_for_this_run = ut.assemble_empirical_parameters(
                U_ll_train, 
                U_hl_train, 
                all_data, 
                hyperparams_diroca
            )
            
            # Set the epsilon and delta for this specific run
            params_for_this_run['epsilon'] = epsilon
            params_for_this_run['delta'] = delta

            opt_args = params_for_this_run.copy()
            opt_args.pop('k_folds', None)

            # Run the optimization
            trained_params, trained_T = optools.run_empirical_erica_optimization(**opt_args)
            
            # Store the results in the nested dictionary
            hyperparam_key = f'eps_{epsilon}_delta_{delta}'
            diroca_cv_results_empirical[fold_key][hyperparam_key] = {
                'T_matrix': trained_T,
                'optimization_params': trained_params,
                'test_indices': fold_info['test'] 
            }
    
    # Save results
    output_path = os.path.join(output_dir, "diroca_cv_results_empirical.pkl")
    joblib.dump(diroca_cv_results_empirical, output_path)
    logger.info(f"DiRoCA empirical results saved to {output_path}")
    
    return diroca_cv_results_empirical

def run_gradca_empirical_optimization(all_data, saved_folds, U_ll_hat, U_hl_hat, hyperparams_gradca, output_dir):
    """Run GRADCA empirical optimization across all folds."""
    logger.info("Starting GRADCA empirical optimization...")
    
    gradca_cv_results_empirical = {}
    
    # Iterate through each cross-validation fold
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Starting Empirical Optimization for Fold {i+1}/{len(saved_folds)}")
        fold_key = f'fold_{i}'
        gradca_cv_results_empirical[fold_key] = {}
        
        # Get the train split for the noise data for this fold
        U_ll_train = U_ll_hat[fold_info['train']]
        U_hl_train = U_hl_hat[fold_info['train']]
        
        params_for_this_run = ut.assemble_empirical_parameters(
            U_ll_train, 
            U_hl_train, 
            all_data, 
            hyperparams_gradca
        )
        
        opt_args = params_for_this_run.copy()
        opt_args.pop('k_folds', None)

        # Run the optimization
        trained_params, trained_T = optools.run_empirical_erica_optimization(**opt_args)
            
        # Store the results in the dictionary
        gradca_cv_results_empirical[fold_key] = { 'gradca_run': {
            'T_matrix': trained_T,
            'optimization_params': trained_params,
            'test_indices': fold_info['test'] 
        }}
    
    # Save results
    output_path = os.path.join(output_dir, "gradca_cv_results_empirical.pkl")
    joblib.dump(gradca_cv_results_empirical, output_path)
    logger.info(f"GRADCA empirical results saved to {output_path}")
    
    return gradca_cv_results_empirical

def run_baryca_empirical_optimization(all_data, saved_folds, U_ll_hat, U_hl_hat, hyperparams_baryca, output_dir):
    """Run BARYCA empirical optimization across all folds."""
    logger.info("Starting BARYCA empirical optimization...")
    
    # Get the SCM instances and intervention sets from your loaded data
    LLmodels = all_data['LLmodel'].get('scm_instances')
    Ill_relevant = all_data['LLmodel']['intervention_set']
    HLmodels = all_data['HLmodel'].get('scm_instances')
    Ihl_relevant = all_data['HLmodel']['intervention_set']

    # Compute the list of matrices for each model
    L_matrices = ut.compute_struc_matrices(LLmodels, Ill_relevant)
    H_matrices = ut.compute_struc_matrices(HLmodels, Ihl_relevant)

    logger.info(f"Pre-computed {len(L_matrices)} low-level and {len(H_matrices)} high-level structural matrices.")
    
    baryca_cv_results_empirical = {}
    
    # Iterate through each cross-validation fold
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Starting Empirical Barycentric Optimization for Fold {i+1}/{len(saved_folds)}")
        fold_key = f'fold_{i}'
        
        # Get the training split for the noise data for this fold
        U_ll_train = U_ll_hat[fold_info['train']]
        U_hl_train = U_hl_hat[fold_info['train']]
        
        # Assemble the arguments dictionary directly for this optimization run
        baryca_args = {
            'U_ll_hat': U_ll_train,
            'U_hl_hat': U_hl_train,
            'L_matrices': L_matrices,
            'H_matrices': H_matrices,
            # Get hyperparameters from the pre-loaded dictionary
            'max_iter': hyperparams_baryca['max_iter'],
            'tol': hyperparams_baryca['tol'],
            'seed': hyperparams_baryca['seed']
        }

        # Run the optimization
        # Note: This function only returns the T matrix
        trained_T = optools.run_empirical_bary_optim(**baryca_args)
        
        # Store the results for this fold
        baryca_cv_results_empirical[fold_key] = {
            'baryca_run': {
                'T_matrix': trained_T,
                'test_indices': fold_info['test'] 
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "baryca_cv_results_empirical.pkl")
    joblib.dump(baryca_cv_results_empirical, output_path)
    logger.info(f"BARYCA empirical results saved to {output_path}")
    
    return baryca_cv_results_empirical

def run_abslingam_optimization(all_data, saved_folds, Dll_obs, Dhl_obs, output_dir):
    """Run Abs-LiNGAM optimization across all folds."""
    logger.info("Starting Abs-LiNGAM optimization...")
    
    abslingam_cv_results_empirical = {}
    
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Running Abs-LiNGAM for Fold {i+1}/{len(saved_folds)}")
        fold_key = f'fold_{i}'
        
        train_idx = fold_info['train']
        Dll_obs_train = Dll_obs[train_idx]
        Dhl_obs_train = Dhl_obs[train_idx]
        
        abslingam_results_for_fold = optools.run_abs_lingam_complete(Dll_obs_train, Dhl_obs_train)

        # Store the results for this fold
        abslingam_cv_results_empirical[fold_key] = {
            'Perfect': {
                'T_matrix': abslingam_results_for_fold['Perfect']['T'].T, # Transpose if needed
                'test_indices': fold_info['test']
            },
            'Noisy': {
                'T_matrix': abslingam_results_for_fold['Noisy']['T'].T, # Transpose if needed
                'test_indices': fold_info['test']
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "abslingam_cv_results_empirical.pkl")
    joblib.dump(abslingam_cv_results_empirical, output_path)
    logger.info(f"Abs-LiNGAM results saved to {output_path}")
    
    return abslingam_cv_results_empirical

def main():
    """Main function to run the empirical optimization pipeline."""
    parser = argparse.ArgumentParser(description='Run empirical optimization experiments')
    parser.add_argument('--experiment', type=str, default='lilucas', 
                       help='Experiment name (default: lilucas)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: data/{experiment}/results_empirical)')
    parser.add_argument('--skip-diroca', action='store_true',
                       help='Skip DiRoCA empirical optimization')
    parser.add_argument('--skip-gradca', action='store_true',
                       help='Skip GRADCA empirical optimization')
    parser.add_argument('--skip-baryca', action='store_true',
                       help='Skip BARYCA empirical optimization')
    parser.add_argument('--skip-abslingam', action='store_true',
                       help='Skip Abs-LiNGAM optimization')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = f"data/{args.experiment}/results_empirical"
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Setup experiment
        configs, all_data, saved_folds, U_ll_hat, U_hl_hat = setup_experiment(args.experiment)
        
        # Get data for Abs-LiNGAM
        Dll_obs = all_data['LLmodel']['data'][None]
        Dhl_obs = all_data['HLmodel']['data'][None]
        
        # Run optimizations
        results = {}
        
        if not args.skip_diroca:
            results['diroca'] = run_diroca_empirical_optimization(
                all_data, saved_folds, U_ll_hat, U_hl_hat, configs['hyperparams_diroca'], args.output_dir
            )
        
        if not args.skip_gradca:
            results['gradca'] = run_gradca_empirical_optimization(
                all_data, saved_folds, U_ll_hat, U_hl_hat, configs['hyperparams_gradca'], args.output_dir
            )
        
        if not args.skip_baryca:
            results['baryca'] = run_baryca_empirical_optimization(
                all_data, saved_folds, U_ll_hat, U_hl_hat, configs['hyperparams_baryca'], args.output_dir
            )
        
        if not args.skip_abslingam:
            results['abslingam'] = run_abslingam_optimization(
                all_data, saved_folds, Dll_obs, Dhl_obs, args.output_dir
            )
        
        logger.info("All empirical optimizations completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("EMPIRICAL OPTIMIZATION COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
        print(f"Experiment: {args.experiment}")
        print(f"Output directory: {args.output_dir}")
        print(f"Results files:")
        for method in ['diroca', 'gradca', 'baryca', 'abslingam']:
            if method in results:
                print(f"  - {method}_cv_results_empirical.pkl")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Error during empirical optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 