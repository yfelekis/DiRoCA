#!/usr/bin/env python3
"""
Gaussian Optimization Script
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import joblib
import numpy as np
import networkx as nx
import yaml
import utilities as ut 
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
import opt_tools as optools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gauss_optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(experiment_name):
    """Setup the experiment configuration and data."""
    logger.info(f"Setting up experiment: {experiment_name}")
    
    def get_config_path(base_name, exp_name):
        """Checks for a specialized config and falls back to default."""
        specialized_path = f"configs/{base_name}_{exp_name}.yaml"
        default_path = f"configs/{base_name}.yaml"
        
        if os.path.exists(specialized_path):
            logger.info(f"Loading specialized config: {specialized_path}")
            return specialized_path
        else:
            logger.info(f"Specialized config not found. Loading default: {default_path}")
            return default_path

    # Load configuration files dynamically
    config_files = {
        'hyperparams_diroca': get_config_path('diroca_opt_config', experiment_name),
        'hyperparams_gradca': get_config_path('gradca_opt_config', experiment_name),
        'hyperparams_baryca': get_config_path('baryca_opt_config', experiment_name)
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
    
    return configs, all_data, saved_folds

def run_diroca_optimization(all_data, saved_folds, hyperparams_diroca, output_dir):
    """Run DiRoCA optimization across all folds and hyperparameters."""
    logger.info("Starting DiRoCA optimization...")
    
    diroca_cv_results = {}
    
    # Iterate through each cross-validation fold
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Starting Cross-Validation for Fold {i+1}/{len(saved_folds)}")
        
        # Create a new sub-dictionary for the current fold
        fold_key = f'fold_{i}'
        diroca_cv_results[fold_key] = {}
        
        # Determine the fold-specific radius bound
        train_n = len(fold_info['train'])
        ll_bound = round(ut.compute_radius_lb(N=train_n, eta=0.05, c=1000), 3)
        eps_delta_values = [4, 8, 1, 2, ll_bound]

        # Iterate through each radii value
        for eps_delta in eps_delta_values:
            logger.info(f"Training for ε=δ = {eps_delta}")

            # Assemble parameters for this specific run
            params_for_this_run = ut.assemble_fold_parameters(fold_info, all_data, hyperparams_diroca)
            params_for_this_run['theta_hatL']['radius'] = eps_delta
            params_for_this_run['theta_hatH']['radius'] = eps_delta
            
            opt_args = params_for_this_run.copy()
            opt_args.pop('k_folds', None)
            
            # Run the optimization
            trained_params, trained_T, monitor = optools.run_erica_optimization(**opt_args)
            
            # Store the results 
            hyperparam_key = f'eps_delta_{eps_delta}'
            diroca_cv_results[fold_key][hyperparam_key] = {
                'T_matrix': trained_T,
                'optimization_params': trained_params,
                'test_indices': fold_info['test'],
                'monitor': monitor
            }
    
    # Save results
    output_path = os.path.join(output_dir, "diroca_cv_results.pkl")
    joblib.dump(diroca_cv_results, output_path)
    logger.info(f"DiRoCA results saved to {output_path}")
    
    return diroca_cv_results

def run_gradca_optimization(all_data, saved_folds, hyperparams_gradca, output_dir):
    """Run GRADCA optimization across all folds."""
    logger.info("Starting GRADCA optimization...")
    
    gradca_cv_results = {}
    
    # Iterate through each cross-validation fold
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Starting Cross-Validation for Fold {i+1}/{len(saved_folds)}")
        
        # Create a new sub-dictionary for the current fold
        fold_key = f'fold_{i}'
        gradca_cv_results[fold_key] = {}
        
        # Assemble parameters for this specific run
        params_for_this_run = ut.assemble_fold_parameters(fold_info, all_data, hyperparams_gradca)
       
        # Prepare arguments for the optimization function
        opt_args = params_for_this_run.copy()
        opt_args.pop('k_folds', None)
        
        # Run the optimization
        trained_params, trained_T, monitor = optools.run_erica_optimization(**opt_args)

        # Store the results
        gradca_cv_results[fold_key] = {
            'gradca_run': {
                'T_matrix': trained_T,
                'test_indices': fold_info['test'] 
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "gradca_cv_results.pkl")
    joblib.dump(gradca_cv_results, output_path)
    logger.info(f"GRADCA results saved to {output_path}")
    
    return gradca_cv_results

def run_baryca_optimization(all_data, saved_folds, hyperparams_baryca, output_dir):
    """Run BARYCA optimization across all folds."""
    logger.info("Starting BARYCA optimization...")
    
    baryca_cv_results = {}
    
    for i, fold_info in enumerate(saved_folds):
        logger.info(f"Starting Barycentric Optimization for Fold {i+1}/{len(saved_folds)}")
        
        opt_args = ut.assemble_barycentric_parameters(fold_info, all_data, hyperparams_baryca)
        opt_args.pop('k_folds', None)

        # Run the optimization
        trained_params, trained_T = optools.barycentric_optimization(**opt_args)
        
        # Store the results 
        fold_key = f'fold_{i}'

        # Store the results
        baryca_cv_results[fold_key] = {
            'baryca_run': {
                'T_matrix': trained_T,
                'test_indices': fold_info['test'] 
            }
        }
    
    # Save results
    output_path = os.path.join(output_dir, "baryca_cv_results.pkl")
    joblib.dump(baryca_cv_results, output_path)
    logger.info(f"BARYCA results saved to {output_path}")
    
    return baryca_cv_results

def generate_summary_plots(diroca_results, all_data, output_dir):
    """Generate summary plots and analysis for the results."""
    logger.info("Generating summary plots...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Example analysis for the first fold and hyperparameter
    fold_to_inspect = 'fold_0'
    hyperparam_to_inspect = 'eps_delta_8'
    
    if fold_to_inspect in diroca_results and hyperparam_to_inspect in diroca_results[fold_to_inspect]:
        result = diroca_results[fold_to_inspect][hyperparam_to_inspect]
        monitor = result['monitor']
        
        # Get variable names
        ll_var_names = list(all_data['LLmodel']['graph'].nodes())
        hl_var_names = list(all_data['HLmodel']['graph'].nodes())
        
        # Print summary to file
        summary_file = os.path.join(output_dir, "optimization_summary.txt")
        with open(summary_file, 'w') as f:
            # Redirect stdout to capture the summary
            import io
            import contextlib
            
            # Capture the summary output
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ut.print_distribution_summary(result['optimization_params']['L'], 
                                            result['optimization_params']['L'], 
                                            name="Low-Level Model")
                ut.print_distribution_summary(result['optimization_params']['H'], 
                                            result['optimization_params']['H'], 
                                            name="High-Level Model")
            
            f.write(output.getvalue())
        
        # Generate trajectory length metrics
        traj_len = monitor.compute_trajectory_length(level='L')
        spreads = monitor.compute_spread_metrics(level='L')
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, "optimization_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Total Trajectory Length (Low-Level): {traj_len:.4f}\n")
            f.write(f"Distribution Spread (Low-Level): μ-Spread={spreads['spread_mu']:.4f}, Σ-Spread={spreads['spread_sigma']:.4f}\n")
        
        logger.info(f"Summary plots and metrics saved to {output_dir}")

def main():
    """Main function to run the optimization pipeline."""
    parser = argparse.ArgumentParser(description='Run Gaussian optimization experiments')
    parser.add_argument('--experiment', type=str, default='slc', 
                       help='Experiment name (default: slc)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: data/{experiment}/results)')
    parser.add_argument('--skip-diroca', action='store_true',
                       help='Skip DiRoCA optimization')
    parser.add_argument('--skip-gradca', action='store_true',
                       help='Skip GRADCA optimization')
    parser.add_argument('--skip-baryca', action='store_true',
                       help='Skip BARYCA optimization')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate summary plots and analysis')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = f"data/{args.experiment}/results"
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Setup experiment
        configs, all_data, saved_folds = setup_experiment(args.experiment)
        
        # Run optimizations
        results = {}
        
        if not args.skip_diroca:
            results['diroca'] = run_diroca_optimization(
                all_data, saved_folds, configs['hyperparams_diroca'], args.output_dir
            )
        
        if not args.skip_gradca:
            results['gradca'] = run_gradca_optimization(
                all_data, saved_folds, configs['hyperparams_gradca'], args.output_dir
            )
        
        if not args.skip_baryca:
            results['baryca'] = run_baryca_optimization(
                all_data, saved_folds, configs['hyperparams_baryca'], args.output_dir
            )
        
        # Generate summary plots if requested
        if args.generate_plots and 'diroca' in results:
            generate_summary_plots(results['diroca'], all_data, args.output_dir)
        
        logger.info("All optimizations completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary
        print(f"\n{'='*50}")
        print("OPTIMIZATION COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
        print(f"Experiment: {args.experiment}")
        print(f"Output directory: {args.output_dir}")
        print(f"Results files:")
        for method in ['diroca', 'gradca', 'baryca']:
            if method in results:
                print(f"  - {method}_cv_results.pkl")
        if args.generate_plots:
            print(f"  - optimization_summary.txt")
            print(f"  - optimization_metrics.txt")
            print(f"  - plots/ (directory)")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 