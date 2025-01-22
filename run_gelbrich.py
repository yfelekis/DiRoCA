import os
import json
import torch
import numpy as np
import argparse
from datetime import datetime
import logging
import linear_additive_noise_models as lanm
from typing import Dict, Tuple, List
import optimization_utils as oput

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Gelbrich optimization experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Name of the experiment to run')
    parser.add_argument('--optimization', type=str, required=True,
                       choices=['bary', 'bary_auto', 'erica', 'erica_prox', 'enrico'],
                       help='Optimization method to use')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon parameter for low-level uncertainty')
    parser.add_argument('--delta', type=float, default=0.1,
                       help='Delta parameter for high-level uncertainty')
    parser.add_argument('--lambda_l', type=float, default=1.0,
                       help='Lambda parameter for low-level regularization')
    parser.add_argument('--lambda_h', type=float, default=1.0,
                       help='Lambda parameter for high-level regularization')
    parser.add_argument('--lambda_param_l', type=float, default=1.0,
                       help='Lambda parameter for low-level parameter regularization')
    parser.add_argument('--lambda_param_h', type=float, default=1.0,
                       help='Lambda parameter for high-level parameter regularization')
    parser.add_argument('--abduction', type=str, default='mean',
                       choices=['mean', 'map'],
                       help='Abduction method')
    parser.add_argument('--coeff_estimation', type=str, default='mle',
                       choices=['mle', 'map'],
                       help='Coefficient estimation method')
    return parser.parse_args()

def setup_logger(experiment_name: str) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    f_handler = logging.FileHandler(f'{log_dir}/{experiment_name}.log')
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def load_experiment_data(experiment: str, abduction: str, coeff_estimation: str, 
                        logger: logging.Logger) -> Tuple:
    """Load experiment data from files"""
    logger.info(f"Loading data for experiment: {experiment}")
    data_dir = f"data/{experiment}"
    
    # Load the required data files
    Tau = np.load(f"{data_dir}/Tau.npy")
    Dll_obs = np.load(f"{data_dir}/Dll_obs.npy")
    Dhl_obs = np.load(f"{data_dir}/Dhl_obs.npy")
    Gll = np.load(f"{data_dir}/Gll.npy")
    Ghl = np.load(f"{data_dir}/Ghl.npy")
    Ill = np.load(f"{data_dir}/Ill.npy")
    Ihl = np.load(f"{data_dir}/Ihl.npy")
    omega = np.load(f"{data_dir}/omega.npy")
    
    # Load coefficients based on estimation method
    coeff_suffix = "_mle.npy" if coeff_estimation == "mle" else "_map.npy"
    ll_coeffs = np.load(f"{data_dir}/ll_coeffs{coeff_suffix}")
    hl_coeffs = np.load(f"{data_dir}/hl_coeffs{coeff_suffix}")
    
    # Load parameters based on abduction method
    param_suffix = "_mean.npy" if abduction == "mean" else "_map.npy"
    mu_U_ll_hat = np.load(f"{data_dir}/mu_U_ll{param_suffix}")
    Sigma_U_ll_hat = np.load(f"{data_dir}/Sigma_U_ll{param_suffix}")
    mu_U_hl_hat = np.load(f"{data_dir}/mu_U_hl{param_suffix}")
    Sigma_U_hl_hat = np.load(f"{data_dir}/Sigma_U_hl{param_suffix}")
    
    return (Tau, Dll_obs, Dhl_obs, Gll, Ghl, Ill, Ihl, omega,
            ll_coeffs, hl_coeffs, mu_U_ll_hat, Sigma_U_ll_hat,
            mu_U_hl_hat, Sigma_U_hl_hat)

def get_optimization_params(opt_type: str, base_params: Dict) -> Dict:
    """
    Adjust optimization parameters based on the optimization type
    """
    if opt_type in ['bary', 'bary_auto']:
        # Barycentric optimization parameters
        params = {
            'theta_L': base_params['theta_hatL'],
            'theta_H': base_params['theta_hatH'],
            'LLmodels': base_params['LLmodels'],
            'HLmodels': base_params['HLmodels'],
            'Ill': base_params['Ill'],
            'Ihl': base_params['Ihl'],
            'projection_method': 'svd',
            'initialization': 'avg',
            'autograd': True if opt_type == 'bary_auto' else False,
            'seed': base_params['seed'],
            'max_iter': base_params['max_iter'],
            'tol': base_params['tol'],
            'display_results': base_params['display_results']
        }
    else:
        # ERiCA/ENRICO parameters
        params = base_params.copy()
        if opt_type == 'erica':
            params['proximal_grad'] = False
            params['robust_L'] = True
            params['robust_H'] = True
        elif opt_type == 'erica_prox':
            params['proximal_grad'] = True
            params['robust_L'] = True
            params['robust_H'] = True
        elif opt_type == 'enrico':
            params['proximal_grad'] = False
            params['robust_L'] = False
            params['robust_H'] = False
    
    return params

def run_optimization(optimization_type: str, **kwargs) -> Tuple:
    """
    Run the selected optimization algorithm
    """
    if optimization_type in ['bary', 'bary_auto']:
        # Use barycentric optimization
        return oput.barycentric_optimization(**kwargs)
    else:
        # Run ERiCA/ENRICO optimization
        return oput.run_erica_optimization(**kwargs)

def run_optimization_experiment(args, logger):
    """Run the main optimization experiment"""
    logger.info(f"Starting optimization experiment using {args.optimization} routine")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    (Tau, Dll_obs, Dhl_obs, Gll, Ghl, Ill, Ihl, omega,
     ll_coeffs, hl_coeffs, mu_U_ll_hat, Sigma_U_ll_hat,
     mu_U_hl_hat, Sigma_U_hl_hat) = load_experiment_data(
        args.experiment, args.abduction, args.coeff_estimation, logger
    )
    
    # Initialize models
    LLmodels = {}
    for iota in Ill:
        LLmodels[iota] = lanm.LinearAddSCM(Gll, ll_coeffs, iota)
    
    HLmodels = {}
    for eta in Ihl:
        HLmodels[eta] = lanm.LinearAddSCM(Ghl, hl_coeffs, eta)
    
    # Set up optimization parameters
    theta_hatL = {'mu_U': mu_U_ll_hat, 'Sigma_U': Sigma_U_ll_hat, 'radius': args.epsilon}
    theta_hatH = {'mu_U': mu_U_hl_hat, 'Sigma_U': Sigma_U_hl_hat, 'radius': args.delta}
    
    # Base optimization parameters
    base_params = {
        'theta_hatL': theta_hatL,
        'theta_hatH': theta_hatH,
        'LLmodels': LLmodels,  # Add these for barycentric optimization
        'HLmodels': HLmodels,
        'Ill': Ill,
        'Ihl': Ihl,
        'initial_theta': 'empirical',
        'epsilon': args.epsilon,
        'delta': args.delta,
        'lambda_L': args.lambda_l,
        'lambda_H': args.lambda_h,
        'lambda_param_L': args.lambda_param_l,
        'lambda_param_H': args.lambda_param_h,
        'xavier': True,
        'project_onto_gelbrich': True,
        'eta_max': 0.001,
        'eta_min': 0.001,
        'max_iter': 100,
        'num_steps_min': 5,
        'num_steps_max': 3,
        'tol': 1e-5,
        'seed': 42,
        'grad_clip': True,
        'plot_steps': False,
        'plot_epochs': False,
        'display_results': True
    }
    
    # Get optimization-specific parameters
    opt_params = get_optimization_params(args.optimization, base_params)
    
    logger.info(f"Starting {args.optimization} optimization")
    params_L, params_H, T_matrix, inobjs, epobjs, condition_num_list = run_optimization(
        args.optimization, **opt_params
    )
    
   # Save results
    logger.info("Saving results")
    results_dir = f"results/{args.experiment}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save T matrix with optimization type in filename
    np.save(f"{results_dir}/T_matrix_{args.optimization}.npy", T_matrix)
    
    # Save parameters and results as JSON
    experiment_results = {
        'experiment_name': args.experiment,
        'optimization_type': args.optimization,
        'timestamp': datetime.now().isoformat(),
        'input_parameters': vars(args),
        'optimization_parameters': opt_params,
        'results': {
            'final_objectives': {
                'inner_objectives': inobjs[-1] if len(inobjs) > 0 else None,
                'epoch_objectives': epobjs[-1] if len(epobjs) > 0 else None
            },
            'condition_numbers': condition_num_list[-1] if len(condition_num_list) > 0 else None
        }
    }
    
    with open(f"{results_dir}/experiment_results_{args.optimization}.json", 'w') as f:
        json.dump(experiment_results, f, indent=4)
    
    logger.info("Experiment completed successfully")
    return T_matrix, params_L, params_H

def main():
    args = parse_args()
    logger = setup_logger(args.experiment)
    
    try:
        T_matrix, params_L, params_H = run_optimization_experiment(args, logger)
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Error during experiment execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()