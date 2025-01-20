import argparse
import json
import pickle
import time
from datetime import datetime, timedelta
from itertools import product
import logging

import numpy as np
import seaborn as sns
import torch
import joblib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local modules
import modularised_utils as mut
import opt_utils as oput
import evaluation_utils as evut
import Linear_Additive_Noise_Models as lanm
import operations as ops
import params

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run optimization experiments')
    
    # Experiment setup
    parser.add_argument('--experiment', type=str, default='synth1',
                      help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Model initialization options
    parser.add_argument('--abduction', action='store_true',
                      help='Whether to use abduction')
    parser.add_argument('--coeff-estimation', action='store_true',
                      help='Whether to estimate coefficients')
    
    # Optimization parameters
    parser.add_argument('--epsilon', type=float, default=4.0,
                      help='Epsilon value for Wasserstein ball radius')
    parser.add_argument('--delta', type=float, default=4.0,
                      help='Delta value for Wasserstein ball radius')
    parser.add_argument('--lambda-l', type=float, default=0.9,
                      help='Lambda L parameter')
    parser.add_argument('--lambda-h', type=float, default=0.9,
                      help='Lambda H parameter')
    parser.add_argument('--lambda-param-l', type=float, default=0.1,
                      help='Lambda parameter L')
    parser.add_argument('--lambda-param-h', type=float, default=0.3,
                      help='Lambda parameter H')
    
    # Optimization settings
    parser.add_argument('--max-iter', type=int, default=100,
                      help='Maximum number of iterations')
    parser.add_argument('--num-steps-min', type=int, default=5,
                      help='Number of minimization steps')
    parser.add_argument('--num-steps-max', type=int, default=3,
                      help='Number of maximization steps')
    parser.add_argument('--eta-max', type=float, default=0.001,
                      help='Maximum learning rate')
    parser.add_argument('--eta-min', type=float, default=0.001,
                      help='Minimum learning rate')
    
    # Additional options
    parser.add_argument('--xavier', action='store_true',
                      help='Use Xavier initialization')
    parser.add_argument('--project-gelbrich', action='store_true',
                      help='Project onto Gelbrich ball')
    parser.add_argument('--proximal-grad', action='store_true',
                      help='Use proximal gradient method')
    parser.add_argument('--grad-clip', action='store_true',
                      help='Use gradient clipping')
    parser.add_argument('--plot-steps', action='store_true',
                      help='Plot optimization steps')
    parser.add_argument('--plot-epochs', action='store_true',
                      help='Plot epoch progress')
    
    return parser.parse_args()

def load_experiment_data(experiment, abduction, coeff_estimation, logger):
    """Load and initialize all experiment data"""
    logger.info(f"Loading experiment data for {experiment}")
    
    # Define parameters
    epsilon, delta = params.radius[experiment]
    ll_num_envs, hl_num_envs = params.n_envs[experiment]
    num_llsamples, num_hlsamples = params.n_samples[experiment]
    
    # Load ground truth abstraction
    Tau = mut.load_T(experiment)
    
    # Load samples and models
    Dll_obs = mut.load_samples(experiment)[None][0] 
    Gll, Ill = mut.load_model(experiment, 'LL')
    
    Dhl_obs = mut.load_samples(experiment)[None][1] 
    Ghl, Ihl = mut.load_model(experiment, 'HL')
    
    omega = mut.load_omega_map(experiment)
    
    # Get coefficients
    if coeff_estimation:
        logger.info("Estimating coefficients")
        ll_coeffs = mut.get_coefficients(Dll_obs, Gll)
        hl_coeffs = mut.get_coefficients(Dhl_obs, Ghl)
    else:
        logger.info("Loading pre-computed coefficients")
        ll_coeffs = mut.load_coeffs(experiment, 'LL')
        hl_coeffs = mut.load_coeffs(experiment, 'HL')
    
    # Get exogenous variables
    if abduction:
        logger.info("Performing abduction")
        U_ll_hat, mu_U_ll_hat, Sigma_U_ll_hat = mut.lan_abduction(Dll_obs, Gll, ll_coeffs)
        U_hl_hat, mu_U_hl_hat, Sigma_U_hl_hat = mut.lan_abduction(Dhl_obs, Ghl, hl_coeffs)
    else:
        logger.info("Loading pre-computed exogenous variables")
        U_ll_hat, mu_U_ll_hat, Sigma_U_ll_hat = mut.load_exogenous(experiment, 'LL')
        U_hl_hat, mu_U_hl_hat, Sigma_U_hl_hat = mut.load_exogenous(experiment, 'HL')
    
    return (Tau, Dll_obs, Dhl_obs, Gll, Ghl, Ill, Ihl, omega, 
            ll_coeffs, hl_coeffs, mu_U_ll_hat, Sigma_U_ll_hat, mu_U_hl_hat, Sigma_U_hl_hat)

def run_optimization_experiment(args, logger):
    """Run the main optimization experiment"""
    logger.info("Starting optimization experiment")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    
    # Run ERiCA optimization
    params_erica = {
        'theta_hatL': theta_hatL,
        'theta_hatH': theta_hatH,
        'initial_theta': 'empirical',
        'epsilon': args.epsilon,
        'delta': args.delta,
        'lambda_L': args.lambda_l,
        'lambda_H': args.lambda_h,
        'lambda_param_L': args.lambda_param_l,
        'lambda_param_H': args.lambda_param_h,
        'xavier': args.xavier,
        'project_onto_gelbrich': args.project_gelbrich,
        'eta_max': args.eta_max,
        'eta_min': args.eta_min,
        'max_iter': args.max_iter,
        'num_steps_min': args.num_steps_min,
        'num_steps_max': args.num_steps_max,
        'proximal_grad': args.proximal_grad,
        'tol': 1e-5,
        'seed': args.seed,
        'robust_L': True,
        'robust_H': True,
        'grad_clip': args.grad_clip,
        'plot_steps': args.plot_steps,
        'plot_epochs': args.plot_epochs,
        'display_results': True
    }
    
    logger.info("Starting ERiCA optimization")
    params_Lerica, params_Herica, T_erica, inobjs, epobjs, condition_num_list = run_optimization(**params_erica)
    
    # Save results
    logger.info("Saving results")
    results_dir = f"data/{args.experiment}"
    os.makedirs(results_dir, exist_ok=True)
    
    type_to_params = {
        'gelbrich_gaussian': {
            'L': theta_hatL,
            'H': theta_hatH
        },
        'boundary_gaussian': {
            'L': params_Lerica,
            'H': params_Herica
        }
    }
    
    joblib.dump(type_to_params, f"{results_dir}/type_to_params.pkl")
    joblib.dump(T_erica, f"{results_dir}/T_erica.pkl")
    
    logger.info("Experiment completed successfully")
    return T_erica, params_Lerica, params_Herica

def main():
    args = parse_args()
    logger = setup_logging()
    
    try:
        T_erica, params_Lerica, params_Herica = run_optimization_experiment(args, logger)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()