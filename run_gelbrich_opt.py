import json
import pickle
import time
from datetime import datetime, timedelta
from itertools import product
import argparse
import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import joblib

import matplotlib.pyplot as plt
from tqdm import tqdm

# Local modules
import modularised_utils as mut
import opt_utils as oput
import Linear_Additive_Noise_Models as lanm
import operations as ops
import params

def load_data(experiment, abduction=False):
    """Load and prepare data for the experiment"""
    # Define the radius of the Wasserstein balls (epsilon, delta) and the size for both models.
    epsilon, delta = params.radius[experiment]
   
    # Load samples and models
    Dll      = mut.load_samples(experiment)[None][0] 
    Gll, Ill = mut.load_model(experiment, 'LL')
    
    Dhl      = mut.load_samples(experiment)[None][1] 
    Ghl, Ihl = mut.load_model(experiment, 'HL')
    
    omega    = mut.load_omega_map(experiment)
    
    # Get coefficients
    ll_coeffs = mut.get_coefficients(Dll, Gll)
    hl_coeffs = mut.get_coefficients(Dhl, Ghl)
    
    # Load or compute exogenous variables
    if abduction:
        U_ll_hat, mu_U_ll_hat, Sigma_U_ll_hat = mut.lan_abduction(Dll, Gll, ll_coeffs)
        U_hl_hat, mu_U_hl_hat, Sigma_U_hl_hat = mut.lan_abduction(Dhl, Ghl, hl_coeffs)
    else:
        U_ll_hat, mu_U_ll_hat, Sigma_U_ll_hat = mut.load_exogenous(experiment, 'LL')
        U_hl_hat, mu_U_hl_hat, Sigma_U_hl_hat = mut.load_exogenous(experiment, 'HL')
    
    return {
        'Dll': Dll, 'Gll': Gll, 'Ill': Ill,
        'Dhl': Dhl, 'Ghl': Ghl, 'Ihl': Ihl,
        'omega': omega,
        'll_coeffs': ll_coeffs, 'hl_coeffs': hl_coeffs,
        'mu_U_ll_hat': mu_U_ll_hat, 'Sigma_U_ll_hat': Sigma_U_ll_hat,
        'mu_U_hl_hat': mu_U_hl_hat, 'Sigma_U_hl_hat': Sigma_U_hl_hat,
        'epsilon': epsilon, 'delta': delta
    }

def prepare_models(data):
    """Prepare the models using loaded data"""
    LLmodels = {}
    for iota in data['Ill']:
        LLmodels[iota] = lanm.LinearAddSCM(data['Gll'], data['ll_coeffs'], iota)
    
    HLmodels = {}
    for eta in data['Ihl']:
        HLmodels[eta] = lanm.LinearAddSCM(data['Ghl'], data['hl_coeffs'], eta)
    
    # Convert to torch tensors
    hat_mu_L = torch.from_numpy(data['mu_U_ll_hat']).float()
    hat_Sigma_L = torch.from_numpy(data['Sigma_U_ll_hat']).float()
    hat_mu_H = torch.from_numpy(data['mu_U_hl_hat']).float()
    hat_Sigma_H = torch.from_numpy(data['Sigma_U_hl_hat']).float()
    
    return {
        'LLmodels': LLmodels,
        'HLmodels': HLmodels,
        'hat_mu_L': hat_mu_L,
        'hat_Sigma_L': hat_Sigma_L,
        'hat_mu_H': hat_mu_H,
        'hat_Sigma_H': hat_Sigma_H,
        'Ill': data['Ill'],
        'Ihl': data['Ihl'],
        'omega': data['omega']
    }

def initialize_parameters(data):
    """Initialize optimization parameters"""
    ll_moments = mut.sample_moments_U(
                                        mu_hat=data['mu_U_ll_hat'],
                                        Sigma_hat=data['Sigma_U_ll_hat'],
                                        bound=data['epsilon'],
                                        num_envs=1
                                     )
    mu_L0, Sigma_L0 = ll_moments[0]
    mu_L0           = torch.from_numpy(mu_L0).float()
    Sigma_L0        = torch.from_numpy(Sigma_L0).float()
    
    hl_moments = mut.sample_moments_U(
                                        mu_hat=data['mu_U_hl_hat'],
                                        Sigma_hat=data['Sigma_U_hl_hat'],
                                        bound=data['delta'],
                                        num_envs=1
                                     )
    mu_H0, Sigma_H0 = hl_moments[0]
    mu_H0           = torch.from_numpy(mu_H0).float()
    Sigma_H0        = torch.from_numpy(Sigma_H0).float()
    
    return mu_L0, Sigma_L0, mu_H0, Sigma_H0

def run_grid_search(args):
    # Load data
    data       = load_data(args.experiment, args.abduction)
    model_data = prepare_models(data)
        
    LLmodels    = model_data['LLmodels']
    HLmodels    = model_data['HLmodels']
    hat_mu_L    = model_data['hat_mu_L']
    hat_Sigma_L = model_data['hat_Sigma_L']
    hat_mu_H    = model_data['hat_mu_H']
    hat_Sigma_H = model_data['hat_Sigma_H']
    Ill         = model_data['Ill']
    Ihl         = model_data['Ihl']
    omega       = model_data['omega']
    
    # Initialize parameters
    mu_L0, Sigma_L0, mu_H0, Sigma_H0 = initialize_parameters(data)
    
    # Define parameter grids
    param_grid = {
        'lambda_param': args.lambda_param,
        'lambda_L': args.lambda_l,  
        'lambda_H': args.lambda_h,  
        'eta_max': args.eta_max,
        'eta_min': args.eta_min
    }
    
    # Fixed parameters
    fixed_params = {
        'epsilon': args.epsilon,
        'delta': args.delta,
        'max_iter': args.max_iter,
        'num_steps_min': args.num_steps_min,
        'num_steps_max': args.num_steps_max,
        'tol': args.tol,
        'seed': args.seed
    }
    
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    # Create results directory if it doesn't exist
    results_dir = f"{args.experiment}_erica_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Iterate over each combination
    for idx, params in enumerate(param_combinations, 1):
        try:
            print(f"Running combination {idx}/{len(param_combinations)}: {params}")
            
            # Run optimization
            final_mu_L, final_Sigma_L, final_mu_H, final_Sigma_H, final_T = oput.optimize_min_max(
                mu_L0, Sigma_L0, mu_H0, Sigma_H0,
                LLmodels, HLmodels,
                hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                Ill=Ill,
                Ihl=Ihl,
                omega=omega,
                **params,
                **fixed_params
            )
            
            # Prepare results
            results = {
                'parameters': params,
                'mu_L': final_mu_L.detach().numpy(),
                'Sigma_L': final_Sigma_L.detach().numpy(),
                'mu_H': final_mu_H.detach().numpy(),
                'Sigma_H': final_Sigma_H.detach().numpy(),
                'T': final_T.detach().numpy(),
                'execution_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save results
            result_filename = f"{results_dir}/result_{idx}.pkl"
            with open(result_filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Results saved to: {result_filename}")
        
        except Exception as e:
            error_message = f"""
            Error in combination {idx}/{total_combinations}
            Parameters: {params}
            Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Error: {str(e)}
            ----------------------------------------
            """
            
            # Log error to file
            with open(error_log_file, 'a') as f:
                f.write(error_message)
            
            print(f"Error occurred in combination {idx}. Logged to: {error_log_file}")
            
            # Save failed result
            failed_result = {
                'parameters': {**params, **fixed_params},
                'execution_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'failed',
                'error_message': str(e)
            }
            
            result_filename = f"{results_dir}/result_{idx}_failed.pkl"
            with open(result_filename, 'wb') as f:
                pickle.dump(failed_result, f)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ERiCA optimization with grid search")
    
    # Add arguments
    parser.add_argument('--experiment', type=str, default='synth1', help='Experiment name')
    parser.add_argument('--abduction', action='store_true', help='Use abduction to compute the exogenous variables. If not, load them from file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Grid search parameters (parameters to test multiple values)
    parser.add_argument('--lambda_param', nargs='+', type=float, default=[0.8], 
                       help='List of Lambda parameter values for grid search')
    parser.add_argument('--lambda_l', nargs='+', type=float, default=[0.5], 
                       help='Lagrange multipliers for the low-level model.')  
    parser.add_argument('--lambda_h', nargs='+', type=float, default=[0.5], 
                       help='Lagrange multipliers for the high-level model.')  
    parser.add_argument('--eta_max', nargs='+', type=float, default=[0.01], 
                       help='Learning rate for the max step.')
    parser.add_argument('--eta_min', nargs='+', type=float, default=[0.001], 
                       help='Learning rate for the min step.')
    
    # Fixed parameters (single values)
    parser.add_argument('--epsilon', type=float, default=0.5, help='Radius for the low-level Wasserstein ball.')
    parser.add_argument('--delta', type=float, default=0.5, help='Radius for the high-level Wasserstein ball.')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum min-max iterations.')
    parser.add_argument('--num_steps_min', type=int, default=1, help='Number of min iterations.')
    parser.add_argument('--num_steps_max', type=int, default=1, help='Number of max iterations.')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance to determine convergence.')
    
    args = parser.parse_args()
    run_grid_search(args)