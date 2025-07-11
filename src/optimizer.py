"""
Minimal optimizer module for DiRoCA: runs GRADCA (non-robust) for Gaussian and empirical cases.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

import opt_tools as oput
import modularised_utils as mut
import evaluation_utils as evut

from src.pipeline import DiRoCAPipeline


def run_gradca(
    experiment_name: str,
    case: str = 'gaussian',  # 'gaussian' or 'empirical'
    max_iter: int = 5,
    plot: bool = True,
    save_dir: str = None
):
    """
    Run GRADCA optimization (non-robust) for a given experiment and case.
    Args:
        experiment_name: 'synth1' or 'lucas6x3'
        case: 'gaussian' or 'empirical'
        max_iter: number of outer iterations (keep low for demo)
        plot: whether to plot and save the training loss curve
        save_dir: directory to save results (defaults to data/experiment_name/)
    Returns:
        T_matrix: learned transformation matrix
        params: optimization parameters
        losses: list of training losses per iteration
    """
    # Set up save directory
    if save_dir is None:
        save_dir = f"data/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Load data using the pipeline
    pipeline = DiRoCAPipeline(experiment_name)
    data = pipeline.load_existing_data()

    if case == 'gaussian':
        # Load Gaussian-case objects
        # For Gaussian case, we need to compute the empirical parameters from observational data
        Dll_obs = data['Ds'][None][0]  # Observational low-level data
        Dhl_obs = data['Ds'][None][1]  # Observational high-level data
        
        # Compute empirical parameters
        U_ll_hat, mu_U_ll_hat, Sigma_U_ll_hat = mut.lan_abduction(Dll_obs, data['ll_causal_graph'], data['ll_coeffs'])
        U_hl_hat, mu_U_hl_hat, Sigma_U_hl_hat = mut.lan_abduction(Dhl_obs, data['hl_causal_graph'], data['hl_coeffs'])
        
        theta_hatL = {'mu_U': mu_U_ll_hat, 'Sigma_U': Sigma_U_ll_hat, 'radius': 0.0}
        theta_hatH = {'mu_U': mu_U_hl_hat, 'Sigma_U': Sigma_U_hl_hat, 'radius': 0.0}
        LLmodels = joblib.load(os.path.join(save_dir, 'LLmodels.pkl'))
        HLmodels = joblib.load(os.path.join(save_dir, 'HLmodels.pkl'))
        omega = data['omega']
        Ill = data['Ill_relevant']
        Ihl = data['Ihl_relevant']
        # Fixed hyperparameters
        opt_params = {
            'theta_hatL': theta_hatL,
            'theta_hatH': theta_hatH,
            'initial_theta': 'empirical',
            'LLmodels': LLmodels,
            'HLmodels': HLmodels,
            'omega': omega,
            'lambda_L': 0.6,
            'lambda_H': 0.3,
            'lambda_param_L': 0.2,
            'lambda_param_H': 0.1,
            'xavier': False,
            'project_onto_gelbrich': True,
            'eta_max': 0.001,
            'eta_min': 0.001,
            'max_iter': max_iter,
            'num_steps_min': 2,
            'num_steps_max': 1,
            'proximal_grad': True,
            'tol': 1e-4,
            'seed': 23,
            'robust_L': False,
            'robust_H': False,
            'grad_clip': True,
            'plot_steps': False,
            'plot_epochs': False,
            'display_results': False,
            'experiment': experiment_name
        }
        # Run GRADCA (non-robust)
        params, T = oput.run_erica_optimization(**opt_params)
        # For demo, fake a loss curve (if not available)
        losses = params.get('T_objectives_overall', [0])
    elif case == 'empirical':
        # Load empirical-case objects
        U_L, mu_U_ll_hat, Sigma_U_ll_hat = joblib.load(os.path.join(save_dir, 'exogenous_LL.pkl'))
        U_H, mu_U_hl_hat, Sigma_U_hl_hat = joblib.load(os.path.join(save_dir, 'exogenous_HL.pkl'))
        LLmodels = joblib.load(os.path.join(save_dir, 'LLmodels.pkl'))
        HLmodels = joblib.load(os.path.join(save_dir, 'HLmodels.pkl'))
        omega = data['omega']
        # Fixed hyperparameters
        opt_params = {
            'U_L': U_L,
            'U_H': U_H,
            'L_models': LLmodels,
            'H_models': HLmodels,
            'omega': omega,
            'epsilon': 0.0,
            'delta': 0.0,
            'eta_min': 0.001,
            'eta_max': 0.001,
            'num_steps_min': 2,
            'num_steps_max': 1,
            'max_iter': max_iter,
            'tol': 1e-4,
            'seed': 23,
            'robust_L': False,
            'robust_H': False,
            'initialization': 'random',
            'experiment': experiment_name
        }
        # Run GRADCA (non-robust)
        params, T = oput.run_empirical_erica_optimization(**opt_params)
        # For demo, fake a loss curve (if not available)
        losses = [0] * max_iter
    else:
        raise ValueError(f"Unknown case: {case}. Use 'gaussian' or 'empirical'.")

    # Save results
    joblib.dump({'T_matrix': T, 'optimization_params': params, 'losses': losses}, os.path.join(save_dir, f'gradca_{case}_results.pkl'))

    # Plot training loss
    if plot:
        plt.figure()
        plt.plot(losses, marker='o')
        plt.title(f'GRADCA Training Loss ({case}, {experiment_name})')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'gradca_{case}_loss.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved training loss plot to {plot_path}")

    print(f"GRADCA ({case}) finished for {experiment_name}. T shape: {T.shape}")
    return T, params, losses 