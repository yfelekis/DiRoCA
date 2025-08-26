import numpy as np
import torch
import time
import os
import ot


import networkx as nx
from tqdm import tqdm
import joblib

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn

from src.CBN import CausalBayesianNetwork as CBN
import operations as ops
import modularised_utils as mut
import evaluation_utils as evut


def monge_map(m_alpha, Sigma_alpha, m_beta, Sigma_beta):
    """
    Compute the Monge map between two multivariate Gaussians and return the transformation
    function T(x) along with the matrix T.

    Args:
    - m_alpha: Mean of the first Gaussian (m_alpha has shape (l,))
    - Sigma_alpha: Covariance matrix of the first Gaussian (Sigma_alpha has shape (l, l))
    - m_beta: Mean of the second Gaussian (m_beta has shape (l,))
    - Sigma_beta: Covariance matrix of the second Gaussian (Sigma_beta has shape (l, l))

    Returns:
    - T(x): A function that applies the Monge map transformation to a point x from the first Gaussian
    - T: The matrix T used for the transformation (T has shape (l, l))
    """
    # Step 1: Compute A using the formula
    Sigma_alpha_half = np.linalg.cholesky(Sigma_alpha)  # Cholesky decomposition of Sigma_alpha
    Sigma_alpha_half_inv = np.linalg.inv(Sigma_alpha_half)  # Inverse of Sigma_alpha_half

    # Compute A using the given formula
    A = Sigma_alpha_half_inv @ np.linalg.cholesky(Sigma_alpha_half.T @ Sigma_beta @ Sigma_alpha_half) @ Sigma_alpha_half_inv.T

    # Step 2: Define the Monge map as a function T(x) = m_beta + A(x - m_alpha)
    def tau(x):
        return m_beta + A @ (x - m_alpha)

    return tau, A


# Proximal operator of a matrix frobenious norm
def prox_operator(A, lambda_param):
    U, S, V        = torch.svd(A)
    frobenius_norm = torch.norm(S, p='fro')
    scaling_factor = torch.max(1 - lambda_param / frobenius_norm, torch.zeros_like(frobenius_norm))
    S_hat          = scaling_factor * S
  
    return U @ torch.diag(S_hat) @ V.T

def diagonalize(A):
    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eig(A)  
    eigvals_real     = eigvals.real  
    eigvals_real     = torch.sqrt(eigvals_real)  # Take the square root of the eigenvalues

    return torch.diag(eigvals_real)

def sqrtm_svd_np(A):
    """
    Compute the matrix square root using SVD for numpy arrays.
    
    Args:
        A: numpy array of shape (n x n)
        
    Returns:
        Matrix square root of A
    """
    # Handle non-finite values
    A = np.nan_to_num(A, nan=1e-6, posinf=1e10, neginf=-1e10)
    
    # Ensure matrix is symmetric
    A = 0.5 * (A + A.T)
    
    # Add small regularization term
    eps = 1e-10
    A = A + eps * np.eye(A.shape[0])
    
    try:
        # Try SVD computation
        U, S, V = np.linalg.svd(A)
        
        # Ensure numerical stability of singular values
        S = np.clip(S, eps, None)
        S_sqrt = np.sqrt(S)
        
        return U @ np.diag(S_sqrt) @ V
        
    except Exception as e:
        print(f"SVD failed: {e}")
        # Return regularized identity matrix as fallback
        return np.eye(A.shape[0]) * np.linalg.norm(A)
    
def sqrtm_svd(A):
    # Handle non-finite values
    A = torch.nan_to_num(A, nan=1e-6, posinf=1e10, neginf=-1e10)
    
    # Ensure matrix is symmetric
    A = 0.5 * (A + A.T)
    
    # Add small regularization term
    eps = 1e-10
    A = A + eps * torch.eye(A.shape[0], device=A.device)
    
    try:
        # Try SVD computation
        U, S, V = torch.svd(A)
        
        # Ensure numerical stability of singular values
        S = torch.clamp(S, min=eps)
        S_sqrt = torch.sqrt(S)
        
        result = U @ torch.diag(S_sqrt) @ V.T
        
        # Additional check for numerical stability of result
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("SVD result contains NaN/Inf, using fallback")
            return torch.eye(A.shape[0], device=A.device) * torch.norm(A)
        
        return result
        
    except Exception as e:
        print(f"SVD failed: {e}")
        # Return regularized identity matrix as fallback
        return torch.eye(A.shape[0], device=A.device) * torch.norm(A)


def regmat(matrix, eps=1e-10):
    # Replace NaN and Inf values with finite numbers
    matrix_new = torch.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)
   
    # Add a small epsilon to the diagonal for numerical stability
    if matrix_new.dim() == 2 and matrix_new.size(0) == matrix_new.size(1):
        matrix_new = matrix_new + eps * torch.eye(matrix_new.size(0), device=matrix_new.device)
    
    return matrix_new

def sqrtm_eig(A):
    eigvals, eigvecs = torch.linalg.eig(A)
    eigvals_real = eigvals.real
    
    # Ensure eigenvalues are non-negative for the square root to be valid
    eigvals_sqrt = torch.sqrt(torch.clamp(eigvals_real, min=0.0))  # Square root of non-negative eigenvalues

    # Reconstruct the square root of the matrix using the eigenvectors
    # Make sure the eigenvectors are also real
    eigvecs_real = eigvecs.real
    
    # Reconstruct the matrix square root
    sqrt_A = eigvecs_real @ torch.diag(eigvals_sqrt) @ eigvecs_real.T
    
    return sqrt_A

def constraints_error_check(satisfied_L, d_L, e, satisfied_H, d_H, d):
    if not satisfied_L:
        print(f"Warning: Constraints not satisfied for mu_L and Sigma_L! Distance: {d_L} and epsilon = {e}")

    if not satisfied_H:
        print(f"Warning: Constraints not satisfied for mu_H and Sigma_H! Distance: {d_H} and delta = {d}")

    return

def plot_inner_loop_objectives(inner_loop_objectives, epoch, erica=True):
    """
    Plot objectives from the inner loop optimization steps for a given epoch.
    
    Args:
        inner_loop_objectives: Dictionary containing 'min_objectives' and optionally 'max_objectives'
        epoch: Current epoch number
        erica: Boolean indicating if both min and max objectives should be plotted
    """
    sns.set_style("whitegrid")
    
    # inner_loop_objectives['min_objectives'] = np.array([t.detach().numpy() for t in inner_loop_objectives['min_objectives']])
    # inner_loop_objectives['max_objectives'] = np.array([t.detach().numpy() for t in inner_loop_objectives['max_objectives']])

    # Convert min objectives to numpy, handling both tensor and float types
    min_objectives = []
    for obj in inner_loop_objectives['min_objectives']:
        if torch.is_tensor(obj):
            min_objectives.append(obj.detach().numpy())
        else:
            min_objectives.append(obj)
    inner_loop_objectives['min_objectives'] = np.array(min_objectives)
    
    # Convert max objectives to numpy, handling both tensor and float types
    max_objectives = []
    for obj in inner_loop_objectives['max_objectives']:
        if torch.is_tensor(obj):
            max_objectives.append(obj.detach().numpy())
        else:
            max_objectives.append(obj)
    inner_loop_objectives['max_objectives'] = np.array(max_objectives)

    plt.figure(figsize=(10, 6))
    if erica:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot minimization objectives
        steps = range(1, len(inner_loop_objectives['min_objectives']) + 1)
        sns.lineplot(
            x=steps,
            y=inner_loop_objectives['min_objectives'],
            color='green',
            ax=ax1
        )
        ax1.set_title(f'Minimization Steps (Epoch {epoch+1})')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Objective T Value')
        
        # Plot maximization objectives
        if 'max_objectives' in inner_loop_objectives:
            steps_max = range(1, len(inner_loop_objectives['max_objectives']) + 1)
            sns.lineplot(
                x=steps_max,
                y=inner_loop_objectives['max_objectives'],
                color='purple',
                ax=ax2
            )
            ax2.set_title(f'Maximization Steps (Epoch {epoch+1})')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Objective θ Value')
    else:
        # Single plot for minimization only
        plt.figure(figsize=(8, 5))
        steps = range(1, len(inner_loop_objectives['min_objectives']) + 1)
        sns.lineplot(
            x=steps,
            y=inner_loop_objectives['min_objectives'],
            color='green'
        )
        plt.title(f'Optimization Progress - Epoch {epoch+1}')
        plt.xlabel('Step')
        plt.ylabel('Objective T Value')
    
    plt.tight_layout()
    plt.show()

def plot_epoch_objectives(epoch_objectives, erica=True):
    """
    Plot overall optimization progress across epochs.
    
    Args:
        epoch_objectives: Dictionary containing 'T_objectives_overall' and optionally 'theta_objectives_overall'
        erica: Boolean indicating if both T and theta objectives should be plotted
    """
    sns.set_style("whitegrid")
    
    epoch_objectives['T_objectives_overall'] = np.array([t.detach().numpy() for t in epoch_objectives['T_objectives_overall']])
    epoch_objectives['theta_objectives_overall'] = np.array([t.detach().numpy() for t in epoch_objectives['theta_objectives_overall']])

    if erica:
        # Create figure with two subplots vertically stacked
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot T objectives
        epochs = range(1, len(epoch_objectives['T_objectives_overall']) + 1)
        sns.lineplot(
            x=epochs,
            y=epoch_objectives['T_objectives_overall'],
            color='green',
            ax=ax1
        )
        ax1.set_title('T Objective across Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('T Objective Value')
        
        # Plot theta objectives
        if 'theta_objectives_overall' in epoch_objectives:
            sns.lineplot(
                x=epochs,
                y=epoch_objectives['theta_objectives_overall'],
                color='purple',
                ax=ax2
            )
            ax2.set_title('θ Objective across Epochs')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('θ Objective Value')
    else:
        # Single plot for T objectives only
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(epoch_objectives['T_objectives_overall']) + 1)
        sns.lineplot(
            x=epochs,
            y=epoch_objectives['T_objectives_overall'],
            color='green'
        )
        plt.title('Overall Optimization Progress')
        plt.xlabel('Epoch')
        plt.ylabel('T Objective Value')
    
    plt.tight_layout()
    plt.show()

def print_results(T, paramsL, paramsH, elapsed_time):
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS".center(50))
    print("="*50 + "\n")

    print("Final Transformation Matrix T:")
    print("-"*30)
    print(np.array2string(T, precision=4, suppress_small=True))
    print(f"\nCondition Number: {evut.condition_number(T):.4f}")
    
    print("\nLow-Level Parameters:")
    print("-"*30)
    print(f"μ_L = {np.array2string(paramsL['mu_U'], precision=4, suppress_small=True)}")
    print(f"Σ_L = \n{np.array2string(paramsL['Sigma_U'], precision=4, suppress_small=True)}")
    
    print("\nHigh-Level Parameters:")
    print("-"*30)
    print(f"μ_H = {np.array2string(paramsH['mu_U'], precision=4, suppress_small=True)}")
    print(f"Σ_H = \n{np.array2string(paramsH['Sigma_U'], precision=4, suppress_small=True)}")

    print(f"\nOptimization time: {elapsed_time:.4f} seconds")

def project_onto_gelbrich_ball(mu_in, Sigma_in, hat_mu, hat_Sigma, epsilon, max_iter=100, tol=1e-4):
    """
    Project (mu, Sigma) onto the Gelbrich ball
    """
    mu = mu_in.clone().detach()
    Sigma = Sigma_in.clone().detach()
    for i in range(max_iter):
        mu_dist_sq     = torch.sum((mu - hat_mu)**2)
        Sigma_sqrt     = sqrtm_svd(Sigma)
        hat_Sigma_sqrt = sqrtm_svd(hat_Sigma)
        Sigma_dist_sq  = torch.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
        
        G_squared      = mu_dist_sq + Sigma_dist_sq
                
        if G_squared <= epsilon**2 + tol:
            #print(f"Projection converged in {i+1} iterations with G_squared = {G_squared.item()} and epsilon_sq = {epsilon**2}")
            break
        
        scale = epsilon / torch.sqrt(G_squared)

        mu         = hat_mu + scale * (mu - hat_mu)
        Sigma_diff = Sigma_sqrt - hat_Sigma_sqrt
        Sigma_sqrt = hat_Sigma_sqrt + scale * Sigma_diff
        Sigma      = torch.matmul(Sigma_sqrt, Sigma_sqrt.T)
        
       
    final_G_squared = torch.sum((mu - hat_mu)**2) + torch.sum((sqrtm_svd(Sigma) - hat_Sigma_sqrt)**2)
    #print(f"difference: {final_G_squared.item() - epsilon**2}")
    if final_G_squared > epsilon**2 + tol:
        print(f"Warning: Final G_squared = {final_G_squared.item()} > {epsilon**2}")

    return mu, Sigma

# Convert inputs to torch tensors and float32 if needed
def to_torch_float(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float32)
    return x.float()
    
def get_gelbrich_boundary(mu, Sigma, hat_mu, hat_Sigma, epsilon, max_iter=100, tol=1e-2):
    """
    Find a point exactly on the boundary of the Gelbrich ball where
    (||μ₁-μ₂||² + ||Σ₁^(1/2) - Σ₂^(1/2)||²) = epsilon²
    """

    mu        = to_torch_float(mu)
    Sigma     = to_torch_float(Sigma)
    hat_mu    = to_torch_float(hat_mu)
    hat_Sigma = to_torch_float(hat_Sigma)
    epsilon   = to_torch_float(epsilon)

    for _ in range(max_iter):
        mu_dist_sq     = torch.sum((mu - hat_mu)**2)
        Sigma_sqrt     = sqrtm_svd(Sigma)
        hat_Sigma_sqrt = sqrtm_svd(hat_Sigma)
        Sigma_dist_sq  = torch.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
        
        G_squared      = mu_dist_sq + Sigma_dist_sq
        
        # Calculate how far we are from the boundary
        diff = abs(G_squared - epsilon**2)
        
        if diff <= tol:
            #print(f"Found boundary point in {i+1} iterations with G_squared = {G_squared.item()}")
            break
            
        # Scale to exactly reach the boundary
        scale = epsilon / torch.sqrt(G_squared)
        
        mu         = hat_mu + scale * (mu - hat_mu)
        Sigma_diff = Sigma_sqrt - hat_Sigma_sqrt
        Sigma_sqrt = hat_Sigma_sqrt + scale * Sigma_diff
        Sigma      = torch.matmul(Sigma_sqrt, Sigma_sqrt.T)
    
    final_G_squared = torch.sum((mu - hat_mu)**2) + torch.sum((sqrtm_svd(Sigma) - hat_Sigma_sqrt)**2)
    if abs(final_G_squared - epsilon**2) > tol:
        print(f"Warning: Final G_squared = {final_G_squared.item()} ≠ {epsilon**2}")

    return mu, Sigma

def verify_gelbrich_constraint(mu, Sigma, hat_mu, hat_Sigma, radius):
    """
    Verify constraint
    """
    mu_dist_sq = torch.sum((mu - hat_mu)**2)
    # print(f"mu distance squared: {mu_dist_sq.item()}")
    
    Sigma_sqrt = sqrtm_svd(Sigma)
    hat_Sigma_sqrt = sqrtm_svd(hat_Sigma)
    Sigma_dist_sq = torch.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
    # print(f"Sigma distance squared: {Sigma_dist_sq.item()}")
    
    G_squared = mu_dist_sq + Sigma_dist_sq
    # print(f"Total G_squared: {G_squared.item()}, epsilon^2: {epsilon**2}")

    G_squared       = round(G_squared.item(), 5)
    radius_squared = round(radius**2, 5)
    
    return G_squared <= radius_squared, G_squared, radius_squared


def compute_objective_value(T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H, 
                            lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H, epsilon, delta, project_onto_gelbrich):
    """
    Compute the terms of the Wasserstein objective function, including regularization terms.
    
    Args:
        T: Transformation matrix
        L_i: Low-level structural matrix
        H_i: High-level structural matrix
        mu_L: Low-level mean
        mu_H: High-level mean
        Sigma_L: Low-level covariance
        Sigma_H: High-level covariance
        lambda_L: Regularization parameter for low-level variables
        lambda_H: Regularization parameter for high-level variables
        hat_mu_L: Target low-level mean
        hat_mu_H: Target high-level mean
        hat_Sigma_L: Target low-level covariance
        hat_Sigma_H: Target high-level covariance
        epsilon: Radius for low-level Gelbrich constraint
        delta: Radius for high-level Gelbrich constraint
        
    Returns:
        val: Value of the objective function
    """

    # Convert all inputs to float32 (torch.float)
    T = T.float()
    L_i = L_i.float()
    H_i = H_i.float()
    mu_L = mu_L.float()
    mu_H = mu_H.float()
    Sigma_L = Sigma_L.float()
    Sigma_H = Sigma_H.float()
    hat_mu_L = hat_mu_L.float()
    hat_mu_H = hat_mu_H.float()
    hat_Sigma_L = hat_Sigma_L.float()
    hat_Sigma_H = hat_Sigma_H.float()

    # Smooth term components
    L_i_mu_L     = L_i @ mu_L
    H_i_mu_H     = H_i @ mu_H
    term1        = torch.norm(T @ L_i_mu_L - H_i_mu_H) ** 2
    
    TL_Sigma_LLT = regmat(T @ L_i @ Sigma_L @ L_i.T @ T.T)
    term2        = torch.trace(TL_Sigma_LLT)

    H_Sigma_HH   = regmat(H_i @ Sigma_H @ H_i.T)
    term3        = torch.trace(H_Sigma_HH)
    
    term4        = -2 * torch.trace(sqrtm_svd(sqrtm_svd(TL_Sigma_LLT) @ H_Sigma_HH @ sqrtm_svd(TL_Sigma_LLT)))

    # Regularization terms
    Sigma_L_sqrt     = sqrtm_svd(Sigma_L)
    hat_Sigma_L_sqrt = sqrtm_svd(hat_Sigma_L)
    Sigma_H_sqrt     = sqrtm_svd(Sigma_H)
    hat_Sigma_H_sqrt = sqrtm_svd(hat_Sigma_H)

    reg_L    = lambda_L * (torch.norm(mu_L - hat_mu_L) ** 2 + torch.norm(Sigma_L_sqrt - hat_Sigma_L_sqrt, p='fro') ** 2 - epsilon**2)
    reg_H    = lambda_H * (torch.norm(mu_H - hat_mu_H) ** 2 + torch.norm(Sigma_H_sqrt - hat_Sigma_H_sqrt, p='fro') ** 2 - delta**2)

    # Total value
    if project_onto_gelbrich:
        val = term1 + term2 + term3 + term4

    else:
        val = term1 + term2 + term3 + term4 + reg_L + reg_H 

    return val

def compute_surrogate_objective_value(T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H, 
                                      lambda_L, lambda_H, hat_mu_L, hat_mu_H, 
                                      hat_Sigma_L, hat_Sigma_H, epsilon, delta, project_onto_gelbrich):
    """
    Computes the simplified surrogate objective F_tilde, where the non-smooth
    term is replaced by the product of Frobenius norms.
    """
    # The Smooth Part (F_s) 
    term1 = torch.norm(T @ L_i @ mu_L - H_i @ mu_H) ** 2
    TL_Sigma_LLT = regmat(T @ L_i @ Sigma_L @ L_i.T @ T.T)
    term2 = torch.trace(TL_Sigma_LLT)
    H_Sigma_HH = regmat(H_i @ Sigma_H @ H_i.T)
    term3 = torch.trace(H_Sigma_HH)

    # The Non-Smooth Part (F*_n) is now the simplified surrogate
    S_l_sqrt = sqrtm_svd(TL_Sigma_LLT)
    S_h_sqrt = sqrtm_svd(H_Sigma_HH)
    term4_surrogate = -2 * torch.norm(S_l_sqrt, p='fro') * torch.norm(S_h_sqrt, p='fro')
    
    # Regularization terms
    Sigma_L_sqrt     = sqrtm_svd(Sigma_L)
    hat_Sigma_L_sqrt = sqrtm_svd(hat_Sigma_L)
    Sigma_H_sqrt     = sqrtm_svd(Sigma_H)
    hat_Sigma_H_sqrt = sqrtm_svd(hat_Sigma_H)

    reg_L    = lambda_L * (torch.norm(mu_L - hat_mu_L) ** 2 + torch.norm(Sigma_L_sqrt - hat_Sigma_L_sqrt, p='fro') ** 2 - epsilon**2)
    reg_H    = lambda_H * (torch.norm(mu_H - hat_mu_H) ** 2 + torch.norm(Sigma_H_sqrt - hat_Sigma_H_sqrt, p='fro') ** 2 - delta**2)


    # Total value using the new surrogate term
    if project_onto_gelbrich:
        val = term1 + term2 + term3 + term4_surrogate
    else:
        val = term1 + term2 + term3 + term4_surrogate + reg_L + reg_H 
    return val


# Generate noise from N(0, σ²I)
def generate_gaussian_noise(dim, sigma=1.0):
    """
    Generate multivariate normal noise with mean 0 and covariance σ²I
    
    Args:
        dim: dimension of the noise vector
        sigma: standard deviation (sqrt of variance)
    
    Returns:
        noise: tensor of shape (dim,) sampled from N(0, σ²I)
    """
    return sigma * torch.randn(dim)


def get_initialization(theta_hatL, theta_hatH, epsilon, delta, initial_theta):
    hat_mu_L, hat_Sigma_L = torch.from_numpy(theta_hatL['mu_U']).float(), torch.from_numpy(theta_hatL['Sigma_U']).float()
    hat_mu_H, hat_Sigma_H = torch.from_numpy(theta_hatH['mu_U']).float(), torch.from_numpy(theta_hatH['Sigma_U']).float()

    l, h = hat_mu_L.shape[0], hat_mu_H.shape[0]

    if initial_theta == 'gelbrich':
        ll_moments    = mut.sample_moments_U(mu_hat = theta_hatL['mu_U'], Sigma_hat = theta_hatL['Sigma_U'], bound = epsilon, num_envs = 1)
        mu_L, Sigma_L = ll_moments[0]
        mu_L, Sigma_L = torch.from_numpy(mu_L).float(), torch.from_numpy(Sigma_L).float()

        hl_moments      = mut.sample_moments_U(mu_hat = theta_hatH['mu_U'], Sigma_hat = theta_hatH['Sigma_U'], bound = delta, num_envs = 1)
        mu_H, Sigma_H = hl_moments[0]
        mu_H, Sigma_H = torch.from_numpy(mu_H).float(), torch.from_numpy(Sigma_H).float()

    elif initial_theta == 'empirical':
        mu_L, Sigma_L = hat_mu_L, hat_Sigma_L
        mu_H, Sigma_H = hat_mu_H, hat_Sigma_H

    elif initial_theta == 'random':
        mu_L, Sigma_L = (torch.rand(l) * 10) - 5, torch.diag(torch.rand(l) * 5)
        mu_H, Sigma_H = (torch.rand(h) * 10) - 5, torch.diag(torch.rand(h) * 5)
    
    elif initial_theta == 'random_invalid':
        raise ValueError(f"Invalid initial_theta value: {initial_theta}")

    else:
        raise ValueError(f"Invalid initial_theta value: {initial_theta}")

    return mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H



def create_psd_matrix(size):
    A = torch.randn(size, size).float()

    return torch.matmul(A, A.T)

# PCA Projection from higher to lower dimension
def pca_projection(Sigma, target_dim):
    """
    Project a d×d matrix to a k×k matrix where k < d
    Args:
        Sigma: source matrix (d×d)
        target_dim: target dimension k
    Returns:
        k×k projected matrix
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Take only the top target_dim eigenvectors
    V = eigenvectors[:, :target_dim]  # d×k matrix
    
    # Project the covariance matrix
    Sigma_projected = torch.matmul(torch.matmul(V.T, Sigma), V)  # k×k matrix
    
    return Sigma_projected, V

# SVD Projection from higher to lower dimension
def svd_projection(Sigma, target_dim):
    """
    Project a d×d matrix to a k×k matrix where k < d using SVD
    Args:
        Sigma: source matrix (d×d)
        target_dim: target dimension k
    Returns:
        k×k projected matrix
    """
    # Perform SVD
    U, S, V = torch.svd(Sigma)
    
    # Take only the first target_dim components
    U_k = U[:, :target_dim]  # d×k matrix
    S_k = S[:target_dim]     # k singular values
    
    # Project the covariance matrix
    Sigma_projected = torch.matmul(torch.matmul(U_k.T, Sigma), U_k)  # k×k matrix
    
    return Sigma_projected, U_k

def project_covariance(Sigma, n, method):
    if method == 'pca':
        return pca_projection(Sigma, n)
    elif method == 'svd':
        return svd_projection(Sigma, n)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
def compute_struc_matrices(models, I):
    matrices = []
    for iota in I:
        M_i = torch.from_numpy(models[iota]._compute_reduced_form()).float()  
        matrices.append(M_i)

    return matrices

def compute_mu_bary(struc_matrices, mu):
    struc_matrices_tensor = torch.stack(struc_matrices)
    mu_barycenter         = torch.sum(struc_matrices_tensor @ mu, dim=0) / len(struc_matrices)

    return mu_barycenter

def compute_Sigma_bary(matrices, Sigma, initialization, max_iter, tol):

    Sigma_matrices = []
    for M in matrices:
        Sigma_matrices.append(M @ Sigma @ M.T)

    return covariance_bary_optim(Sigma_matrices, initialization, max_iter, tol)

def covariance_bary_optim(Sigma_list, initialization, max_iter, tol):
    
    if initialization == 'psd':
        S_0 = create_psd_matrix(Sigma_list[0].shape[0])
    elif initialization == 'avg':
        S_0 = sum(Sigma_list) / len(Sigma_list)
    
    S_n = S_0.clone()
    n   = len(Sigma_list)  # Number of matrices
    lambda_j = 1.0 / n   # Equal weights
    
    for n in range(max_iter):
        S_n_old = S_n.clone()

        S_n_inv_half = sqrtm_svd(regmat(torch.inverse(S_n)))
        
        # Compute the sum of S_n^(1/2) Σ_j S_n^(1/2)
        sum_term = torch.zeros_like(S_n)
        for Sigma_j in Sigma_list:
            S_n_half   = sqrtm_svd(regmat(S_n))
            inner_term = torch.matmul(torch.matmul(S_n_half, Sigma_j), S_n_half)
            sqrt_term  = sqrtm_svd(regmat(inner_term))
            sum_term  += lambda_j * sqrt_term
        # Square the sum term
        squared_sum = torch.matmul(sum_term, sum_term.T)

        S_n_next = torch.matmul(torch.matmul(S_n_inv_half, squared_sum), S_n_inv_half)
        S_n = S_n_next

        if torch.norm(S_n - S_n_old, p='fro') < tol:
            #print(f"Converged after {n+1} iterations")
            break
            
    return S_n

def monge(m1, S1, m2, S2):
    inner      = torch.matmul(sqrtm_svd(S1), torch.matmul(S2, sqrtm_svd(S1)))
    sqrt_inner = sqrtm_svd(inner)
    A          = torch.matmul(torch.inverse(sqrtm_svd(regmat(S1))), torch.matmul(sqrt_inner, torch.inverse(sqrtm_svd(regmat(S1)))))  

    # Define the Monge map as a function τ(x) = m_2 + A(x - m_1)
    def tau(x):
        return m2 + A @ (x - m1)

    return tau, A

def regmat(matrix, eps=1e-10):
    # Replace NaN and Inf values with finite numbers
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Add a small epsilon to the diagonal for numerical stability
    if matrix.dim() == 2 and matrix.size(0) == matrix.size(1):
        matrix = matrix + eps * torch.eye(matrix.size(0), device=matrix.device)
    
    return matrix


def auto_bary_optim(theta_baryL, theta_baryH, max_iter, tol, seed):

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    mu_L, Sigma_L = torch.from_numpy(theta_baryL['mu_U']).float(), torch.from_numpy(theta_baryL['Sigma_U']).float()
    mu_H, Sigma_H = torch.from_numpy(theta_baryH['mu_U']).float(), torch.from_numpy(theta_baryH['Sigma_U']).float()


    T = torch.randn(mu_H.shape[0], mu_L.shape[0], requires_grad=True)

    optimizer_T        = torch.optim.Adam([T], lr=0.001)
    previous_objective = float('inf')
    objective_T        = 0  # Reset objective at the start of each step
    # Optimization loop
    for step in range(max_iter):
        objective_T = 0  # Reset objective at the start of each step
        
        # Calculate each term of the Wasserstein distance
        term1 = torch.norm(T @ mu_L - mu_H) ** 2  # Squared Euclidean distance between transformed means
        term2 = torch.trace(T @ Sigma_L @ T.T)   # Trace term for low-level covariance
        term3 = torch.trace(Sigma_H)             # Trace term for high-level covariance
        
        # Compute the intermediate covariance matrices
        T_Sigma_L_T      = torch.matmul(T, torch.matmul(Sigma_L, T.T))
        T_Sigma_L_T_sqrt = sqrtm_svd(T_Sigma_L_T)
        Sigma_H_sqrt     = sqrtm_svd(Sigma_H)
        
        # Coupling term using nuclear norm
        term4 = -2 * torch.norm(T_Sigma_L_T_sqrt @ Sigma_H_sqrt, p='nuc')

        # Total objective is the sum of terms
        objective_T += term1 + term2 + term3 + term4

        if abs(previous_objective - objective_T.item()) < tol:
            print(f"Converged at step {step + 1}/{max_iter} with objective: {objective_T.item()}")
            break

        # Update previous objective
        previous_objective = objective_T.item()

        # Perform optimization step
        optimizer_T.zero_grad()  # Clear gradients
        objective_T.backward(retain_graph=True)  # Backpropagate
        optimizer_T.step()  # Update T

    return T  # Return final objective and optimized T

def barycentric_optimization(theta_L, theta_H, LLmodels, HLmodels, Ill, Ihl, projection_method, initialization, autograd, seed, max_iter, tol, display_results):

    # Start timing
    start_time = time.time()

    mu_L, Sigma_L = torch.from_numpy(theta_L['mu_U']).float(), torch.from_numpy(theta_L['Sigma_U']).float()
    mu_H, Sigma_H = torch.from_numpy(theta_H['mu_U']).float(), torch.from_numpy(theta_H['Sigma_U']).float()

    epsilon, delta = theta_L['radius'], theta_H['radius']

    h, l = mu_H.shape[0], mu_L.shape[0]

    # Initialize the structural matrices    
    L_matrices   = compute_struc_matrices(LLmodels, Ill)
    H_matrices   = compute_struc_matrices(HLmodels, Ihl)

    # Initilize the barycenteric means and covariances
    mu_bary_L    = compute_mu_bary(L_matrices, mu_L)
    mu_bary_H    = compute_mu_bary(H_matrices, mu_H)

    Sigma_bary_L = compute_Sigma_bary(L_matrices, Sigma_L, initialization, max_iter, tol)
    Sigma_bary_H = compute_Sigma_bary(H_matrices, Sigma_H, initialization, max_iter, tol)
    
    proj_Sigma_bary_L, Tp = project_covariance(Sigma_bary_L, h, projection_method)
    proj_mu_bary_L        = torch.matmul(Tp.T, mu_bary_L)

    paramsL = {'mu_U': mu_bary_L.detach().numpy(), 'Sigma_U': Sigma_bary_L.detach().numpy(), 'radius': epsilon}
    paramsH = {'mu_U': mu_bary_H.detach().numpy(), 'Sigma_U': Sigma_bary_H.detach().numpy(), 'radius': delta}

    if autograd == True:
        params_bary_autograd = {
                                'theta_baryL': paramsL,
                                'theta_baryH': paramsH,
                                'max_iter': 10,
                                'tol': 1e-5,
                                'seed': seed
                               }
        
        T = auto_bary_optim(**params_bary_autograd)

    else:
        tau, A = monge(proj_mu_bary_L, proj_Sigma_bary_L, mu_bary_H, Sigma_bary_H)
        T = torch.matmul(A, Tp.T)

    T  = T.detach().numpy()
    Tp = Tp.detach().numpy()

    opt_params = {'L': paramsL, 'H': paramsH}

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Display results
    if display_results == True: 
        print_results(T, paramsL, paramsH, elapsed_time)

    return opt_params, T



def check_for_invalid_values(matrix):
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        #print("Matrix contains NaN or Inf values!")
        return True
    return False

def handle_nans(matrix, replacement_value=0.0):
    # Replace NaNs with a given value (default is 0)
    if torch.isnan(matrix).any():
        print("Warning: NaN values found! Replacing with zero.")
        matrix = torch.nan_to_num(matrix, nan=replacement_value)
    return matrix


def compute_grad_mu_L(T, mu_L, mu_H, LLmodels, HLmodels, omega, lambda_L, hat_mu_L):
    Ill = list(LLmodels.keys())
    sum_term     = torch.zeros_like(mu_L)
    for n, iota in enumerate(Ill):
        L_i   = torch.from_numpy(LLmodels[iota].F).float() 
        V_i   = T @ L_i  
        H_i   = torch.from_numpy(HLmodels[omega[iota]].F).float() 

        sum_term = sum_term + V_i.T @ V_i @ mu_L.float() - V_i.T @ H_i @ mu_H.float()
    
    reg_term     = 2 * lambda_L * (mu_L - hat_mu_L) # reg_term = -2 * lambda_L * (mu_L - hat_mu_L)
    grad_mu_L = (2 / (len(Ill))) * sum_term + reg_term

    return grad_mu_L

def compute_grad_mu_H(T, mu_L, mu_H, LLmodels, HLmodels, omega, lambda_H, hat_mu_H):

    Ill = list(LLmodels.keys())
    sum_term     = torch.zeros_like(mu_H)
    for n, iota in enumerate(Ill):
        L_i   = torch.from_numpy(LLmodels[iota].F).float()  
        V_i   = T @ L_i  
        H_i   = torch.from_numpy(HLmodels[omega[iota]].F).float()  

        sum_term = sum_term + H_i.T @ H_i @ mu_H.float() - (V_i.T @ H_i).T @ mu_L.float()
    
    reg_term     = 2 * lambda_H * (mu_H - hat_mu_H) # reg_term = -2 * lambda_H * (mu_H - hat_mu_H)
    grad_mu_H = (2 / len(Ill)) * sum_term + reg_term

    return grad_mu_H


def compute_grad_Sigma_L_half(T, Sigma_L, LLmodels, omega, lambda_L, hat_Sigma_L):
    Ill = list(LLmodels.keys())
    sum_term         = torch.zeros_like(Sigma_L)
    for n, iota in enumerate(Ill):
        L_i          = torch.from_numpy(LLmodels[iota].F)
        V_i          = T @ L_i.float()

        sum_term        = sum_term + V_i.T @ V_i

    Sigma_L_sqrt     = sqrtm_svd(Sigma_L)  
    hat_Sigma_L_sqrt = sqrtm_svd(hat_Sigma_L) 

    reg_term         = 2 * lambda_L * (Sigma_L_sqrt - hat_Sigma_L_sqrt) @ torch.inverse(Sigma_L_sqrt) # reg_term = -2 * lambda_L * (Sigma_L_sqrt - hat_Sigma_L_sqrt) @ torch.inverse(Sigma_L_sqrt)
    grad_Sigma_L     = (2 / (n+1)) * sum_term + reg_term
    
    return grad_Sigma_L

def compute_grad_Sigma_H_half(T, Sigma_H, LLmodels, HLmodels, omega, lambda_H, hat_Sigma_H):

    Ill = list(LLmodels.keys())
    sum_term            = torch.zeros_like(Sigma_H)
    for n, iota in enumerate(Ill):
        H_i          = torch.from_numpy(HLmodels[omega[iota]].F).float()
        
        sum_term     = sum_term + H_i.T @ H_i

    Sigma_H_sqrt     = sqrtm_svd(Sigma_H)  
    hat_Sigma_H_sqrt = sqrtm_svd(hat_Sigma_H) 

    reg_term            = 2 * lambda_H * (Sigma_H_sqrt - hat_Sigma_H_sqrt) @ torch.inverse(Sigma_H_sqrt) # reg_term = -2 * lambda_H * (Sigma_H_sqrt - hat_Sigma_H_sqrt) @ torch.inverse(Sigma_H_sqrt)
    grad_Sigma_H     = (2 / (n+1)) * sum_term + reg_term

    return grad_Sigma_H

def prox_grad_Sigma_L(T, Sigma_L_half, LLmodels, Sigma_H, HLmodels, omega, lambda_param_L):

    Ill = list(LLmodels.keys())

    Sigma_L               = torch.zeros_like(Sigma_L_half, dtype=torch.float32)  
    for n, iota in enumerate(Ill):
        L_i               = torch.from_numpy(LLmodels[iota].F).float()  
        V_i               = regmat(T @ L_i)  
        H_i               = torch.from_numpy(HLmodels[omega[iota]].F).float()  
        
        Sigma_L_half      = Sigma_L_half.float()
        V_Sigma_V         = V_i @ Sigma_L_half @ V_i.T
        sqrtm_V_Sigma_V   = sqrtm_svd(regmat(V_Sigma_V)) 
        
        prox_Sigma_L_half = prox_operator(sqrtm_V_Sigma_V, lambda_param_L) @ prox_operator(sqrtm_V_Sigma_V, lambda_param_L).T
        
        ll_term           = regmat(torch.linalg.pinv(V_i)) @ regmat(prox_Sigma_L_half) @ regmat(torch.linalg.pinv(V_i).T)

        Sigma_H           = Sigma_H.float()  
        H_Sigma_H         = H_i @ Sigma_H @ H_i.T

        hl_term           = torch.norm(sqrtm_svd(regmat(H_Sigma_H)), p='fro') 

        Sigma_L           = Sigma_L + (ll_term * hl_term)

    Sigma_L_final         = (2 / (n+1)) * Sigma_L 
    Sigma_L_final         = diagonalize(Sigma_L_final)
    
    # Add numerical stability check and regularization
    try:
        Sigma_L_np = Sigma_L_final.detach().cpu().numpy()
        Sigma_L_np = np.nan_to_num(Sigma_L_np, nan=1e-6, posinf=1e10, neginf=-1e10)
        Sigma_L_np = 0.5 * (Sigma_L_np + Sigma_L_np.T)  # Ensure symmetry
        
        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(Sigma_L_np)
        if np.any(eigenvalues < -1e-10):
            # Regularize to make positive semi-definite
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            U, _ = np.linalg.eigh(Sigma_L_np)
            Sigma_L_np = U @ np.diag(eigenvalues) @ U.T
            Sigma_L_final = torch.from_numpy(Sigma_L_np).float().to(Sigma_L_final.device)
    except Exception as e:
        print(f"WARNING: Numerical instability in prox_grad_Sigma_L: {e}")
        # Return regularized identity matrix
        n = Sigma_L_final.shape[0]
        Sigma_L_final = torch.eye(n, device=Sigma_L_final.device) * 1e-6

    return Sigma_L_final

def prox_grad_Sigma_H(T, Sigma_H_half, LLmodels, Sigma_L, HLmodels, omega, lambda_param_H):

    Ill = list(LLmodels.keys())
    Sigma_H               = torch.zeros_like(Sigma_H_half, dtype=torch.float32)
    for n, iota in enumerate(Ill):
        L_i               = torch.from_numpy(LLmodels[iota].F).float()
        V_i               = T @ L_i #oput.regmat(T @ L_i)
        H_i               = torch.from_numpy(HLmodels[omega[iota]].F).float()

        Sigma_H_half      = Sigma_H_half.float()
        H_Sigma_H         = H_i @ Sigma_H_half @ H_i.T
        sqrtm_H_Sigma_H   = sqrtm_svd(H_Sigma_H)
        #H_Sigma_H         = oput.regmat(H_i @ Sigma_H_half @ H_i.T)
  
        prox_Sigma_H_half = prox_operator(sqrtm_H_Sigma_H, lambda_param_H) @ prox_operator(sqrtm_H_Sigma_H, lambda_param_H).T
        hl_term_iota      = torch.inverse(H_i) @ prox_Sigma_H_half @ torch.inverse(H_i).T
        
        Sigma_L           = Sigma_L.float()
        V_Sigma_V         = V_i @ Sigma_L @ V_i.T
        ll_term_iota      = torch.norm(sqrtm_svd(V_Sigma_V), p='fro')

        Sigma_H           = Sigma_H + (ll_term_iota * hl_term_iota)
    
    Sigma_H_final         = (2 / (n+1)) * Sigma_H
    Sigma_H_final         = diagonalize(Sigma_H_final)
    
    # Add numerical stability check and regularization
    try:
        Sigma_H_np = Sigma_H_final.detach().cpu().numpy()
        Sigma_H_np = np.nan_to_num(Sigma_H_np, nan=1e-6, posinf=1e10, neginf=-1e10)
        Sigma_H_np = 0.5 * (Sigma_H_np + Sigma_H_np.T)  # Ensure symmetry
        
        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(Sigma_H_np)
        if np.any(eigenvalues < -1e-10):
            # Regularize to make positive semi-definite
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            U, _ = np.linalg.eigh(Sigma_H_np)
            Sigma_H_np = U @ np.diag(eigenvalues) @ U.T
            Sigma_H_final = torch.from_numpy(Sigma_H_np).float().to(Sigma_H_final.device)
    except Exception as e:
        print(f"WARNING: Numerical instability in prox_grad_Sigma_H: {e}")
        # Return regularized identity matrix
        n = Sigma_H_final.shape[0]
        Sigma_H_final = torch.eye(n, device=Sigma_H_final.device) * 1e-6

    return Sigma_H_final


def optimize_max(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, omega, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                 lambda_L, lambda_H, lambda_param_L, lambda_param_H, eta, num_steps_max, epsilon, delta, seed, project_onto_gelbrich, max_grad_norm):
    
    torch.manual_seed(seed)
    Ill = list(LLmodels.keys())
    cur_T = T.clone()
    theta_objectives_epoch = []
    objective_theta_step = torch.tensor(0.0)
    
    max_converged = False
    # Pre-compute constant terms
    L_matrices = {iota: torch.from_numpy(LLmodels[iota].F).float() for iota in Ill}
    H_matrices = {iota: torch.from_numpy(HLmodels[omega[iota]].F).float() for iota in Ill}
        
    # Initialize running averages for adaptive learning rate
    avg_grad_norm = 0
    beta = 0.9  # momentum factor
    
    for step in range(num_steps_max): 
        # Compute all gradients first
        grad_mu_L = compute_grad_mu_L(cur_T, mu_L, mu_H, LLmodels, HLmodels, omega, lambda_L, hat_mu_L)
        grad_mu_H = compute_grad_mu_H(cur_T, mu_L, mu_H, LLmodels, HLmodels, omega, lambda_H, hat_mu_H)
        grad_Sigma_L = compute_grad_Sigma_L_half(cur_T, Sigma_L, LLmodels, omega, lambda_L, hat_Sigma_L)
        grad_Sigma_H = compute_grad_Sigma_H_half(cur_T, Sigma_H, LLmodels, HLmodels, omega, lambda_H, hat_Sigma_H)

        # Clip gradients
        if max_grad_norm < float('inf'):
            grad_mu_L = torch.clamp(grad_mu_L, -max_grad_norm, max_grad_norm)
            grad_mu_H = torch.clamp(grad_mu_H, -max_grad_norm, max_grad_norm)
            grad_Sigma_L = torch.clamp(grad_Sigma_L, -max_grad_norm, max_grad_norm)
            grad_Sigma_H = torch.clamp(grad_Sigma_H, -max_grad_norm, max_grad_norm)

        # Compute current gradient norm
        current_grad_norm = (torch.norm(grad_mu_L) + torch.norm(grad_mu_H) + 
                           torch.norm(grad_Sigma_L) + torch.norm(grad_Sigma_H))
        
        # Update running average
        avg_grad_norm = beta * avg_grad_norm + (1 - beta) * current_grad_norm
        
        # Adjust learning rate if gradients are getting too large
        current_eta = eta
        if avg_grad_norm > 1.0:
            current_eta = eta / avg_grad_norm

        # Update parameters
        if delta == 0:
            mu_L = mu_L + current_eta * grad_mu_L
        elif epsilon == 0:
            mu_H = mu_H + current_eta * grad_mu_H
        else:
            mu_L = mu_L + current_eta * grad_mu_L
            mu_H = mu_H + current_eta * grad_mu_H
        
        if delta == 0:
            Sigma_L_half = Sigma_L + current_eta * grad_Sigma_L
        elif epsilon == 0:
            Sigma_H_half = Sigma_H + current_eta * grad_Sigma_H
        else:
            Sigma_L_half = Sigma_L + current_eta * grad_Sigma_L
            Sigma_H_half = Sigma_H + current_eta * grad_Sigma_H
        
        # Proximal updates
        if delta == 0:
            Sigma_L = prox_grad_Sigma_L(cur_T, Sigma_L_half, LLmodels, Sigma_H, HLmodels, omega, lambda_param_L)
        elif epsilon == 0:
            Sigma_H = prox_grad_Sigma_H(cur_T, Sigma_H_half, LLmodels, Sigma_L, HLmodels, omega, lambda_param_H)
        else:
            Sigma_L = prox_grad_Sigma_L(cur_T, Sigma_L_half, LLmodels, Sigma_H, HLmodels, omega, lambda_param_L)
            Sigma_H = prox_grad_Sigma_H(cur_T, Sigma_H_half, LLmodels, Sigma_L, HLmodels, omega, lambda_param_H)

        
        if project_onto_gelbrich:
            if delta == 0:
                mu_L, Sigma_L = project_onto_gelbrich_ball(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
            elif epsilon == 0:
                mu_H, Sigma_H = project_onto_gelbrich_ball(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
            else:
                mu_L, Sigma_L = project_onto_gelbrich_ball(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
                mu_H, Sigma_H = project_onto_gelbrich_ball(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)

            satisfied_L, dist_L, epsi = verify_gelbrich_constraint(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
            satisfied_H, dist_H, delt = verify_gelbrich_constraint(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
            
            #constraints_error_check(satisfied_L, dist_L, epsi, satisfied_H, dist_H, delt)

        objective_iota = 0
        for iota in Ill:
            obj_value_iota = compute_objective_value(
                                                    cur_T, L_matrices[iota], H_matrices[iota], 
                                                    mu_L, mu_H, Sigma_L, Sigma_H,
                                                    lambda_L, lambda_H, hat_mu_L, hat_mu_H, 
                                                    hat_Sigma_L, hat_Sigma_H, epsilon, delta, project_onto_gelbrich
                                                    )
            objective_iota += obj_value_iota

        objective_theta_step = objective_iota/len(Ill)
        theta_objectives_epoch.append(objective_theta_step)

    return mu_L, Sigma_L, mu_H, Sigma_H, objective_theta_step, theta_objectives_epoch, max_converged


def compute_worst_case_distance(mu_worst, Sigma_worst, params_hat):

    mu_hat = params_hat['mu_U']
    Sigma_hat = params_hat['Sigma_U']
    radius = params_hat['radius']
    
    mu_hat = torch.tensor(mu_hat, dtype=torch.float32) if not torch.is_tensor(mu_hat) else mu_hat
    Sigma_hat = torch.tensor(Sigma_hat, dtype=torch.float32) if not torch.is_tensor(Sigma_hat) else Sigma_hat
    

    mu_dist_sq     = torch.sum((mu_worst - mu_hat)**2)

    Sigma_sqrt     = sqrtm_svd(Sigma_worst)
    hat_Sigma_sqrt = sqrtm_svd(Sigma_hat)
    Sigma_dist_sq  = torch.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
    G_squared      = mu_dist_sq + Sigma_dist_sq
    G_squared = G_squared.item()
    radius_squared = round(radius**2, 5)
    
    diff = abs(G_squared - radius_squared)
    print(f"G_squared: {G_squared}, radius_squared: {radius_squared}, diff: {diff}, constraint satisfied: {G_squared <= radius_squared}")
    return #diff, G_squared


def empirical_objective(U_L, U_H, T, Theta, Phi, L_models, H_models, Ill, omega):

    loss_iota = 0
    for iota in Ill:
        L_i = torch.from_numpy(L_models[iota].F).float()
        H_i = torch.from_numpy(H_models[omega[iota]].F).float()
        
        pert_L_i = U_L + Theta
        pert_H_i = U_H + Phi

        diff = T @ L_i @ pert_L_i.T - H_i @ pert_H_i.T
        # Normalize by matrix size
        loss_iota += torch.norm(diff, p='fro')**2 / (diff.shape[0] * diff.shape[1])
    
    loss = loss_iota / len(Ill)
    return loss

def empirical_objective_no_max(U_L, U_H, T, L_models, H_models, Ill, omega):

    loss_iota = 0
    for iota in Ill:
        L_i = torch.from_numpy(L_models[iota].F).float()
        H_i = torch.from_numpy(H_models[omega[iota]].F).float()

        diff = T @ L_i @ U_L.T - H_i @ U_H.T

        loss_iota += torch.norm(diff, p='fro')**2 / (diff.shape[0] * diff.shape[1])

    loss = loss_iota / len(Ill)
    return loss

def project_onto_frobenius_ball(matrix, radius):
    """
    Projects matrix onto the ball defined by ||matrix||_F^2 <= radius_squared
    
    Args:
        matrix: The matrix to project
        radius_squared: The squared radius (N*epsilon^2 or N*delta^2)
    """
    N = matrix.shape[0]
    squared_norm = torch.norm(matrix, p='fro')**2
    if squared_norm > N * radius**2:
        return matrix * torch.sqrt(N * radius**2 / squared_norm)
    return matrix

def init_in_frobenius_ball(shape, epsilon):
    """
    Initialize a matrix inside the Frobenius ball with ||X||_F^2 <= N*epsilon^2
    """
    num_samples      = shape[0]
    matrix           = torch.randn(*shape)  # Standard normal initialization
    squared_norm     = torch.norm(matrix, p='fro')**2
    max_squared_norm = num_samples * epsilon**2
    
    # Scale to ensure it's inside the ball
    scaling_factor = torch.sqrt(max_squared_norm / squared_norm) * torch.rand(1)  # random scaling between 0 and max radius
    matrix = matrix * scaling_factor
    
    return nn.Parameter(matrix)  # This ensures requires_grad=True

def run_empirical_erica_optimization(U_L, U_H, L_models, H_models, omega, epsilon, delta, eta_min, eta_max,
                                    num_steps_min, num_steps_max, max_iter, tol, seed, robust_L, robust_H, initialization, experiment):
    
    torch.manual_seed(seed)
    Ill = list(L_models.keys())

    method  = 'erica' if robust_L or robust_H else 'enrico'
    num_steps_min = 1 if method == 'enrico' else num_steps_min

    # Convert inputs to torch tensors
    U_L = torch.as_tensor(U_L, dtype=torch.float32)
    U_H = torch.as_tensor(U_H, dtype=torch.float32)
    
    # Get dimensions
    N, l = U_L.shape
    _, h = U_H.shape
    
    # Initialize variables
    T     = torch.randn(h, l, requires_grad=True)
    if initialization == 'random':
        Theta = torch.randn(N, l, requires_grad=True)
        Phi   = torch.randn(N, h, requires_grad=True)

    elif initialization == 'projected':
        Theta = init_in_frobenius_ball((N, l), epsilon)
        Phi   = init_in_frobenius_ball((N, h), delta)

    
    # Create optimizers
    optimizer_T   = torch.optim.Adam([T], lr=eta_min)
    optimizer_max = torch.optim.Adam([Theta, Phi], lr=eta_max)
    
    prev_T_objective = float('inf')
    
    for iteration in tqdm(range(max_iter)):
         
        objs_T, objs_max = [], []
        # Step 1: Minimize with respect to T
        for _ in range(num_steps_min):
            optimizer_T.zero_grad()
            T_objective = empirical_objective(U_L, U_H, T, Theta, Phi, L_models, H_models, Ill, omega)
            objs_T.append(T_objective.item())
            T_objective.backward()
            optimizer_T.step()
        #plot_progress(objs_T, 'T')
        # Step 2: Maximize with respect to Theta and Phi
        if method == 'erica':
            for _ in range(num_steps_max):
                optimizer_max.zero_grad()
                max_objective = -empirical_objective(U_L, U_H, T, Theta, Phi, L_models, H_models, Ill, omega)
                max_objective.backward()
                optimizer_max.step()
                
                # Project onto constraint sets
                with torch.no_grad():
                    Theta.data = project_onto_frobenius_ball(Theta, epsilon)
                    Phi.data   = project_onto_frobenius_ball(Phi, delta)

                mobj = empirical_objective(U_L, U_H, T, Theta, Phi, L_models, H_models, Ill, omega)
                objs_max.append(mobj.item())

        with torch.no_grad():
            current_T_objective = T_objective.item()
            if abs(prev_T_objective - current_T_objective) < tol:
                print(f"Converged at iteration {iteration + 1}")
                break
            prev_T_objective = current_T_objective
            
    T       = T.detach().numpy()
    paramsL = {'pert_U': Theta.detach().numpy(), 'radius_worst': epsilon,
                    'pert_hat': U_L, 'radius': epsilon}
    paramsH = {'pert_U': Phi.detach().numpy(), 'radius_worst': delta,
                    'pert_hat': U_H, 'radius': delta}
    
    if method == 'erica':
        
        radius_worst_L          = evut.compute_empirical_worst_case_distance(paramsL)
        paramsL['radius_worst'] = radius_worst_L

        radius_worst_H          = evut.compute_empirical_worst_case_distance(paramsH)
        paramsH['radius_worst'] = radius_worst_H

    opt_params = {'L': paramsL, 'H': paramsH}

    save_dir = f"data/{experiment}/{method}"
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(opt_params, f"data/{experiment}/{method}/opt_params.pkl")

    return opt_params, T

def optimize_min(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, omega,
                 lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                 epsilon, delta, num_steps_min, optimizer_T, max_grad_norm, seed,
                 project_onto_gelbrich, xavier, monitor): 
    torch.manual_seed(seed)
    Ill = list(LLmodels.keys())
    if xavier:
        #T = torch.nn.init.xavier_normal_(T, gain=0.1)
        T0 = torch.eye(T.shape[0], T.shape[1])
        T = T + 1e-3 * torch.randn_like(T0)
        #T = torch.nn.init.orthogonal_(T, gain=0.1)
        # T = torch.nn.init.xavier_normal_(T, gain=0.01) for slc gaussian

    for step in range(num_steps_min):
        objective_iota = torch.tensor(0.0, device=T.device)
        for n, iota in enumerate(Ill):
            L_i = torch.from_numpy(LLmodels[iota].F).float().to(T.device)
            H_i = torch.from_numpy(HLmodels[omega[iota]].F).float().to(T.device)

            obj_value_iota = compute_objective_value(
                T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H,
                lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                epsilon, delta, project_onto_gelbrich
            )
            objective_iota += obj_value_iota

        objective_T_step = objective_iota / (n + 1)
        
        # Track progress using the monitor
        monitor.track_min_step(objective_T_step.item())

        if torch.isnan(T).any():
            print("T contains NaN! Returning previous matrix.")
            print('Failed at step:', step + 1)
            return T.detach().clone().requires_grad_(True), objective_T_step, True

        optimizer_T.zero_grad()
        objective_T_step.backward(retain_graph=True) 

        if max_grad_norm < float('inf'):
            torch.nn.utils.clip_grad_norm_([T], max_grad_norm)

        optimizer_T.step()

    return T, objective_T_step, False

def optimize_max_surrogate(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, omega, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, 
                           lambda_L, lambda_H, lambda_param_L, lambda_param_H, eta, num_steps_max, epsilon, delta, seed, project_onto_gelbrich, monitor, experiment):

    torch.manual_seed(seed)
    Ill = list(LLmodels.keys())
    cur_T = T.clone().detach()
    mu_L_optim, mu_H_optim = mu_L.clone().detach().requires_grad_(True), mu_H.clone().detach().requires_grad_(True)
    Sigma_L_optim, Sigma_H_optim = Sigma_L.clone().detach().requires_grad_(True), Sigma_H.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([mu_L_optim, mu_H_optim, Sigma_L_optim, Sigma_H_optim], lr=eta)
    
    for _ in range(num_steps_max):
        optimizer.zero_grad()
        
        obj_values = []
        for n, iota in enumerate(Ill):
            L_i = torch.from_numpy(LLmodels[iota].F).float()
            H_i = torch.from_numpy(HLmodels[omega[iota]].F).float()
            obj_value_iota = compute_surrogate_objective_value(
                cur_T, L_i, H_i, mu_L_optim, mu_H_optim, Sigma_L_optim, Sigma_H_optim, 
                lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                epsilon, delta, project_onto_gelbrich
            )
            obj_values.append(obj_value_iota)
        
        objective = -(torch.stack(obj_values).sum() / (n + 1))
        
        monitor.track_max_step(-objective.item())
        objective.backward()
        
        Sigma_L_before_step = Sigma_L_optim.detach().clone()
        Sigma_H_before_step = Sigma_H_optim.detach().clone()
        
        optimizer.step()
        
        # Perform the proximal and stabilization steps
        with torch.no_grad():
            Sigma_L_new = prox_grad_Sigma_L(cur_T, Sigma_L_before_step, LLmodels, Sigma_H_before_step, HLmodels, omega, lambda_param_L)
            Sigma_H_new = prox_grad_Sigma_H(cur_T, Sigma_H_before_step, LLmodels, Sigma_L_before_step, HLmodels, omega, lambda_param_H)
            
            Sigma_L_optim.data.copy_(Sigma_L_new.data)
            Sigma_H_optim.data.copy_(Sigma_H_new.data)
            
            # Jitter: Ensures matrices are positive definite
            jitter_L = 1e-5 * torch.eye(Sigma_L_optim.shape[0], device=Sigma_L_optim.device)
            jitter_H = 1e-5 * torch.eye(Sigma_H_optim.shape[0], device=Sigma_H_optim.device)
            Sigma_L_optim.add_(jitter_L)
            Sigma_H_optim.add_(jitter_H)
            
            if experiment == 'lilucas':
                mu_clip_val = 20.0
                sigma_clip_val = 20.0 
            else: # Default for other experiments
                mu_clip_val = 50.0
                sigma_clip_val = 100.0

            mu_L_optim.clamp_(-mu_clip_val, mu_clip_val)
            mu_H_optim.clamp_(-mu_clip_val, mu_clip_val)
            Sigma_L_optim.clamp_(-sigma_clip_val, sigma_clip_val)
            Sigma_H_optim.clamp_(-sigma_clip_val, sigma_clip_val)
     
            
    return (mu_L_optim.detach(), Sigma_L_optim.detach(), mu_H_optim.detach(), Sigma_H_optim.detach(), -objective.detach(), False)

def run_erica_optimization(theta_hatL, theta_hatH, initial_theta, LLmodels, HLmodels, omega,
                           lambda_L, lambda_H, lambda_param_L, lambda_param_H,
                           xavier, project_onto_gelbrich, eta_min, eta_max, max_iter,
                           num_steps_min, num_steps_max, proximal_grad, tol, seed,
                           robust_L, robust_H, grad_clip, plot_steps, plot_epochs,
                           display_results, experiment):

    torch.manual_seed(seed)
    start_time = time.time()
    run_device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Running optimization on device: {run_device}")

    epsilon = theta_hatL['radius'] if robust_L else 0
    delta = theta_hatH['radius'] if robust_H else 0
    method = 'erica' if robust_L or robust_H else 'enrico'
    #num_steps_min = 1 if method == 'enrico' else num_steps_min
    max_grad_norm = 1.0 if grad_clip else float('inf')

    mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H = get_initialization(
        theta_hatL, theta_hatH, epsilon, delta, initial_theta)
    
    mu_L, Sigma_L = mu_L.to(run_device), Sigma_L.to(run_device)
    mu_H, Sigma_H = mu_H.to(run_device), Sigma_H.to(run_device)
    hat_mu_L, hat_Sigma_L = hat_mu_L.to(run_device), hat_Sigma_L.to(run_device)
    hat_mu_H, hat_Sigma_H = hat_mu_H.to(run_device), hat_Sigma_H.to(run_device)

    T = torch.randn(mu_H.shape[0], mu_L.shape[0], requires_grad=True, device=run_device)
    optimizer_T = torch.optim.Adam([T], lr=eta_min)
    #optimizer_T = torch.optim.Adam([T], lr=eta_min, eps=1e-8, amsgrad=True)
    monitor = TrainingMonitor()
    previous_objective = float('inf')

    for epoch in tqdm(range(max_iter), desc="Optimizing"):
        monitor.start_epoch()
        T_prev = T.detach().clone()
        mu_L_last_good, Sigma_L_last_good = mu_L.clone(), Sigma_L.clone()
        mu_H_last_good, Sigma_H_last_good = mu_H.clone(), Sigma_H.clone()

        try:
            T_new, objective_T, hit_nan = optimize_min(
                T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, omega,
                lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                epsilon, delta, num_steps_min, optimizer_T, max_grad_norm, seed,
                project_onto_gelbrich, xavier, monitor)

            if hit_nan: raise RuntimeError("Unrecoverable NaN in optimize_min.")

            if method == 'erica':
                mu_L, Sigma_L, mu_H, Sigma_H, obj_theta, max_converged = optimize_max_surrogate(
                    T_new, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, omega,
                    hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                    lambda_L, lambda_H, lambda_param_L, lambda_param_H,
                    eta_max, num_steps_max, epsilon, delta, seed, project_onto_gelbrich, monitor, experiment)

                if project_onto_gelbrich:
                    mu_L, Sigma_L = project_onto_gelbrich_ball(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
                    mu_H, Sigma_H = project_onto_gelbrich_ball(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
            
            T = T_new
            current_objective = objective_T.item()

        except Exception as e:
            print(f"\nWARNING: Epoch {epoch+1} failed due to instability: {e}")
            print("         Reverting to parameters from the previous stable epoch.")
            mu_L, Sigma_L = mu_L_last_good, Sigma_L_last_good
            mu_H, Sigma_H = mu_H_last_good, Sigma_H_last_good
            current_objective = previous_objective

        monitor.end_epoch()
        if plot_steps: monitor.plot_epoch_progress(epoch)
        
        delta_objective = abs(previous_objective - current_objective)
        condition_num = evut.condition_number(T.detach().cpu().numpy())
        delta_T_norm = torch.norm(T - T_prev).item()
        
        monitor.track_epoch_metrics(
            delta_obj=delta_objective, cond_num=condition_num, delta_T=delta_T_norm,
            mu_L=mu_L, sigma_L=Sigma_L, mu_H=mu_H, sigma_H=Sigma_H)
        
        # if delta_objective < tol:
        #     print(f"Convergence reached at epoch {epoch+1}")
        #     break

        if (epoch + 1) >= 50 and delta_objective < tol:
            print(f"Convergence reached at epoch {epoch+1}")
            break
        
        previous_objective = current_objective

    if plot_epochs: monitor.plot_run_summary()
    
    paramsL = {
        'mu_U': mu_L.detach().numpy(),
        'Sigma_U': Sigma_L.detach().numpy(),
        'g_squared': epsilon,
        'mu_hat': theta_hatL['mu_U'],
        'Sigma_hat': theta_hatL['Sigma_U'],
        'radius': epsilon
    }
    paramsH = {
        'mu_U': mu_H.detach().numpy(),
        'Sigma_U': Sigma_H.detach().numpy(),
        'g_squared': delta,
        'mu_hat': theta_hatH['mu_U'],
        'Sigma_hat': theta_hatH['Sigma_U'],
        'radius': delta
    }

    T_final = T.detach().cpu().numpy()
    end_time = time.time()
    
    if display_results:
        elapsed_time = end_time - start_time
        print_results(T_final, paramsL, paramsH, elapsed_time)

    return {'L': paramsL, 'H': paramsH}, T_final, monitor


import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingMonitor:
    """
    A class to track and visualize metrics during optimization.
    """
    def __init__(self):
        self.epoch_history = []
        self.current_epoch_data = {}
        # Initialize the missing attributes for track_epoch_metrics
        self.delta_objectives = []
        self.condition_numbers = []
        self.delta_T_norms = []
        self.mu_L_history = []
        self.sigma_L_history = []
        self.mu_H_history = []
        self.sigma_H_history = []

    def compute_trajectory_length(self, level='L', lambda_reg=1.0):
        """Calculates the total path length of the optimization in parameter space."""
        mu_hist = self.mu_L_history if level == 'L' else self.mu_H_history
        sigma_hist = self.sigma_L_history if level == 'L' else self.sigma_H_history
        
        total_dist = 0.0
        for i in range(len(mu_hist) - 1):
            mu_dist = np.linalg.norm(mu_hist[i+1] - mu_hist[i])
            sigma_dist = np.linalg.norm(sigma_hist[i+1] - sigma_hist[i], 'fro')
            total_dist += mu_dist + lambda_reg * sigma_dist
        return total_dist

    def compute_spread_metrics(self, level='L'):
        """Calculates the spread of the visited distributions."""
        mu_hist = np.array(self.mu_L_history if level == 'L' else self.mu_H_history)
        sigma_hist = np.array(self.sigma_L_history if level == 'L' else self.sigma_H_history)
        
        # Spread of the mean vectors
        spread_mu = np.trace(np.cov(mu_hist, rowvar=False))
        
        # Spread of the covariance matrices
        sigma_bar = np.mean(sigma_hist, axis=0)
        spread_sigma = np.mean([np.linalg.norm(s - sigma_bar, 'fro')**2 for s in sigma_hist])
        
        return {'spread_mu': spread_mu, 'spread_sigma': spread_sigma}

    def compute_effective_rank_history(self, level='L'):
        """Calculates the effective rank for each covariance matrix in the trajectory."""
        sigma_hist = self.sigma_L_history if level == 'L' else self.sigma_H_history
        eff_ranks = []
        for sigma in sigma_hist:
            # Get real-valued eigenvalues for the symmetric matrix
            eigvals = np.linalg.eigvalsh(sigma)
            eigvals = np.maximum(eigvals, 1e-9) # Prevent issues with zero eigenvalues
            
            # Normalize eigenvalues to form a probability distribution
            normalized_eigvals = eigvals / np.sum(eigvals)
            
            # Calculate Shannon entropy of the spectrum
            entropy = -np.sum(normalized_eigvals * np.log(normalized_eigvals))
            eff_ranks.append(np.exp(entropy))
            
        return eff_ranks

    def _w2_sq_dist(self, mu1, sigma1, mu2, sigma2):
        """
        Helper to compute squared Wasserstein-2 distance using an SVD-based matrix sqrt
        to be consistent with the optimization's projection function.
        """
        # Helper function to compute sqrt via SVD, mimicking your 'sqrtm_svd_np'
        def _sqrtm_svd_internal(A):
            A = 0.5 * (A + A.T) # Symmetrize
            eps = 1e-10
            A = A + eps * np.eye(A.shape[0]) # Regularize
            U, S, Vh = np.linalg.svd(A)
            S_sqrt = np.sqrt(np.maximum(S, eps)) # Ensure non-negative before sqrt
            return U @ np.diag(S_sqrt) @ Vh

        mu_dist_sq = np.sum((mu1 - mu2)**2)
        
        sigma1_sqrt = _sqrtm_svd_internal(sigma1)
        sigma2_sqrt = _sqrtm_svd_internal(sigma2)
        
        sigma_dist_sq = np.sum((sigma1_sqrt - sigma2_sqrt)**2)
        
        return mu_dist_sq + sigma_dist_sq


    def compute_utilization_history(self, initial_params, level='L'):
        """Calculates how close the optimizer gets to the edge of the ball."""
        if level == 'L':
            mu_hist, sigma_hist = self.mu_L_history, self.sigma_L_history
            mu_hat, sigma_hat, radius = initial_params['mu_U'], initial_params['Sigma_U'], initial_params['radius']
        else:
            mu_hist, sigma_hist = self.mu_H_history, self.sigma_H_history
            mu_hat, sigma_hat, radius = initial_params['mu_U'], initial_params['Sigma_U'], initial_params['radius']

        utilization = []
        if radius == 0: return [0.0] * len(mu_hist) # Avoid division by zero
        
        for mu_t, sigma_t in zip(mu_hist, sigma_hist):
            w2_dist = np.sqrt(self._w2_sq_dist(mu_t, sigma_t, mu_hat, sigma_hat))
            utilization.append(w2_dist / radius)
        return utilization
    
    def start_epoch(self):
        """Resets the step history for a new epoch."""
        self.current_epoch_data = {
            'min_step_losses': [],
            'max_step_losses': []
        }

    def track_min_step(self, loss):
        """Tracks the objective value during a minimization step."""
        self.current_epoch_data['min_step_losses'].append(loss)

    def track_max_step(self, loss):
        """Tracks the objective value during a maximization step."""
        self.current_epoch_data['max_step_losses'].append(loss)

    def end_epoch(self):
        """Stores the data for the completed epoch."""
        self.epoch_history.append(self.current_epoch_data)

    def plot_epoch_progress(self, epoch_num):
        """Plots the min and max objective values for the given epoch."""
        min_losses = self.current_epoch_data['min_step_losses']
        max_losses = self.current_epoch_data['max_step_losses']

        if not min_losses and not max_losses:
            print(f"Epoch {epoch_num + 1}: No data to plot.")
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        fig.suptitle(f'Epoch {epoch_num + 1} Learning Progress', fontsize=16)

        # Plot minimization steps 
        ax1 = sns.lineplot(x=range(len(min_losses)), y=min_losses, ax=axes[0], color='green', marker='o', label='Objective')
        ax1.set_title('Minimization over T (Loss should decrease)')
        ax1.set_xlabel('Min-Step')
        ax1.set_ylabel('Objective Value')
        ax1.legend()

        # Plot maximization steps 
        if max_losses:
            ax2 = sns.lineplot(x=range(len(max_losses)), y=max_losses, ax=axes[1], color='purple', marker='o', label='Objective')
            ax2.set_title('Maximization over θ (Loss should increase)')
            ax2.set_xlabel('Max-Step')
            ax2.legend()
        else:
            ax2.set_title('Maximization over θ')
            ax2.text(0.5, 0.5, 'No max steps in this epoch.', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_run_summary(self):
        """Plots a 2x2 grid summarizing the entire optimization run."""
        if not self.epoch_history:
            print("No epoch history to plot.")
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Optimization Run Summary', fontsize=20)
        
        epochs = range(1, len(self.epoch_history) + 1)

        # 1. T Objective Across Epochs
        t_objectives = [epoch['min_step_losses'][-1] for epoch in self.epoch_history if epoch['min_step_losses']]
        sns.lineplot(x=epochs, y=t_objectives, ax=axes[0, 0], color='green').set(title='T Objective Convergence', xlabel='Epoch', ylabel='Final Objective Value')

        # 2. Delta Objective (Change in Objective)
        sns.lineplot(x=epochs, y=self.delta_objectives, ax=axes[0, 1], color='blue').set(title='Delta Objective (Convergence)', xlabel='Epoch', ylabel='|Obj_t - Obj_t-1|')
        axes[0, 1].set_yscale('log') # Log scale is often useful for convergence plots

        # 3. Condition Number of T
        sns.lineplot(x=epochs, y=self.condition_numbers, ax=axes[1, 0], color='red').set(title='Condition Number of T (Stability)', xlabel='Epoch', ylabel='Cond(T)')
        
        # 4. Delta T Norm (Change in T matrix)
        sns.lineplot(x=epochs, y=self.delta_T_norms, ax=axes[1, 1], color='orange').set(title='||T_t - T_t-1|| (T Matrix Change)', xlabel='Epoch', ylabel='Frobenius Norm')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    # --- NEW: Method to track all end-of-epoch metrics at once ---
    def track_epoch_metrics(self, delta_obj, cond_num, delta_T, mu_L, sigma_L, mu_H, sigma_H):
        self.delta_objectives.append(delta_obj)
        self.condition_numbers.append(cond_num)
        self.delta_T_norms.append(delta_T)
        self.mu_L_history.append(mu_L.detach().cpu().numpy())
        self.sigma_L_history.append(sigma_L.detach().cpu().numpy())
        self.mu_H_history.append(mu_H.detach().cpu().numpy())
        self.sigma_H_history.append(sigma_H.detach().cpu().numpy())



def empirical_bary(M):
    return  sum(M) / len(M)

def compute_transformed_samples(matrices, U):
    return [(M @ U.T).T for M in matrices]

def run_empirical_bary_optim(U_ll_hat, U_hl_hat, L_matrices, H_matrices, max_iter, tol, seed):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    UL, UH = torch.from_numpy(U_ll_hat).float(), torch.from_numpy(U_hl_hat).float()
    
    L_samples = compute_transformed_samples(L_matrices, U_ll_hat)
    H_samples = compute_transformed_samples(H_matrices, U_hl_hat)

   
    bary_L = empirical_bary(L_samples).float()
    bary_H = empirical_bary(H_samples).float()
    
    h = bary_H.shape[1]
    l = bary_L.shape[1]
    T = torch.randn(h, l, requires_grad=True)

    optimizer_T        = torch.optim.Adam([T], lr=0.001)
    previous_objective = float('inf')
    objective_T        = 0  # Reset objective at the start of each step
    # Optimization loop
    #for step in tqdm(range(max_iter)):
    for step in tqdm(range(int(max_iter))):
        objective_T = 0  # Reset objective at the start of each step
        
        diff = T @ bary_L.T  - bary_H.T
        # Normalize by matrix size
        objective_T += torch.norm(diff, p='fro')**2 / (diff.shape[0] * diff.shape[1])

        if abs(previous_objective - objective_T.item()) < tol:
            print(f"Converged at step {step + 1}/{max_iter} with objective: {objective_T.item()}")
            break

        # Update previous objective
        previous_objective = objective_T.item()

        # Perform optimization step
        optimizer_T.zero_grad()  # Clear gradients
        objective_T.backward(retain_graph=True)  # Backpropagate
        optimizer_T.step()  # Update T

    return T.detach().numpy()  # Return final objective and optimized T


def compute_empirical_radius(N, eta, c1=1.0, c2=1.0, alpha=2.0, m=3):
    """
    Compute epsilon_N(eta) for empirical Wasserstein case.

    Parameters:
    - N: int, number of samples
    - eta: float, confidence level (0 < eta < 1)
    - c1: float, constant from theorem (default 1.0, adjust if needed)
    - c2: float, constant from theorem (default 1.0, adjust if needed)
    - alpha: float, light-tail exponent (P[exp(||ξ||^α)] ≤ A)
    - m: int, ambient dimension

    Returns:
    - epsilon: float, the concentration radius
    """
    assert 0 < eta < 1, "eta must be in (0,1)"
    threshold = np.log(c1 / eta) / c2
    if N >= threshold:
        exponent = min(1/m, 0.5)
    else:
        exponent = 1 / alpha

    epsilon = (np.log(c1 / eta) / (c2 * N)) ** exponent
    return epsilon


import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

def perfect_abstraction(px_samples, py_samples, tau_threshold=1e-2):
    """
    Fits an abstraction function assumed to be perfect.
    
    Args:
        px_samples: concrete/base samples (your df_base)
        py_samples: abstract samples (your df_abst)
        tau_threshold: threshold for filtering small values
    """
    tau_adj = np.linalg.pinv(px_samples) @ py_samples
    tau_adj_mask = np.abs(tau_adj) > tau_threshold
    tau_adj = tau_adj * tau_adj_mask
    return tau_adj

def noisy_abstraction(px_samples, py_samples, tau_threshold=1e-1, refit_coeff=False):
    """
    Fits an abstraction function assumed to be noisy.
    """
    # Reconstruct T
    tau_adj_hat = np.linalg.pinv(px_samples) @ py_samples
    
    # Max entries
    tau_mask_hat = np.argmax(np.abs(tau_adj_hat), axis=1)
    # To one hot
    abs_nodes = py_samples.shape[1]
    tau_mask_hat = np.eye(abs_nodes)[tau_mask_hat]
    # Filter out small values
    tau_mask_hat *= np.array(np.abs(tau_adj_hat) > tau_threshold, dtype=np.int32)
    
    # Eventually refit coefficients
    if refit_coeff:
        for y in range(tau_mask_hat.shape[1]):
            block = np.where(tau_mask_hat[:, y] == 1)[0]
            if len(block) > 0:
                tau_adj_hat[block, y] = np.linalg.pinv(px_samples[:, block]) @ py_samples[:, y]
    
    # Compute final abstraction
    tau_adj_hat = tau_mask_hat * tau_adj_hat
    return tau_adj_hat

def abs_lingam_reconstruction(df_base, df_abst, style="Perfect", tau_threshold=1e-2):
    """
    Implement Abs-LiNGAM's T-reconstruction following the paper's implementation.
    
    Args:
        df_base: numpy array of concrete observations
        df_abst: numpy array of abstract observations
        style: "Perfect" or "Noisy"
        tau_threshold: threshold for filtering small values
    """
    # Verify shapes
    n_samples_base, d = df_base.shape
    n_samples_abst, b = df_abst.shape
    
    assert n_samples_base == n_samples_abst, \
        f"Number of samples must match: {n_samples_base} != {n_samples_abst}"
    
    # Fit abstraction function based on style
    if style == "Perfect":
        T = perfect_abstraction(df_base, df_abst, tau_threshold)
    elif style == "Noisy":
        T = noisy_abstraction(df_base, df_abst, tau_threshold, False)
    else:
        raise ValueError(f"Unknown style {style}")
    
    # Compute reconstruction error
    error = np.sum((df_base @ T - df_abst) ** 2)
    
    return T, error

def evaluate_abstraction(df_base, df_abst, T, tau_threshold=1e-2):
    """
    Evaluate the abstraction quality using metrics from the paper.
    """
    # Predictions
    predictions = df_base @ T
    
    # Create binary masks for evaluation
    T_mask = np.abs(T) > tau_threshold
    T_pred = np.abs(T) > tau_threshold
    
    # Basic metrics
    mse = np.mean((df_abst - predictions) ** 2)
    r2 = 1 - np.sum((df_abst - predictions) ** 2) / np.sum((df_abst - np.mean(df_abst, axis=0)) ** 2)
    
    # Structure metrics (from paper)
    def false_positive_rate(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return fp / (fp + tn) if (fp + tn) > 0 else 0
        return 0.0
    
    # Flatten for binary metrics
    T_flat = T_mask.flatten()
    T_pred_flat = T_pred.flatten()
    
    # Handle cases where all predictions are the same class
    try:
        prec = precision_score(T_flat, T_pred_flat)
    except:
        prec = 1.0 if np.array_equal(T_flat, T_pred_flat) else 0.0
        
    try:
        rec = recall_score(T_flat, T_pred_flat)
    except:
        rec = 1.0 if np.array_equal(T_flat, T_pred_flat) else 0.0
        
    try:
        f1 = f1_score(T_flat, T_pred_flat)
    except:
        f1 = 1.0 if np.array_equal(T_flat, T_pred_flat) else 0.0
    
    # Skip ROC AUC if we have only one class
    unique_classes = np.unique(T_flat)
    if len(unique_classes) == 2:
        try:
            roc_auc = roc_auc_score(T_flat, np.abs(T).flatten())
        except:
            roc_auc = np.nan
    else:
        roc_auc = np.nan
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': false_positive_rate(T_flat, T_pred_flat)
    }
    
    # Add sparsity metric
    metrics['sparsity'] = 1.0 - (np.sum(T_mask) / T_mask.size)
    
    return metrics

def abs_lingam_reconstruction_v2(df_base, df_abst, n_paired_samples=None, style="Perfect", tau_threshold=1e-2):
    """
    Modified version to better match the paper's approach.
    
    Args:
        df_base: numpy array of concrete observations (D_L)
        df_abst: numpy array of abstract observations
        n_paired_samples: number of samples to use for joint dataset (D_J)
        style: "Perfect" or "Noisy"
        tau_threshold: threshold for filtering small values
    """
    # Get dimensions
    n_samples_base, d = df_base.shape
    n_samples_abst, b = df_abst.shape
    
    # Create joint dataset D_J by taking a subset of paired samples
    if n_paired_samples is None:
        n_paired_samples = min(n_samples_base, n_samples_abst)
    
    # Ensure n_paired_samples is smaller than both datasets
    n_paired_samples = min(n_paired_samples, n_samples_base, n_samples_abst)
    
    # Create D_J using the first n_paired_samples
    D_J_base = df_base[:n_paired_samples]
    D_J_abst = df_abst[:n_paired_samples]
    
    # Fit abstraction function based on style using only paired data
    if style == "Perfect":
        T = perfect_abstraction(D_J_base, D_J_abst, tau_threshold)
    elif style == "Noisy":
        T = noisy_abstraction(D_J_base, D_J_abst, tau_threshold, False)
    else:
        raise ValueError(f"Unknown style {style}")
    
    # Compute reconstruction error using all available data
    error = np.sum((df_base @ T - df_abst) ** 2)
    
    return T, error

def run_abs_lingam_complete(df_base, df_abst, n_paired_samples=1000):
    """
    Run complete Abs-LiNGAM algorithm with different abstraction styles.
    """
    styles = ["Perfect", "Noisy"]
    results = {}
    
    for style in styles:
        T, error = abs_lingam_reconstruction_v2(
            df_base, 
            df_abst,
            n_paired_samples=n_paired_samples,
            style=style
        )
        metrics = evaluate_abstraction(df_base, df_abst, T)
        
        results[style] = {
            'T': T,
            'error': error,
            'metrics': metrics
        }
    
    return results