import numpy as np
import torch
import time

import networkx as nx
from tqdm import tqdm
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
        
        return U @ torch.diag(S_sqrt) @ V.T
        
    except Exception as e:
        print(f"SVD failed: {e}")
        # Return regularized identity matrix as fallback
        return torch.eye(A.shape[0], device=A.device) * torch.norm(A)


def are_matrices_equal(matrix1, matrix2, tol=1e-11):
    """
    Check if two matrices are equal within a given tolerance.
    
    Args:
        matrix1 (np.ndarray): The first matrix to compare.
        matrix2 (np.ndarray): The second matrix to compare.
        tol (float): Tolerance for element-wise comparison (default: 1e-8).
    
    Returns:
        bool: True if the matrices are equal within the given tolerance, False otherwise.
    """
    if matrix1.shape != matrix2.shape:
        return False  # Matrices must have the same shape to be equal

    return torch.allclose(matrix1, matrix2, atol=tol)

def regmat(matrix, eps=1e-10):
    # Replace NaN and Inf values with finite numbers
    matrix_new = torch.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)
    # if not are_matrices_equal(matrix, matrix_new):
    #     print('O')
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
    
    inner_loop_objectives['min_objectives'] = np.array([t.detach().numpy() for t in inner_loop_objectives['min_objectives']])
    inner_loop_objectives['max_objectives'] = np.array([t.detach().numpy() for t in inner_loop_objectives['max_objectives']])
    
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

def project_onto_gelbrich_ball(mu, Sigma, hat_mu, hat_Sigma, epsilon, max_iter=100, tol=1e-4):
    """
    Project (mu, Sigma) onto the Gelbrich ball
    """
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
    if final_G_squared > epsilon**2 + tol:
        print(f"Warning: Final G_squared = {final_G_squared.item()} > {epsilon**2}")
    
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


def compute_objective_value_old(T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H):
    """
    Compute the terms of the Wasserstein objective function.
    
    Args:
        T: Transformation matrix
        L_i: Low-level structural matrix
        H_i: High-level structural matrix
        mu_L: Low-level mean
        mu_H: High-level mean
        Sigma_L: Low-level covariance
        Sigma_H: High-level covariance
        
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

    L_i_mu_L     = L_i @ mu_L
    H_i_mu_H     = H_i @ mu_H
    term1        = torch.norm(T @ L_i_mu_L - H_i_mu_H) ** 2
    
    TL_Sigma_LLT = regmat(T @ L_i @ Sigma_L @ L_i.T @ T.T)
    term2        = torch.trace(TL_Sigma_LLT)

    H_Sigma_HH   = regmat(H_i @ Sigma_H @ H_i.T)
    term3        = torch.trace(H_Sigma_HH)

    term4       = -2 * torch.trace(sqrtm_svd(sqrtm_svd(TL_Sigma_LLT) @ H_Sigma_HH @ sqrtm_svd(TL_Sigma_LLT)))
    
    val         = term1 + term2 + term3 + term4

    return val

def compute_objective_value(T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H, 
                            lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H, epsilon, delta):
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

    #penalty_term = (reg_L - epsilon**2)**2 + (reg_H - delta**2)**2
    # Total value
    val = term1 + term2 + term3 + term4 + reg_L + reg_H #+ penalty_term

    return val

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


#======================= BARYCENTRIC OPTIMISATION =======================
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

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Display results
    if display_results == True: 
        print_results(T, paramsL, paramsH, elapsed_time)

    return paramsL, paramsH, T

#======================= GELBRICH OPTIMISATION =======================

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


def compute_grad_mu_L(T, mu_L, mu_H, LLmodels, HLmodels, lambda_L, hat_mu_L):

    sum_term     = torch.zeros_like(mu_L)
    for n, iota in enumerate(Ill):
        L_i   = torch.from_numpy(LLmodels[iota].F).float() 
        V_i   = T @ L_i  
        H_i   = torch.from_numpy(HLmodels[omega[iota]].F).float() 

        sum_term = sum_term + V_i.T @ V_i @ mu_L.float() - V_i.T @ H_i @ mu_H.float()
    
    reg_term     = 2 * lambda_L * (mu_L - hat_mu_L) # reg_term = -2 * lambda_L * (mu_L - hat_mu_L)
    grad_mu_L = (2 / (len(Ill))) * sum_term + reg_term

    return grad_mu_L

def compute_grad_mu_H(T, mu_L, mu_H, LLmodels, HLmodels, lambda_H, hat_mu_H):

    sum_term     = torch.zeros_like(mu_H)
    for n, iota in enumerate(Ill):
        L_i   = torch.from_numpy(LLmodels[iota].F).float()  
        V_i   = T @ L_i  
        H_i   = torch.from_numpy(HLmodels[omega[iota]].F).float()  

        sum_term = sum_term + H_i.T @ H_i @ mu_H.float() - (V_i.T @ H_i).T @ mu_L.float()
    
    reg_term     = 2 * lambda_H * (mu_H - hat_mu_H) # reg_term = -2 * lambda_H * (mu_H - hat_mu_H)
    grad_mu_H = (2 / len(Ill)) * sum_term + reg_term

    return grad_mu_H


def compute_grad_Sigma_L_half(T, Sigma_L, LLmodels, lambda_L, hat_Sigma_L):

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

def compute_grad_Sigma_H_half(T, Sigma_H, HLmodels, lambda_H, hat_Sigma_H):

    sum_term            = torch.zeros_like(Sigma_H)
    for n, iota in enumerate(Ill):
        H_i          = torch.from_numpy(HLmodels[omega[iota]].F).float()
        
        sum_term     = sum_term + H_i.T @ H_i

    Sigma_H_sqrt     = sqrtm_svd(Sigma_H)  
    hat_Sigma_H_sqrt = sqrtm_svd(hat_Sigma_H) 

    reg_term            = 2 * lambda_H * (Sigma_H_sqrt - hat_Sigma_H_sqrt) @ torch.inverse(Sigma_H_sqrt) # reg_term = -2 * lambda_H * (Sigma_H_sqrt - hat_Sigma_H_sqrt) @ torch.inverse(Sigma_H_sqrt)
    grad_Sigma_H     = (2 / (n+1)) * sum_term + reg_term

    return grad_Sigma_H

def prox_grad_Sigma_L(T, Sigma_L_half, LLmodels, Sigma_H, HLmodels, lambda_param_L):

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

    return Sigma_L_final

def prox_grad_Sigma_H(T, Sigma_H_half, LLmodels, Sigma_L, HLmodels, lambda_param_H):

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
    
    return Sigma_H_final


def optimize_min(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels,lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                                                            epsilon, delta, num_steps_min, optimizer_T, max_grad_norm, seed, xavier):

    torch.manual_seed(seed)
    if xavier:
        T = torch.nn.init.xavier_normal_(T, gain=0.01)

    cur_mu_L    = mu_L.clone()
    cur_mu_H    = mu_H.clone()
    cur_Sigma_L = Sigma_L.clone()
    cur_Sigma_H = Sigma_H.clone()

    objective_T_step = torch.tensor(0.0)  
    T_objectives_epoch = []
   
    for step in range(num_steps_min):

        objective_iota = torch.tensor(0.0)        
        for n, iota in enumerate(Ill):
            L_i = torch.from_numpy(LLmodels[iota].F).float()
            H_i = torch.from_numpy(HLmodels[omega[iota]].F).float()
            
            obj_value_iota   = compute_objective_value(T, L_i, H_i, cur_mu_L, cur_mu_H, cur_Sigma_L, cur_Sigma_H,
                                                            lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                                                            epsilon, delta)
            
            objective_iota = objective_iota + obj_value_iota

        objective_T_step = objective_iota/(n+1)
        T_objectives_epoch.append(objective_T_step)

        if torch.isnan(T).any():
            print("T contains NaN! Returning zero matrix.")
            print('Failed at step:', step+1)
            # Return zero matrix of same shape as T
            return torch.zeros_like(T), objective_T_step, T_objectives_epoch

        optimizer_T.zero_grad() 
        objective_T_step.backward(retain_graph=True)
        # Add gradient clipping before optimizer step
        if max_grad_norm < float('inf'):
            torch.nn.utils.clip_grad_norm_([T], max_grad_norm)

        optimizer_T.step()  
        #print(f"Min step {step+1}/{num_steps_min}, Objective: {objective_T_step.item()}, Gradient Norm: {torch.norm(T.grad)}")
    
    return T, objective_T_step, T_objectives_epoch


def optimize_max(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                 lambda_L, lambda_H, lambda_param_L, lambda_param_H, eta, num_steps_max, epsilon, delta, seed, project_onto_gelbrich, max_grad_norm):
    
    torch.manual_seed(seed)

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
        grad_mu_L = compute_grad_mu_L(cur_T, mu_L, mu_H, LLmodels, HLmodels, lambda_L, hat_mu_L)
        grad_mu_H = compute_grad_mu_H(cur_T, mu_L, mu_H, LLmodels, HLmodels, lambda_H, hat_mu_H)
        grad_Sigma_L = compute_grad_Sigma_L_half(cur_T, Sigma_L, LLmodels, lambda_L, hat_Sigma_L)
        grad_Sigma_H = compute_grad_Sigma_H_half(cur_T, Sigma_H, HLmodels, lambda_H, hat_Sigma_H)

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
        mu_L = mu_L + current_eta * grad_mu_L
        mu_H = mu_H + current_eta * grad_mu_H
        
        Sigma_L_half = Sigma_L + current_eta * grad_Sigma_L
        Sigma_H_half = Sigma_H + current_eta * grad_Sigma_H
        
        # Proximal updates
        Sigma_L = prox_grad_Sigma_L(cur_T, Sigma_L_half, LLmodels, Sigma_H, HLmodels, lambda_param_L)
        Sigma_H = prox_grad_Sigma_H(cur_T, Sigma_H_half, LLmodels, Sigma_L, HLmodels, lambda_param_H)
        
        if project_onto_gelbrich:
            mu_L, Sigma_L = project_onto_gelbrich_ball(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
            mu_H, Sigma_H = project_onto_gelbrich_ball(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
            
            satisfied_L, dist_L, epsi = verify_gelbrich_constraint(mu_L, Sigma_L, hat_mu_L, hat_Sigma_L, epsilon)
            satisfied_H, dist_H, delt = verify_gelbrich_constraint(mu_H, Sigma_H, hat_mu_H, hat_Sigma_H, delta)
            
            constraints_error_check(satisfied_L, dist_L, epsi, satisfied_H, dist_H, delt)

        objective_iota = 0
        for iota in Ill:
            obj_value_iota = compute_objective_value(
                                                            cur_T, L_matrices[iota], H_matrices[iota], 
                                                            mu_L, mu_H, Sigma_L, Sigma_H,
                                                            lambda_L, lambda_H, hat_mu_L, hat_mu_H, 
                                                            hat_Sigma_L, hat_Sigma_H, epsilon, delta
                                                         )
            objective_iota += obj_value_iota

        objective_theta_step = objective_iota/len(Ill)
        theta_objectives_epoch.append(objective_theta_step)

        # Early stopping check
        if step > 0 and abs(theta_objectives_epoch[-1] - theta_objectives_epoch[-2]) < 1e-5:
            max_converged = True

    return mu_L, Sigma_L, mu_H, Sigma_H, objective_theta_step, theta_objectives_epoch, max_converged

def optimize_max_proxgrad(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H, 
                lambda_L, lambda_H, lambda_param_L, lambda_param_H, eta, num_steps_max, epsilon, delta, seed):

    torch.manual_seed(seed)
    cur_T   = T.clone()

    mu_L    = mu_L.clone().detach().requires_grad_(True)
    mu_H    = mu_H.clone().detach().requires_grad_(True)
    Sigma_L = Sigma_L.clone().detach().requires_grad_(True)
    Sigma_H = Sigma_H.clone().detach().requires_grad_(True)

    max_converged = False
    
    # Create optimizers for means and covariances
    optimizer_mu    = torch.optim.Adam([mu_L, mu_H], lr=eta)
    optimizer_sigma = torch.optim.Adam([Sigma_L, Sigma_H], lr=eta)
    
    theta_objectives_epoch = []
    
    for step in range(num_steps_max):
        # Zero gradients
        optimizer_mu.zero_grad()
        optimizer_sigma.zero_grad()
        
        # Compute objective for this step
        obj_values = []
        for n, iota in enumerate(Ill):
            L_i = torch.from_numpy(LLmodels[iota].F).float()
            H_i = torch.from_numpy(HLmodels[omega[iota]].F).float()
            
            # Compute objective value
            obj_value_iota = compute_objective_value(cur_T, L_i, H_i, mu_L, mu_H, Sigma_L, Sigma_H, 
                                                            lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                                                            epsilon, delta)
            obj_values.append(obj_value_iota)
        
        # Sum objectives without in-place operations
        objective_iota = torch.stack(obj_values).sum()
        # Total objective
        objective = -(objective_iota/(n+1))
        
        # Store objective value
        theta_objectives_epoch.append(-objective.item())
        
        # Early stopping check
        if step > 0 and abs(theta_objectives_epoch[-1] - theta_objectives_epoch[-2]) < 1e-5:
            max_converged = True

        # Backward pass
        objective.backward()
        
        # Update means using optimizer
        optimizer_mu.step()
        
        # Store current Sigma values before proximal update
        Sigma_L_half = Sigma_L.detach().clone()
        Sigma_H_half = Sigma_H.detach().clone()
        
        # Update Sigma_half values using optimizer
        optimizer_sigma.step()
        
        # Manual proximal gradient steps for Sigma_L and Sigma_H
        with torch.no_grad():
            # Proximal step for Sigma_L
            Sigma_L = prox_grad_Sigma_L(cur_T, Sigma_L_half, LLmodels, Sigma_H_half, HLmodels, lambda_param_L)
            
            # Proximal step for Sigma_H
            Sigma_H = prox_grad_Sigma_H(cur_T, Sigma_H_half, LLmodels, Sigma_L, HLmodels, lambda_param_H)
            
            # Update the tensor values for next iteration
            Sigma_L.requires_grad_(True)
            Sigma_H.requires_grad_(True)

    return (mu_L.detach(), Sigma_L.detach(), mu_H.detach(), Sigma_H.detach(), objective.detach(), theta_objectives_epoch, max_converged)


def run_erica_optimization(theta_hatL, theta_hatH, initial_theta,
                     epsilon, delta, lambda_L, lambda_H, lambda_param_L, lambda_param_H, 
                     xavier, project_onto_gelbrich, eta_min, eta_max, max_iter, num_steps_min, num_steps_max, proximal_grad,
                     tol, seed, robust_L, robust_H, grad_clip, plot_steps, plot_epochs, display_results):
    
    torch.manual_seed(seed) 

    # Start timing
    start_time = time.time()

    if not robust_L:
        epsilon = 0
    if not robust_H:
        delta = 0

    erica = robust_L or robust_H

    if not erica:
        num_steps_min = 1
    
    if grad_clip:
        max_grad_norm = 1.0
    else:
        max_grad_norm = float('inf')
    
    if xavier and not project_onto_gelbrich:
        print("Forcing projection onto Gelbrich ball!")
        project_onto_gelbrich = True
    
    max_converged = False
    
    mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H = get_initialization(theta_hatL, theta_hatH, epsilon, delta, initial_theta)

    T           = torch.randn(mu_H.shape[0], mu_L.shape[0], requires_grad=True)
    optimizer_T = torch.optim.Adam([T], lr=eta_min, eps=1e-8, amsgrad=True)
    condition_num_list = []
    condition_num_list.append(evut.condition_number(T.detach().numpy()))
    previous_objective = float('inf') 
    epoch_objectives   = {'T_objectives_overall': [], 'theta_objectives_overall': []}
    for epoch in tqdm(range(max_iter)): 
        
        inner_loop_objectives = {'min_objectives': [], 'max_objectives': []}
        
        # Minimize T
        T_new, objective_T, T_objectives_epoch = optimize_min(T, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels,
                                                              lambda_L, lambda_H, hat_mu_L, hat_mu_H, hat_Sigma_L, hat_Sigma_H,
                                                            epsilon, delta, num_steps_min, optimizer_T, max_grad_norm, seed, xavier=xavier)

        #condition_num_list.append(evut.condition_number(T_new.detach().numpy()))
        inner_loop_objectives['min_objectives'] = T_objectives_epoch
        epoch_objectives['T_objectives_overall'].append(objective_T)

        # Maximize μ, Σ
        if erica and not max_converged:  
            if proximal_grad:
                mu_L, Sigma_L, mu_H, Sigma_H, obj_theta, theta_objectives_epoch, max_converged = optimize_max_proxgrad(T_new, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels,
                                                                                                        hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                                                                                                        lambda_L, lambda_H, lambda_param_L, lambda_param_H,
                                                                                                        eta_max, num_steps_max, epsilon, delta, seed
                                                                                                        )
            else:
                mu_L, Sigma_L, mu_H, Sigma_H, obj_theta, theta_objectives_epoch, max_converged = optimize_max(
                                                                                                T_new, mu_L, Sigma_L, mu_H, Sigma_H, LLmodels, HLmodels,
                                                                                                hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H,
                                                                                                lambda_L, lambda_H, lambda_param_L, lambda_param_H,
                                                                                                eta_max, num_steps_max, epsilon, delta, seed, project_onto_gelbrich, max_grad_norm
                                                                                              )
            
            if max_converged:
                num_steps_min = 1
                print(f"Max converged at epoch {epoch+1}")
            # Store objectives
            inner_loop_objectives['max_objectives'] = theta_objectives_epoch
            epoch_objectives['theta_objectives_overall'].append(obj_theta)
    
        if plot_steps:
            plot_inner_loop_objectives(inner_loop_objectives, epoch, erica)
        
        # Check convergence
        criterion = abs(previous_objective - objective_T.item()) 
        if criterion < tol:
            print(f"Convergence reached at epoch {epoch+1} with objective {objective_T.item()}")
            break
        
        
        T                  = T_new
        previous_objective = objective_T.item()

    if plot_epochs:
        plot_epoch_objectives(epoch_objectives, erica)

    paramsL      = {'mu_U': mu_L.detach().numpy(), 'Sigma_U': Sigma_L.detach().numpy(), 'radius': epsilon}
    paramsH      = {'mu_U': mu_H.detach().numpy(), 'Sigma_U': Sigma_H.detach().numpy(), 'radius': delta}
    T            = T.detach().numpy()
    end_time     = time.time()
    elapsed_time = end_time - start_time

    if display_results == True:
        print_results(T, paramsL, paramsH, elapsed_time)

    return paramsL, paramsH, T, inner_loop_objectives, epoch_objectives, condition_num_list



#======================= EMPIRICA OPTIMISATION =======================

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