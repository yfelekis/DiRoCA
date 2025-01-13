
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from src.CBN import CausalBayesianNetwork as CBN
import operations as ops
import modularised_utils as mut
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
    print(f"\nCondition Number: {condition_number(T):.4f}")
    
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


def condition_number(matrix):
    """
    Computes the condition number of a matrix using the 2-norm.

    Parameters:
        matrix (np.ndarray): Input matrix (can be square or rectangular).

    Returns:
        float: The condition number of the matrix.
    """
    # Compute the singular values of the matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    # Condition number is the ratio of the largest to smallest singular value
    cond_number = singular_values.max() / singular_values.min()

    return cond_number

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

    reg_L    = lambda_L * (epsilon**2 - torch.norm(mu_L - hat_mu_L) ** 2 - torch.norm(Sigma_L_sqrt - hat_Sigma_L_sqrt, p='fro') ** 2)
    reg_H    = lambda_H * (delta**2 - torch.norm(mu_H - hat_mu_H) ** 2 - torch.norm(Sigma_H_sqrt - hat_Sigma_H_sqrt, p='fro') ** 2)

    # Total value
    val = term1 + term2 + term3 + term4 + reg_L + reg_H

    return val

def get_initialization(theta_hatL, theta_hatH, epsilon, delta, initial_theta):
    hat_mu_L, hat_Sigma_L = torch.from_numpy(theta_hatL['mu_U']).float(), torch.from_numpy(theta_hatL['Sigma_U']).float()
    hat_mu_H, hat_Sigma_H = torch.from_numpy(theta_hatH['mu_U']).float(), torch.from_numpy(theta_hatH['Sigma_U']).float()

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

    return mu_L, Sigma_L, mu_H, Sigma_H, hat_mu_L, hat_Sigma_L, hat_mu_H, hat_Sigma_H
