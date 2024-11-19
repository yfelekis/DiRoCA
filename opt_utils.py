
import numpy as np

import networkx as nx
import numpy as np

from scipy.linalg import sqrtm

import networkx as nx

from src.CBN import CausalBayesianNetwork as CBN
import operations as ops

def compute_gauss_barycenter(struc_matrices, mu, Sigma, max_iter=100, tol=1e-6):
    """
    Compute the barycenter of a family of multivariate Gaussian distributions.
    
    Args:
    struc_matrices (list of np.array): List of transformation matrices for every intervention.
    mu (np.array): Shared mean vector for all Gaussians.
    Sigma (np.array): Shared covariance matrix for all Gaussians.
    max_iter (int): Maximum number of iterations to compute the covariance barycenter.
    tol (float): Tolerance for convergence.
    
    Returns:
    mu_barycenter (np.array): Barycenter mean vector.
    Sigma_barycenter (np.array): Barycenter covariance matrix.
    """
    # Number of distributions
    n = len(struc_matrices)
    
    # Compute the barycenter mean
    mu_barycenter = np.sum([L @ mu for L in struc_matrices], axis=0) / n
    
    # Initialize the covariance matrix
    Sigma_barycenter = Sigma  # Start with the shared covariance matrix
    
    # Iterate to refine Sigma_barycenter
    for iteration in range(max_iter):
        Sigma_barycenter_half = sqrtm(Sigma_barycenter)  # Compute sqrt(Σ_barycenter)
        sum_term = np.zeros_like(Sigma)
        
        for i in range(n):
            # Calculate the transformed covariance L_i * Sigma_L * L_i^T
            struc_i_transformed = struc_matrices[i] @ Sigma @ struc_matrices[i].T
            # Compute the square root of the term
            term = sqrtm(Sigma_barycenter_half @ struc_i_transformed @ Sigma_barycenter_half)
            sum_term += term
        
        # Update Sigma_barycenter (take the average)
        new_Sigma_barycenter = sum_term / n
        
        # Check for convergence
        if np.linalg.norm(new_Sigma_barycenter - Sigma_barycenter) < tol:
            break
        
        # Update for the next iteration
        Sigma_barycenter = new_Sigma_barycenter
    
    return mu_barycenter, Sigma_barycenter

def sample_projection(l, h, use_stiefel=False):
    """
    Sample a matrix of shape (l, h).
    If `use_stiefel` is True, sample from the Stiefel manifold (orthonormal rows).
    Otherwise, sample a random matrix (naive projection).

    Args:
    - l: the number of rows in the matrix (source dimension)
    - h: the number of columns in the matrix (target dimension)
    - use_stiefel: whether to sample from the Stiefel manifold (True) or a random matrix (False)

    Returns:
    - A: a matrix of shape (l, h)
    """
    if use_stiefel:
        # Step 1: Sample a random matrix X of shape (l, h)
        X = np.random.randn(h, l)
        
        # Step 2: Perform QR decomposition to obtain an orthonormal matrix A
        Q, R = np.linalg.qr(X)
        
        # Step 3: Return the orthonormal matrix A
        return Q
    else:
        # Step 1: Sample a random matrix (naive projection) of shape (l, h)
        G = np.random.randn(h, l)
        
        # Step 2: Return the randomly sampled matrix A
        return  G
    
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
