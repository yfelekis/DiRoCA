import numpy as np
from scipy.linalg import sqrtm

def compute_wasserstein(mu1, cov1, mu2, cov2):
    """
    Compute the 2-Wasserstein distance between two multivariate Gaussian distributions.
    
    Args:
        mu1 (np.array): Mean vector of the first distribution
        cov1 (np.array): Covariance matrix of the first distribution
        mu2 (np.array): Mean vector of the second distribution
        cov2 (np.array): Covariance matrix of the second distribution
        
    Returns:
        float: The 2-Wasserstein distance between the distributions
    """
    # Compute mean term
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Compute covariance term
    cov_term = np.trace(cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1)))
    
    # Return Wasserstein distance
    return mean_diff + cov_term 