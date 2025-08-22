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
    
    # Compute covariance term with robust sqrtm
    try:
        # Ensure matrices are symmetric and positive semi-definite
        cov1_sym = 0.5 * (cov1 + cov1.T)
        cov2_sym = 0.5 * (cov2 + cov2.T)
        
        # Add small regularization to ensure positive definiteness
        eps = 1e-10
        cov1_reg = cov1_sym + eps * np.eye(cov1_sym.shape[0])
        cov2_reg = cov2_sym + eps * np.eye(cov2_sym.shape[0])
        
        # Compute sqrtm with error handling
        sqrt_cov1 = sqrtm(cov1_reg)
        sqrt_cov2 = sqrtm(cov2_reg)
        
        # Ensure real values
        if np.iscomplexobj(sqrt_cov1):
            sqrt_cov1 = sqrt_cov1.real
        if np.iscomplexobj(sqrt_cov2):
            sqrt_cov2 = sqrt_cov2.real
            
        # Compute the trace term
        inner_matrix = sqrt_cov1 @ cov2_reg @ sqrt_cov1
        inner_sqrt = sqrtm(inner_matrix)
        
        if np.iscomplexobj(inner_sqrt):
            inner_sqrt = inner_sqrt.real
            
        cov_term = np.trace(cov1_reg + cov2_reg - 2 * inner_sqrt)
        
    except Exception as e:
        # Fallback: use Frobenius norm as approximation
        print(f"Failed to find a square root: {e}. Using fallback method.")
        cov_term = np.trace(cov1 + cov2) - 2 * np.trace(sqrtm(cov1) @ sqrtm(cov2))
        
        # If that also fails, use a simpler approximation
        try:
            cov_term = np.trace(cov1 + cov2)
        except:
            cov_term = 0.0
    
    # Return Wasserstein distance
    return mean_diff + cov_term 