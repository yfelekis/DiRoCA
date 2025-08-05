import numpy as np
from scipy.stats import laplace
from scipy.stats import gennorm

class Intervention:
    
    def __init__(self, intervention):
            
        self.intervention = intervention
        
    def phi(self):
        return list(self.intervention.values())
        
    def Phi(self):
        return list(self.intervention.keys())
    
    def vv(self):
        return self.intervention
    
    def __eq__(self, other):
        if isinstance(other, Intervention):
            return self.intervention == other.intervention
        return False

    def __hash__(self):
        return hash(frozenset(self.intervention.items()))

class MultivariateLaplace:
    def __init__(self, loc_vec, scale_vec):
        """
        Initialize a multivariate Laplace distribution.
        
        Args:
        - loc_vec (array-like): Mean (location) vector for the Laplacian distribution.
        - scale_vec (array-like): Scale vector for the Laplacian distribution.
        """
        self.loc = np.array(loc_vec)    # Location (mean) vector
        self.scale = np.array(scale_vec)  # Scale vector
        self.dim = len(loc_vec)         # Dimension of the distribution
    
    def sample(self, n):
        """
        Sample from the multivariate Laplace distribution.
        
        Args:
        - n (int): Number of samples.
        
        Returns:
        - samples (ndarray): n x k dataset where k is the number of dimensions.
        """
        # Generate samples for each dimension and stack them
        samples = np.array([laplace(loc=loc, scale=scale).rvs(size=n) for loc, scale in zip(self.loc, self.scale)]).T
        return samples


class MultivariateGeneralizedNormal:
    def __init__(self, loc_vec, scale_vec, shape_vec):
        """
        Initialize a multivariate Generalized Normal distribution.
        
        Args:
        - loc_vec (array-like): Mean (location) vector for the Generalized Normal distribution.
        - scale_vec (array-like): Scale vector for the Generalized Normal distribution.
        - shape_vec (array-like): Shape vector for the Generalized Normal distribution (beta).
        """
        self.loc = np.array(loc_vec)        # Location (mean) vector
        self.scale = np.array(scale_vec)    # Scale vector
        self.shape = np.array(shape_vec)    # Shape vector
        self.dim = len(loc_vec)             # Dimension of the distribution
    
    def sample(self, n):
        """
        Sample from the multivariate Generalized Normal distribution.
        
        Args:
        - n (int): Number of samples.
        
        Returns:
        - samples (ndarray): n x k dataset where k is the number of dimensions.
        """
        # Generate samples for each dimension and stack them
        samples = np.array([gennorm(beta=shape, loc=loc, scale=scale).rvs(size=n)
                            for loc, scale, shape in zip(self.loc, self.scale, self.shape)]).T
        return samples
    


class MatrixDistances:
    @staticmethod
    def frobenius_distance(A, B):
        """Frobenius norm (Euclidean norm of matrices)"""
        diff = A - B
        return np.sqrt(np.sum(diff * diff))
    
    @staticmethod
    def squared_frobenius_distance(A, B):
        """Squared Frobenius norm"""
        diff = A - B
        return np.sum(diff * diff)
    
    @staticmethod
    def nuclear_norm_distance(A, B):
        """Nuclear norm (sum of singular values)"""
        diff = A - B
        return np.linalg.norm(diff, ord='nuc')
    
    @staticmethod
    def spectral_norm_distance(A, B):
        """Spectral norm (largest singular value)"""
        diff = A - B
        return np.linalg.norm(diff, ord=2)
    
    @staticmethod
    def l1_distance(A, B):
        """Manhattan distance (sum of absolute differences)"""
        return np.sum(np.abs(A - B))
    
    @staticmethod
    def l2_distance(A, B):
        """
        Compute L2 (Euclidean) distance between matrices A and B.
        """
        diff = A - B
        return np.sqrt(np.sum(diff**2))
    