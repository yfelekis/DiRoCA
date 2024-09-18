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
    
class Pair:
    
    def __init__(self, ll_model, hl_model, iota, omega):
        self.ll_model = ll_model
        self.hl_model = hl_model
        self.iota     = iota
        self.eta      = omega[iota]

class DPair:
    
    def __init__(self, ll_dist, hl_dist, iota, omega):
        self.ll_dist = ll_dist
        self.hl_dist = hl_dist
        self.iota     = iota
        self.eta      = omega[iota]
        
class Environment:
    
    def __init__(self, distribution, coefficients, num_samples):
        
        self.distribution = distribution
        self.coefficients = list(coefficients.values())
        self.variables    = list(coefficients.keys())
        self.num_samples  = num_samples
        
        if self.distribution == "gaussian":
            self.mu_vector  = np.array([self.coefficients[var][0] for var in self.variables])
            self.std_vector = np.array([self.coefficients[var][1] for var in self.variables])
            self.cov_matrix = np.diag(self.std_vector)
            
            self.noise_sample = np.random.multivariate_normal(mean=self.mu_vector, cov=self.cov_matrix, size=self.num_samples)


        elif self.distribution == "exponential":

            self.scales       = [slef.coefficients[var] for var in variables]

            self.noise_sample = np.array([np.random.exponential(scale=scale, size=num_samples) for scale in scales]).T

        elif self.distribution == 'uniform':

            self.lows  = [self.coefficients[var][0] for var in self.variables]
            self.highs = [self.coefficients[var][1] for var in self.variables]

            self.noise_sample = np.array([np.random.uniform(low=low, high=high, size=self.num_samples) for low, high in zip(self.lows, self.highs)]).T

            
#         def sample(self):

#             exogenous_coefficients = {}
#             for node in self.variables:
#                 exogenous_coefficients[node] = [sample_mean(mean_range), sample_variance(variance_range)]
#             return exogenous_coefficients