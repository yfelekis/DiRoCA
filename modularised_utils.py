import numpy as np
import ot

import networkx as nx
import random
import networkx as nx
import numpy as np
import joblib

from scipy.stats import wishart
from scipy.linalg import sqrtm
from scipy.optimize import minimize

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

from scipy.stats import norm
import networkx as nx

from scipy.stats import norm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from src.CBN import CausalBayesianNetwork as CBN
import operations as ops

def sample_contexts(num_samples, ex_distribution, ex_coefficients):
        
    variables = list(ex_coefficients.keys())

    if ex_distribution == "gaussian":

        mu_vector  = np.array([ex_coefficients[var][0] for var in variables])
        std_vector = np.array([ex_coefficients[var][1] for var in variables])
        cov_matrix = np.diag(std_vector)

        noise_sample = np.random.multivariate_normal(mean=mu_vector, cov=cov_matrix, size=num_samples)

    elif ex_distribution == "exponential":

        scales       = [ex_coefficients[var] for var in variables]

        noise_sample = np.array([np.random.exponential(scale=scale, size=num_samples) for scale in scales]).T

    elif ex_distribution == 'uniform':

        lows  = [ex_coefficients[var][0] for var in variables]
        highs = [ex_coefficients[var][1] for var in variables]

        noise_sample = np.array([np.random.uniform(low=low, high=high, size=num_samples) for low, high in zip(lows, highs)]).T

    return noise_sample
    
def get_endogenous_distribution(data):
    return GaussianMixture(n_components=1).fit(data[:,:])

def get_exogenous_distribution(data):
    return GaussianMixture(n_components=1, covariance_type = 'diag').fit(data[:,:])

def sample_weights(edges, weight_range):
    """
    Assign random weights to a list of edges.

    Parameters:
    edges (list of tuples): List of edges, where each edge is a tuple of two nodes.
    weight_range (tuple): A tuple specifying the range (min_weight, max_weight) from which to sample weights.

    Returns:
    dict: A dictionary with edges as keys and random weights as values.
    """
    minw, maxw = weight_range
    weights    = {edge: round(random.uniform(minw, maxw), 9) for edge in edges}
    
    return weights


def sample_mean(mean_range):
    return random.uniform(mean_range[0], mean_range[1])

def sample_variance(variance_range):
    return random.uniform(variance_range[0], variance_range[1])

def sample_environment(dag, mean_range, variance_range):
    nodes = dag.nodes()
    exogenous_coefficients = {}
    for node in nodes:
        exogenous_coefficients[node] = [sample_mean(mean_range), sample_variance(variance_range)]
    return exogenous_coefficients

def create_pairs(Ill_relevant, omega, LLs, HLs):

    pairs = []
    for iota in Ill_relevant:
        pll, phl    = LLs[iota], HLs[omega[iota]]          
        pairs.append(ops.Pair(pll, phl, iota, omega))

    return pairs

def create_dist_pairs(Ill_relevant, omega, distLLs, distHLs):

    pairs = {}
    for iota in Ill_relevant:
        dpll, dphl    = distLLs[iota], distHLs[omega[iota]]          
        pairs[iota] = ops.Pair(dpll, dphl, iota, omega)

    return pairs


def gelbrich_dist(mu_P, Sigma_P, mu_Q, Sigma_Q):

    mean_diff    = np.linalg.norm(mu_P - mu_Q)**2
    sqrt_Sigma_P = sqrtm(Sigma_P)
    trace        = np.trace(Sigma_P + Sigma_Q - 2 * sqrtm(sqrt_Sigma_P @ Sigma_Q @ sqrt_Sigma_P))
    # gb           = np.sqrt(mean_diff + trace)
    gb           = mean_diff + trace
    
    return gb


def mechpush(dist, mechanism):
    """
    Apply a linear mechanism to a GMM by transforming its means and covariances.
    """
    diagonal_elements = dist.covariances_[0]
    dcovariances_     = np.zeros((len(diagonal_elements), len(diagonal_elements)))
    
    np.fill_diagonal(dcovariances_, diagonal_elements)

    new_means         = mechanism @ dist.means_.T
    new_covariances   = mechanism @ dcovariances_@ mechanism.T
    
    gmm               = GaussianMixture(n_components=1)
    gmm.weights_      = [1 / len(new_means)] * len(new_means) #???????
    gmm.means_        = new_means
    gmm.covariances_  = new_covariances
    
    return gmm

def ui_error_dist(error, dlcm, dhcm, L_iota, H_eta, T):
    
    ll_transformation = (L_iota @ T).T
    hl_transformation = H_eta

    f_ll = mechpush(dlcm, ll_transformation)
    f_hl = mechpush(dhcm, hl_transformation)
    
    if error == 'wass':
        d_ui = wasserstein_dist(f_ll, f_hl)
        
    elif error == 'jsd':
        d_ui = jensenshannon(f_ll, f_hl)
        
    return d_ui

def taupush(dist, mechanism):
    """
    Apply a linear mechanism to a GMM by transforming its means and covariances.
    """
    #diagonal_elements = dist.covariances_[0]
    #dcovariances_     = np.zeros((len(diagonal_elements), len(diagonal_elements)))
    
    #np.fill_diagonal(dcovariances_, diagonal_elements)

    new_means         = mechanism.T @ dist.means_.T
    new_covariances   = mechanism.T @ dist.covariances_[0]@ mechanism
    
    gmm               = GaussianMixture(n_components=1)
    gmm.weights_      = [1 / len(new_means)] * len(new_means) #???????
    gmm.means_        = new_means
    gmm.covariances_  = new_covariances
    
    return gmm

def i_error_dist(error, dlcm, dhcm, T):
    
    if error == 'wass':
        d_ui = wasserstein_dist(taupush(dlcm,T) , dhcm)
        
    elif error == 'jsd':
        d_ui = jensenshannon(taupush(dlcm,T), dhcm)
        
    return d_ui

def wasserstein_dist(P, Q):
    
    mu_P, mu_Q       = P.means_, Q.means_
    Sigma_P, Sigma_Q = P.covariances_.squeeze(), Q.covariances_.squeeze()
    mean_diff        = np.linalg.norm(mu_P - mu_Q)**2
    sqrt_Sigma_P     = sqrtm(Sigma_P)
    trace            = np.trace(Sigma_P + Sigma_Q - 2 * sqrtm(sqrt_Sigma_P @ Sigma_Q @ sqrt_Sigma_P))
    #dist             = np.sqrt(mean_diff + trace)
    dist             = mean_diff + trace
    
    return dist

def wasserstein_moments(mu1, cov1, mu2, cov2):
    
    mean_diff  = np.linalg.norm(mu1 - mu2)
    cov_sqrt   = sqrtm(np.dot(np.dot(sqrtm(cov2), cov1), sqrtm(cov2)))
    
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
        
    trace_term = np.trace(cov1 + cov2 - 2 * cov_sqrt)
    #dist      = np.sqrt(mean_diff**2 + trace_term)
    dist       = mean_diff**2 + trace_term
    
    return dist

def mat_wasserstein_distance(matrix1, matrix2):
    """
    Computes the average Wasserstein distance between corresponding rows of two matrices.

    Parameters:
    - matrix1: np.ndarray of shape (n, m), first distribution matrix
    - matrix2: np.ndarray of shape (n, m), second distribution matrix

    Returns:
    - avg_wasserstein_distance: float, average Wasserstein distance between the rows of the two matrices
    """
    # Ensure the matrices have the same shape
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape"

    # Compute the Wasserstein distance for each pair of corresponding rows
    wasserstein_distances = []
    for row1, row2 in zip(matrix1, matrix2):
        distance = wasserstein_distance(row1, row2)
        wasserstein_distances.append(distance)
    
    # Average the Wasserstein distances across all rows
    avg_wasserstein_distance = np.mean(wasserstein_distances)
    
    return avg_wasserstein_distance


def mat_ot_wasserstein_distance(matrix1, matrix2, metric='euclidean'):
    """
    Computes the Wasserstein distance between two matrices using optimal transport.

    Parameters:
    - matrix1: np.ndarray of shape (n, k), first dataset
    - matrix2: np.ndarray of shape (m, k), second dataset
    - metric: str, distance metric for computing ground cost (default: 'euclidean')

    Returns:
    - float, Wasserstein distance
    """
    # Number of samples
    n, k1 = matrix1.shape
    m, k2 = matrix2.shape
    assert k1 == k2, "Both matrices must have the same number of features"

    # Uniform weights for each point (assuming empirical distributions)
    weights1 = np.ones(n) / n
    weights2 = np.ones(m) / m

    # Compute the cost matrix (ground metric between points)
    cost_matrix = ot.dist(matrix1, matrix2, metric=metric)
    
    # Compute the Wasserstein distance
    wasserstein_dist = ot.emd2(weights1, weights2, cost_matrix)
    
    return wasserstein_dist

def mat_jsd_distance(matrix1, matrix2):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two matrices.

    Parameters:
    - matrix1: np.ndarray of shape (n, m), first distribution matrix
    - matrix2: np.ndarray of shape (n, m), second distribution matrix

    Returns:
    - jsd: float, Jensen-Shannon Divergence between the two matrices
    """
    # Normalize the matrices row-wise to represent probability distributions
    matrix1_normalized = matrix1 / np.sum(matrix1, axis=1, keepdims=True)
    matrix2_normalized = matrix2 / np.sum(matrix2, axis=1, keepdims=True)

    # Compute the Jensen-Shannon Divergence for each row (distribution)
    jsd_values = []
    for row1, row2 in zip(matrix1_normalized, matrix2_normalized):
        # Ensure no zero probabilities before calculating JSD
        row1 = np.clip(row1, 1e-10, None)
        row2 = np.clip(row2, 1e-10, None)
        
        jsd = jensenshannon(row1, row2)
        jsd_values.append(jsd)
    
    # Average the JSD across all rows
    jsd_mean = np.mean(jsd_values)
    
    return jsd_mean

def sample_covariance(Sigma_hat, epsilon, method, scale):
    """
    Sample a new diagonal covariance matrix using specified method (uniform or Wishart).

    Parameters:
    - Sigma_hat: Reference diagonal covariance matrix (must be diagonal).
    - m: Dimensionality of the covariance matrix.
    - epsilon: Perturbation range for the uniform method.
    - method: Method to use for sampling ('uniform' or 'wishart').

    Returns:
    - A diagonal covariance matrix that is positive semi-definite.
    """
    m = Sigma_hat.shape[0]
    # Ensure Sigma_hat is diagonal
    diag_sigma_hat = np.diag(np.diag(Sigma_hat))  # Keep the diagonal elements

    if method == 'uniform':
        # Sample perturbations uniformly
        perturbation = np.random.uniform(-scale, scale, size=diag_sigma_hat.shape[0])  # Sample perturbations
        new_diag = np.diag(np.clip(np.diag(diag_sigma_hat) + perturbation, 0, None))  # Ensure positivity

    elif method == 'wishart':
        # Generate a diagonal covariance perturbation using Wishart distribution
        perturbation_scale = np.linalg.inv(diag_sigma_hat)  # Scale for Wishart
        perturbation = wishart.rvs(df=m + 1, scale=perturbation_scale)  # Sample a perturbation
        new_diag = np.diag(np.clip(np.diag(perturbation), 0, None))  # Take diagonal and ensure positivity

    return new_diag

def sample_meanvec(mu_hat, scale, mu_method):
    
    if mu_method == 'perturbation':
        m            = len(mu_hat)
        perturbation = np.random.randn(m) * scale 
        new_mu       = mu_hat + perturbation
    else:
        radius = np.random.uniform(-scale, scale)  
        direction = np.random.randn(m)
        direction /= np.linalg.norm(direction)
        mu = mu_hat + radius * direction  
        
    return new_mu

"""
The Wishart distribution is used to generate random positive semi-definite matrices, 
which makes it a natural choice for sampling covariance matrices. 
Since covariance matrices must be symmetric and positive semi-definite 
(i.e., they have non-negative eigenvalues), the Wishart distribution is 
the go-to method for sampling such matrices.
"""
def sample_moments_U(mu_hat, Sigma_hat, bound, mu_method = 'perturbation', Sigma_method = 'uniform', 
                     mu_scale = 0.1, Sigma_scale = 0.2, num_envs = 1, max_attempts = 1000, dag = None):
    
    samples = [] # To store multiple samples
    
    while len(samples) < num_envs:
        for _ in range(max_attempts):
            
            mu    = sample_meanvec(mu_hat, mu_scale, mu_method)
            Sigma = sample_covariance(Sigma_hat, bound, Sigma_method, Sigma_scale)
                               
            # 3. Check if the new mean and covariance satisfy the Wasserstein distance constraint
            if wasserstein_moments(mu_hat, Sigma_hat, mu, Sigma) <= bound**2:
                samples.append((mu, Sigma))
                #environments.append(Environment('gaussian', (mu, Sigma), dag))
                break

        if len(samples) == num_envs:
            return samples
    
    raise ValueError(f"Failed to find {num_envs} valid samples within the specified Wasserstein distance.")

def sample_distros_Gelbrich(samples):
    
    distributions = []
    
    for mu, Sigma in samples:
        
        Sigma = Sigma.reshape(1, -1)  
        
        gmm   = GaussianMixture(n_components=1)
        
        # Set the means and covariances
        gmm.means_       = mu.reshape(1, -1)  # Reshape to (1, m)
        gmm.covariances_ = Sigma.reshape(1, mu.size, mu.size)  # Reshape to (1, m, m)
        gmm.weights_     = np.array([1.0])  # Set the weight of the component
        
        distributions.append(gmm)

    return distributions

def sample_stoch_matrix(n, m, axis=0):
    # Generate random n x m matrix
    matrix = np.random.rand(n, m)
    
    if axis == 0:  # Stochastic along rows
        # Normalize each row to sum to 1
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    elif axis == 1:  # Stochastic along columns
        # Normalize each column to sum to 1
        matrix = matrix / matrix.sum(axis=0, keepdims=True)
    
    return matrix

def generate_perturbed_datasets(D, bound, num_envs=1, p=2):
    """
    Generates multiple perturbed datasets with perturbation constraint.
    
    Parameters:
    - D: np.ndarray of shape (n, k), original dataset where n is the number of samples, k is the number of variables.
    - bound: float, the constraint parameter.
    - num: int, number of perturbed datasets to generate (default is 1).
    - p: int, norm degree (default is 2 for Euclidean norm).
    
    Returns:
    - D_perturbed_list: list of np.ndarray, each of shape (n, k), perturbed datasets.
    - Theta_list: list of np.ndarray, each of shape (n, k), perturbation matrices.
    """
    n, k = D.shape
    D_perturbed_list = []
    Theta_list = []
    
    for _ in range(num_envs):
        # Generate random perturbation matrix Theta
        Theta = np.random.randn(n, k)

        # Compute the norm of each row of Theta
        norm_theta_p = np.linalg.norm(Theta, ord=p, axis=1)

        # Calculate the current average of ||theta_i||^p
        current_avg_norm_p = (1 / n) * np.sum(norm_theta_p ** p)
        
        # Scale Theta to satisfy the constraint
        scaling_factor = (bound / current_avg_norm_p) ** (1 / p)
        Theta *= scaling_factor

        # Apply perturbation to the original dataset
        D_perturbed = D + Theta
        
        # Store the perturbed dataset and Theta matrix
        D_perturbed_list.append(D_perturbed)
        Theta_list.append(Theta)
    
    return D_perturbed_list #, Theta_list


# ######################## KEEP THIS! ########################
# def get_coefficients(data, G):
#     """
#     Estimates the structural coefficients for the edges in the causal model.
    
#     Args:
#     - data (ndarray): n x k dataset where n is the number of samples and k is the number of variables.
#     - G (DiGraph)   : The underlying DAG of the causal model.
    
#     Returns:
#     - coeffs (dict) : Dictionary of estimated structural coefficients for each edge.
#     """
#     nodes  = list(G.nodes)
#     coeffs = {}
    
#     for node in nx.topological_sort(G):
#         node_idx = nodes.index(node)
#         parents  = list(G.predecessors(node))
        
#         if parents:
#             parent_indices = [nodes.index(p) for p in parents]
            
#             # Perform linear regression of the node on its parents
#             X_parents = data[:, parent_indices]
#             y         = data[:, node_idx]
#             reg       = LinearRegression().fit(X_parents, y)
            
#             for p, coef in zip(parents, reg.coef_):
#                 coeffs[(p, node)] = coef  # Store as (parent, child) pair

#     return coeffs
# ######################## KEEP THIS! ########################


def get_coefficients(data_list, G, weights=None):
    """
    Estimates the structural coefficients for the edges in the causal model
    using both observational and interventional datasets.
    
    Args:
    - data_list (list of ndarray): List of n x k datasets where n is the number of samples and k is the number of variables.
    - G (DiGraph): The underlying DAG of the causal model.
    - weights (list of float): List of weights corresponding to each dataset (optional).
    
    Returns:
    - coeffs (dict): Dictionary of estimated structural coefficients for each edge.
    """
    nodes = list(G.nodes)
    coeffs = {}

    # Check if only a single dataset is provided
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]  # Convert to a list for consistency

    # Default weights to 1 if not provided
    if weights is None:
        weights = [1] * len(data_list)

    # Initialize an empty list to hold concatenated datasets
    combined_data = []
    sample_weights = []

    # Combine datasets and create sample weights
    for data, weight in zip(data_list, weights):
        num_samples = data.shape[0]  # Number of samples in the dataset
        combined_data.append(data)  # Append the current dataset
        sample_weights.extend([weight] * num_samples)  # Extend weights list

    # Concatenate all datasets into a single array
    combined_data = np.vstack(combined_data)
    sample_weights = np.array(sample_weights)  # Convert to numpy array

    # Ensure that sample_weights matches the number of samples
    if sample_weights.shape[0] != combined_data.shape[0]:
        raise ValueError("Sample weights shape does not match combined data shape!")

    for node in nx.topological_sort(G):
        node_idx = nodes.index(node)
        parents = list(G.predecessors(node))
        
        if parents:
            parent_indices = [nodes.index(p) for p in parents]
            X_parents = combined_data[:, parent_indices]  # Combined features
            y = combined_data[:, node_idx]  # Combined target variable
            
            # Perform weighted linear regression
            reg = LinearRegression().fit(X_parents, y, sample_weight=sample_weights)
            
            for p, coef in zip(parents, reg.coef_):
                coeffs[(p, node)] = coef  # Store as (parent, child) pair

    return coeffs


def lan_abduction(data, G, coeffs):
    """
    Computes the exogenous variables for a linear additive Gaussian noise SCM using the coefficients.
    
    Args:
    - data (ndarray)  : n x k dataset where n is the number of samples and k is the number of variables.
    - G (DiGraph)     : The underlying DAG of the causal model.
    - coeffs (dict)   : Dictionary of estimated structural coefficients for each edge.
    
    Returns:
    - U (ndarray)     : n x k dataset of the exogenous variables (residuals).
    - mean_U (ndarray): Mean vector of the exogenous variables.
    - cov_U (ndarray) : Covariance matrix of the exogenous variables.
    """
    
    # Initialize residual matrix (exogenous variables)
    n_samples, n_vars = data.shape
    U                 = np.zeros((n_samples, n_vars))
    nodes             = list(G.nodes)
    
    for node in nx.topological_sort(G):
        node_idx = nodes.index(node)
        parents  = list(G.predecessors(node))
        
        if not parents:
            # If there are no parents, this is an exogenous node
            U[:, node_idx] = data[:, node_idx]  # U_node = X_node (no parents)
        else:
            X_parents = np.zeros((n_samples, len(parents)))
            for i, p in enumerate(parents):
                # Fill the values based on the coefficients
                X_parents[:, i] = data[:, nodes.index(p)] * coeffs[(p, node)]
                
            # Calculate the residuals (exogenous noise)
            y = data[:, node_idx]
            U[:, node_idx] = y - X_parents.sum(axis=1)  # Residuals are the exogenous noise

    # Estimate the mean and covariance of the exogenous variables
    mean_U = np.mean(U, axis=0)
    cov_U  = np.cov(U, rowvar=False)
    
    return U, mean_U, cov_U

def weighted_likelihood(params, X_parents, y, weights, beta=1.5, sigma=1.0):
    """
    Custom weighted likelihood function for MLE with Generalized Normal noise.
    
    Args:
    - params (ndarray): Coefficients for the linear model.
    - X_parents (ndarray): Parent data matrix.
    - y (ndarray): Target data.
    - weights (ndarray): Weights for each sample.
    - beta (float): Shape parameter for the Generalized Normal distribution (controls tails).
    - sigma (float): Scale parameter for the distribution (assumed fixed or can be estimated separately).
    
    Returns:
    - weighted_likelihood (float): Weighted negative log-likelihood (to be minimized).
    """
    predicted = np.dot(X_parents, params)  # Linear relationship
    residuals = np.abs(y - predicted)      # Residuals (errors)

    # Generalized normal loss (L_beta loss)
    loss = np.sum(weights * (residuals / sigma) ** beta)  # Weighted Generalized Normal loss
    
    return loss

def weighted_likelihood(params, X_parents, y, weights, beta=1.5, sigma=1.0):
    """
    Custom weighted likelihood function for MLE with Generalized Normal noise.
    
    Args:
    - params (ndarray): Coefficients for the linear model.
    - X_parents (ndarray): Parent data matrix.
    - y (ndarray): Target data.
    - weights (ndarray): Weights for each sample.
    - beta (float): Shape parameter for the Generalized Normal distribution (controls tails).
    - sigma (float): Scale parameter for the distribution (assumed fixed or can be estimated separately).
    
    Returns:
    - weighted_likelihood (float): Weighted negative log-likelihood (to be minimized).
    """
    predicted = np.dot(X_parents, params)  # Linear relationship
    residuals = np.abs(y - predicted)      # Residuals (errors)

    # Generalized normal loss (L_beta loss)
    loss = np.sum(weights * (residuals / sigma) ** beta)  # Weighted Generalized Normal loss
    
    return loss

def get_mle_coefficients(data_list, G, weights=None, beta=1.5, sigma=1.0):
    """
    Estimates the structural coefficients for the edges in the causal model using MLE with Generalized Normal noise.
    
    Args:
    - data_list (list of ndarray): List of n x k datasets where n is the number of samples and k is the number of variables.
    - G (DiGraph): The underlying DAG of the causal model.
    - weights (list of float): List of weights corresponding to each dataset (optional).
    - beta (float): Shape parameter for the Generalized Normal distribution (default=1.5).
    - sigma (float): Scale parameter for the distribution (optional, default=1.0).
    
    Returns:
    - coeffs (dict): Dictionary of estimated structural coefficients for each edge.
    """
    nodes = list(G.nodes)
    coeffs = {}

    if isinstance(data_list, np.ndarray):
        data_list = [data_list]

    if weights is None:
        weights = [1] * len(data_list)

    combined_data = []
    sample_weights = []

    for data, weight in zip(data_list, weights):
        num_samples = data.shape[0]
        combined_data.append(data)
        sample_weights.extend([weight] * num_samples)

    combined_data = np.vstack(combined_data)
    sample_weights = np.array(sample_weights)

    if sample_weights.shape[0] != combined_data.shape[0]:
        raise ValueError("Sample weights shape does not match combined data shape!")

    for node in nx.topological_sort(G):
        node_idx = nodes.index(node)
        parents = list(G.predecessors(node))
        
        if parents:
            parent_indices = [nodes.index(p) for p in parents]
            X_parents = combined_data[:, parent_indices]
            y = combined_data[:, node_idx]
            
            init_params = np.zeros(len(parents))  # Initial guess for coefficients
            
            # Optimize the likelihood using Generalized Normal noise
            result = minimize(weighted_likelihood, init_params, args=(X_parents, y, sample_weights, beta, sigma), method='Powell')
            
            for p, coef in zip(parents, result.x):
                    coeffs[(p, node)] = coef  # Store as (parent, child) pair
        
    return coeffs


def weighted_huber_loss(params, X_parents, y, weights, delta=1.0):
    """
    Custom weighted likelihood function for MLE using Huber loss to handle noise uncertainty.
    
    Args:
    - params (ndarray): Coefficients for the linear model.
    - X_parents (ndarray): Parent data matrix.
    - y (ndarray): Target data.
    - weights (ndarray): Weights for each sample.
    - delta (float): Huber loss parameter; defines the threshold between L1 and L2 losses.
    
    Returns:
    - loss (float): Weighted Huber loss.
    """
    predicted = np.dot(X_parents, params)  # Linear relationship
    residuals = y - predicted               # Residuals (errors)
    
    # Huber loss: Quadratic for small residuals, linear for large residuals
    loss = np.where(np.abs(residuals) <= delta,
                    0.5 * residuals**2,  # L2 for small residuals
                    delta * (np.abs(residuals) - 0.5 * delta))  # L1 for large residuals
    
    return np.sum(weights * loss)  # Weighted Huber loss

def get_mle_coefficients_huber(data_list, G, weights=None, delta=1.0):
    """
    Estimates the structural coefficients using MLE with Huber loss, robust to unknown noise.
    
    Args:
    - data_list (list of ndarray): List of datasets.
    - G (DiGraph): The underlying DAG of the causal model.
    - weights (list of float): Weights corresponding to each dataset.
    - delta (float): Huber loss parameter; defines the threshold between L1 and L2 losses.
    
    Returns:
    - coeffs (dict): Dictionary of estimated structural coefficients for each edge.
    """
    nodes = list(G.nodes)
    coeffs = {}

    # Ensure data is a list of datasets
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]

    # Default weights to 1 if not provided
    if weights is None:
        weights = [1] * len(data_list)

    # Combine datasets and create sample weights
    combined_data = []
    sample_weights = []
    for data, weight in zip(data_list, weights):
        num_samples = data.shape[0]
        combined_data.append(data)
        sample_weights.extend([weight] * num_samples)

    combined_data = np.vstack(combined_data)
    sample_weights = np.array(sample_weights)

    for node in nx.topological_sort(G):
        node_idx = nodes.index(node)
        parents = list(G.predecessors(node))
        
        if parents:
            parent_indices = [nodes.index(p) for p in parents]
            X_parents = combined_data[:, parent_indices]
            y = combined_data[:, node_idx]
            
            init_params = np.zeros(len(parents))  # Initial guess for coefficients
            
            # Optimize using Huber loss
            result = minimize(weighted_huber_loss, init_params, args=(X_parents, y, sample_weights, delta), method='Powell')
            
            for p, coef in zip(parents, result.x):
                coeffs[(p, node)] = coef  # Store as (parent, child) pair

    return coeffs


def weighted_gmm_likelihood(params, X_parents, y, weights, n_components=2):
    """
    Custom likelihood function using a Gaussian Mixture Model for the residuals.
    
    Args:
    - params (ndarray): Coefficients for the linear model.
    - X_parents (ndarray): Parent data matrix.
    - y (ndarray): Target data.
    - weights (ndarray): Weights for each sample.
    - n_components (int): Number of components in the Gaussian Mixture Model.
    
    Returns:
    - loss (float): Negative log-likelihood from GMM.
    """
    predicted = np.dot(X_parents, params)  # Linear relationship
    residuals = y - predicted               # Residuals (errors)

    # Fit a Gaussian Mixture Model to the residuals
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(residuals.reshape(-1, 1))  # Fit GMM on residuals
    
    # Get the log-likelihood of the residuals under the GMM
    log_likelihood = gmm.score_samples(residuals.reshape(-1, 1))
    
    # We minimize the negative log-likelihood weighted by the sample weights
    return -np.sum(weights * log_likelihood)

def get_mle_coefficients_gmm(data_list, G, weights=None, n_components=2):
    """
    Estimates the structural coefficients using MLE with Gaussian Mixture Model for residuals.
    
    Args:
    - data_list (list of ndarray): List of datasets.
    - G (DiGraph): The underlying DAG of the causal model.
    - weights (list of float): Weights corresponding to each dataset.
    - n_components (int): Number of components in the Gaussian Mixture Model.
    
    Returns:
    - coeffs (dict): Dictionary of estimated structural coefficients for each edge.
    """
    nodes = list(G.nodes)
    coeffs = {}

    # Ensure data is a list of datasets
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]

    # Default weights to 1 if not provided
    if weights is None:
        weights = [1] * len(data_list)

    # Combine datasets and create sample weights
    combined_data = []
    sample_weights = []
    for data, weight in zip(data_list, weights):
        num_samples = data.shape[0]
        combined_data.append(data)
        sample_weights.extend([weight] * num_samples)

    combined_data = np.vstack(combined_data)
    sample_weights = np.array(sample_weights)

    for node in nx.topological_sort(G):
        node_idx = nodes.index(node)
        parents = list(G.predecessors(node))
        
        if parents:
            parent_indices = [nodes.index(p) for p in parents]
            X_parents = combined_data[:, parent_indices]
            y = combined_data[:, node_idx]
            
            init_params = np.zeros(len(parents))  # Initial guess for coefficients
            
            # Optimize using Gaussian Mixture Model for residuals
            result = minimize(weighted_gmm_likelihood, init_params, args=(X_parents, y, sample_weights, n_components), method='Powell')
            
            for p, coef in zip(parents, result.x):
                coeffs[(p, node)] = coef  # Store as (parent, child) pair

    return coeffs

# def weighted_likelihood(params, X_parents, y, weights):
#     """
#     Custom weighted likelihood function for MLE with non-Gaussian noise.
    
#     Args:
#     - params (ndarray): Coefficients for the linear model.
#     - X_parents (ndarray): Parent data matrix.
#     - y (ndarray): Target data.
#     - weights (ndarray): Weights for each sample.
    
#     Returns:
#     - weighted_likelihood (float): Weighted negative log-likelihood (to be minimized).
#     """
#     predicted = np.dot(X_parents, params)  # Linear relationship
#     residuals = y - predicted               # Residuals (errors)

#     # Use L1 loss for robust estimation, weighted by the sample weights
#     return np.sum(weights * np.abs(residuals))  # Weighted L1 loss

# def get_mle_coefficients(data_list, G, weights=None):
#     """
#     Estimates the structural coefficients for the edges in the causal model using MLE and
#     multiple datasets (observational and/or interventional).
    
#     Args:
#     - data_list (list of ndarray): List of n x k datasets where n is the number of samples and k is the number of variables.
#     - G (DiGraph): The underlying DAG of the causal model.
#     - weights (list of float): List of weights corresponding to each dataset (optional).
    
#     Returns:
#     - coeffs (dict): Dictionary of estimated structural coefficients for each edge.
#     """
#     nodes = list(G.nodes)
#     coeffs = {}

#     # Check if only a single dataset is provided
#     if isinstance(data_list, np.ndarray):
#         data_list = [data_list]  # Convert to a list for consistency

#     # Default weights to 1 if not provided
#     if weights is None:
#         weights = [1] * len(data_list)

#     # Initialize an empty list to hold concatenated datasets
#     combined_data = []
#     sample_weights = []

#     # Combine datasets and create sample weights
#     for data, weight in zip(data_list, weights):
#         num_samples = data.shape[0]  # Number of samples in the dataset
#         combined_data.append(data)  # Append the current dataset
#         sample_weights.extend([weight] * num_samples)  # Extend weights list

#     # Concatenate all datasets into a single array
#     combined_data = np.vstack(combined_data)
#     sample_weights = np.array(sample_weights)  # Convert to numpy array

#     # Ensure that sample_weights matches the number of samples
#     if sample_weights.shape[0] != combined_data.shape[0]:
#         raise ValueError("Sample weights shape does not match combined data shape!")

#     for node in nx.topological_sort(G):
#         node_idx = nodes.index(node)
#         parents = list(G.predecessors(node))
        
#         if parents:
#             parent_indices = [nodes.index(p) for p in parents]
#             X_parents = combined_data[:, parent_indices]  # Combined features
#             y = combined_data[:, node_idx]  # Combined target variable
            
#             # Use MLE to estimate coefficients
#             init_params = np.zeros(len(parents))  # Initial guess for coefficients
            
#             # Optimize the negative likelihood (maximize likelihood by minimizing -logL)
#             result = minimize(weighted_likelihood, init_params, args=(X_parents, y, sample_weights), method='BFGS')
            
#             for p, coef in zip(parents, result.x):
#                     coeffs[(p, node)] = coef  # Store as (parent, child) pair
        
#     return coeffs

######################## LOADERS ########################

def load_ll_model(experiment):
    return joblib.load(f'data/{experiment}/LL.pkl')

def load_hl_model(experiment):
    return joblib.load(f'data/{experiment}/HL.pkl')

def load_omega_map(experiment):
    return joblib.load(f'data/{experiment}/omega.pkl')

def load_pairs(experiment):
    return joblib.load(f'data/{experiment}/pairs.pkl')

def load_samples(experiment):
    return joblib.load(f'data/{experiment}/Ds.pkl')
# def gauss_lan_abduction(mu_X, Sigma_X, A):
    
#     mu_X     = np.asarray(mu_X)
#     Sigma_X  = np.asarray(Sigma_X)
#     #dSigma_X = np.zeros((len(Sigma_X), len(Sigma_X)))
    
#     #np.fill_diagonal(dSigma_X, Sigma_X)
#     size    = len(mu_X)    
    
#     A = np.asarray(A)
    
#     if not np.allclose(A, np.triu(A)):
#         raise ValueError("Matrix A must be upper triangular.")
    
#     I = np.eye(A.shape[0])
#     K = I - A
   
#     mu_E    = K @ mu_X.T
#     Sigma_E = K @ Sigma_X @ K.T 

#     return mu_E, Sigma_E

# def sample_distros_Gelbrich(sampled_means, sampled_covariances):
    
#     distributions = []
    
#     for mu, Sigma in zip(sampled_means, sampled_covariances):
        
#         Sigma = Sigma.reshape(1, -1)  
        
#         gmm   = GaussianMixture(n_components=1)
        
#         # Set the means and covariances
#         gmm.means_       = mu.reshape(1, -1)  # Reshape to (1, m)
#         gmm.covariances_ = Sigma.reshape(1, mu.size, mu.size)  # Reshape to (1, m, m)
#         gmm.weights_     = np.array([1.0])  # Set the weight of the component
        
#         distributions.append(gmm)

#     return distributions