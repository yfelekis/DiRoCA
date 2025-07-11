import numpy as np
import ot

import networkx as nx
import random
import networkx as nx
import numpy as np
import joblib
import torch

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

from sklearn.linear_model import LinearRegression, Ridge
import scipy.stats as stats

from src.CBN import CausalBayesianNetwork as CBN
import operations as ops
import opt_tools as oput 
from math_utils import compute_wasserstein


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
    gmm.weights_      = [1 / len(new_means)] * len(new_means)
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

    new_means         = mechanism.T @ dist.means_.T
    new_covariances   = mechanism.T @ dist.covariances_[0]@ mechanism
    
    gmm               = GaussianMixture(n_components=1)
    gmm.weights_      = [1 / len(new_means)] * len(new_means) 
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
    
    dist             = mean_diff + trace
    
    return dist

def compute_wasserstein(mu1, cov1, mu2, cov2):
    
    mean_diff  = np.linalg.norm(mu1 - mu2)
    cov_sqrt   = sqrtm(np.dot(np.dot(sqrtm(cov2), cov1), sqrtm(cov2)))
    
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
        
    trace_term = np.trace(cov1 + cov2 - 2 * cov_sqrt)
    dist       = mean_diff**2 + trace_term
    
    return dist

def compute_gaussian_kl(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two multivariate Gaussians:
    
    Formula:
    0.5 * (tr(sigma2^(-1) @ sigma1) + (mu2-mu1)^T @ sigma2^(-1) @ (mu2-mu1) - k + ln(det(sigma2)/det(sigma1)))
    where k is the dimension of the distributions
    """
    k = len(mu1)
    
    # Compute inverse of sigma2
    sigma2_inv = np.linalg.inv(sigma2)
    
    # Compute difference in means
    mu_diff = mu2 - mu1
    
    # Compute terms
    term1 = np.trace(sigma2_inv @ sigma1)
    term2 = mu_diff.T @ sigma2_inv @ mu_diff
    term3 = k
    term4 = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    
    kl = 0.5 * (term1 + term2 - term3 + term4)
    return kl

def compute_jensenshannon(samples1, samples2, bins=100):
    """
    Compute Jensen-Shannon divergence between two sets of samples
    
    Args:
        samples1: (n_samples x n_vars) array of samples from first distribution
        samples2: (n_samples x n_vars) array of samples from second distribution
        bins: number of bins for histogram
    """
    # For each dimension
    js_div = 0
    n_dims = samples1.shape[1]
    
    for dim in range(n_dims):
        # Compute histograms for this dimension
        min_val = min(samples1[:, dim].min(), samples2[:, dim].min())
        max_val = max(samples1[:, dim].max(), samples2[:, dim].max())
        
        hist1, bin_edges = np.histogram(samples1[:, dim], bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(samples2[:, dim], bins=bins, range=(min_val, max_val), density=True)
        
        # Convert to probabilities
        p = hist1 / hist1.sum()
        q = hist2 / hist2.sum()
        
        # Add small constant to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # Normalize again
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute JS divergence for this dimension
        js_div += jensenshannon(p, q)
    
    # Return average JS divergence across dimensions
    return js_div / n_dims


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


def noise_generation(center, radius, sample_form, level, hat_dict, worst_dict, coverage, normalize):
    #np.random.seed(42) 
    
    if center == 'hat':
        mu = hat_dict[level][0]
        Sigma = hat_dict[level][1]
    elif center == 'worst':
        mu = worst_dict[level][0]
        Sigma = worst_dict[level][1]
    else:
        raise ValueError(f"Unknown center: {center}")
    n_vars = mu.shape[0]
    
    if sample_form == 'boundary':
        random_mu    = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars)) 

        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu, random_Sigma, mu, Sigma, radius)
        
    elif sample_form == 'sample':
        noise_mu, noise_Sigma = sample_moments_U(mu, Sigma, bound=radius, coverage=coverage)

    else:
        raise ValueError(f"Unknown sample form: {sample_form}")
    
    if normalize:
        if torch.is_tensor(noise_mu):
            noise_mu = noise_mu.numpy()
        if torch.is_tensor(noise_Sigma):
            noise_Sigma = noise_Sigma.numpy()
            
        noise_mu = noise_mu / np.std(noise_mu)
        noise_Sigma = noise_Sigma / np.std(noise_Sigma)

    if torch.is_tensor(noise_mu):
        noise_mu = noise_mu.numpy()
    if torch.is_tensor(noise_Sigma):
        noise_Sigma = noise_Sigma.numpy()

    return noise_mu, noise_Sigma

def sample_moments_U(mu_hat, Sigma_hat, bound, mu_method='perturbation', Sigma_method='uniform', 
                    mu_scale=0.1, Sigma_scale=0.2, coverage='rand', seed=None):
    """
    Sample a single pair of moments (mu, Sigma) with option for uniform coverage in Wasserstein ball.
    
    Args:
        mu_hat: Center mean vector
        Sigma_hat: Center covariance matrix
        bound: Radius of Wasserstein ball
        mu_scale, Sigma_scale: Scaling factors
        coverage: 'rand' (original) or 'uniform' for better ball coverage
        seed: Optional random seed
    Returns:
        tuple: (mu, Sigma) - A single pair of mean vector and covariance matrix
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = len(mu_hat)  # dimension of the space
    max_attempts = 10000
    
    for attempt in range(max_attempts):
        if coverage == 'rand':
            # Original random sampling
            mu = sample_meanvec(mu_hat, mu_scale, mu_method)
            Sigma = sample_covariance(Sigma_hat, bound, Sigma_method, Sigma_scale)
            
        elif coverage == 'uniform':
            # Uniform coverage in Wasserstein ball
            # Sample radius uniformly in [0, bound^2]
            total_dims = n + (n * (n + 1)) // 2
            u = np.random.random()
            radius = bound * u**(1.0/total_dims)
            
            # Sample direction uniformly
            direction = np.random.randn(total_dims)
            direction = direction / np.linalg.norm(direction)
            
            # Split into mean and covariance components
            mean_dims = n
            mean_component = direction[:mean_dims] * radius * np.sqrt(mu_scale)
            cov_component = direction[mean_dims:] * radius * np.sqrt(Sigma_scale)
            
            # Convert to mu and Sigma
            mu = mu_hat + mean_component
            
            # Convert cov_component to symmetric matrix
            cov_matrix = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    cov_matrix[i,j] = cov_matrix[j,i] = cov_component[idx]
                    idx += 1
            
            Sigma = Sigma_hat + cov_matrix
            
            # Ensure Sigma is positive definite
            min_eigenval = np.min(np.linalg.eigvals(Sigma))
            if min_eigenval < 0:
                Sigma -= (min_eigenval - 1e-6) * np.eye(n)
        
        else:
            raise ValueError(f"Unknown coverage type: {coverage}. Use 'rand' or 'uniform'")
        
        # Check Wasserstein constraint
        if compute_wasserstein(mu_hat, Sigma_hat, mu, Sigma) <= bound**2:
            return mu, Sigma
            
    raise ValueError(f"Failed to find valid sample in {max_attempts} attempts")


def sample_moments_U2(mu_hat, Sigma_hat, bound, mu_method = 'perturbation', Sigma_method = 'uniform', 
                     mu_scale = 0.1, Sigma_scale = 0.2, num_envs = 1, max_attempts = 10000, dag = None):
    
    samples = [] 
    
    while len(samples) < num_envs:
        num_attempts = 0
        for i in range(max_attempts):
            num_attempts += 1
            mu    = sample_meanvec(mu_hat, mu_scale, mu_method)
            Sigma = sample_covariance(Sigma_hat, bound, Sigma_method, Sigma_scale)
            # 3. Check if the new mean and covariance satisfy the Wasserstein distance constraint
            if compute_wasserstein(mu_hat, Sigma_hat, mu, Sigma) <= bound**2:
                samples.append((mu, Sigma))
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


def generate_shifted_gaussian_family(mu, Sigma, k, r_mu=1.0, r_sigma=1.0, coverage='rand', seed=None):
    """
    Generate k shifted multivariate Gaussians with option for uniform coverage.

    Args:
        mu: Base mean vector (d,)
        Sigma: Base covariance matrix (d, d)
        k: Number of shifted distributions
        r_mu: Max norm of mean shifts
        r_sigma: Max Frobenius norm of covariance shifts
        coverage: 'rand' (original) or 'uniform' for better ball coverage
        seed: Optional random seed

    Returns:
        List of (mu_i, Sigma_i) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    d = mu.shape[0]
    shifted = []
    total_dims = d + (d * (d + 1)) // 2  # Total dimensions (mean + unique covariance elements)

    for _ in range(k):
        if coverage == 'rand':
            # Original random sampling
            # Mean shift: random direction, scaled to r_mu
            delta_mu = np.random.randn(d)
            delta_mu = r_mu * delta_mu / np.linalg.norm(delta_mu)

            # Covariance shift: random symmetric matrix with Frobenius norm = r_sigma
            A = np.random.randn(d, d)
            sym_A = (A + A.T) / 2
            delta_Sigma = r_sigma * sym_A / np.linalg.norm(sym_A, ord='fro')

        elif coverage == 'uniform':
            # Uniform coverage in combined space
            # Sample radius using proper power-law scaling
            u = np.random.random()
            radius_mu = r_mu * u**(1.0/total_dims)
            radius_sigma = r_sigma * u**(1.0/total_dims)

            # Sample direction for mean uniformly on unit sphere
            mean_dir = np.random.randn(d)
            mean_dir = mean_dir / np.linalg.norm(mean_dir)
            delta_mu = radius_mu * mean_dir

            # Sample direction for covariance uniformly
            cov_vec = np.random.randn((d * (d + 1)) // 2)
            cov_vec = cov_vec / np.linalg.norm(cov_vec)
            
            # Convert to symmetric matrix
            delta_Sigma = np.zeros((d, d))
            idx = 0
            for i in range(d):
                for j in range(i, d):
                    delta_Sigma[i,j] = delta_Sigma[j,i] = cov_vec[idx] * radius_sigma
                    idx += 1

        else:
            raise ValueError(f"Unknown coverage type: {coverage}. Use 'rand' or 'uniform'")

        mu_i = mu + delta_mu
        Sigma_i = Sigma + delta_Sigma

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(Sigma_i)
        Sigma_i = eigvecs @ np.diag(np.clip(eigvals, 1e-4, None)) @ eigvecs.T

        shifted.append((mu_i, Sigma_i))

    return shifted

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

def get_coefficients(data_list, G, weights=None, use_ridge=False, alpha=1.0):
    """
    Estimate structural coefficients for a linear SCM using observational data.
    
    Args:
        data_list: List of datasets or single dataset (n_samples, n_variables)
        G: NetworkX DiGraph representing causal structure
        weights: Optional weights for different datasets
        use_ridge: Whether to use Ridge regression (default: False)
        alpha: Ridge regression parameter (default: 1.0)
    
    Returns:
        dict: Edge coefficients {(parent, child): coefficient}
    """
    # Input validation
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")
    
    # Convert single dataset to list
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
        
    # Setup weights
    if weights is None:
        weights = [1.0] * len(data_list)
    
    # Combine datasets and weights
    data = np.vstack(data_list)
    sample_weights = np.repeat(weights, [d.shape[0] for d in data_list])
    sample_weights = sample_weights / sample_weights.sum()
    
    # Get node ordering
    nodes = list(G.nodes)
    coeffs = {}
    
    # Estimate coefficients for each node
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        if not parents:
            continue
            
        # Prepare regression data
        y = data[:, nodes.index(node)]
        X = data[:, [nodes.index(p) for p in parents]]
        
        # Fit regression
        model = Ridge(alpha=alpha) if use_ridge else LinearRegression()
        model.fit(X, y, sample_weight=sample_weights)
        
        # Store coefficients
        for parent, coef in zip(parents, model.coef_):
            coeffs[(parent, node)] = coef
            
    return coeffs

def get_coefficients_with_known_noise(data, noise, G):
    """
    Estimate coefficients when noise terms are known.
    This gives exact solutions since X = BX + U becomes a determined system.
    """
    nodes = list(G.nodes)
    coeffs = {}
    
    for node in nx.topological_sort(G):
        node_idx = nodes.index(node)
        parents = list(G.predecessors(node))
        
        if parents:
            y = data[:, node_idx] - noise[:, node_idx]  # Subtract known noise
            X = data[:, [nodes.index(p) for p in parents]]
            
            # Now we're fitting X = BX exactly (no error term)
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            
            for parent, coef in zip(parents, model.coef_):
                coeffs[(parent, node)] = coef
                
    return coeffs

def compute_radius_lb(N, eta, c):
    """
    Computes the concentration radius epsilon_N(eta) lower bound

    Parameters:
    - N: int, number of samples
    - eta: float, confidence parameter (0 < eta < 1)
    - c: float, constant from the theorem (c > 1)

    Returns:
    - epsilon: float, the concentration bound
    """
    assert 0 < eta <= 1, "eta must be in (0, 1]"
    assert c > 1, "c must be greater than 1"
    return np.log(c / eta) / np.sqrt(N)

def lan_abduction(data, G, coeffs):
    
    n_samples, n_vars = data.shape
    noise = np.zeros((n_samples, n_vars))
    nodes = list(G.nodes)
    
    # For each variable in topological order
    for node in nx.topological_sort(G):
        idx = nodes.index(node)
        parents = list(G.predecessors(node))
        
        if not parents:
            # No parents -> observed value is the noise
            noise[:, idx] = data[:, idx]
        else:
            # Compute predicted value from parents
            predicted = sum(data[:, nodes.index(p)] * coeffs[(p, node)] 
                          for p in parents)
            # Noise is observed minus predicted
            noise[:, idx] = data[:, idx] - predicted
    
    
    mean = np.mean(noise, axis=0).round(3)
    var = np.var(noise, axis=0).round(3)
    cov = np.diag(var)  # Assuming independent noise
    
    return noise, mean, cov

def estimate_shape_parameter(noise, loc, scale, initial_beta=2.0):
    """
    Estimate the shape parameter beta using MLE.
    """
    def neg_log_likelihood(beta):
        total_ll = 0
        for dim in range(noise.shape[1]):
            z = np.abs(noise[:, dim] - loc[dim]) / scale[dim]
            total_ll += np.sum(z**beta)
        return total_ll
    
    # Optimize to find beta (shape parameter)
    result = minimize(neg_log_likelihood, x0=initial_beta, 
                     bounds=[(0.1, 10.0)],  # Reasonable bounds for beta
                     method='L-BFGS-B')
    
    return result.x[0]


############################# LOADERS #############################
def load_model(experiment, model):
    if model == 'LL':
        return joblib.load(f'data/{experiment}/LL.pkl')
    elif model == 'HL':
        return joblib.load(f'data/{experiment}/HL.pkl')

def load_coeffs(experiment, model):
    if model == 'LL':
        return joblib.load(f'data/{experiment}/ll_coeffs.pkl')
    elif model == 'HL':
        return joblib.load(f'data/{experiment}/hl_coeffs.pkl')

def load_omega_map(experiment):
    return joblib.load(f'data/{experiment}/omega.pkl')

def load_pairs(experiment):
    return joblib.load(f'data/{experiment}/pairs.pkl')

def load_samples(experiment):
    return joblib.load(f'data/{experiment}/Ds.pkl')

def load_exogenous(experiment, model):
    if model == 'LL':
        return joblib.load(f'data/{experiment}/exogenous_LL.pkl')
    elif model == 'HL':
        return joblib.load(f'data/{experiment}/exogenous_HL.pkl')
    
def load_T(experiment):
    return joblib.load(f'data/{experiment}/Tau.pkl')

def load_type_to_params(experiment, noise_type, level):
    type_to_params_dict = joblib.load(f'data/{experiment}/type_to_params.pkl')
    return type_to_params_dict[noise_type][level]

def load_optimization_params(experiment, level, method='erica'):
    save_dir = f"data/{experiment}/{method}"
    try:
        opt_params = joblib.load(f"{save_dir}/opt_params.pkl")
        params_L, params_H = opt_params['L'], opt_params['H']

        return {'L': params_L, 'H': params_H}[level]
       
    except FileNotFoundError:
        print(f"No saved parameters found in {save_dir}/opt_params.pkl")
        return None, None
    
def load_empirical_boundary_params(experiment, pert_level):
    return joblib.load(f'data/{experiment}/empirical_boundary_params.pkl')[pert_level]

def load_abstraction(experiment, method):
    return joblib.load(f"data/{experiment}/{method}.pkl")