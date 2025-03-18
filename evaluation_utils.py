import numpy as np
import matplotlib.pyplot as plt
from modularised_utils import compute_wasserstein   
import torch
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm
import seaborn as sns
import modularised_utils as mut
import opt_utils as oput
import operations as ops

def compute_worst_case_distance(params_worst):
    mu_worst = params_worst['mu_U']
    Sigma_worst = params_worst['Sigma_U']
    mu_hat = params_worst['mu_hat']
    Sigma_hat = params_worst['Sigma_hat']
    radius = params_worst['radius']

    mu_dist_sq     = np.sum((mu_worst - mu_hat)**2)

    std_worst = np.std(Sigma_worst)
    std_hat   = np.std(Sigma_hat)
    std_diff  = np.sum((std_worst - std_hat)**2)

    Sigma_sqrt     = oput.sqrtm_svd_np(Sigma_worst)
    hat_Sigma_sqrt = oput.sqrtm_svd_np(Sigma_hat)
    Sigma_dist_sq  = np.sum((Sigma_sqrt - hat_Sigma_sqrt)**2)
    G_squared      = mu_dist_sq + Sigma_dist_sq
    G_squared      = G_squared.item()
    #radius_squared = round(radius**2, 5)
    #diff = abs(G_squared - radius_squared)
    
    return G_squared

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

def contaminate_linear_relationships(data, contamination_fraction, contamination_type, k=1.345):
    """
    Contaminate linear relationships between variables by applying a specific transformation.
    
    Args:
        data: numpy array of shape (n_samples, n_vars)
        contamination_fraction: fraction of samples to contaminate (default: 0.3)
        contamination_type: type of transformation to apply (default: 'multiplicative')
                          options: ['multiplicative', 'threshold', 'exponential', 'sinusoidal', 'huber']
        k: Huber parameter (default=1.345 for 95% efficiency), only used if contamination_type='huber'
    
    Returns:
        Contaminated data array
    """
    if contamination_type not in ['multiplicative', 'threshold', 'exponential', 'sinusoidal', 'huber']:
        raise ValueError(f"Unknown contamination type: {contamination_type}. "
                       f"Must be one of: ['multiplicative', 'threshold', 'exponential', 'sinusoidal', 'huber']")
    
    contaminated = data.copy()
    n_samples, n_vars = data.shape
    
    # Select samples to contaminate
    n_contaminate = int(n_samples * contamination_fraction)
    contaminate_idx = np.random.choice(n_samples, n_contaminate, replace=False)
    
    # Apply the specified contamination
    if contamination_type == 'huber':
        # For each variable
        for j in range(n_vars):
            # Compute median and MAD for robust statistics
            median = np.median(data[:, j])
            mad = np.median(np.abs(data[:, j] - median)) * 1.4826  # consistent with normal distribution
            
            # Apply Huber function to selected indices
            for idx in contaminate_idx:
                x = data[idx, j]
                if np.abs(x - median) > k * mad:
                    # Apply Huber transformation
                    sign = np.sign(x - median)
                    contaminated[idx, j] = median + sign * k * mad
    else:
        for idx in contaminate_idx:
            if contamination_type == 'multiplicative':
                # Multiply pairs of variables
                for i in range(n_vars-1):
                    contaminated[idx, i+1] *= contaminated[idx, i]
                    
            elif contamination_type == 'threshold':
                # Create discontinuous jumps
                thresholds = np.random.randn(n_vars)
                for i in range(n_vars):
                    if contaminated[idx, i] > thresholds[i]:
                        contaminated[idx, i] *= 2
                    else:
                        contaminated[idx, i] *= 0.5
                        
            elif contamination_type == 'exponential':
                # Create exponential relationships
                contaminated[idx] = np.exp(contaminated[idx] * 0.5) - 1
                    
            elif contamination_type == 'sinusoidal':
                # Add sinusoidal transformations
                contaminated[idx] = np.sin(contaminated[idx])
    
    # Normalize to keep similar scale as original data
    for i in range(n_vars):
        orig_std = np.std(data[:, i])
        orig_mean = np.mean(data[:, i])
        cont_std = np.std(contaminated[:, i])
        
        # Avoid division by zero by checking if std is too small
        if cont_std < 1e-10:  # numerical threshold for "zero"
            contaminated[:, i] = orig_mean  # set all values to mean if no variation
        else:
            contaminated[:, i] = ((contaminated[:, i] - np.mean(contaminated[:, i])) 
                                / cont_std * orig_std + orig_mean)
            
    return contaminated

def plot_contamination_effects(original, contaminated):
    """
    Visualize the effects of contamination on the data relationships using Seaborn.
    """
    
    n_vars = original.shape[1]
    fig, axes = plt.subplots(2, n_vars-1, figsize=(15, 10))
    
    # Plot relationships between consecutive variables
    for i in range(n_vars-1):
        # Original data
        sns.scatterplot(data=None, 
                       x=original[:,i], 
                       y=original[:,i+1], 
                       alpha=0.5, 
                       s=10,
                       color='purple',
                       ax=axes[0,i])
        axes[0,i].set_title(f'Original: Var{i+1} vs Var{i+2}')
        axes[0,i].set_xlabel(f'Var{i+1}')
        axes[0,i].set_ylabel(f'Var{i+2}')
        
        # Contaminated data
        sns.scatterplot(data=None, 
                       x=contaminated[:,i], 
                       y=contaminated[:,i+1], 
                       alpha=0.5, 
                       s=10,
                       color='green',
                       ax=axes[1,i])
        axes[1,i].set_title(f'Contaminated: Var{i+1} vs Var{i+2}')
        axes[1,i].set_xlabel(f'Var{i+1}')
        axes[1,i].set_ylabel(f'Var{i+2}')
    
    # Optional: Set style for all subplots
    sns.set_style("whitegrid")
    
    plt.tight_layout()
    plt.show()

def plot_abstraction_error(abstraction_error_dict, spacing_factor=0.2):
    """
    Plot abstraction errors with error bars using Seaborn.
    """
    # Extract data from dictionary
    methods = list(abstraction_error_dict.keys())
    means = [v[0] for v in abstraction_error_dict.values()]
    errors = [v[1] for v in abstraction_error_dict.values()]
    
    # Calculate width first
    width = max(4, len(methods) * spacing_factor)
    
    # Set style and font sizes with LaTeX
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.dpi': 300,  # Increase DPI
        'savefig.dpi': 300,  # Increase saving DPI
        'figure.figsize': (width, 5),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'text.antialiased': True,
        'axes.linewidth': 0.5
    })
    
    # Create figure with higher quality
    plt.figure(figsize=(width, 5), dpi=100)
    
    sns.scatterplot(
        x=range(len(methods)),
        y=means,
        color='purple',
        s=60
    )
    
    plt.errorbar(
        x=range(len(methods)),
        y=means,
        yerr=errors,
        fmt='none',
        color='green',
        capsize=7,
        capthick=2,
        elinewidth=1
    )
    
    plt.yscale('log')
    plt.xticks(
        range(len(methods)),
        methods,
        rotation=45,
        ha='right'
    )
    
    plt.margins(x=0.1)
    
    plt.title('')
    plt.xlabel(r'Method')
    plt.ylabel(r'$e(T)$')
    
    plt.tight_layout(pad=1.0)
    plt.show()
    
    return

def plot_condition_nums(cn_dict, spacing_factor=0.2):
    """
    Plot condition numbers with consistent spacing logic.
    """
    # Extract data from dictionary
    methods = list(cn_dict.keys())
    condition_number = [v for v in cn_dict.values()]
    
    # Calculate width first
    width = max(4, len(methods) * spacing_factor)
    
    # Set style and font sizes with LaTeX
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.dpi': 300,  # Increase DPI
        'savefig.dpi': 300,  # Increase saving DPI
        'figure.figsize': (width, 5),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'text.antialiased': True,
        'axes.linewidth': 0.5
    })
    
    # Create figure with higher quality
    plt.figure(figsize=(width, 5), dpi=100)
    
    # Create scatter plot
    sns.scatterplot(
        x=range(len(methods)),
        y=condition_number,
        color='purple',
        s=200
    )
    
    # Add markers
    plt.errorbar(
        x=range(len(methods)),
        y=condition_number,
        yerr=None,
        fmt='o',
        color='purple',
        capsize=5,
        capthick=5,
        elinewidth=2
    )
    
    # Customize plot with tighter spacing
    plt.yscale('log')
    plt.xticks(
        range(len(methods)),
        methods,
        rotation=45,
        ha='right'
    )
    
    plt.margins(x=0.1)
    plt.title('')  # Removed title as in previous function
    plt.xlabel(r'Method')
    plt.ylabel(r'$\kappa(T)$')  # Using LaTeX for condition number symbol
    
    plt.tight_layout(pad=1.0)
    plt.show()
    return

def plot_distribution_shifts(original, modified):
    """
    Visualize the changes in distributions using Seaborn.
    """    
    n_vars = original.shape[1]
    plt.figure(figsize=(15, 5))
    
    for i in range(n_vars):
        plt.subplot(1, n_vars, i+1)
        
        # Plot original distribution with Seaborn (purple)
        sns.kdeplot(data=original[:, i], 
                   color='green', 
                   alpha=0.5,
                   label=f'Original (μ={np.mean(original[:, i]):.2f}, σ²={np.var(original[:, i]):.2f})',
                   fill=True)
        
        # Plot modified distribution with Seaborn (green)
        sns.kdeplot(data=modified[:, i], 
                   color='purple', 
                   alpha=0.5,
                   label=f'Modified (μ={np.mean(modified[:, i]):.2f}, σ²={np.var(modified[:, i]):.2f})',
                   fill=True)
        
        plt.title(f'Variable {i+1} Distribution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
def plot_distribution_changes(original, modified, title="Distribution Changes"):
    """
    Visualize the changes in marginal distributions for each dimension.
    
    Args:
        original: Original data array of shape (n_samples, n_dims)
        modified: Modified data array of shape (n_samples, n_dims)
        title: Optional title for the plot
    """
    n_vars = original.shape[1]
    fig = plt.figure(figsize=(15, 5))
    
    for i in range(n_vars):
        plt.subplot(1, n_vars, i+1)
        
        # Plot original distribution
        plt.hist(original[:, i], bins=50, alpha=0.5, color='lightblue',
                label=f'Original (μ={np.mean(original[:, i]):.2f}, σ²={np.var(original[:, i]):.2f})', 
                density=True)
        
        # Plot modified distribution
        plt.hist(modified[:, i], bins=50, alpha=0.5, color='orange',
                label=f'Modified (μ={np.mean(modified[:, i]):.2f}, σ²={np.var(modified[:, i]):.2f})', 
                density=True)
        
        plt.title(f'Variable {i+1} Distribution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def generate_noise(shape, noise_type, form, level, experiment, normalize, rad=None):
    n_samples, n_vars = shape
    
    if noise_type in ['gelbrich_gaussian', 'boundary_gaussian', 'rand_epsilon_delta', 'gelbrich_boundary_gaussian']:
        params = mut.load_type_to_params(experiment, noise_type, level)
            
    if noise_type == 'gelbrich_gaussian':

        mu_U_hat    = params['mu_U'] #+ np.ones(params_Lerica['mu_U'].shape[0])
        Sigma_U_hat = params['Sigma_U'] #+ np.random.normal(0, 0.1, size=params_Lerica['Sigma_U'].shape)
        radius      = params['radius']

        # Sample moments from Gelbrich ball
        moments = mut.sample_moments_U(
                                        mu_hat=mu_U_hat, 
                                        Sigma_hat=Sigma_U_hat, 
                                        bound=radius, 
                                        num_envs=1
                                      )
        
        noise_mu, noise_Sigma = moments[0]

        noise = np.random.multivariate_normal(
                                                mean=noise_mu, 
                                                cov=noise_Sigma, 
                                                size=n_samples
                                                )

    elif noise_type == 'boundary_gaussian':
        noise_mu, noise_Sigma = params['mu_U'], params['Sigma_U']

        noise = np.random.multivariate_normal(
                                                mean=noise_mu, 
                                                cov=noise_Sigma, 
                                                size=n_samples
                                                )
    
    elif noise_type == 'rand_epsilon_delta':
        mu_U_hat    = params['mu_U'] #+ np.ones(params_Lerica['mu_U'].shape[0])
        Sigma_U_hat = params['Sigma_U'] #+ np.random.normal(0, 0.1, size=params_Lerica['Sigma_U'].shape)
        radius      = rad
        

        # Sample moments from Gelbrich ball
        moments = mut.sample_moments_U(
                                        mu_hat=mu_U_hat, 
                                        Sigma_hat=Sigma_U_hat, 
                                        bound=radius, 
                                        num_envs=1
                                        )
        
        noise_mu, noise_Sigma = moments[0]

        noise = np.random.multivariate_normal(
                                                mean=noise_mu, 
                                                cov=noise_Sigma, 
                                                size=n_samples
                                                )
        
    elif noise_type == 'gelbrich_boundary_gaussian':

        mu_U_hat    = params['mu_U'] #+ np.ones(params_Lerica['mu_U'].shape[0])
        Sigma_U_hat = params['Sigma_U'] #+ np.random.normal(0, 0.1, size=params_Lerica['Sigma_U'].shape)
        radius      = rad
        random_mu   = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars))

        # Convert to PyTorch tensors before calling get_gelbrich_boundary
        random_mu_tensor = torch.tensor(random_mu, dtype=torch.float32)
        random_Sigma_tensor = torch.tensor(random_Sigma, dtype=torch.float32)
        mu_U_hat_tensor = torch.tensor(mu_U_hat, dtype=torch.float32)
        Sigma_U_hat_tensor = torch.tensor(Sigma_U_hat, dtype=torch.float32)

        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu_tensor, random_Sigma_tensor, mu_U_hat_tensor, Sigma_U_hat_tensor, radius)
        
        noise = np.random.multivariate_normal(
                                                mean=noise_mu, 
                                                cov=noise_Sigma, 
                                                size=n_samples
                                                )
    # Generate noise based on type
    elif noise_type == 'uniform':
        noise = np.random.uniform(low=-1, high=1, size=(n_samples, n_vars))
    
    elif noise_type == 'exponential':
        noise = np.random.exponential(scale=4.0, size=(n_samples, n_vars))
        noise = noise - 1  # Center to mean 0
    
    elif noise_type == 'laplace':
        noise = np.random.laplace(loc=0.5, scale=2.0, size=(n_samples, n_vars))
    
    elif noise_type == 'chi_square':
        noise = np.random.chisquare(df=1, size=(n_samples, n_vars))
        noise = noise - 1  # Center to mean 0
    
    elif noise_type == 'random_normal':
        low, high = (-2, 2)
        noise_mu = np.random.uniform(low=low, high=high, size=n_vars)
        noise_Sigma = np.diag(np.random.uniform(0, high, size=n_vars))

        noise = np.random.multivariate_normal(
                                                mean=noise_mu, 
                                                cov=noise_Sigma, 
                                                size=n_samples
                                                )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Normalize noise to have similar scale
    if normalize == True:
        noise = noise / np.std(noise, axis=0)

    if form == 'sample':
        return noise
    
    elif form == 'distributional':
        return noise_mu, noise_Sigma

def generate_noise_fixed(n_samples, noise_type, form, level, experiment, normalize, rad=None):

    params = load_optimization_params(experiment, level)
    n_vars = params['mu_hat'].shape[0]

    if noise_type == 'gelbrich_fixed_hat': # Sample moments from the Gelbrich ball centered at theta_hat

        mu     = params['mu_hat'] 
        Sigma  = params['Sigma_hat']
        radius = params['radius']

        # Sample moments from Gelbrich ball
        noise_mu, noise_Sigma = mut.sample_moments_U(mu, Sigma, bound=radius, num_envs=1)[0]
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    elif noise_type == 'gelbrich_boundary_hat': # Sample moments from the boundary of the Gelbrich ball centered at theta_hat

        mu     = params['mu_hat'] 
        Sigma  = params['Sigma_hat'] 
        radius = params['g_squared']
        
        random_mu    = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars))
    
        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu, random_Sigma, mu, Sigma, radius)
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    elif noise_type == 'gelbrich_boundary_worst': # Sample moments from the boundary of the Gelbrich ball centered at theta_worst

        mu     = params['mu_U'] 
        Sigma  = params['Sigma_U'] 
        radius = params['g_squared']
        random_mu   = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars))
    
        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu, random_Sigma, mu, Sigma, radius)
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    elif noise_type == 'gelbrich_random_hat': # Sample moments from Gelbrich ball centered at theta_hat
        mu     = params['mu_hat'] 
        Sigma  = params['Sigma_hat']
        radius = rad

        noise_mu, noise_Sigma = mut.sample_moments_U(mu, Sigma, bound=radius, num_envs=1)[0]
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)
    
    elif noise_type == 'gelbrich_random_worst': # Sample moments from Gelbrich ball centered at theta_worst
        mu     = params['mu_U'] 
        Sigma  = params['Sigma_U']
        radius = rad
        
        noise_mu, noise_Sigma = mut.sample_moments_U(mu, Sigma, bound=radius, num_envs=1)[0]
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    elif noise_type == 'boundary_fixed_worst': # Return theta_worst
        noise_mu, noise_Sigma = params['mu_U'], params['Sigma_U']

        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    elif noise_type == 'gelbrich_boundary_gaussian':

        mu_U_hat     = params['mu_hat'] 
        Sigma_U_hat  = params['Sigma_hat'] 
        radius       = params['g_squared']
        random_mu    = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars))
    
        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu, random_Sigma, mu_U_hat, Sigma_U_hat, radius)
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    elif noise_type == 'gelbrich_boundary_gaussian2':

        mu_U_hat     = params['mu_U'] 
        Sigma_U_hat  = params['Sigma_U'] 
        radius       = rad
        random_mu    = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars))


        # Convert to PyTorch tensors before calling get_gelbrich_boundary
        random_mu_tensor = torch.tensor(random_mu, dtype=torch.float32)
        random_Sigma_tensor = torch.tensor(random_Sigma, dtype=torch.float32)
        mu_U_hat_tensor = torch.tensor(mu_U_hat, dtype=torch.float32)
        Sigma_U_hat_tensor = torch.tensor(Sigma_U_hat, dtype=torch.float32)

        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu_tensor, random_Sigma_tensor, mu_U_hat_tensor, Sigma_U_hat_tensor, radius)


        #noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu, random_Sigma, mu_U_hat, Sigma_U_hat, radius)
        
        noise = np.random.multivariate_normal(mean=noise_mu, cov=noise_Sigma, size=n_samples)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Normalize noise to have similar scale
    if normalize == True:
        noise = noise / np.std(noise, axis=0)

    if form == 'sample':
        return noise
    
    elif form == 'distributional':
        return noise_mu, noise_Sigma
    
def generate_pertubation(data, pert_type, pert_level, experiment):
    N, n = data.shape
    
    boundary_matrix, radius = mut.load_empirical_boundary_params(experiment, pert_level)
    if pert_type == 'constraint_set':
        P = oput.init_in_frobenius_ball((N, n), radius).detach().numpy()

    elif pert_type == 'boundary':
        P = boundary_matrix

    elif pert_type == 'random_normal':
        P = np.random.randn(N, n)
        
    elif pert_type == 'random_uniform':
        P = np.random.rand(N, n)

    return P

def compute_abstraction_error(T, base, abst, metric):
    tau_base   = base @ T.T
    tau_muL    = np.mean(tau_base, axis=0)
    tau_sigmaL = np.cov(tau_base, rowvar=False)
    muH        = np.mean(abst, axis=0)
    sigmaH     = np.cov(abst, rowvar=False)

    if metric == 'wass':
        dist = mut.compute_wasserstein(tau_muL, tau_sigmaL, muH, sigmaH)
        #dist = 1 - np.exp(-dist)
    elif metric == 'js':
        dist = mut.compute_jensenshannon(tau_base, abst)

    return dist

def compute_empirical_distance(tbase, abst, metric):

    if metric == 'fro':
        dist     = ops.MatrixDistances.frobenius_distance(tbase, abst)
    elif metric == 'sq_fro':
        dist     = ops.MatrixDistances.squared_frobenius_distance(tbase, abst)
    elif metric == 'nuclear':
        dist     = ops.MatrixDistances.nuclear_norm_distance(tbase, abst)
    elif metric == 'spectral':
        dist     = ops.MatrixDistances.spectral_norm_distance(tbase, abst)
    elif metric == 'l1':
        dist     = ops.MatrixDistances.l1_distance(tbase, abst)    
    else:
        raise ValueError(f"Invalid metric: {metric}")

    return dist

def generate_data(LLmodels, HLmodels, omega, num_llsamples, num_hlsamples, mu_U_ll_hat, Sigma_U_ll_hat, mu_U_hl_hat, Sigma_U_hl_hat):
    """
    Generates data for the linear additive noise SCMs.
    """
    Ill = list(LLmodels.keys())
    Ihl = list(HLmodels.keys())
    dbase = {}
    for iota in Ill:
        lenv_iota   = mut.sample_distros_Gelbrich([(mu_U_ll_hat, Sigma_U_ll_hat)])[0] 
        noise_iota  = lenv_iota.sample(num_llsamples)[0]
        dbase[iota] = LLmodels[iota].simulate(noise_iota, iota)

    dabst = {}
    for eta in Ihl:
        henv_eta   = mut.sample_distros_Gelbrich([(mu_U_hl_hat, Sigma_U_hl_hat)])[0] 
        noise_eta  = henv_eta.sample(num_hlsamples)[0]
        dabst[eta] = HLmodels[eta].simulate(noise_eta, eta)

    data = {}
    for iota in Ill:
        data[iota] = (dbase[iota], dabst[omega[iota]])

    return data


def generate_empirical_data(LLmodels, HLmodels, omega, U_L, U_H):
    Ill = list(LLmodels.keys())
    Ihl = list(HLmodels.keys())
   
    dbase = {}
    for iota in Ill:
        dbase[iota] = U_L @ LLmodels[iota].F

    dabst = {}
    for eta in Ihl:
        dabst[eta] = U_H @ HLmodels[eta].F

    data = {}
    for iota in Ill:
        data[iota] = (dbase[iota], dabst[omega[iota]])

    return data

def compute_empirical_worst_case_distance(params_worst):
    pert_worst = params_worst['pert_U']
    N          = pert_worst.shape[0]

    return (1/np.sqrt(N)) * np.linalg.norm(pert_worst, 'fro') #torch.norm(pert_worst, p='fro') maybe torch different output
