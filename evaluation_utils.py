import numpy as np
import matplotlib.pyplot as plt
from math_utils import compute_wasserstein   
import torch
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm
import seaborn as sns
import modularised_utils as mut
import opt_utils as oput
import operations as ops
import joblib


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

def plot_empirical_abstraction_error(results, methods, sample_form, figsize=(12, 8)):
    """
    Plot empirical abstraction error vs radius for specified methods and sample form.
    
    Parameters:
    -----------
    results : dict
        The results dictionary containing the empirical data
    methods : list
        List of method names to plot
    sample_form : str
        Either 'boundary' or 'sample'
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Define method styles for basic methods
    method_styles = {
        'T_0.00': {'color': 'purple', 'label': r'$\mathrm{T}_{0,0}$', 'marker': 'o'},
        'T_b': {'color': 'red', 'label': r'$\mathrm{T}_{b}$', 'marker': 's'},
        'T_s': {'color': 'green', 'label': r'$\mathrm{T}_{s}$', 'marker': '^'},
        'T_ba': {'color': 'orange', 'label': r'$\mathrm{T}_{ba}$', 'marker': 'D'},
        'T_pa': {'color': 'brown', 'label': r'$\mathrm{T}_{pa}$', 'marker': 'P'},  # Added T_pa
        'T_na': {'color': 'cyan', 'label': r'$\mathrm{T}_{na}$', 'marker': 'X'}    # Added T_na
    }
    # Generate color map for epsilon-delta methods
    n_colors = len([m for m in methods if '-' in m])  # Count methods with epsilon-delta format
    if n_colors > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
        color_idx = 0
        
        # Add styles for epsilon-delta methods
        for method in methods:
            if '-' in method:
                eps, delta = method.replace('T_', '').split('-')
                method_styles[method] = {
                    'color': colors[color_idx],
                    'label': f'$\\mathrm{{T}}_{{\\varepsilon_l={eps},\\varepsilon_h={delta}}}$',
                    'marker': 'h'
                }
                color_idx += 1
    
    # Add styles for single parameter methods using blues
    blues = plt.cm.Blues(np.linspace(0.6, 1, 5))
    markers = ['h', 'v', 'p', '*', 'x']
    for i, eps in enumerate(['0.031', '1', '2', '4', '8']):
        method_name = f'T_{eps}'
        if method_name in methods:  # Only add if method is actually used
            method_styles[method_name] = {
                'color': blues[i],
                'label': f'$\\mathrm{{T}}_{{\\varepsilon={eps}}}$',
                'marker': markers[i]
            }
    
    # Extract radius values and sort them
    radius_values = sorted([float(r) for r in results.keys()])
    
    # Plot each method
    for method in methods:
        if method not in method_styles:
            print(f"Warning: No style defined for method {method}")
            continue
            
        style = method_styles[method]
        errors = []
        error_stds = []
        
        # Collect data for each radius
        for rad in radius_values:
            values = results[rad][sample_form]['empirical'][method]
            mean = np.mean(values)
            std = np.std(values)
            errors.append(mean)
            error_stds.append(1.96 * std)  # 95% confidence interval
        
        # Plot with error bands
        plt.plot(radius_values, errors,
                f"{style['marker']}-",
                color=style['color'],
                label=style['label'],
                alpha=0.8,
                markersize=12,
                linewidth=2)
        
        # plt.fill_between(radius_values,
        #                 np.array(errors) - np.array(error_stds),
        #                 np.array(errors) + np.array(error_stds),
        #                 color=style['color'],
        #                 alpha=0.2)
    
    # Customize plot
    plt.xlabel(r'Perturbation Radius ($\varepsilon_{\ell}=\varepsilon_{h}$)', fontsize=30)
    plt.ylabel('Empirical Abstraction Error', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    # Add legend with two columns if many methods
    n_methods = len([m for m in methods if m in method_styles])
    ncols = 2 if n_methods > 6 else 1
    
    plt.legend(prop={'size': 24},
              frameon=True,
              framealpha=1,
              borderpad=0.5,
              handletextpad=0.5,
              handlelength=1.5,
              ncol=ncols,
              loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

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

def load_optimization_params(experiment, level, method='erica'):
    save_dir = f"data/{experiment}/{method}"
    try:
        opt_params = joblib.load(f"{save_dir}/opt_params.pkl")
        params_L, params_H = opt_params['L'], opt_params['H']

        return {'L': params_L, 'H': params_H}[level]
       
    except FileNotFoundError:
        print(f"No saved parameters found in {save_dir}/opt_params.pkl")
        return None, None
    
def generate_noise2(center, radius, sample_form, level, experiment, normalize, rad=None):
    params = load_optimization_params(experiment, level)
    n_vars = params['mu_hat'].shape[0]
    #np.random.seed(seed) NEED FOR THE RADIUS VS ERROR?
    if center == 'hat':
        mu = params['mu_hat']
        Sigma = params['Sigma_hat']
    elif center == 'worst':
        mu = params['mu_U']
        Sigma = params['Sigma_U']
    else:
        raise ValueError(f"Unknown center: {center}")
    
    if radius == 'hat':
        radius = params['radius']
    elif radius == 'random':
        radius = rad
    elif radius == 'worst':
        radius = params['g_squared']
    else:
        raise ValueError(f"Unknown radius: {radius}")
    
    if sample_form == 'boundary':
        random_mu    = np.random.randn(n_vars)
        random_Sigma = np.diag(np.random.rand(n_vars)) 

        noise_mu, noise_Sigma = oput.get_gelbrich_boundary(random_mu, random_Sigma, mu, Sigma, radius)
        
    elif sample_form == 'sample':
        noise_mu, noise_Sigma = mut.sample_moments_U(mu, Sigma, bound=radius, num_envs=1)[0]
    else:
        raise ValueError(f"Unknown sample form: {sample_form}")
    
    if normalize:
        # Convert to numpy if tensor
        if torch.is_tensor(noise_mu):
            noise_mu = noise_mu.numpy()
        if torch.is_tensor(noise_Sigma):
            noise_Sigma = noise_Sigma.numpy()
            
        # Normalize using numpy
        noise_mu = noise_mu / np.std(noise_mu)
        noise_Sigma = noise_Sigma / np.std(noise_Sigma)

    # Convert to numpy if still tensor
    if torch.is_tensor(noise_mu):
        noise_mu = noise_mu.numpy()
    if torch.is_tensor(noise_Sigma):
        noise_Sigma = noise_Sigma.numpy()

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
    
def generate_pertubation11(data, pert_type, pert_level, experiment):
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

def generate_perturbation_matrix2(radius, sample_form, level, hat_dict, seed=None):
    """
    Generate a perturbation matrix either on the boundary or sampled from within a Frobenius ball.
    
    Args:
        radius: Radius of the Frobenius ball
        sample_form: Either 'sample' (from within ball) or 'boundary' (on boundary)
        level: The level key to access in the dictionaries
        hat_dict: Dictionary containing matrices for 'hat' case
        worst_dict: Dictionary containing matrices for 'worst' case
        seed: Optional random seed for reproducibility
    
    Returns:
        Perturbation matrix that can be added to a center matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples, num_vars = hat_dict[level].shape

    # Generate random perturbation matrix
    perturbation = np.random.randn(num_samples, num_vars)
    
    # Calculate norms
    squared_norm = np.linalg.norm(perturbation, 'fro')**2
    max_squared_norm = num_samples * radius**2
    
    # Calculate scaling factor based on sample type
    if sample_form == 'boundary':
        # Scale exactly to the radius
        scaling_factor = np.sqrt(max_squared_norm / squared_norm)
    elif sample_form == 'sample':
        # Scale to be within the radius
        scaling_factor = np.sqrt(max_squared_norm / squared_norm) * np.random.rand(1)
    
    return perturbation * scaling_factor

def generate_perturbation_family(center_matrix, k, r_mu, r_sigma, coverage, seed=None):
    """
    Generate k perturbation matrices with option for uniform coverage.
    
    Args:
        center_matrix: Base matrix to perturb (n, m)
        k: Number of perturbations to generate
        r_mu: Max norm of mean shifts
        r_sigma: Max Frobenius norm of covariance shifts
        coverage: Either 'rand' (original) or 'uniform' for better ball coverage
        seed: Optional random seed
        
    Returns:
        List of perturbation matrices
    """
    if seed is not None:
        np.random.seed(seed)
        
    n, m = center_matrix.shape
    total_dims = n * m  # Total dimensionality for uniform sampling
    perturbations = []
    
    for i in range(k):
        if coverage == 'rand':
            # Original random perturbation logic
            # Covariance perturbation
            A = np.random.randn(n, m)
            A = r_sigma * A / np.linalg.norm(A, ord='fro')
            
            # Mean shift
            delta_mu = np.random.randn(n, m)
            delta_mu = r_mu * delta_mu / np.linalg.norm(delta_mu)
            
        elif coverage == 'uniform':
            # Uniform coverage for both components
            
            # Covariance perturbation - uniform in r_sigma ball
            A = np.random.randn(n, m)
            A_norm = np.linalg.norm(A, ord='fro')
            A = A / A_norm
            # Use proper radius distribution for uniform sampling
            u_sigma = np.random.random()
            r_sigma_actual = r_sigma * u_sigma**(1.0/total_dims)
            A = r_sigma_actual * A
            
            # Mean shift - uniform in r_mu ball
            delta_mu = np.random.randn(n, m)
            mu_norm = np.linalg.norm(delta_mu, ord='fro')
            delta_mu = delta_mu / mu_norm
            # Use proper radius distribution for uniform sampling
            u_mu = np.random.random()
            r_mu_actual = r_mu * u_mu**(1.0/total_dims)
            delta_mu = r_mu_actual * delta_mu
            
        else:
            raise ValueError(f"Unknown coverage type: {coverage}. Use 'rand' or 'uniform'")
        
        # Combine perturbations
        perturbation = A + delta_mu
        perturbations.append(perturbation)
        
    return perturbations

def generate_perturbation_matrix(radius, sample_form, level, hat_dict, coverage, seed=None):
    """
    Generate a perturbation matrix with flexible sampling strategies.
    
    Args:
        radius: Radius of the Frobenius ball
        sample_form: Either 'sample' (from within ball) or 'boundary' (on boundary)
        level: The level key to access in the dictionaries
        hat_dict: Dictionary containing matrices for 'hat' case
        coverage: Either 'rand' (original random sampling) or 'uniform' (uniform ball coverage)
        seed: Optional random seed for reproducibility
    
    Returns:
        Perturbation matrix that can be added to a center matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples, num_vars = hat_dict[level].shape
    total_dims = num_samples * num_vars

    # Generate initial random perturbation
    perturbation = np.random.randn(num_samples, num_vars)
    
    # Boundary case - always use original logic
    if sample_form == 'boundary':
        squared_norm = np.linalg.norm(perturbation, 'fro')**2
        max_squared_norm = num_samples * radius**2
        scaling_factor = np.sqrt(max_squared_norm / squared_norm)
    
    # Sample case - choose between random and uniform coverage
    elif sample_form == 'sample':
        if coverage == 'rand':
            # Original random sampling logic
            squared_norm = np.linalg.norm(perturbation, 'fro')**2
            max_squared_norm = num_samples * radius**2
            scaling_factor = np.sqrt(max_squared_norm / squared_norm) * np.random.rand(1)
        
        elif coverage == 'uniform':
            # Uniform ball coverage logic
            norms = np.linalg.norm(perturbation, 'fro')
            perturbation = perturbation / norms
            u = np.random.random()
            r = radius * u**(1.0/total_dims)
            scaling_factor = r * np.sqrt(num_samples)
        
        else:
            raise ValueError(f"Unknown coverage type: {coverage}. Use 'rand' or 'uniform'")
    
    else:
        raise ValueError(f"Unknown sample form: {sample_form}. Use 'sample' or 'boundary'")

    return perturbation * scaling_factor

def generate_perturbation_family2(center_matrix, k, r_mu=1.0, r_sigma=1.0, seed=None):
    """
    Generate k perturbation matrices from a base matrix using random shifts.
    
    Args:
        center_matrix: Base matrix to perturb (n, m)
        k: Number of perturbations to generate
        r_mu: Max norm of mean shifts
        r_sigma: Max Frobenius norm of covariance shifts
        seed: Optional random seed
        
    Returns:
        List of perturbation matrices
    """
    if seed is not None:
        np.random.seed(seed)
        
    n, m = center_matrix.shape
    perturbations = []
    
    for _ in range(k):
        # Generate random perturbation matrix
        A = np.random.randn(n, m)
        # Scale to have Frobenius norm = r_sigma
        A = r_sigma * A / np.linalg.norm(A, ord='fro')
        
        # Add mean shift
        delta_mu = np.random.randn(n, m)
        delta_mu = r_mu * delta_mu / np.linalg.norm(delta_mu)
        
        # Combine perturbations
        perturbation = A + delta_mu
        perturbations.append(perturbation)
        
    return perturbations

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
    elif metric == 'l2': 
        dist = ops.MatrixDistances.l2_distance(tbase, abst)  
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


######################################### CausAbs FUNCTIONS###############################################
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

def perfect_abstraction(px_samples, py_samples, tau_threshold=1e-2):
    """
    Fits an abstraction function assumed to be perfect.
    
    Args:
        px_samples: concrete/base samples (your df_base)
        py_samples: abstract samples (your df_abst)
        tau_threshold: threshold for filtering small values
    """
    tau_adj = np.linalg.pinv(px_samples) @ py_samples
    tau_adj_mask = np.abs(tau_adj) > tau_threshold
    tau_adj = tau_adj * tau_adj_mask
    return tau_adj

def noisy_abstraction(px_samples, py_samples, tau_threshold=1e-1, refit_coeff=False):
    """
    Fits an abstraction function assumed to be noisy.
    """
    # Reconstruct T
    tau_adj_hat = np.linalg.pinv(px_samples) @ py_samples
    
    # Max entries
    tau_mask_hat = np.argmax(np.abs(tau_adj_hat), axis=1)
    # To one hot
    abs_nodes = py_samples.shape[1]
    tau_mask_hat = np.eye(abs_nodes)[tau_mask_hat]
    # Filter out small values
    tau_mask_hat *= np.array(np.abs(tau_adj_hat) > tau_threshold, dtype=np.int32)
    
    # Eventually refit coefficients
    if refit_coeff:
        for y in range(tau_mask_hat.shape[1]):
            block = np.where(tau_mask_hat[:, y] == 1)[0]
            if len(block) > 0:
                tau_adj_hat[block, y] = np.linalg.pinv(px_samples[:, block]) @ py_samples[:, y]
    
    # Compute final abstraction
    tau_adj_hat = tau_mask_hat * tau_adj_hat
    return tau_adj_hat

def abs_lingam_reconstruction(df_base, df_abst, style="Perfect", tau_threshold=1e-2):
    """
    Implement Abs-LiNGAM's T-reconstruction following the paper's implementation.
    
    Args:
        df_base: numpy array of concrete observations
        df_abst: numpy array of abstract observations
        style: "Perfect" or "Noisy"
        tau_threshold: threshold for filtering small values
    """
    # Verify shapes
    n_samples_base, d = df_base.shape
    n_samples_abst, b = df_abst.shape
    
    assert n_samples_base == n_samples_abst, \
        f"Number of samples must match: {n_samples_base} != {n_samples_abst}"
    
    # Fit abstraction function based on style
    if style == "Perfect":
        T = perfect_abstraction(df_base, df_abst, tau_threshold)
    elif style == "Noisy":
        T = noisy_abstraction(df_base, df_abst, tau_threshold, False)
    else:
        raise ValueError(f"Unknown style {style}")
    
    # Compute reconstruction error
    error = np.sum((df_base @ T - df_abst) ** 2)
    
    return T, error

def evaluate_abstraction(df_base, df_abst, T, tau_threshold=1e-2):
    """
    Evaluate the abstraction quality using metrics from the paper.
    """
    # Predictions
    predictions = df_base @ T
    
    # Create binary masks for evaluation
    T_mask = np.abs(T) > tau_threshold
    T_pred = np.abs(T) > tau_threshold
    
    # Basic metrics
    mse = np.mean((df_abst - predictions) ** 2)
    r2 = 1 - np.sum((df_abst - predictions) ** 2) / np.sum((df_abst - np.mean(df_abst, axis=0)) ** 2)
    
    # Structure metrics (from paper)
    def false_positive_rate(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return fp / (fp + tn) if (fp + tn) > 0 else 0
        return 0.0
    
    # Flatten for binary metrics
    T_flat = T_mask.flatten()
    T_pred_flat = T_pred.flatten()
    
    # Handle cases where all predictions are the same class
    try:
        prec = precision_score(T_flat, T_pred_flat)
    except:
        prec = 1.0 if np.array_equal(T_flat, T_pred_flat) else 0.0
        
    try:
        rec = recall_score(T_flat, T_pred_flat)
    except:
        rec = 1.0 if np.array_equal(T_flat, T_pred_flat) else 0.0
        
    try:
        f1 = f1_score(T_flat, T_pred_flat)
    except:
        f1 = 1.0 if np.array_equal(T_flat, T_pred_flat) else 0.0
    
    # Skip ROC AUC if we have only one class
    unique_classes = np.unique(T_flat)
    if len(unique_classes) == 2:
        try:
            roc_auc = roc_auc_score(T_flat, np.abs(T).flatten())
        except:
            roc_auc = np.nan
    else:
        roc_auc = np.nan
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': false_positive_rate(T_flat, T_pred_flat)
    }
    
    # Add sparsity metric
    metrics['sparsity'] = 1.0 - (np.sum(T_mask) / T_mask.size)
    
    return metrics

# def run_abs_lingam_complete(df_base, df_abst):
#     """
#     Run complete Abs-LiNGAM algorithm with different abstraction styles.
#     """
#     styles = ["Perfect", "Noisy"]
#     results = {}
    
#     for style in styles:
#         # print(f"\nTesting {style} abstraction:")
#         T, error = abs_lingam_reconstruction(df_base, df_abst, style=style)
#         metrics = evaluate_abstraction(df_base, df_abst, T)
        
#         results[style] = {
#             'T': T,
#             'error': error,
#             'metrics': metrics
#         }
    
#     return results

def abs_lingam_reconstruction_v2(df_base, df_abst, n_paired_samples=None, style="Perfect", tau_threshold=1e-2):
    """
    Modified version to better match the paper's approach.
    
    Args:
        df_base: numpy array of concrete observations (D_L)
        df_abst: numpy array of abstract observations
        n_paired_samples: number of samples to use for joint dataset (D_J)
        style: "Perfect" or "Noisy"
        tau_threshold: threshold for filtering small values
    """
    # Get dimensions
    n_samples_base, d = df_base.shape
    n_samples_abst, b = df_abst.shape
    
    # Create joint dataset D_J by taking a subset of paired samples
    if n_paired_samples is None:
        n_paired_samples = min(n_samples_base, n_samples_abst)
    
    # Ensure n_paired_samples is smaller than both datasets
    n_paired_samples = min(n_paired_samples, n_samples_base, n_samples_abst)
    
    # Create D_J using the first n_paired_samples
    D_J_base = df_base[:n_paired_samples]
    D_J_abst = df_abst[:n_paired_samples]
    
    # Fit abstraction function based on style using only paired data
    if style == "Perfect":
        T = perfect_abstraction(D_J_base, D_J_abst, tau_threshold)
    elif style == "Noisy":
        T = noisy_abstraction(D_J_base, D_J_abst, tau_threshold, False)
    else:
        raise ValueError(f"Unknown style {style}")
    
    # Compute reconstruction error using all available data
    error = np.sum((df_base @ T - df_abst) ** 2)
    
    return T, error

# Usage example:
def run_abs_lingam_complete(df_base, df_abst, n_paired_samples=1000):
    """
    Run complete Abs-LiNGAM algorithm with different abstraction styles.
    """
    styles = ["Perfect", "Noisy"]
    results = {}
    
    for style in styles:
        T, error = abs_lingam_reconstruction_v2(
            df_base, 
            df_abst,
            n_paired_samples=n_paired_samples,
            style=style
        )
        metrics = evaluate_abstraction(df_base, df_abst, T)
        
        results[style] = {
            'T': T,
            'error': error,
            'metrics': metrics
        }
    
    return results

def compute_empirical_radius(N, eta, c1=1.0, c2=1.0, alpha=2.0, m=3):
    """
    Compute epsilon_N(eta) for empirical Wasserstein case.

    Parameters:
    - N: int, number of samples
    - eta: float, confidence level (0 < eta < 1)
    - c1: float, constant from theorem (default 1.0, adjust if needed)
    - c2: float, constant from theorem (default 1.0, adjust if needed)
    - alpha: float, light-tail exponent (P[exp(||ξ||^α)] ≤ A)
    - m: int, ambient dimension

    Returns:
    - epsilon: float, the concentration radius
    """
    assert 0 < eta < 1, "eta must be in (0,1)"
    threshold = np.log(c1 / eta) / c2
    if N >= threshold:
        exponent = min(1/m, 0.5)
    else:
        exponent = 1 / alpha

    epsilon = (np.log(c1 / eta) / (c2 * N)) ** exponent
    return epsilon