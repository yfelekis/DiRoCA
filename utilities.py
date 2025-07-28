# You'll need this import
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
import networkx as nx  
import numpy as np
import joblib
import torch
import yaml
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def get_coefficients(data, G, return_noise=False, use_ridge=False, alpha=1.0):
    """
    Estimates coefficients and computes noise (residuals) simultaneously.
    """
    nodes = list(G.nodes)
    coeffs = {}
    
    # Initialize noise matrix with the data; we'll subtract predictions later
    noise = data.copy()

    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        if not parents:
            continue
            
        # Prepare regression data
        node_idx = nodes.index(node)
        parent_indices = [nodes.index(p) for p in parents]
        
        y = data[:, node_idx]
        X = data[:, parent_indices]
        
        # Fit regression
        model = Ridge(alpha=alpha) if use_ridge else LinearRegression()
        model.fit(X, y)
        
        # Store coefficients
        for parent, coef in zip(parents, model.coef_):
            coeffs[(parent, node)] = coef
            
        # OVERWRITE the noise for this node with the residuals
        # Residuals = y - y_predicted
        y_pred = model.predict(X)
        noise[:, node_idx] = y - y_pred
        
    if return_noise:
        return coeffs, noise
    else:
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

def load_all_data(experiment_name):
    """Loads all model blueprints and abstraction data for a given experiment."""
    path = f"data/{experiment_name}"
    data = {
        'LLmodel': joblib.load(f"{path}/LLmodel.pkl"),
        'HLmodel': joblib.load(f"{path}/HLmodel.pkl"),
        'abstraction_data': joblib.load(f"{path}/abstraction_data.pkl")
    }
    print(f"Data loaded for '{experiment_name}'.")

    return data

def prepare_cv_folds(observational_data, k, random_state, save_path):
    """Generates and saves K-Fold train/test indices."""
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    num_samples = observational_data.shape[0]
    
    fold_indices = [{'train': train_idx, 'test': test_idx} 
                    for train_idx, test_idx in kf.split(np.arange(num_samples))]
    
    joblib.dump(fold_indices, save_path)
    print(f"Created and saved {len(fold_indices)} folds to '{save_path}'")
    return fold_indices

def assemble_fold_parameters(fold_indices, all_data, hyperparameters):
    """Assembles the final opt_params dictionary for a specific fold."""
    # Start with the general hyperparameters
    opt_params = hyperparameters.copy()

    # Add the core models and mappings
    opt_params['LLmodels']      = all_data['LLmodel'].get('scm_instances')
    opt_params['HLmodels']      = all_data['HLmodel'].get('scm_instances')
    opt_params['omega']         = all_data['abstraction_data']['omega']
    opt_params['experiment']    = all_data['experiment_name']
    opt_params['initial_theta'] = 'empirical'
    
    # Calculate fold-specific radius
    train_n  = len(fold_indices['train'])
    ll_bound = round(compute_radius_lb(N=train_n, eta=0.05, c=1000), 3)
    hl_bound = round(compute_radius_lb(N=train_n, eta=0.05, c=1000), 3)

    # Add the final theta parameters
    opt_params['theta_hatL'] = {
                                    'mu_U': all_data['LLmodel']['noise_dist']['mu'], 
                                    'Sigma_U': all_data['LLmodel']['noise_dist']['sigma'], 
                                    'radius': ll_bound
                                }
    opt_params['theta_hatH'] = {
                                    'mu_U': all_data['HLmodel']['noise_dist']['mu'], 
                                    'Sigma_U': all_data['HLmodel']['noise_dist']['sigma'], 
                                    'radius': hl_bound
                                }
    
    return opt_params

def assemble_barycentric_parameters(fold_info, all_data, baryca_hyperparams):
    """
    Assembles the final arguments dictionary specifically for barycentric_optimization.
    """
    # 1. Start with the specific hyperparameters for this algorithm
    opt_args = baryca_hyperparams.copy()

    # 2. Add the required model and data components
    opt_args['LLmodels'] = all_data['LLmodel'].get('scm_instances')
    opt_args['HLmodels'] = all_data['HLmodel'].get('scm_instances')
    opt_args['Ill'] = all_data['LLmodel']['intervention_set']
    opt_args['Ihl'] = all_data['HLmodel']['intervention_set']
    
    # 3. Calculate fold-specific radius
    train_n  = len(fold_info['train'])
    # Assuming 'ut' is your imported utilities module
    ll_bound = round(compute_radius_lb(N=train_n, eta=0.05, c=1000), 3)
    hl_bound = round(compute_radius_lb(N=train_n, eta=0.05, c=1000), 3)

    # 4. Add the theta parameters (note the key is 'theta_L', not 'theta_hatL')
    opt_args['theta_L'] = {
        'mu_U': all_data['LLmodel']['noise_dist']['mu'], 
        'Sigma_U': all_data['LLmodel']['noise_dist']['sigma'], 
        'radius': ll_bound
    }
    opt_args['theta_H'] = {
        'mu_U': all_data['HLmodel']['noise_dist']['mu'], 
        'Sigma_U': all_data['HLmodel']['noise_dist']['sigma'], 
        'radius': hl_bound
    }
    
    return opt_args

def assemble_empirical_parameters(U_ll_train, U_hl_train, all_data, empirical_hyperparams):
    """
    Assembles the final arguments dictionary specifically for the empirical optimization.
    """
    # 1. Start with the specific hyperparameters for this algorithm
    opt_args = empirical_hyperparams.copy()

    # 2. Add the required model and data components
    opt_args['U_L'] = U_ll_train
    opt_args['U_H'] = U_hl_train
    opt_args['L_models'] = all_data['LLmodel'].get('scm_instances')
    opt_args['H_models'] = all_data['HLmodel'].get('scm_instances')
    opt_args['omega'] = all_data['abstraction_data']['omega']
    opt_args['experiment'] = all_data['experiment_name']
    
    return opt_args

def compute_struc_matrices(models, intervention_set):
    """Computes the reduced-form matrix F for each SCM instance."""
    return [torch.from_numpy(models[iota].F).float() for iota in intervention_set]


def print_distribution_summary(final_params, initial_params, name=""):
    """Prints a summary comparing initial and final Gaussian parameters."""
    
    mu_final = final_params['mu_U']
    sigma_final = final_params['Sigma_U']
    
    mu_initial = initial_params['mu_U']
    sigma_initial = initial_params['Sigma_U']
    
    print(f"\n--- Distribution Summary: {name} ---")
    
    # Compare Means
    print("\nMean (μ):")
    print(f"  - Initial: {np.round(mu_initial, 3)}")
    print(f"  - Final  : {np.round(mu_final, 3)}")
    
    # Compare Variances (diagonal of the covariance matrix)
    print("\nVariances (diag(Σ)):")
    print(f"  - Initial: {np.round(np.diag(sigma_initial), 3)}")
    print(f"  - Final  : {np.round(np.diag(sigma_final), 3)}")
    
    # Show the final correlation matrix to see if the adversary induced correlations
    std_devs = np.sqrt(np.diag(sigma_final))
    # Add a small epsilon to avoid division by zero if variance is zero
    corr_matrix = sigma_final / np.outer(std_devs + 1e-8, std_devs + 1e-8)
    print("\nFinal Correlation Matrix:")
    print(np.round(corr_matrix, 2))
    print("-"*(25 + len(name)))
    
def plot_marginal_distributions(final_params, initial_params, var_names, model_name=""):
    """Plots a comparison of the 1D marginals for each variable."""
    
    # Set LaTeX font
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18
    })
    
    mu_final, sigma_final = final_params['mu_U'], final_params['Sigma_U']
    mu_initial, sigma_initial = initial_params['mu_U'], initial_params['Sigma_U']
    
    n_vars = len(var_names)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols # Calculate rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'Marginal Noise Distributions: {model_name}', fontsize=20)

    for i in range(n_vars):
        # Initial Distribution
        mean_i, std_i = mu_initial[i], np.sqrt(sigma_initial[i, i])
        x = np.linspace(mean_i - 3*std_i, mean_i + 3*std_i, 200)
        axes[i].plot(x, norm.pdf(x, mean_i, std_i), 'b-', lw=3, label='Initial (Empirical)')

        # Final (Worst-Case) Distribution
        mean_f, std_f = mu_final[i], np.sqrt(sigma_final[i, i])
        x = np.linspace(mean_f - 3*std_f, mean_f + 3*std_f, 200)
        axes[i].plot(x, norm.pdf(x, mean_f, std_f), 'r--', lw=3, label='Final (Worst-Case)')
        
        axes[i].set_title(var_names[i], fontsize=18)
        axes[i].legend(fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

    # Hide any unused subplots
    for j in range(n_vars, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_optimization_animation(monitor, initial_params, var_names, model_level='L', filename='optimization.gif'):
    """Creates and saves an animation of the distribution's evolution."""
    
    # Select which history to use (Low-Level or High-Level)
    if model_level == 'L':
        mu_history = monitor.mu_L_history
        sigma_history = monitor.sigma_L_history
        initial_mu = initial_params['theta_hatL']['mu_U']
        initial_sigma = initial_params['theta_hatL']['Sigma_U']
    else:
        mu_history = monitor.mu_H_history
        sigma_history = monitor.sigma_H_history
        initial_mu = initial_params['theta_hatH']['mu_U']
        initial_sigma = initial_params['theta_hatH']['Sigma_U']

    num_epochs = len(mu_history)
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
    if n_vars == 1: axes = [axes]
    fig.suptitle(f'Evolution of Worst-Case Distribution ({model_level} model)', fontsize=16)

    # This function will be called for each frame of the animation
    def update(epoch):
        for i in range(n_vars):
            ax = axes[i]
            ax.clear()

            # Plot Initial distribution (static blue line)
            mean_i, std_i = initial_mu[i], np.sqrt(initial_sigma[i, i] + 1e-8)
            x_i = np.linspace(mean_i - 4*std_i, mean_i + 4*std_i, 200)
            ax.plot(x_i, norm.pdf(x_i, mean_i, std_i), 'b-', lw=2, label='Initial (Empirical)')

            # Plot Final distribution (static red line)
            mean_f, std_f = mu_history[-1][i], np.sqrt(sigma_history[-1][i, i] + 1e-8)
            x_f = np.linspace(mean_f - 4*std_f, mean_f + 4*std_f, 200)
            ax.plot(x_f, norm.pdf(x_f, mean_f, std_f), 'r--', lw=2, label='Final (Worst-Case)')

            # Plot the CURRENT distribution for this epoch (moving green line)
            mean_e, std_e = mu_history[epoch][i], np.sqrt(sigma_history[epoch][i, i] + 1e-8)
            x_e = np.linspace(mean_e - 4*std_e, mean_e + 4*std_e, 200)
            ax.plot(x_e, norm.pdf(x_e, mean_e, std_e), 'g--', lw=2.5, label=f'Epoch {epoch+1}')
            
            ax.set_title(var_names[i])
            ax.set_ylim(bottom=0)
            ax.legend()
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_epochs, blit=False, repeat=False)
    
    # Save the animation as a GIF
    print(f"Saving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=5)
    plt.close() # Prevent static plot from showing
    print("Done.")
    return HTML(ani.to_jshtml()) # Display animation in the notebook

def load_configs(config_files):
    """
    Load multiple YAML configuration files.
    
    Args:
        config_files (dict): Dictionary mapping variable names to config file paths
                           e.g., {'hyperparams_diroca': 'configs/diroca_opt_config.yaml'}
    
    Returns:
        dict: Dictionary with loaded configs, keys are the variable names
    """
    configs = {}
    for var_name, file_path in config_files.items():
        try:
            with open(file_path, 'r') as f:
                configs[var_name] = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {file_path} not found")
            configs[var_name] = {}
        except yaml.YAMLError as e:
            print(f"Error parsing {file_path}: {e}")
            configs[var_name] = {}
    
    return configs