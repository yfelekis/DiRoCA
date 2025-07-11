# You'll need this import
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
import networkx as nx  
import numpy as np
import joblib

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