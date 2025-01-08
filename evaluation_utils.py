import numpy as np
import matplotlib.pyplot as plt
from modularised_utils import compute_wasserstein   
import torch
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm


def contaminate_linear_relationships(data, contamination_fraction=0.3, contamination_type='multiplicative'):
    """
    Contaminate linear relationships between variables by applying a specific non-linear transformation.
    
    Args:
        data: numpy array of shape (n_samples, n_vars)
        contamination_fraction: fraction of samples to contaminate (default: 0.3)
        contamination_type: type of non-linear transformation to apply (default: 'multiplicative')
                          options: ['multiplicative', 'threshold', 'exponential', 'sinusoidal']
    
    Returns:
        Contaminated data array
    """
    if contamination_type not in ['multiplicative', 'threshold', 'exponential', 'sinusoidal']:
        raise ValueError(f"Unknown contamination type: {contamination_type}. "
                       f"Must be one of: ['multiplicative', 'threshold', 'exponential', 'sinusoidal']")
    
    contaminated = data.copy()
    n_samples, n_vars = data.shape
    
    # Select samples to contaminate
    n_contaminate = int(n_samples * contamination_fraction)
    contaminate_idx = np.random.choice(n_samples, n_contaminate, replace=False)
    
    # Apply the specified contamination
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
        contaminated[:, i] = ((contaminated[:, i] - np.mean(contaminated[:, i])) 
                            / np.std(contaminated[:, i]) * orig_std + orig_mean)
    
    return contaminated

def plot_contamination_effects(original, contaminated):
    """
    Visualize the effects of contamination on the data relationships.
    """
    n_vars = original.shape[1]
    fig, axes = plt.subplots(2, n_vars-1, figsize=(15, 10))
    
    # Plot relationships between consecutive variables
    for i in range(n_vars-1):
        # Original data
        axes[0,i].scatter(original[:,i], original[:,i+1], alpha=0.5, s=1)
        axes[0,i].set_title(f'Original: Var{i+1} vs Var{i+2}')
        axes[0,i].set_xlabel(f'Var{i+1}')
        axes[0,i].set_ylabel(f'Var{i+2}')
        
        # Contaminated data
        axes[1,i].scatter(contaminated[:,i], contaminated[:,i+1], alpha=0.5, s=1)
        axes[1,i].set_title(f'Contaminated: Var{i+1} vs Var{i+2}')
        axes[1,i].set_xlabel(f'Var{i+1}')
        axes[1,i].set_ylabel(f'Var{i+2}')
    
    plt.tight_layout()
    plt.show()
