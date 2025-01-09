import numpy as np
import matplotlib.pyplot as plt
from modularised_utils import compute_wasserstein   
import torch
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm
import seaborn as sns


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

def plot_abstraction_error(abstraction_error_dict):
    """
    Plot abstraction errors with error bars using Seaborn.
    """
    
    # Extract data from dictionary
    methods = list(abstraction_error_dict.keys())
    means = [v[0] for v in abstraction_error_dict.values()]
    errors = [v[1] for v in abstraction_error_dict.values()]
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create error bar plot
    sns.scatterplot(
        x=methods,
        y=means,
        color='purple',
        s=100  # marker size
    )
    
    # Add error bars
    plt.errorbar(
        x=methods,
        y=means,
        yerr=errors,
        fmt='none',  # no connecting lines
        color='green',
        capsize=5,
        capthick=2,
        elinewidth=2
    )
    
    # Customize plot
    plt.yscale('log')  # log scale for y-axis
    plt.xticks(rotation=45, ha='right')  # rotate x labels
    plt.title('Abstraction Error by Method')
    plt.xlabel('Method')
    plt.ylabel('Error')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show plot
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
