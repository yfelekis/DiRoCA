#!/usr/bin/env python3
"""
Example usage of both gaussian and empirical evaluation scripts

This script demonstrates how to:
1. Run both types of evaluations
2. Load and compare results
3. Analyze the differences between gaussian and empirical evaluations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from run_evaluation import run_evaluation as run_gaussian_evaluation
from run_empirical_evaluation import run_evaluation as run_empirical_evaluation
from load_results import load_gaussian_latest, load_empirical_latest

def run_comparison_evaluations():
    """Run both gaussian and empirical evaluations with the same parameters."""
    print("Running comparison evaluations...")
    
    # Define parameters for comparison
    alpha_values = np.linspace(0, 1.0, 3)  # Fewer for quick comparison
    noise_levels = np.linspace(0, 5.0, 3)  # Fewer for quick comparison
    
    # Run gaussian evaluation
    print("\n=== Running Gaussian Evaluation ===")
    gaussian_results = run_gaussian_evaluation(
        experiment='slc',
        alpha_values=alpha_values,
        noise_levels=noise_levels,
        num_trials=2,
        zero_mean=True,
        shift_type='additive',
        distribution='gaussian'
    )
    
    # Run empirical evaluation
    print("\n=== Running Empirical Evaluation ===")
    empirical_results = run_empirical_evaluation(
        experiment='slc',
        alpha_values=alpha_values,
        noise_levels=noise_levels,
        num_trials=2,
        zero_mean=True,
        shift_type='additive',
        distribution='gaussian'
    )
    
    return gaussian_results, empirical_results

def load_and_compare_results():
    """Load latest results from both evaluation types and compare them."""
    print("Loading and comparing results...")
    
    # Load latest results
    gaussian_df = load_gaussian_latest('slc')
    empirical_df = load_empirical_latest('slc')
    
    if gaussian_df is None or empirical_df is None:
        print("Could not load one or both result files")
        return None, None
    
    # Add evaluation type column
    gaussian_df['evaluation_type'] = 'gaussian'
    empirical_df['evaluation_type'] = 'empirical'
    
    # Combine results
    combined_df = pd.concat([gaussian_df, empirical_df], ignore_index=True)
    
    return gaussian_df, empirical_df, combined_df

def analyze_differences(gaussian_df, empirical_df):
    """Analyze differences between gaussian and empirical evaluations."""
    print("\n=== Analysis of Differences ===")
    
    # Basic statistics
    print(f"Gaussian evaluation records: {len(gaussian_df)}")
    print(f"Empirical evaluation records: {len(empirical_df)}")
    
    # Compare error statistics by method
    print("\nError statistics by method and evaluation type:")
    stats_comparison = []
    
    for method in gaussian_df['method'].unique():
        gaussian_method = gaussian_df[gaussian_df['method'] == method]
        empirical_method = empirical_df[empirical_df['method'] == method]
        
        if len(empirical_method) > 0:
            gaussian_mean = gaussian_method['error'].mean()
            empirical_mean = empirical_method['error'].mean()
            difference = empirical_mean - gaussian_mean
            
            stats_comparison.append({
                'method': method,
                'gaussian_mean': gaussian_mean,
                'empirical_mean': empirical_mean,
                'difference': difference,
                'relative_diff': (difference / gaussian_mean) * 100 if gaussian_mean > 0 else 0
            })
    
    stats_df = pd.DataFrame(stats_comparison)
    print(stats_df.round(4))
    
    return stats_df

def create_comparison_plots(gaussian_df, empirical_df):
    """Create comparison plots between gaussian and empirical evaluations."""
    print("\nCreating comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gaussian vs Empirical Evaluation Comparison', fontsize=16)
    
    # 1. Error vs Alpha comparison
    ax1 = axes[0, 0]
    for method in gaussian_df['method'].unique():
        # Gaussian
        gaussian_method = gaussian_df[gaussian_df['method'] == method]
        alpha_means_g = gaussian_method.groupby('alpha')['error'].mean()
        ax1.plot(alpha_means_g.index, alpha_means_g.values, 
                label=f'{method} (Gaussian)', marker='o', linestyle='-')
        
        # Empirical
        empirical_method = empirical_df[empirical_df['method'] == method]
        if len(empirical_method) > 0:
            alpha_means_e = empirical_method.groupby('alpha')['error'].mean()
            ax1.plot(alpha_means_e.index, alpha_means_e.values, 
                    label=f'{method} (Empirical)', marker='s', linestyle='--')
    
    ax1.set_xlabel('Alpha (Contamination Level)')
    ax1.set_ylabel('Mean Error')
    ax1.set_title('Error vs Alpha by Method and Evaluation Type')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Error vs Noise comparison
    ax2 = axes[0, 1]
    for method in gaussian_df['method'].unique():
        # Gaussian
        gaussian_method = gaussian_df[gaussian_df['method'] == method]
        noise_means_g = gaussian_method.groupby('noise_scale')['error'].mean()
        ax2.plot(noise_means_g.index, noise_means_g.values, 
                label=f'{method} (Gaussian)', marker='o', linestyle='-')
        
        # Empirical
        empirical_method = empirical_df[empirical_df['method'] == method]
        if len(empirical_method) > 0:
            noise_means_e = empirical_method.groupby('noise_scale')['error'].mean()
            ax2.plot(noise_means_e.index, noise_means_e.values, 
                    label=f'{method} (Empirical)', marker='s', linestyle='--')
    
    ax2.set_xlabel('Noise Scale')
    ax2.set_ylabel('Mean Error')
    ax2.set_title('Error vs Noise by Method and Evaluation Type')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = axes[1, 0]
    combined_data = []
    combined_labels = []
    
    for method in gaussian_df['method'].unique():
        gaussian_method = gaussian_df[gaussian_df['method'] == method]
        empirical_method = empirical_df[empirical_df['method'] == method]
        
        combined_data.extend([gaussian_method['error'].values, empirical_method['error'].values])
        combined_labels.extend([f'{method} (G)', f'{method} (E)'])
    
    ax3.boxplot(combined_data, labels=combined_labels)
    ax3.set_title('Error Distribution Comparison')
    ax3.set_ylabel('Error')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot of gaussian vs empirical errors
    ax4 = axes[1, 1]
    for method in gaussian_df['method'].unique():
        gaussian_method = gaussian_df[gaussian_df['method'] == method]
        empirical_method = empirical_df[empirical_df['method'] == method]
        
        if len(empirical_method) > 0:
            # Group by alpha and noise to get comparable points
            gaussian_grouped = gaussian_method.groupby(['alpha', 'noise_scale'])['error'].mean()
            empirical_grouped = empirical_method.groupby(['alpha', 'noise_scale'])['error'].mean()
            
            # Align indices
            common_indices = gaussian_grouped.index.intersection(empirical_grouped.index)
            if len(common_indices) > 0:
                gaussian_errors = gaussian_grouped.loc[common_indices]
                empirical_errors = empirical_grouped.loc[common_indices]
                
                ax4.scatter(gaussian_errors, empirical_errors, label=method, alpha=0.7)
    
    # Add diagonal line
    min_val = min(ax4.get_xlim()[0], ax4.get_ylim()[0])
    max_val = max(ax4.get_xlim()[1], ax4.get_ylim()[1])
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax4.set_xlabel('Gaussian Error')
    ax4.set_ylabel('Empirical Error')
    ax4.set_title('Gaussian vs Empirical Error Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gaussian_vs_empirical_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plots saved to 'gaussian_vs_empirical_comparison.png'")
    
    # Show the plot
    plt.show()

def main():
    """Main function to run the comparison."""
    print("=== Gaussian vs Empirical Evaluation Comparison ===\n")
    
    try:
        # Option 1: Run new evaluations (uncomment to run)
        # gaussian_results, empirical_results = run_comparison_evaluations()
        
        # Option 2: Load existing results
        gaussian_df, empirical_df, combined_df = load_and_compare_results()
        
        if gaussian_df is not None and empirical_df is not None:
            # Analyze differences
            stats_df = analyze_differences(gaussian_df, empirical_df)
            
            # Create comparison plots
            create_comparison_plots(gaussian_df, empirical_df)
            
            print("\n=== Comparison completed successfully ===")
        else:
            print("Could not load results for comparison")
            
    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 