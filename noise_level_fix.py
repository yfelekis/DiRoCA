"""
Solution to the "No data found for noise_scale ≈ 1.0" warning in gauss_evaluation.ipynb

This script provides a diagnostic and fix for the noise level issue.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def diagnose_noise_levels(final_results_df):
    """
    Diagnose the noise level issue and provide a solution.
    """
    print("=== DIAGNOSIS ===")
    print("Available noise levels in your data:")
    available_levels = sorted(final_results_df['noise_scale'].unique())
    print(f"  {available_levels}")

    print(f"\nExpected noise levels from np.linspace(0, 1.0, 5):")
    expected_levels = np.linspace(0, 1.0, 5)
    print(f"  {expected_levels}")

    print(f"\n=== SOLUTION ===")
    # Use the highest available noise level for plotting
    noise_level_to_plot = available_levels[-1]
    print(f"Using noise_level_to_plot = {noise_level_to_plot}")

    # Filter the data using the available noise level
    df_filtered = final_results_df[np.isclose(final_results_df['noise_scale'], noise_level_to_plot)]

    if not df_filtered.empty:
        print(f"✓ Found {len(df_filtered)} data points for noise_scale ≈ {noise_level_to_plot}")
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Use seaborn.lineplot with 'alpha' as the new x-axis
        sns.lineplot(
            data=df_filtered,
            x='alpha',
            y='error',
            hue='method',
            style='run_id',
            marker='o',
            linewidth=2.5,
            errorbar=('ci', 95)
        )

        plt.title(f'Robustness to Outlier Fraction (at Noise Scale ≈ {noise_level_to_plot})', fontsize=18)
        plt.xlabel('Contamination Fraction (alpha)', fontsize=14)
        plt.ylabel('Average Abstraction Error', fontsize=14)
        plt.legend(title='Method')
        plt.grid(True, which='both', linestyle='--')
        plt.ylim(bottom=0)
        plt.show()
        
        return df_filtered
    else:
        print("✗ Still no data found. This indicates a deeper issue with the data generation.")
        return None

def fix_noise_level_issue():
    """
    Instructions to fix the noise level issue in the notebook.
    """
    print("=== HOW TO FIX THE NOISE LEVEL ISSUE ===")
    print("\n1. In your notebook, replace the problematic cell (Cell 27) with this code:")
    print("""
# ======================================================================
# 1. CONTROL PANEL: Choose which noise level to analyze
# ======================================================================

# First, let's check what noise levels are actually available in the data
print("Available noise levels in your data:")
available_levels = sorted(final_results_df['noise_scale'].unique())
print(f"  {available_levels}")

# Select a noise level that exists in the data
noise_level_to_plot = available_levels[-1]  # Use the highest available level
print(f"Using noise_level_to_plot = {noise_level_to_plot}")

# ======================================================================
# 2. FILTER AND PLOT (Corrected)
# ======================================================================

# Use np.isclose for safe floating-point comparison instead of '=='
df_filtered = final_results_df[np.isclose(final_results_df['noise_scale'], noise_level_to_plot)]

if df_filtered.empty:
    print(f"Warning: No data found for noise_scale ≈ {noise_level_to_plot}.")
    print("Available noise levels in your data:")
    print(f"  {available_levels}")
    print("Please check the 'noise_levels' list in your experiment to ensure this value exists.")
else:
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Use seaborn.lineplot with 'alpha' as the new x-axis
    sns.lineplot(
        data=df_filtered,
        x='alpha',
        y='error',
        hue='method',
        style='run_id',
        marker='o',
        linewidth=2.5,
        errorbar=('ci', 95)
    )

    plt.title(f'Robustness to Outlier Fraction (at Noise Scale ≈ {noise_level_to_plot})', fontsize=18)
    plt.xlabel('Contamination Fraction (alpha)', fontsize=14)
    plt.ylabel('Average Abstraction Error', fontsize=14)
    plt.legend(title='Method')
    plt.grid(True, which='both', linestyle='--')
    plt.ylim(bottom=0)
    plt.show()
""")
    
    print("\n2. Alternative: If you want to use a specific noise level, make sure it exists:")
    print("   - Check the 'noise_levels' definition in Cell 21")
    print("   - Use np.linspace(0, 1.0, 5) which creates [0.0, 0.25, 0.5, 0.75, 1.0]")
    print("   - Or change the noise_level_to_plot to one of the available values")

if __name__ == "__main__":
    fix_noise_level_issue() 