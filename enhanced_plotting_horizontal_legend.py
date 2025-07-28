# Enhanced Plotting Code with Horizontal Legend Below
# Copy this code into a new cell in your empirical_analysis.ipynb

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
})

methods_to_plot = [
    'DiRoCA_star',
    'DIROCA_1',
    'DIROCA_2', 
    'DIROCA_4',
    'GradCA',
    'BARYCA',
    'Abslin_p',
    'Abslin_n'
]

alpha_to_plot = 1.0

# Filter data for selected alpha and methods
df_subset = final_results_df[
    (final_results_df['alpha'] == alpha_to_plot) & 
    (final_results_df['method'].isin(methods_to_plot))
]

if df_subset.empty:
    print(f"Warning: No data found for alpha ≈ {alpha_to_plot}. Please check the 'alpha_values' in your experiment.")
else:
    df_for_plotting = df_subset.copy()
    df_for_plotting['display_name'] = df_for_plotting['method'].map(print_label_map)

    plt.figure(figsize=(12, 8))  # Increased height to accommodate legend below
    sns.lineplot(
        data=df_for_plotting,
        x='noise_scale',
        y='error',
        hue='display_name',
        marker='o',
        linewidth=2.0,
        markersize=8,
        errorbar=('ci', 95) # Shaded area is the 95% CI across folds and trials
    )
    
    # No title as requested
    plt.xlabel(r'$\sigma_{noise}$', fontsize=34)
    plt.ylabel(r'Abstraction Error', fontsize=32)
    
    # Horizontal legend below the plot
    plt.legend(fontsize=20, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Increase tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    # Adjust layout to make room for legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    plt.show()

# Alternative version for plotting against alpha (fixing noise)
print("\n" + "="*60)
print("Alternative: Plotting against alpha (fixing noise)")
print("="*60)

noise_level_to_plot = 5.0

# Filter data for selected noise level and methods
df_subset_alpha = final_results_df[
    (final_results_df['noise_scale'] == noise_level_to_plot) & 
    (final_results_df['method'].isin(methods_to_plot))
]

if df_subset_alpha.empty:
    print(f"Warning: No data found for noise_scale ≈ {noise_level_to_plot}.")
else:
    df_for_plotting_alpha = df_subset_alpha.copy()
    df_for_plotting_alpha['display_name'] = df_for_plotting_alpha['method'].map(print_label_map)

    plt.figure(figsize=(12, 8))  # Increased height to accommodate legend below
    sns.lineplot(
        data=df_for_plotting_alpha,
        x='alpha',
        y='error',
        hue='display_name',
        marker='o',
        linewidth=2.0,
        markersize=8,
        errorbar=('ci', 95)
    )
    
    plt.xlabel(r'$\alpha$', fontsize=34)
    plt.ylabel(r'Abstraction Error', fontsize=32)
    
    # Horizontal legend below the plot
    plt.legend(fontsize=20, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=4)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Increase tick label sizes
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    # Adjust layout to make room for legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    plt.show() 