"""
Fix for contamination_analysis_new.ipynb

This script provides the corrected code to fix the two main issues:

1. TypeError: No matching signature found - caused by NaN values in DataFrame aggregation
2. "Failed to find a square root" messages - caused by numerical issues in sqrtm operations

To use this fix:

1. Run the existing notebook up to Cell 7 (the contamination loop)
2. Instead of running Cell 8, run the corrected code below
3. The sqrtm issues should be reduced by the fixes in math_utils.py and modularised_utils.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fix_dataframe_processing(f_spec_records):
    """
    Fixed version of the DataFrame processing that handles NaN values properly.
    
    Args:
        f_spec_records: List of dictionaries containing the contamination analysis results
        
    Returns:
        tuple: (cleaned_dataframe, summary_statistics)
    """
    # Compile results into a DataFrame
    f_spec_df = pd.DataFrame(f_spec_records)
    
    # Handle NaN values before aggregation
    f_spec_df_clean = f_spec_df.dropna(subset=['error'])
    
    print("\n--- F-Misspecification Evaluation Complete (Fixed) ---")
    print("="*65)
    print("Overall Performance (Averaged Across All Nonlinearity Strengths)")
    print("="*65)
    print(f"{'Method/Run':<35} | {'Mean ± Std'}")
    print("="*65)
    
    # Use the clean DataFrame for aggregation
    if len(f_spec_df_clean) > 0:
        summary = f_spec_df_clean.groupby('method')['error'].agg(['mean', 'std', 'count'])
        summary['sem'] = summary['std']  # You can uncomment next line for CI
        # summary['sem'] = summary['std'] / np.sqrt(summary['count'])
        summary['ci95'] = summary['sem']  # or 1.96 * sem
        
        for method_name, row in summary.sort_values('mean').iterrows():
            print(f"{method_name:<35} | {row['mean']:.4f} ± {row['ci95']:.4f}")
        
        # Set plot style
        sns.set(style="whitegrid")
        
        # Plot abstraction error vs. contamination strength
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=f_spec_df_clean,  # Use clean DataFrame for plotting
            x="contamination",
            y="error",
            hue="method",
            estimator="mean",
            errorbar='sd',  # Updated parameter name
            marker="o"
        )
        
        plt.title("Abstraction Error vs. Contamination Strength", fontsize=14)
        plt.xlabel("Contamination Strength (s)", fontsize=12)
        plt.ylabel("Abstraction Error", fontsize=12)
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        return f_spec_df_clean, summary
    else:
        print("No valid data points found after removing NaN values.")
        print("Check if the contamination process is generating valid errors.")
        print(f"Original DataFrame shape: {f_spec_df.shape}")
        print(f"NaN values in error column: {f_spec_df['error'].isna().sum()}")
        return f_spec_df, None

# Usage instructions:
print("""
INSTRUCTIONS FOR FIXING THE NOTEBOOK:

1. The sqrtm issues have been fixed in math_utils.py and modularised_utils.py
2. For the DataFrame aggregation issue, replace Cell 8 with this code:

# After running the contamination loop (Cell 7), run this instead of Cell 8:
from fix_contamination_analysis import fix_dataframe_processing

# Use the fixed processing function
f_spec_df_clean, summary_stats = fix_dataframe_processing(f_spec_records)

# If you want to continue with the Pareto frontier analysis, use f_spec_df_clean:
mean_var_df = f_spec_df_clean.groupby('method')['error'].agg(['mean', 'std']).reset_index()
# ... rest of your analysis code
""") 