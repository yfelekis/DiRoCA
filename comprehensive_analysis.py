#!/usr/bin/env python3
"""
Comprehensive Analysis Script for DiRoCA TBS

This script automatically processes all available experiments and generates
organized outputs (tables, plots, animations) that can be easily selected
for inclusion in papers.

Usage:
    python comprehensive_analysis.py

Outputs:
    - Organized tables for each experiment
    - Plots and animations for each experiment
    - Summary dashboard
    - LaTeX-ready tables
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
import random
import glob
import os
import re
from datetime import datetime
import json

import utilities as ut
import modularised_utils as mut
from load_results import load_results, list_available_results



# Set LaTeX font for all plots
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

sns.set_theme(style="whitegrid")
seed = 42
np.random.seed(seed)

class ComprehensiveAnalyzer:
    def __init__(self, output_dir="analysis_outputs"):
        """Initialize the comprehensive analyzer."""
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.experiments = self.discover_experiments()
        self.results_summary = {}
        
    def ensure_output_dir(self):
        """Create output directory structure."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/tables", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/latex", exist_ok=True)
        
    def discover_experiments(self):
        """Discover all available experiments and their results."""
        experiments = {}
        
        # Find all experiment directories
        for exp_dir in glob.glob("data/*/"):
            exp_name = os.path.basename(os.path.dirname(exp_dir))
            if exp_name in ['.ipynb_checkpoints']:
                continue
                
            experiments[exp_name] = {
                'name': exp_name,
                'data_path': exp_dir,
                'results': self.find_experiment_results(exp_name)
            }
            
        return experiments
    
    def find_experiment_results(self, experiment):
        """Find all result files for an experiment."""
        results = {
            'gaussian': [],
            'empirical': []
        }
        
        # Check for gaussian evaluation results
        gaussian_pattern = f"data/{experiment}/evaluation_results/evaluation_*.csv"
        gaussian_files = glob.glob(gaussian_pattern)
        results['gaussian'] = gaussian_files
        
        # Check for empirical evaluation results
        empirical_pattern = f"data/{experiment}/evaluation_results/empirical_evaluation_*.csv"
        empirical_files = glob.glob(empirical_pattern)
        results['empirical'] = empirical_files
        
        return results
    
    def analyze_experiment(self, experiment_name, evaluation_type='gaussian'):
        """Analyze a single experiment."""
        print(f"\n{'='*60}")
        print(f"Analyzing {experiment_name.upper()} - {evaluation_type.upper()} evaluation")
        print(f"{'='*60}")
        
        # Load the latest results for this experiment
        try:
            if evaluation_type == 'gaussian':
                df = load_results(experiment=experiment_name, latest=True)
            else:
                df = load_results(experiment=experiment_name, evaluation_type='empirical', latest=True)
                
            if df is None:
                print(f"No results found for {experiment_name} - {evaluation_type}")
                return None
                
        except Exception as e:
            print(f"Error loading results for {experiment_name}: {e}")
            return None
            
        # Create experiment-specific output directory
        exp_output_dir = f"{self.output_dir}/{experiment_name}_{evaluation_type}"
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Run all analyses
        analysis_results = {
            'experiment': experiment_name,
            'evaluation_type': evaluation_type,
            'dataframe': df,
            'output_dir': exp_output_dir
        }
        
        # 1. Generate summary tables
        analysis_results['tables'] = self.generate_summary_tables(df, exp_output_dir)
        
        # 2. Generate plots
        analysis_results['plots'] = self.generate_plots(df, exp_output_dir)
        
        # 3. Generate LaTeX tables
        analysis_results['latex'] = self.generate_latex_tables(df, exp_output_dir)
        
        return analysis_results
    
    def generate_summary_tables(self, df, output_dir):
        """Generate summary tables for different scenarios."""
        tables = {}
        
        # 0-shift scenario
        df_clean = df[df['alpha'] == 0.0]
        if not df_clean.empty:
            summary_stats = df_clean.groupby(['method'])['error'].agg(['mean', 'std', 'count'])
            summary_stats['sem'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
            tables['zero_shift'] = summary_stats
            
            # Save to CSV
            summary_stats.to_csv(f"{output_dir}/zero_shift_summary.csv")
            
            # Print formatted table
            self.print_formatted_table(summary_stats, "Zero-Shift Scenario", output_dir)
        
        # Point comparison (alpha=1.0, noise=2.5)
        alpha_point = 1.0
        noise_level_point = 2.5
        df_point = df[(df['alpha'] == alpha_point) & (df['noise_scale'] == noise_level_point)]
        if not df_point.empty:
            summary_stats = df_point.groupby(['method'])['error'].agg(['mean', 'std', 'count'])
            summary_stats['sem'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
            tables['point_comparison'] = summary_stats
            
            # Save to CSV
            summary_stats.to_csv(f"{output_dir}/point_comparison_summary.csv")
            
            # Print formatted table
            self.print_formatted_table(summary_stats, f"Point Comparison (α={alpha_point}, noise={noise_level_point})", output_dir)
        
        return tables
    
    def print_formatted_table(self, summary_stats, title, output_dir):
        """Print and save a formatted table."""
        print(f"\n--- {title} ---")
        print("="*65)
        print(f"{'Method':<45} | {'Mean Error ± SEM'}")
        print("="*65)
        
        table_content = []
        table_content.append(f"--- {title} ---")
        table_content.append("="*65)
        table_content.append(f"{'Method':<45} | {'Mean Error ± SEM'}")
        table_content.append("="*65)
        
        for method_name, row in summary_stats.iterrows():
            mean_val = row['mean']
            sem_val = row['sem']
            print(f"{method_name:<45} | {mean_val:>7.4f} ± {sem_val:.4f}")
            table_content.append(f"{method_name:<45} | {mean_val:>7.4f} ± {sem_val:.4f}")
        
        print("="*65)
        table_content.append("="*65)
        
        # Save to text file
        with open(f"{output_dir}/{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('α', 'alpha')}.txt", 'w') as f:
            f.write('\n'.join(table_content))
    
    def generate_plots(self, df, output_dir):
        """Generate various plots for the experiment."""
        plots = {}
        
        # Define label mapping for LaTeX
        label_map = {
            'DIROCA (eps_delta_0.111)': r'DiRoCA$_{\epsilon_\ell^*, \epsilon_h^*}$',
            'DIROCA (eps_delta_1)':     r'DiRoCA$_{1,1}$',
            'DIROCA (eps_delta_2)':     r'DiRoCA$_{2,2}$',
            'DIROCA (eps_delta_4)':     r'DiRoCA$_{4,4}$',
            'DIROCA (eps_delta_8)':     r'DiRoCA$_{8,8}$',
            'GradCA':                   r'GRAD$_{(\tau, \omega)}$',
            'BARYCA':                   r'GRAD$_{\text{bary}}$',
            'Abs-LiNGAM (Perfect)':     r'AbsLin$_{\text{p}}$', 
            'Abs-LiNGAM (Noisy)':       r'AbsLin$_{\text{n}}$'
        }
        
        df_for_plotting = df.copy()
        df_for_plotting['display_name'] = df_for_plotting['method'].map(label_map)
        
        # 1. Fix alpha plot
        alpha_to_plot = 1.0
        df_subset = df_for_plotting[df_for_plotting['alpha'] == alpha_to_plot]
        if not df_subset.empty:
            plt.figure(figsize=(14, 8))
            sns.lineplot(
                data=df_subset,
                x='noise_scale',
                y='error',
                hue='display_name',
                marker='o',
                linewidth=2,
                errorbar=('ci', 95)
            )
            plt.title(f'Robustness to Noise Intensity (α = {alpha_to_plot})', fontsize=18)
            plt.xlabel('Contamination Strength (Noise Scale)', fontsize=14)
            plt.ylabel('Average Abstraction Error', fontsize=14)
            plt.legend(title='Method', fontsize=12)
            plt.grid(True, which='both', linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/robustness_vs_noise_alpha_{alpha_to_plot}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_dir}/robustness_vs_noise_alpha_{alpha_to_plot}.png", dpi=300, bbox_inches='tight')
            plt.close()
            plots['robustness_vs_noise'] = f"robustness_vs_noise_alpha_{alpha_to_plot}.pdf"
        
        # 2. Fix noise plot
        noise_level_to_plot = 2.5
        df_subset = df_for_plotting[df_for_plotting['noise_scale'] == noise_level_to_plot]
        if not df_subset.empty:
            plt.figure(figsize=(14, 8))
            sns.lineplot(
                data=df_subset,
                x='alpha',
                y='error',
                hue='display_name',
                marker='o',
                linewidth=2,
                errorbar=('ci', 95)
            )
            plt.title(f'Robustness to Outlier Fraction (Noise Scale = {noise_level_to_plot})', fontsize=18)
            plt.xlabel('Contamination Fraction (α)', fontsize=14)
            plt.ylabel('Average Abstraction Error', fontsize=14)
            plt.legend(title='Method', fontsize=12)
            plt.grid(True, which='both', linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/robustness_vs_alpha_noise_{noise_level_to_plot}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_dir}/robustness_vs_alpha_noise_{noise_level_to_plot}.png", dpi=300, bbox_inches='tight')
            plt.close()
            plots['robustness_vs_alpha'] = f"robustness_vs_alpha_noise_{noise_level_to_plot}.pdf"
        
        return plots
    

    
    def generate_latex_tables(self, df, output_dir):
        """Generate LaTeX-ready tables."""
        latex_tables = {}
        
        # 0-shift scenario
        df_clean = df[df['alpha'] == 0.0]
        if not df_clean.empty:
            summary_stats = df_clean.groupby(['method'])['error'].agg(['mean', 'std', 'count'])
            summary_stats['sem'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
            
            latex_content = self.create_latex_table(summary_stats, "Zero-Shift Scenario")
            with open(f"{output_dir}/zero_shift_latex.tex", 'w') as f:
                f.write(latex_content)
            latex_tables['zero_shift'] = 'zero_shift_latex.tex'
        
        # Point comparison
        alpha_point = 1.0
        noise_level_point = 2.5
        df_point = df[(df['alpha'] == alpha_point) & (df['noise_scale'] == noise_level_point)]
        if not df_point.empty:
            summary_stats = df_point.groupby(['method'])['error'].agg(['mean', 'std', 'count'])
            summary_stats['sem'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
            
            latex_content = self.create_latex_table(summary_stats, f"Point Comparison (α={alpha_point}, noise={noise_level_point})")
            with open(f"{output_dir}/point_comparison_latex.tex", 'w') as f:
                f.write(latex_content)
            latex_tables['point_comparison'] = 'point_comparison_latex.tex'
        
        return latex_tables
    
    def create_latex_table(self, summary_stats, title):
        """Create a LaTeX table from summary statistics."""
        latex_content = []
        latex_content.append(r"\begin{table}[h]")
        latex_content.append(r"\centering")
        latex_content.append(r"\caption{" + title + r"}")
        latex_content.append(r"\begin{tabular}{lcc}")
        latex_content.append(r"\hline")
        latex_content.append(r"Method & Mean Error & SEM \\")
        latex_content.append(r"\hline")
        
        for method_name, row in summary_stats.iterrows():
            mean_val = row['mean']
            sem_val = row['sem']
            # Clean method name for LaTeX
            clean_method = method_name.replace('_', r'\_')
            latex_content.append(f"{clean_method} & {mean_val:.4f} & {sem_val:.4f} \\\\")
        
        latex_content.append(r"\hline")
        latex_content.append(r"\end{tabular}")
        latex_content.append(r"\end{table}")
        
        return '\n'.join(latex_content)
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis for all experiments."""
        print("Starting Comprehensive Analysis")
        print("="*60)
        
        all_results = {}
        
        for exp_name, exp_info in self.experiments.items():
            print(f"\nProcessing experiment: {exp_name}")
            
            exp_results = {}
            
            # Analyze gaussian evaluation if available
            if exp_info['results']['gaussian']:
                gaussian_results = self.analyze_experiment(exp_name, 'gaussian')
                if gaussian_results:
                    exp_results['gaussian'] = gaussian_results
            
            # Analyze empirical evaluation if available
            if exp_info['results']['empirical']:
                empirical_results = self.analyze_experiment(exp_name, 'empirical')
                if empirical_results:
                    exp_results['empirical'] = empirical_results
            
            if exp_results:
                all_results[exp_name] = exp_results
        
        # Generate summary dashboard
        self.generate_summary_dashboard(all_results)
        
        # Save analysis summary
        self.save_analysis_summary(all_results)
        
        return all_results
    
    def generate_summary_dashboard(self, all_results):
        """Generate a summary dashboard of all experiments."""
        dashboard_content = []
        dashboard_content.append("# Comprehensive Analysis Dashboard")
        dashboard_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dashboard_content.append("")
        
        for exp_name, exp_results in all_results.items():
            dashboard_content.append(f"## {exp_name.upper()}")
            dashboard_content.append("")
            
            for eval_type, results in exp_results.items():
                dashboard_content.append(f"### {eval_type.upper()} Evaluation")
                dashboard_content.append("")
                
                if 'tables' in results:
                    dashboard_content.append("#### Summary Tables:")
                    for table_name, table_data in results['tables'].items():
                        dashboard_content.append(f"- `{table_name}`: {results['output_dir']}/{table_name}_summary.csv")
                    dashboard_content.append("")
                
                if 'plots' in results:
                    dashboard_content.append("#### Generated Plots:")
                    for plot_name, plot_file in results['plots'].items():
                        dashboard_content.append(f"- `{plot_name}`: {results['output_dir']}/{plot_file}")
                    dashboard_content.append("")
                

                
                if 'latex' in results:
                    dashboard_content.append("#### LaTeX Tables:")
                    for latex_name, latex_file in results['latex'].items():
                        dashboard_content.append(f"- `{latex_name}`: {results['output_dir']}/{latex_file}")
                    dashboard_content.append("")
        
        # Save dashboard
        with open(f"{self.output_dir}/analysis_dashboard.md", 'w') as f:
            f.write('\n'.join(dashboard_content))
        
        print(f"\nDashboard saved to: {self.output_dir}/analysis_dashboard.md")
    
    def save_analysis_summary(self, all_results):
        """Save a JSON summary of the analysis."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(all_results),
            'experiments': {}
        }
        
        for exp_name, exp_results in all_results.items():
            summary['experiments'][exp_name] = {
                'evaluation_types': list(exp_results.keys()),
                'output_files': {}
            }
            
            for eval_type, results in exp_results.items():
                summary['experiments'][exp_name]['output_files'][eval_type] = {
                    'tables': list(results.get('tables', {}).keys()),
                    'plots': list(results.get('plots', {}).keys()),
                    'latex_tables': list(results.get('latex', {}).keys())
                }
        
        with open(f"{self.output_dir}/analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis summary saved to: {self.output_dir}/analysis_summary.json")

def main():
    """Main function to run the comprehensive analysis."""
    analyzer = ComprehensiveAnalyzer()
    
    print("DiRoCA TBS Comprehensive Analysis")
    print("="*60)
    print(f"Found {len(analyzer.experiments)} experiments:")
    for exp_name, exp_info in analyzer.experiments.items():
        print(f"  - {exp_name}: {len(exp_info['results']['gaussian'])} gaussian, {len(exp_info['results']['empirical'])} empirical results")
    
    print(f"\nOutput directory: {analyzer.output_dir}")
    
    # Run the analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Check the '{analyzer.output_dir}' directory for all outputs.")
    print("Use the analysis_dashboard.md file to navigate the results.")

if __name__ == "__main__":
    main() 