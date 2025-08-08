# ======================================================================
# F-MISSPECIFICATION EVALUATION: CONTROLLED NONLINEARITY
# ======================================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import os
import utilities as ut
import modularised_utils as mut

# Set up plotting
sns.set_theme(style="whitegrid")
seed = 42
np.random.seed(seed)

# ======================================================================
# CONFIGURATION
# ======================================================================

experiment = 'slc'
setting = 'empirical'

if setting == 'gaussian':
    path = f"data/{experiment}/results"
elif setting == 'empirical':
    path = f"data/{experiment}/results_empirical"

# ======================================================================
# DATA LOADING
# ======================================================================

print(f"Loading data for experiment: {experiment}")
saved_folds = joblib.load(f"data/{experiment}/cv_folds.pkl")
all_data = ut.load_all_data(experiment)

Dll_samples = all_data['LLmodel']['data']
Dhl_samples = all_data['HLmodel']['data']
I_ll_relevant = all_data['LLmodel']['intervention_set']
omega = all_data['abstraction_data']['omega']
ll_var_names = list(all_data['LLmodel']['graph'].nodes())
hl_var_names = list(all_data['HLmodel']['graph'].nodes())

# ======================================================================
# LOAD RESULTS
# ======================================================================

print("Loading optimization results...")

if setting == 'gaussian':
    diroca_results = joblib.load(f"{path}/diroca_cv_results.pkl")
    gradca_results = joblib.load(f"{path}/gradca_cv_results.pkl")
    baryca_results = joblib.load(f"{path}/baryca_cv_results.pkl")
elif setting == 'empirical':
    diroca_results = joblib.load(f"{path}/diroca_cv_results_empirical.pkl")
    gradca_results = joblib.load(f"{path}/gradca_cv_results_empirical.pkl")
    baryca_results = joblib.load(f"{path}/baryca_cv_results_empirical.pkl")
    abslingam_results = joblib.load(f"{path}/abslingam_cv_results_empirical.pkl")

results_to_evaluate = {}

# Process results (simplified version)
if setting == 'empirical':
    if abslingam_results:
        first_fold_key = list(abslingam_results.keys())[0]
        for style in abslingam_results[first_fold_key].keys():
            method_name = f"Abs-LiNGAM ({style})"
            new_abslingam_dict = {}
            for fold_key, fold_results in abslingam_results.items():
                if style in fold_results:
                    new_abslingam_dict[fold_key] = {style: fold_results[style]}
            results_to_evaluate[method_name] = new_abslingam_dict
    
    # Process DIROCA results
    if diroca_results:
        first_fold_key = list(diroca_results.keys())[0]
        for run_id in diroca_results[first_fold_key].keys():
            method_name = f"DIROCA ({run_id})"
            new_diroca_dict = {}
            for fold_key, fold_results in diroca_results.items():
                if run_id in fold_results:
                    new_diroca_dict[fold_key] = {run_id: fold_results[run_id]}
            results_to_evaluate[method_name] = new_diroca_dict

    results_to_evaluate['GradCA'] = gradca_results
    results_to_evaluate['BARYCA'] = baryca_results

elif setting == 'gaussian':
    results_to_evaluate['GradCA'] = gradca_results
    results_to_evaluate['BARYCA'] = baryca_results
    
    if diroca_results:
        first_fold_key = list(diroca_results.keys())[0]
        for run_id in diroca_results[first_fold_key].keys():
            method_name = f"DIROCA ({run_id})"
            new_diroca_dict = {}
            for fold_key, fold_results in diroca_results.items():
                if run_id in fold_results:
                    new_diroca_dict[fold_key] = {run_id: fold_results[run_id]}
            results_to_evaluate[method_name] = new_diroca_dict

print(f"Loaded {len(results_to_evaluate)} methods for evaluation")

# ======================================================================
# CONTROLLED NONLINEARITY SCM
# ======================================================================

class ControlledNonlinearSCM:
    """A class for simulating data with controlled nonlinearity."""
    
    def __init__(self, causal_graph, linear_coeffs, nonlinearity_strength=0.0, intervention=None):
        self.graph = causal_graph
        self.linear_coeffs = linear_coeffs
        self.nonlinearity_strength = nonlinearity_strength
        self.intervention = intervention
        self.var_names = list(self.graph.nodes())
        self.topological_order = list(nx.topological_sort(self.graph))
        self.intervention_dict = intervention.vv() if intervention else {}
        
    def _create_blended_function(self, child_var, parent_vars):
        """Creates a function that blends linear and nonlinear components."""
        if not parent_vars:
            # Root variable - no parents
            return lambda parent_values, noise: noise
            
        # Get linear coefficients for this child
        linear_weights = []
        for parent in parent_vars:
            coeff = self.linear_coeffs.get((parent, child_var), 0.0)
            linear_weights.append(coeff)
        linear_weights = np.array(linear_weights)
        
        def blended_function(parent_values, noise):
            # Convert parent_values dict to array in correct order
            parent_array = np.column_stack([parent_values[parent] for parent in parent_vars])
            
            # Linear component
            linear_component = parent_array @ linear_weights
            
            # Nonlinear component (sin-based for smoothness)
            nonlinear_component = np.sin(parent_array @ linear_weights)
            
            # Blend them based on nonlinearity strength
            blended = (1 - self.nonlinearity_strength) * linear_component + self.nonlinearity_strength * nonlinear_component
            
            return blended + noise
            
        return blended_function
    
    def simulate(self, exogenous_noise):
        """Simulates data with controlled nonlinearity."""
        n_samples, n_dims = exogenous_noise.shape
        data = np.zeros((n_samples, n_dims))
        
        for var_name in self.topological_order:
            var_pos = self.var_names.index(var_name)
            
            # Handle interventions
            if var_name in self.intervention_dict:
                data[:, var_pos] = self.intervention_dict[var_name]
                continue
            
            # Get parents and create function
            parents = list(self.graph.predecessors(var_name))
            func = self._create_blended_function(var_name, parents)
            
            # Get parent values and noise
            parent_values = {p: data[:, self.var_names.index(p)] for p in parents}
            noise = exogenous_noise[:, var_pos]
            
            # Compute value
            data[:, var_pos] = func(parent_values, noise)
            
        return data

def generate_controlled_nonlinear_data(strength, n_samples, all_data, I_ll_relevant):
    """Generate data with controlled nonlinearity."""
    ll_graph = all_data['LLmodel']['graph']
    ll_coeffs = all_data['LLmodel'].get('coeffs', {})
    ll_mu = all_data['LLmodel']['noise_dist']['mu']
    ll_sigma = all_data['LLmodel']['noise_dist']['sigma']
    
    Dll_samples = {}
    
    for iota in I_ll_relevant:
        # Create SCM with controlled nonlinearity
        scm = ControlledNonlinearSCM(ll_graph, ll_coeffs, strength, iota)
        
        # Generate noise
        noise = np.random.multivariate_normal(mean=ll_mu, cov=ll_sigma, size=n_samples)
        
        # Simulate data
        Dll_samples[iota] = scm.simulate(noise)
    
    return Dll_samples

# ======================================================================
# ERROR CALCULATION FUNCTIONS
# ======================================================================

def calculate_abstraction_error_gaussian(T_matrix, Dll_test, Dhl_test):
    """Calculates error for the GAUSSIAN setting via Wasserstein distance."""
    mu_L_test = np.mean(Dll_test, axis=0)
    Sigma_L_test = np.cov(Dll_test, rowvar=False)
    mu_H_test = np.mean(Dhl_test, axis=0)
    Sigma_H_test = np.cov(Dhl_test, rowvar=False)
    mu_V_predicted = mu_L_test @ T_matrix.T
    Sigma_V_predicted = T_matrix @ Sigma_L_test @ T_matrix.T
    try:
        return np.sqrt(mut.compute_wasserstein(mu_V_predicted, Sigma_V_predicted, mu_H_test, Sigma_H_test))
    except Exception:
        return np.nan

def calculate_empirical_error(T_matrix, Dll_test, Dhl_test, metric='fro'):
    """Calculates error for the EMPIRICAL setting via Frobenius distance."""
    if Dll_test.shape[0] == 0: return np.nan
    try:
        Dhl_predicted = Dll_test @ T_matrix.T
        return mut.compute_empirical_distance(Dhl_predicted.T, Dhl_test.T, metric)
    except Exception:
        return np.nan

# Select the correct error function
if setting == 'gaussian':
    calculate_error = calculate_abstraction_error_gaussian
    print("✓ Using GAUSSIAN error metric (Wasserstein Distance).")
else: # empirical
    calculate_error = calculate_empirical_error
    print("✓ Using EMPIRICAL error metric (Frobenius Distance).")

# ======================================================================
# EVALUATION PARAMETERS
# ======================================================================

# Range of nonlinearity strengths to test (0 = linear, 1 = fully nonlinear)
nonlinearity_strengths = np.linspace(0, 1.0, 11)

# Number of trials for each strength level
num_trials = 5

# Sample size for each trial
n_samples = 500

# ======================================================================
# EVALUATION LOOP
# ======================================================================

f_misspec_records = []
print(f"🚀 Starting F-misspecification (controlled nonlinearity) evaluation...")

for strength in tqdm(nonlinearity_strengths, desc="Nonlinearity Strength"):
    for trial in range(num_trials):
        # Generate new nonlinear data for this trial
        Dll_samples_nl = generate_controlled_nonlinear_data(strength, n_samples, all_data, I_ll_relevant)
        
        # Generate corresponding HL data using the ground truth T matrix
        T_matrix_gt = all_data['abstraction_data']['T']
        Dhl_samples = {eta: Dll_samples_nl[iota] @ T_matrix_gt.T for iota, eta in omega.items()}
        
        # Evaluate each method on this nonlinear data
        for fold_id, fold_info in enumerate(saved_folds):
            for method_name, results_dict in results_to_evaluate.items():
                fold_results = results_dict.get(f'fold_{fold_id}', {})
                for run_key, run_data in fold_results.items():
                    T_learned = run_data['T_matrix']
                    test_indices = run_data['test_indices']
                    
                    errors_per_intervention = []
                    for iota in I_ll_relevant:
                        eta = omega[iota]
                        Dll_test = Dll_samples_nl[iota][test_indices]
                        Dhl_test = Dhl_samples[eta][test_indices]
                        
                        # Calculate error using the learned T on nonlinear data
                        error = calculate_error(T_learned, Dll_test, Dhl_test)
                        if not np.isnan(error):
                            errors_per_intervention.append(error)
                    
                    avg_error = np.mean(errors_per_intervention) if errors_per_intervention else np.nan
                    
                    record = {
                        'method': method_name,
                        'nonlinearity_strength': strength,
                        'trial': trial,
                        'fold': fold_id,
                        'error': avg_error
                    }
                    f_misspec_records.append(record)

# ======================================================================
# RESULTS PROCESSING
# ======================================================================

f_misspec_df = pd.DataFrame(f_misspec_records)
print("\n--- F-Misspecification Evaluation Complete ---")

# Summary statistics
summary = f_misspec_df.groupby(['method', 'nonlinearity_strength'])['error'].agg(['mean', 'std']).reset_index()

# Display results
print("\n" + "="*80)
print("F-MISSPECIFICATION RESULTS: Controlled Nonlinearity")
print("="*80)
print(f"{'Method':<25} | {'Nonlinearity':<12} | {'Mean Error':<12} | {'Std Error':<12}")
print("="*80)

for method in sorted(f_misspec_df['method'].unique()):
    method_data = summary[summary['method'] == method]
    for _, row in method_data.iterrows():
        print(f"{method:<25} | {row['nonlinearity_strength']:<12.2f} | {row['mean']:<12.4f} | {row['std']:<12.4f}")
print("="*80)

# ======================================================================
# VISUALIZATION
# ======================================================================

plt.figure(figsize=(12, 8))

# Plot each method's performance
for method in sorted(f_misspec_df['method'].unique()):
    method_data = summary[summary['method'] == method]
    plt.plot(method_data['nonlinearity_strength'], method_data['mean'], 
             marker='o', linewidth=2, label=method)
    
    # Add error bars
    plt.fill_between(method_data['nonlinearity_strength'], 
                     method_data['mean'] - method_data['std'],
                     method_data['mean'] + method_data['std'], 
                     alpha=0.2)

plt.xlabel('Nonlinearity Strength', fontsize=12)
plt.ylabel('Abstraction Error', fontsize=12)
plt.title('F-Misspecification: Robustness to Controlled Nonlinearity', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save plot
plt.savefig(f'results/f_misspecification_nonlinearity_{experiment}_{setting}.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Save results
f_misspec_df.to_csv(f'results/f_misspecification_nonlinearity_{experiment}_{setting}.csv', index=False)
print(f"\n✓ Results saved to: results/f_misspecification_nonlinearity_{experiment}_{setting}.csv")
print(f"✓ Plot saved to: results/f_misspecification_nonlinearity_{experiment}_{setting}.png")

# ======================================================================
# SUMMARY STATISTICS
# ======================================================================

print("\n" + "="*80)
print("OVERALL PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Method':<25} | {'Mean Error':<12} | {'Std Error':<12}")
print("="*80)

overall_summary = f_misspec_df.groupby('method')['error'].agg(['mean', 'std']).reset_index()
for _, row in overall_summary.sort_values('mean').iterrows():
    print(f"{row['method']:<25} | {row['mean']:<12.4f} | {row['std']:<12.4f}")
print("="*80) 