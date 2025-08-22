import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
import random
import re
import utilities as ut
import modularised_utils as mut
import networkx as nx
from scipy.linalg import sqrtm

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

sns.set_theme(style="whitegrid")
seed = 42
np.random.seed(seed)

def robust_compute_wasserstein(mu1, cov1, mu2, cov2):
    """
    Robust version of Wasserstein distance computation that handles numerical issues.
    """
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    try:
        # Ensure matrices are symmetric and positive semi-definite
        cov1_sym = 0.5 * (cov1 + cov1.T)
        cov2_sym = 0.5 * (cov2 + cov2.T)
        
        # Add small regularization to ensure positive definiteness
        eps = 1e-10
        cov1_reg = cov1_sym + eps * np.eye(cov1_sym.shape[0])
        cov2_reg = cov2_sym + eps * np.eye(cov2_sym.shape[0])
        
        # Compute sqrtm with error handling
        sqrt_cov1 = sqrtm(cov1_reg)
        sqrt_cov2 = sqrtm(cov2_reg)
        
        # Ensure real values
        if np.iscomplexobj(sqrt_cov1):
            sqrt_cov1 = sqrt_cov1.real
        if np.iscomplexobj(sqrt_cov2):
            sqrt_cov2 = sqrt_cov2.real
            
        # Compute the trace term
        inner_matrix = sqrt_cov1 @ cov2_reg @ sqrt_cov1
        inner_sqrt = sqrtm(inner_matrix)
        
        if np.iscomplexobj(inner_sqrt):
            inner_sqrt = inner_sqrt.real
            
        cov_term = np.trace(cov1_reg + cov2_reg - 2 * inner_sqrt)
        
    except Exception as e:
        # Fallback: use simpler approximation
        print(f"Failed to find a square root: {e}. Using fallback method.")
        try:
            cov_term = np.trace(cov1 + cov2)
        except:
            cov_term = 0.0
    
    return mean_diff + cov_term

def calculate_abstraction_error_robust(T_matrix, Dll_test, Dhl_test):
    """
    Robust version of abstraction error calculation.
    """
    # 1. Estimate parameters from the low-level test data
    mu_L_test = np.mean(Dll_test, axis=0)
    Sigma_L_test = np.cov(Dll_test, rowvar=False)

    # 2. Estimate parameters from the high-level test data
    mu_H_test = np.mean(Dhl_test, axis=0)
    Sigma_H_test = np.cov(Dhl_test, rowvar=False)

    # 3. Transform the low-level parameters using the T matrix
    mu_V_predicted = mu_L_test @ T_matrix.T
    Sigma_V_predicted = T_matrix @ Sigma_L_test @ T_matrix.T
    
    # 4. Compute the Wasserstein distance between the two resulting Gaussians
    try:
        wasserstein_dist = np.sqrt(robust_compute_wasserstein(mu_V_predicted, Sigma_V_predicted, mu_H_test, Sigma_H_test))
    except Exception as e:
        print(f"  - Warning: Could not compute Wasserstein distance. Error: {e}. Returning NaN.")
        return np.nan

    return wasserstein_dist

def apply_structural_contamination_flexible(
    linear_data,
    graph,
    coeffs,
    noise,
    strength,
    scaled=True,
    nonlinear_func=np.sin,
    reuse=False,
    scm_instance=None
):
    """
    Applies structural contamination to SCM data with exact linear simulation.
    """
    if reuse:
        linear_part = linear_data.copy()
    else:
        if scm_instance is None:
            raise ValueError("You must provide `scm_instance` when reuse=False.")
        linear_part = scm_instance.simulate(noise)

    topo_order = list(nx.topological_sort(graph))
    var_index = {var: idx for idx, var in enumerate(topo_order)}
    nonlinear_part = np.zeros_like(linear_part)

    # Compute nonlinear effect using *true parent values from linear SCM output*
    for var in topo_order:
        var_idx = var_index[var]
        parents = list(graph.predecessors(var))

        if not parents:
            continue
        else:
            parent_indices = [var_index[p] for p in parents]
            parent_vals = linear_part[:, parent_indices] 
            nonlinear_effect = nonlinear_func(parent_vals).sum(axis=1)
            nonlinear_part[:, var_idx] = nonlinear_effect

    if scaled:
        contaminated = (1 - strength) * linear_part + strength * nonlinear_part
    else:
        contaminated = linear_part + strength * nonlinear_part

    return contaminated

def run_contamination_analysis():
    """
    Main function to run the contamination analysis with fixes.
    """
    # Load data
    experiment = 'lilucas'
    setting = 'gaussian'
    
    if setting == 'gaussian':
        path = f"data/{experiment}/results"
    elif setting == 'empirical':
        path = f"data/{experiment}/results_empirical"

    saved_folds = joblib.load(f"data/{experiment}/cv_folds.pkl")
    all_data = ut.load_all_data(experiment)

    Dll_samples = all_data['LLmodel']['data']
    Dhl_samples = all_data['HLmodel']['data']
    LLmodel = all_data['LLmodel']
    HLmodel = all_data['HLmodel']
    ll_graph = all_data['LLmodel']['graph']
    hl_graph = all_data['HLmodel']['graph']
    T_matrix_gt = all_data['abstraction_data']['T']
    I_ll_relevant = all_data['LLmodel']['intervention_set']
    omega = all_data['abstraction_data']['omega']

    # Load results
    if setting == 'gaussian':
        diroca_results = joblib.load(f"{path}/diroca_cv_results.pkl")
        gradca_results = joblib.load(f"{path}/gradca_cv_results.pkl")
        baryca_results = joblib.load(f"{path}/baryca_cv_results.pkl")
    elif setting == 'empirical':
        diroca_results = joblib.load(f"{path}/diroca_cv_results_empirical.pkl")
        gradca_results = joblib.load(f"{path}/gradca_cv_results_empirical.pkl")
        baryca_results = joblib.load(f"{path}/baryca_cv_results_empirical.pkl")

    results_to_evaluate = {}
    results_to_evaluate['GradCA'] = gradca_results
    results_to_evaluate['BARYCA'] = baryca_results

    if diroca_results:
        first_fold_key = list(diroca_results.keys())[0]
        diroca_run_ids = list(diroca_results[first_fold_key].keys())

        for run_id in diroca_run_ids:
            method_name = f"DIROCA ({run_id})"
            new_diroca_dict = {}
            for fold_key, fold_results in diroca_results.items():
                if run_id in fold_results:
                    new_diroca_dict[fold_key] = {run_id: fold_results[run_id]}
            results_to_evaluate[method_name] = new_diroca_dict

    # Label mapping
    label_map_gaussian = {
        'DIROCA (eps_delta_0.111)': 'DiRoCA_star',
        'DIROCA (eps_delta_1)': 'DIROCA_1',
        'DIROCA (eps_delta_2)': 'DIROCA_2',
        'DIROCA (eps_delta_4)': 'DIROCA_4',
        'DIROCA (eps_delta_8)': 'DIROCA_8',
        'GradCA': 'GradCA',
        'BARYCA': 'BARYCA'
    }

    if setting == 'gaussian':
        results_to_evaluate = {label_map_gaussian.get(key, key): value for key, value in results_to_evaluate.items()}

    print("\nMethods available for evaluation:")
    for key in results_to_evaluate.keys():
        print(f"  - {key}")

    # Run contamination analysis
    contamination_strengths = np.linspace(0, 1.0, 10)
    num_trials = 1
    nonlinear_func = np.tanh
    scaled = True
    reuse = False

    f_spec_records = []

    for strength in tqdm(contamination_strengths, desc="Contamination Strength"):
        for trial in range(num_trials):
            for i, fold_info in enumerate(saved_folds):
                for method_name, results_dict in results_to_evaluate.items():
                    fold_results = results_dict.get(f'fold_{i}', {})
                    for run_key, run_data in fold_results.items():
                        T_learned = run_data['T_matrix']
                        test_indices = run_data['test_indices']

                        errors_per_intervention = []

                        for iota in I_ll_relevant:
                            # Prepare inputs
                            Dll_clean = Dll_samples[iota][test_indices]
                            Dhl_clean = Dhl_samples[omega[iota]][test_indices]

                            noise_ll = LLmodel['noise'][iota][test_indices]
                            noise_hl = HLmodel['noise'][omega[iota]][test_indices]

                            scm_ll = LLmodel['scm_instances'][iota]
                            scm_hl = HLmodel['scm_instances'][omega[iota]]

                            # Apply contamination
                            Dll_cont = apply_structural_contamination_flexible(
                                linear_data=Dll_clean,
                                graph=ll_graph,
                                coeffs=LLmodel['coeffs'],
                                noise=noise_ll,
                                strength=strength,
                                scaled=scaled,
                                nonlinear_func=nonlinear_func,
                                reuse=reuse,
                                scm_instance=scm_ll
                            )

                            Dhl_cont = apply_structural_contamination_flexible(
                                linear_data=Dhl_clean,
                                graph=hl_graph,
                                coeffs=HLmodel['coeffs'],
                                noise=noise_hl,
                                strength=strength,
                                scaled=scaled,
                                nonlinear_func=nonlinear_func,
                                reuse=reuse,
                                scm_instance=scm_hl
                            )

                            # Compute abstraction error using robust method
                            if setting == 'gaussian':
                                error = calculate_abstraction_error_robust(T_learned, Dll_cont, Dhl_cont)
                            elif setting == 'empirical':
                                error = ut.calculate_empirical_error(T_learned, Dll_cont, Dhl_cont)
                            else:
                                raise ValueError(f"Unknown setting: {setting}")

                            if not np.isnan(error):
                                errors_per_intervention.append(error)

                        avg_error = np.mean(errors_per_intervention) if errors_per_intervention else np.nan
                        f_spec_records.append({
                            'method': method_name,
                            'contamination': strength,
                            'trial': trial,
                            'fold': i,
                            'error': avg_error
                        })

    # Process results with proper NaN handling
    f_spec_df = pd.DataFrame(f_spec_records)
    f_spec_df_clean = f_spec_df.dropna(subset=['error'])

    print("\n--- F-Misspecification Evaluation Complete (Fixed) ---")
    print("="*65)
    print("Overall Performance (Averaged Across All Nonlinearity Strengths)")
    print("="*65)
    print(f"{'Method/Run':<35} | {'Mean ± Std'}")
    print("="*65)

    if len(f_spec_df_clean) > 0:
        summary = f_spec_df_clean.groupby('method')['error'].agg(['mean', 'std', 'count'])
        summary['sem'] = summary['std']
        summary['ci95'] = summary['sem']

        for method_name, row in summary.sort_values('mean').iterrows():
            print(f"{method_name:<35} | {row['mean']:.4f} ± {row['ci95']:.4f}")

        # Plot results
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=f_spec_df_clean,
            x="contamination",
            y="error",
            hue="method",
            estimator="mean",
            errorbar='sd',
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
        print(f"Original DataFrame shape: {f_spec_df.shape}")
        print(f"NaN values in error column: {f_spec_df['error'].isna().sum()}")
        return f_spec_df, None

if __name__ == "__main__":
    results_df, summary_stats = run_contamination_analysis() 