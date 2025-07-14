import argparse
import os
import joblib
import numpy as np
import yaml
# Import the new NonlinearAddSCM class as well
from models import LinearAddSCM, NonlinearAddSCM, CausalBayesianNetwork, Intervention
import utilities as ut

def generate_and_save(config):
    """
    Generates and saves causal abstraction data based on a config dictionary.
    Handles both linear and non-linear model types.
    """
    # --- 1. Unpack Configuration ---
    experiment = config['experiment_name']
    model_type = config['model_type']
    ll_config = config['low_level_model']
    hl_config = config['high_level_model']
    abs_config = config['abstraction']
    num_llsamples = config['num_llsamples']
    num_hlsamples = config['num_hlsamples']
    
    # --- 2. Low-Level Model Setup & Sampling (MODEL-TYPE-SPECIFIC LOGIC) ---
    print(f"--- Generating data for {model_type} model: {experiment} ---")
    
    # These will be populated by the specific model branch
    Dll_samples, Dll_noise = {}, {}
    ll_causal_graph = None

    # Get shared noise parameters
    ll_mu_hat = np.array(ll_config['noise_params']['mu'])
    ll_Sigma_hat = np.diag(ll_config['noise_params']['sigma_diag'])

    # BRANCH for Linear Models
    if model_type == 'linear_anm':
        ll_coeffs_list = ll_config['coefficients']
        ll_endogenous_coeff_dict = {tuple(item[0]): item[1] for item in ll_coeffs_list}
        ll_causal_graph = CausalBayesianNetwork(list(ll_endogenous_coeff_dict.keys()))
        
        # Helper to create Intervention objects from the config definitions
        def create_intervention(spec):
            if spec is None or spec == 'None': return None
            return Intervention(spec)
        interventions = {name: create_intervention(spec) for name, spec in abs_config.get('interventions', {}).items()}
        omega = {interventions[ll_name]: interventions[hl_name] for ll_name, hl_name in abs_config.get('omega_map', {}).items()}
        Ill_relevant = list(set(omega.keys()))

        for iota in Ill_relevant:
            llcm = LinearAddSCM(ll_causal_graph, ll_endogenous_coeff_dict, iota)
            noise = np.random.multivariate_normal(mean=ll_mu_hat, cov=ll_Sigma_hat, size=num_llsamples)
            Dll_noise[iota] = noise
            Dll_samples[iota] = llcm.simulate(Dll_noise[iota])
        print("✓ Linear low-level sampling complete.")


    # BRANCH for Non-Linear Models (NEW)
    elif model_type == 'continuous_nonlinear_anm':
        # For non-linear models, the graph structure must be defined explicitly
        ll_causal_graph = CausalBayesianNetwork(ll_config['graph_edges'])

        # WARNING: eval() is powerful but can be a security risk if the config 
        # is not from a trusted source. It is safe for your use case here.
        safe_eval_scope = {'np': np} 
        functions = {var: eval(func_str, safe_eval_scope) for var, func_str in ll_config['structural_functions'].items()}

        # Create interventions (same logic as linear)
        def create_intervention(spec):
            if spec is None or spec == 'None': return None
            return Intervention(spec)
        interventions = {name: create_intervention(spec) for name, spec in abs_config.get('interventions', {}).items()}
        omega = {interventions[ll_name]: interventions[hl_name] for ll_name, hl_name in abs_config.get('omega_map', {}).items()}
        Ill_relevant = list(set(omega.keys()))

        for iota in Ill_relevant:
            # Use the new NonlinearAddSCM class
            llcm = NonlinearAddSCM(ll_causal_graph, functions, iota)
            noise = np.random.multivariate_normal(mean=ll_mu_hat, cov=ll_Sigma_hat, size=num_llsamples)
            Dll_noise[iota] = noise
            Dll_samples[iota] = llcm.simulate(Dll_noise[iota])
        print("✓ Non-linear low-level sampling complete.")
    
    else:
        raise ValueError(f"Unknown model_type in config: {model_type}")

    # --- 3. Abstraction & High-Level Model Inference (SHARED LOGIC) ---
    # This logic runs for BOTH model types, as it only needs the generated data.
    hl_initial_coeff_dict = {tuple(item[0]): item[1] for item in hl_config['initial_coefficients']}
    hl_causal_graph = CausalBayesianNetwork(list(hl_initial_coeff_dict.keys()))
    
    # Re-fetch interventions and omega map for the HL part
    def create_intervention(spec):
        if spec is None or spec == 'None': return None
        return Intervention(spec)
    
    interventions = {name: create_intervention(spec) for name, spec in abs_config.get('interventions', {}).items()}
    omega = {interventions[ll_name]: interventions[hl_name] for ll_name, hl_name in abs_config.get('omega_map', {}).items()}
    Ill_relevant = list(set(omega.keys()))
    Ihl_relevant = list(set(omega.values()))

    T = np.array(abs_config['T_matrix'])
    data_observational_hl = Dll_samples[None] @ T.T
    
    hl_endogenous_coeff_dict, U_hl = ut.get_coefficients(data_observational_hl, hl_causal_graph, return_noise=True)
    hl_mu_hat = np.mean(U_hl, axis=0)
    hl_Sigma_hat = np.diag(np.var(U_hl, axis=0))
    print("✓ High-level model inferred.")

    # --- 4. High-Level Sampling (SHARED LOGIC) ---
    Dhl_samples, Dhl_noise = {}, {}
    for eta in Ihl_relevant:
        if eta is not None:
            hlcm = LinearAddSCM(hl_causal_graph, hl_endogenous_coeff_dict, eta)
            Dhl_noise[eta] = np.random.multivariate_normal(mean=hl_mu_hat, cov=hl_Sigma_hat, size=num_hlsamples)
            Dhl_samples[eta] = hlcm.simulate(Dhl_noise[eta])
        else:
            Dhl_noise[eta] = U_hl
            Dhl_samples[eta] = data_observational_hl
    print("✓ High-level sampling complete.")

    # --- 5. Save the Data (SHARED LOGIC) ---
    if model_type == 'linear_anm':
        LLmodels = {iota: LinearAddSCM(ll_causal_graph, ll_endogenous_coeff_dict, iota) for iota in Ill_relevant}
        HLmodels = {eta: LinearAddSCM(hl_causal_graph, hl_endogenous_coeff_dict, eta) for eta in Ihl_relevant}
    elif model_type == 'continuous_nonlinear_anm':
        LLmodels = {iota: NonlinearAddSCM(ll_causal_graph, functions, iota) for iota in Ill_relevant}
        HLmodels = {eta: LinearAddSCM(hl_causal_graph, hl_endogenous_coeff_dict, eta) for eta in Ihl_relevant}

    LLmodel = {'graph': ll_causal_graph, 'intervention_set': Ill_relevant,
               'noise_dist': {'mu': ll_mu_hat, 'sigma': ll_Sigma_hat}, 'data': Dll_samples,
               'scm_instances': LLmodels, 'noise': Dll_noise}
    if model_type == 'linear_anm':
        LLmodel['coeffs'] = {tuple(item[0]): item[1] for item in ll_config['coefficients']}
    elif model_type == 'continuous_nonlinear_anm':
        LLmodel['functions'] = ll_config['structural_functions']

    HLmodel = {'graph': hl_causal_graph, 'intervention_set': Ihl_relevant, 'coeffs': hl_endogenous_coeff_dict,
               'noise_dist': {'mu': hl_mu_hat, 'sigma': hl_Sigma_hat}, 'data': Dhl_samples,
               'scm_instances': HLmodels, 'noise': Dhl_noise}
    abstraction_data = {'T': T, 'omega': omega}
    
    path = f"data/{experiment}"
    os.makedirs(path, exist_ok=True)
    
    joblib.dump(LLmodel, f"{path}/LLmodel.pkl")
    joblib.dump(HLmodel, f"{path}/HLmodel.pkl")
    joblib.dump(abstraction_data, f"{path}/abstraction_data.pkl")
    print(f"✓ Data saved successfully to {path}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save data for a causal abstraction experiment.")
    parser.add_argument('config_path', type=str, help="Path to the experiment's YAML configuration file.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        print(f"Error: Configuration file '{args.config_path}' is empty or invalid.")
        exit(1)

    generate_and_save(config)