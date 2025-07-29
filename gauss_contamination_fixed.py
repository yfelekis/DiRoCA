# Load dictionaries containing the results for each optimization method
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

def create_diroca_label(run_id):
    """Parses a run_id and creates a simplified label if epsilon and delta are equal."""
    # Use regular expression to find numbers for epsilon and delta
    matches = re.findall(r'(\d+\.?\d*)', run_id)
    if len(matches) == 2:
        eps, delta = matches
        # If they are the same, use the simplified format
        if eps == delta:
            # Handle integer conversion for clean labels like '1' instead of '1.0'
            val = int(float(eps)) if float(eps).is_integer() else float(eps)
            return f"DIROCA (eps_delta_{val})"
    # Otherwise, or if parsing fails, use the full original name
    return f"DIROCA ({run_id})"

# Add baseline methods
results_to_evaluate['GradCA'] = gradca_results
results_to_evaluate['BARYCA'] = baryca_results

# Add Abs-LiNGAM methods if empirical setting
if setting == 'empirical' and abslingam_results:
    first_fold_key = list(abslingam_results.keys())[0]
    for style in abslingam_results[first_fold_key].keys():
        method_name = f"Abs-LiNGAM ({style})"
        new_abslingam_dict = {}
        for fold_key, fold_results in abslingam_results.items():
            if style in fold_results:
                new_abslingam_dict[fold_key] = {style: fold_results[style]}
        results_to_evaluate[method_name] = new_abslingam_dict

# Add DIROCA methods with clean labels
if diroca_results:
    first_fold_key = list(diroca_results.keys())[0]
    for run_id in diroca_results[first_fold_key].keys():
        method_name = create_diroca_label(run_id)  # Use the helper to create the name
        new_diroca_dict = {}
        for fold_key, fold_results in diroca_results.items():
            if run_id in fold_results:
                new_diroca_dict[fold_key] = {run_id: fold_results[run_id]}
        results_to_evaluate[method_name] = new_diroca_dict

# Define label mappings for final renaming
label_map_gaussian = {
    'DIROCA (eps_delta_0.111)': 'DiRoCA_star',
    'DIROCA (eps_delta_1)': 'DIROCA_1',
    'DIROCA (eps_delta_2)': 'DIROCA_2',
    'DIROCA (eps_delta_4)': 'DIROCA_4',
    'DIROCA (eps_delta_8)': 'DIROCA_8',
    'GradCA': 'GradCA',
    'BARYCA': 'BARYCA'
}

label_map_empirical = {
    'DIROCA (eps_0.328_delta_0.107)': 'DiRoCA_star',
    'DIROCA (eps_delta_1)': 'DIROCA_1',
    'DIROCA (eps_delta_2)': 'DIROCA_2',
    'DIROCA (eps_delta_4)': 'DIROCA_4',
    'DIROCA (eps_delta_8)': 'DIROCA_8',
    'GradCA': 'GradCA',
    'BARYCA': 'BARYCA',
    'Abs-LiNGAM (Perfect)': 'Abslin_p',
    'Abs-LiNGAM (Noisy)': 'Abslin_n'
}

# Apply final label mapping
if setting == 'empirical':
    results_to_evaluate = {label_map_empirical.get(key, key): value for key, value in results_to_evaluate.items()}
elif setting == 'gaussian':
    results_to_evaluate = {label_map_gaussian.get(key, key): value for key, value in results_to_evaluate.items()}

print("\nMethods available for evaluation:")
for key in results_to_evaluate.keys():
    print(f"  - {key}") 