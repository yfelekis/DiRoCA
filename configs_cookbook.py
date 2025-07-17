# ===================================================================
# CONFIGURATION COOKBOOK
# ===================================================================

# --- 1. Simple Shifts ---

# Add a constant value of 0.5 to all data points
config_translation = {
    'type': 'translation',
    'c': 0.5 
}

# Scale all data points by 150%
config_scaling = {
    'type': 'scaling',
    'c': 1.5
}

# Apply a random rotation to the data
config_rotation = {
    'type': 'rotation'
}


# --- 2. Additive Gaussian Noise ---

# Add N(0, I) noise to ALL variables
config_add_gauss_all = {
    'type': 'additive',
    'distribution': 'gaussian',
    'll_params': {'mu': [0, 0, 0], 'sigma': [1, 1, 1]},
    'hl_params': {'mu': [0, 0], 'sigma': [1, 1]}
}

# Add correlated noise to a SUBSET of variables
config_add_gauss_selective = {
    'type': 'additive',
    'distribution': 'gaussian',
    'll_params': {
        'apply_to_vars': ['Smoking', 'Cancer'],
        'mu': [0, 0, 0],
        'sigma': [[1.0, 0.5, 0], [0.5, 1.0, 0], [0, 0, 1.0]]
    },
    'hl_params': {
        'apply_to_vars': ['Cancer_'],
        'mu': [0, 0],
        'sigma': [0.5, 0.5]
    }
}


# --- 3. Multiplicative Gaussian Noise ---

# Multiply by N(1, 0.1*I) noise for ALL variables
config_mul_gauss_all = {
    'type': 'multiplicative',
    'distribution': 'gaussian',
    'll_params': {'mu': [1, 1, 1], 'sigma': [0.1, 0.1, 0.1]},
    'hl_params': {'mu': [1, 1], 'sigma': [0.1, 0.1]}
}


# --- 4. Additive Student-T Noise (Heavy-Tailed) ---

# Add heavy-tailed noise to a SUBSET of variables
config_add_student_t_selective = {
    'type': 'additive',
    'distribution': 'student-t',
    'll_params': {
        'apply_to_vars': ['Smoking'],
        'df': 3, # Low df = heavy tails
        'loc': [0, 0, 0],
        'shape': [1, 1, 1]
    },
    'hl_params': {
        'apply_to_vars': ['Smoking_'],
        'df': 5,
        'loc': [0, 0],
        'shape': [1, 1]
    }
}


# --- 5. Additive Exponential Noise ---

config_add_exp_all = {
    'type': 'additive',
    'distribution': 'exponential',
    'll_params': {'scale': 0.5},
    'hl_params': {'scale': 0.2}
}