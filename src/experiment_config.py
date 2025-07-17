"""
Experiment configuration management for DiRoCA experiments.
Handles experiment parameters, data paths, and configuration loading.
"""

import os
from typing import Dict, List, Tuple, Any
import joblib
import numpy as np

# Import your existing params
import configs_cookbook


class ExperimentConfig:
    """Manages configuration for DiRoCA experiments."""
    
    def __init__(self, experiment_name: str):
        """
        Initialize configuration for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'synth1', 'lucas6x3')
        """
        self.experiment_name = experiment_name
        self.data_dir = f"data/{experiment_name}"
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load experiment-specific parameters
        self._load_experiment_params()
    
    def _load_experiment_params(self):
        """Load experiment-specific parameters from params.py."""
        if self.experiment_name not in configs_cookbook.n_samples:
            raise ValueError(f"Experiment '{self.experiment_name}' not found in params.py")
        
        self.n_samples = configs_cookbook.n_samples[self.experiment_name]
        self.n_envs = configs_cookbook.n_envs[self.experiment_name]
    
    def get_data_path(self, filename: str) -> str:
        """Get full path for a data file."""
        return os.path.join(self.data_dir, filename)
    
    def save_data(self, data: Any, filename: str):
        """Save data to experiment directory."""
        filepath = self.get_data_path(filename)
        joblib.dump(data, filepath)
        print(f"Saved {filename} to {filepath}")
    
    def load_data(self, filename: str) -> Any:
        """Load data from experiment directory."""
        filepath = self.get_data_path(filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return joblib.load(filepath)
    
    def data_exists(self, filename: str) -> bool:
        """Check if a data file exists."""
        return os.path.exists(self.get_data_path(filename))


class Synth1Config(ExperimentConfig):
    """Configuration for synth1 experiment."""
    
    def __init__(self):
        super().__init__('synth1')
        
        # Synth1 specific parameters
        self.variables = {
            'low_level': ['Smoking', 'Tar', 'Cancer'],
            'high_level': ['Smoking_', 'Cancer_']
        }
        
        self.ll_endogenous_coeff_dict = {
            ('Smoking', 'Tar'): 0.3,
            ('Tar', 'Cancer'): 0.2
        }
        
        self.hl_endogenous_coeff_dict = {
            ('Smoking_', 'Cancer_'): 0.0
        }
        
        # Nominal distributions
        self.ll_mu_hat = np.array([0, 0, 0])
        self.ll_Sigma_hat = np.diag([1, 1, 1])
        
        self.hl_mu_hat = np.array([0, 0])
        self.hl_Sigma_hat = np.diag([1, 1])


class Lucas6x3Config(ExperimentConfig):
    """Configuration for lucas6x3 experiment."""
    
    def __init__(self):
        super().__init__('lucas6x3')
        
        # Lucas6x3 specific parameters
        self.variables = {
            'low_level': ['Smoking', 'Genetics', 'Lung Cancer', 'Allergy', 'Coughing', 'Fatigue'],
            'high_level': ['Environment', 'Genetics_', 'Lung Cancer_']
        }
        
        self.ll_endogenous_coeff_dict = {
            ('Smoking', 'Lung Cancer'): 0.9,
            ('Genetics', 'Lung Cancer'): 0.8,
            ('Lung Cancer', 'Coughing'): 0.6,
            ('Lung Cancer', 'Fatigue'): 0.9,
            ('Coughing', 'Fatigue'): 0.5,
            ('Allergy', 'Coughing'): 0.4
        }
        
        self.hl_endogenous_coeff_dict = {
            ('Environment', 'Lung Cancer_'): 0.0,
            ('Genetics_', 'Lung Cancer_'): 0.0
        }
        
        # Nominal distributions
        self.ll_mu_hat = np.array([0, 0, 0.1, 0.1, 0.3, 0.2])
        self.ll_Sigma_hat = np.diag([0.5, 2.0, 1.0, 1.5, 0.8, 1.2])


def get_config(experiment_name: str) -> ExperimentConfig:
    """Factory function to get the appropriate config for an experiment."""
    if experiment_name == 'synth1':
        return Synth1Config()
    elif experiment_name == 'lucas6x3':
        return Lucas6x3Config()
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}") 