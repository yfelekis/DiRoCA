"""
Data generation module for DiRoCA experiments.
Handles generation of low-level and high-level data for causal abstraction experiments.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

from .CBN import CausalBayesianNetwork as CBN
from .experiment_config import ExperimentConfig
import modularised_utils as mut
import Linear_Additive_Noise_Models as lanm
import operations as ops


class DataGenerator:
    """Handles data generation for causal abstraction experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize data generator with experiment configuration.
        
        Args:
            config: Experiment configuration object
        """
        self.config = config
        self.experiment_name = config.experiment_name
        
        # Initialize causal graphs
        self.ll_causal_graph = CBN(list(config.ll_endogenous_coeff_dict.keys()))
        self.hl_causal_graph = CBN(list(config.hl_endogenous_coeff_dict.keys()))
        
        # Get sample sizes
        self.num_llsamples = config.n_samples[0]
        self.num_hlsamples = config.n_samples[1]
        
        # Initialize data storage
        self.Dll_samples = {}
        self.Dhl_samples = {}
        self.Dll_noise = {}
        self.Dhl_noise = {}
        self.Ds = {}
        
    def define_interventions(self) -> Tuple[List, List, Dict]:
        """
        Define interventions and mapping function ω.
        This method should be overridden by specific experiment classes.
        
        Returns:
            Tuple of (Ill_relevant, Ihl_relevant, omega)
        """
        raise NotImplementedError("Subclasses must implement define_interventions")
    
    def define_abstraction_matrix(self) -> np.ndarray:
        """
        Define the abstraction matrix T.
        This method should be overridden by specific experiment classes.
        
        Returns:
            Abstraction matrix T
        """
        raise NotImplementedError("Subclasses must implement define_abstraction_matrix")
    
    def generate_low_level_samples(self, Ill_relevant: List) -> Dict:
        """Generate samples for low-level model under different interventions."""
        print(f"Generating low-level samples for {len(Ill_relevant)} interventions...")
        
        for iota in Ill_relevant:
            llcm = lanm.LinearAddSCM(
                self.ll_causal_graph, 
                self.config.ll_endogenous_coeff_dict, 
                iota
            )
            
            # Sample from nominal distribution
            lenv_iota = mut.sample_distros_Gelbrich([
                (self.config.ll_mu_hat, self.config.ll_Sigma_hat)
            ])[0]
            
            self.Dll_noise[iota] = lenv_iota.sample(self.num_llsamples)[0]
            self.Dll_samples[iota] = llcm.simulate(self.Dll_noise[iota], iota)
        
        return self.Dll_samples
    
    def generate_high_level_samples(self, Ihl_relevant: List, omega: Dict) -> Dict:
        """Generate samples for high-level model under different interventions."""
        print(f"Generating high-level samples for {len(Ihl_relevant)} interventions...")
        
        # Compute high-level parameters from observational data
        data_observational_hl = self.Dll_samples[None] @ self.T.T
        hl_endogenous_coeff_dict = mut.get_coefficients(
            data_observational_hl, 
            self.hl_causal_graph
        )
        
        U_hl, hl_mu_hat, hl_Sigma_hat = mut.lan_abduction(
            data_observational_hl, 
            self.hl_causal_graph, 
            hl_endogenous_coeff_dict
        )
        
        # Generate samples for each intervention
        for eta in Ihl_relevant:
            if eta is not None:
                hlcm = lanm.LinearAddSCM(
                    self.hl_causal_graph, 
                    hl_endogenous_coeff_dict, 
                    eta
                )
                
                lenv_eta = mut.sample_distros_Gelbrich([
                    (hl_mu_hat, hl_Sigma_hat)
                ])[0]
                
                self.Dhl_noise[eta] = lenv_eta.sample(self.num_hlsamples)[0]
                self.Dhl_samples[eta] = hlcm.simulate(self.Dhl_noise[eta], eta)
            else:
                self.Dhl_noise[eta] = U_hl
                self.Dhl_samples[eta] = data_observational_hl
        
        # Store high-level parameters
        self.hl_endogenous_coeff_dict = hl_endogenous_coeff_dict
        self.hl_mu_hat = hl_mu_hat
        self.hl_Sigma_hat = hl_Sigma_hat
        self.U_hl = U_hl
        
        return self.Dhl_samples
    
    def create_sample_pairs(self, omega: Dict) -> Dict:
        """Create pairs of low-level and high-level samples."""
        for iota in omega.keys():
            self.Ds[iota] = (self.Dll_samples[iota], self.Dhl_samples[omega[iota]])
        
        return self.Ds
    
    def save_generated_data(self, Ill_relevant: List, Ihl_relevant: List, omega: Dict):
        """Save all generated data to files."""
        print("Saving generated data...")
        
        # Save causal graphs and interventions
        self.config.save_data((self.ll_causal_graph, Ill_relevant), "LL.pkl")
        self.config.save_data(self.config.ll_endogenous_coeff_dict, "ll_coeffs.pkl")
        
        self.config.save_data((self.hl_causal_graph, Ihl_relevant), "HL.pkl")
        self.config.save_data(self.hl_endogenous_coeff_dict, "hl_coeffs.pkl")
        
        # Save sample pairs
        self.config.save_data(self.Ds, "Ds.pkl")
        
        # Save abstraction matrix and mapping
        self.config.save_data(self.T, "Tau.pkl")
        self.config.save_data(omega, "omega.pkl")
        
        # Save exogenous variables
        self.config.save_data(
            (self.Dll_noise[None], self.config.ll_mu_hat, self.config.ll_Sigma_hat), 
            "exogenous_LL.pkl"
        )
        self.config.save_data(
            (self.U_hl, self.hl_mu_hat, self.hl_Sigma_hat), 
            "exogenous_HL.pkl"
        )
        
        print("Data generation completed successfully!")
    
    def generate_data(self) -> Dict:
        """
        Main method to generate all data for the experiment.
        
        Returns:
            Dictionary containing all generated data
        """
        print(f"Starting data generation for experiment: {self.experiment_name}")
        
        # Define interventions and mapping
        Ill_relevant, Ihl_relevant, omega = self.define_interventions()
        
        # Define abstraction matrix
        self.T = self.define_abstraction_matrix()
        
        # Generate samples
        self.generate_low_level_samples(Ill_relevant)
        self.generate_high_level_samples(Ihl_relevant, omega)
        
        # Create sample pairs
        self.create_sample_pairs(omega)
        
        # Save data
        self.save_generated_data(Ill_relevant, Ihl_relevant, omega)
        
        return {
            'Dll_samples': self.Dll_samples,
            'Dhl_samples': self.Dhl_samples,
            'Ds': self.Ds,
            'T': self.T,
            'omega': omega,
            'Ill_relevant': Ill_relevant,
            'Ihl_relevant': Ihl_relevant
        }


class Synth1DataGenerator(DataGenerator):
    """Data generator for synth1 experiment."""
    
    def define_interventions(self) -> Tuple[List, List, Dict]:
        """Define interventions for synth1 experiment."""
        # Low-level interventions
        iota0 = None
        iota1 = ops.Intervention({'Smoking': 0})
        iota2 = ops.Intervention({'Smoking': 0, 'Tar': 1})
        iota3 = ops.Intervention({'Smoking': 1})
        iota4 = ops.Intervention({'Smoking': 1, 'Tar': 0})
        iota5 = ops.Intervention({'Smoking': 1, 'Tar': 1})
        
        # High-level interventions
        eta0 = None
        eta1 = ops.Intervention({'Smoking_': 0})
        eta2 = ops.Intervention({'Smoking_': 1})
        
        # Mapping function
        omega = {
            iota0: eta0,
            iota1: eta1,
            iota2: eta1,
            iota3: eta2,
            iota4: eta2,
            iota5: eta2
        }
        
        Ill_relevant = list(set(omega.keys()))
        Ihl_relevant = list(set(omega.values()))
        
        return Ill_relevant, Ihl_relevant, omega
    
    def define_abstraction_matrix(self) -> np.ndarray:
        """Define abstraction matrix for synth1 experiment."""
        return np.array([[1, 2, 1], [0, 1, 0]])


class Lucas6x3DataGenerator(DataGenerator):
    """Data generator for lucas6x3 experiment."""
    
    def define_interventions(self) -> Tuple[List, List, Dict]:
        """Define interventions for lucas6x3 experiment."""
        # Low-level interventions
        iota0 = None  # No intervention
        iota1 = ops.Intervention({'Smoking': 0})
        iota2 = ops.Intervention({'Smoking': 1})
        iota3 = ops.Intervention({'Lung Cancer': 0})
        iota4 = ops.Intervention({'Lung Cancer': 1})
        iota5 = ops.Intervention({'Smoking': 0, 'Lung Cancer': 0})
        iota6 = ops.Intervention({'Smoking': 1, 'Lung Cancer': 1})
        iota7 = ops.Intervention({'Smoking': 0, 'Lung Cancer': 1})
        iota8 = ops.Intervention({'Smoking': 1, 'Lung Cancer': 0})
        iota9 = ops.Intervention({'Genetics': 0})
        iota10 = ops.Intervention({'Genetics': 1})
        iota11 = ops.Intervention({'Genetics': 0, 'Smoking': 0})
        iota12 = ops.Intervention({'Genetics': 1, 'Smoking': 1})
        iota13 = ops.Intervention({'Genetics': 0, 'Smoking': 1})
        iota14 = ops.Intervention({'Genetics': 1, 'Smoking': 0})
        iota15 = ops.Intervention({'Allergy': 0})
        iota16 = ops.Intervention({'Allergy': 1})
        iota17 = ops.Intervention({'Coughing': 0})
        iota18 = ops.Intervention({'Coughing': 1, 'Fatigue': 1})
        iota19 = ops.Intervention({'Coughing': 1, 'Fatigue': 0})
        iota20 = ops.Intervention({'Coughing': 0, 'Fatigue': 1})
        
        # High-level interventions
        eta0 = None  # No intervention
        eta1 = ops.Intervention({'Environment': 0})
        eta2 = ops.Intervention({'Environment': 1})
        eta3 = ops.Intervention({'Genetics_': 0})
        eta4 = ops.Intervention({'Genetics_': 1})
        eta5 = ops.Intervention({'Environment': 0, 'Genetics_': 0})
        eta6 = ops.Intervention({'Environment': 1, 'Genetics_': 1})
        eta7 = ops.Intervention({'Environment': 0, 'Genetics_': 1})
        eta8 = ops.Intervention({'Environment': 1, 'Genetics_': 0})
        eta9 = ops.Intervention({'Lung Cancer_': 0})
        eta10 = ops.Intervention({'Lung Cancer_': 1})
        
        # Mapping function
        omega = {
            iota0: eta0,
            iota1: eta1,
            iota2: eta2,
            iota3: eta3,
            iota4: eta4,
            iota5: eta5,
            iota6: eta6,
            iota7: eta9,
            iota8: eta10,
            iota10: eta5,
            iota11: eta6,
            iota12: eta3,
            iota13: eta4,
            iota14: eta8,
            iota15: eta7,
            iota16: eta5,
            iota17: eta6,
            iota18: eta10,
            iota19: eta9,
            iota20: eta8
        }
        
        Ill_relevant = list(set(omega.keys()))
        Ihl_relevant = list(set(omega.values()))
        
        return Ill_relevant, Ihl_relevant, omega
    
    def define_abstraction_matrix(self) -> np.ndarray:
        """Define abstraction matrix for lucas6x3 experiment."""
        return np.array([
            [2, 1, 1, 0.5, 1, 0.5],
            [0.5, 2, 0.8, 1.5, 0.7, 1],
            [1, 0.5, 1, 2, 0.9, 1.5]
        ])


def create_data_generator(experiment_name: str) -> DataGenerator:
    """Factory function to create the appropriate data generator."""
    if experiment_name == 'synth1':
        from .experiment_config import Synth1Config
        return Synth1DataGenerator(Synth1Config())
    elif experiment_name == 'lucas6x3':
        from .experiment_config import Lucas6x3Config
        return Lucas6x3DataGenerator(Lucas6x3Config())
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}") 