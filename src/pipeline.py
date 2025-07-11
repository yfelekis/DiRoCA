"""
Pipeline module for DiRoCA experiments.
Provides a simple interface to run data generation and manage experiments.
"""

from typing import Dict, Any, Optional
import os

from .data_generator import create_data_generator
from .experiment_config import get_config


class DiRoCAPipeline:
    """Main pipeline for DiRoCA experiments."""
    
    def __init__(self, experiment_name: str):
        """
        Initialize pipeline for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment ('synth1' or 'lucas6x3')
        """
        self.experiment_name = experiment_name
        self.config = get_config(experiment_name)
        self.data_generator = create_data_generator(experiment_name)
        
    def check_data_exists(self) -> bool:
        """Check if data already exists for this experiment."""
        required_files = [
            "LL.pkl", "HL.pkl", "Ds.pkl", "Tau.pkl", 
            "omega.pkl", "ll_coeffs.pkl", "hl_coeffs.pkl",
            "exogenous_LL.pkl", "exogenous_HL.pkl"
        ]
        
        for filename in required_files:
            if not self.config.data_exists(filename):
                return False
        return True
    
    def generate_data(self, force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Generate data for the experiment.
        
        Args:
            force_regenerate: If True, regenerate data even if it exists
            
        Returns:
            Dictionary containing generated data
        """
        if self.check_data_exists() and not force_regenerate:
            print(f"Data already exists for experiment '{self.experiment_name}'")
            print("Use force_regenerate=True to regenerate data")
            return self.load_existing_data()
        
        print(f"Generating data for experiment: {self.experiment_name}")
        return self.data_generator.generate_data()
    
    def load_existing_data(self) -> Dict[str, Any]:
        """Load existing data for the experiment."""
        print(f"Loading existing data for experiment: {self.experiment_name}")
        
        data = {}
        data['ll_causal_graph'], data['Ill_relevant'] = self.config.load_data("LL.pkl")
        data['hl_causal_graph'], data['Ihl_relevant'] = self.config.load_data("HL.pkl")
        data['Ds'] = self.config.load_data("Ds.pkl")
        data['T'] = self.config.load_data("Tau.pkl")
        data['omega'] = self.config.load_data("omega.pkl")
        data['ll_coeffs'] = self.config.load_data("ll_coeffs.pkl")
        data['hl_coeffs'] = self.config.load_data("hl_coeffs.pkl")
        
        return data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment data."""
        if not self.check_data_exists():
            return {"error": "Data does not exist. Run generate_data() first."}
        
        data = self.load_existing_data()
        
        summary = {
            "experiment": self.experiment_name,
            "low_level_nodes": len(data['ll_causal_graph'].nodes()),
            "high_level_nodes": len(data['ll_causal_graph'].nodes()),
            "num_interventions_ll": len(data['Ill_relevant']),
            "num_interventions_hl": len(data['Ihl_relevant']),
            "num_sample_pairs": len(data['Ds']),
            "abstraction_matrix_shape": data['T'].shape,
            "data_directory": self.config.data_dir
        }
        
        return summary


def run_data_generation(experiment_name: str, force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run data generation for an experiment.
    
    Args:
        experiment_name: Name of the experiment ('synth1' or 'lucas6x3')
        force_regenerate: If True, regenerate data even if it exists
        
    Returns:
        Dictionary containing generated data
    """
    pipeline = DiRoCAPipeline(experiment_name)
    return pipeline.generate_data(force_regenerate=force_regenerate)


def get_experiment_summary(experiment_name: str) -> Dict[str, Any]:
    """
    Get a summary of an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary containing experiment summary
    """
    pipeline = DiRoCAPipeline(experiment_name)
    return pipeline.get_data_summary()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <experiment_name> [--force]")
        print("Available experiments: synth1, lucas6x3")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    force_regenerate = "--force" in sys.argv
    
    try:
        pipeline = DiRoCAPipeline(experiment_name)
        data = pipeline.generate_data(force_regenerate=force_regenerate)
        
        # Print summary
        summary = pipeline.get_data_summary()
        print("\nExperiment Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 