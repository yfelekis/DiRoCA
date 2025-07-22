#!/usr/bin/env python3
"""
Script to fix the pickling error by removing TrainingMonitor objects from results.
Run this script after your optimization is complete but before saving.
"""

import joblib
import os

def clean_results_for_saving(results_dict):
    """
    Remove TrainingMonitor objects from results to make them pickleable.
    
    Args:
        results_dict: Dictionary containing optimization results with TrainingMonitor objects
        
    Returns:
        cleaned_dict: Dictionary with TrainingMonitor objects removed
    """
    cleaned_dict = {}
    
    for fold_key, fold_data in results_dict.items():
        cleaned_dict[fold_key] = {}
        
        # Handle different result structures
        if isinstance(fold_data, dict):
            for hyperparam_key, hyperparam_data in fold_data.items():
                if isinstance(hyperparam_data, dict):
                    cleaned_dict[fold_key][hyperparam_key] = {}
                    for key, value in hyperparam_data.items():
                        if key != 'monitor':  # Skip the TrainingMonitor object
                            cleaned_dict[fold_key][hyperparam_key][key] = value
                else:
                    # Handle cases where hyperparam_data is not a dict
                    cleaned_dict[fold_key][hyperparam_key] = hyperparam_data
        else:
            # Handle cases where fold_data is not a dict
            cleaned_dict[fold_key] = fold_data
    
    return cleaned_dict

def save_results_safely(experiment_name, diroca_results, gradca_results, baryca_results):
    """
    Save results safely by removing TrainingMonitor objects first.
    
    Args:
        experiment_name: Name of the experiment
        diroca_results: DiRoCA optimization results
        gradca_results: GradCA optimization results  
        baryca_results: BaryCA optimization results
    """
    
    # Create data directory if it doesn't exist
    data_dir = f"data/{experiment_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Clean the results
    print("Cleaning results to remove TrainingMonitor objects...")
    diroca_clean = clean_results_for_saving(diroca_results)
    gradca_clean = clean_results_for_saving(gradca_results)
    baryca_clean = clean_results_for_saving(baryca_results)
    
    # Save the cleaned results
    print("Saving cleaned results...")
    joblib.dump(diroca_clean, f"{data_dir}/diroca_cv_results.pkl")
    joblib.dump(gradca_clean, f"{data_dir}/gradca_cv_results.pkl")
    joblib.dump(baryca_clean, f"{data_dir}/baryca_cv_results.pkl")
    
    print("All results have been saved successfully!")
    print(f"Files saved in: {data_dir}/")

if __name__ == "__main__":
    # Example usage - you would call this after your optimization
    # save_results_safely("lilucas", diroca_cv_results, gradca_cv_results, baryca_cv_results)
    print("This script provides functions to fix the pickling error.")
    print("Use the save_results_safely() function after your optimization is complete.") 