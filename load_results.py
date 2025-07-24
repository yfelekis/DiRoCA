#!/usr/bin/env python3
"""
Comprehensive helper to load evaluation results by any parameter.
"""

import pandas as pd
import glob
import os
import re
from datetime import datetime

def parse_evaluation_filename(filename):
    """
    Parse detailed evaluation filename to extract all parameters.
    
    Example: evaluation_additive_gaussian_alpha10-0.0-1.0_noise20-0.0-10.0_trials20_zero_meanTrue_20241201_143022.csv
    """
    pattern = r'evaluation_(\w+)_(\w+)_alpha(\d+)-([\d.]+)-([\d.]+)_noise(\d+)-([\d.]+)-([\d.]+)_trials(\d+)_zero_mean(\w+)_(\d{8})_(\d{6})\.csv'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'shift_type': match.group(1),
            'distribution': match.group(2),
            'alpha_steps': int(match.group(3)),
            'alpha_min': float(match.group(4)),
            'alpha_max': float(match.group(5)),
            'noise_steps': int(match.group(6)),
            'noise_min': float(match.group(7)),
            'noise_max': float(match.group(8)),
            'trials': int(match.group(9)),
            'zero_mean': match.group(10) == 'True',
            'date': match.group(11),
            'time': match.group(12),
            'timestamp': f"{match.group(11)}_{match.group(12)}"
        }
    return None

def list_available_results(experiment=None, show_details=True):
    """
    List all available evaluation results with detailed parameter information.
    
    Args:
        experiment (str, optional): Filter by experiment
        show_details (bool): Show detailed parameter breakdown
    """
    if experiment:
        pattern = f"data/{experiment}/evaluation_results/*.csv"
    else:
        pattern = "data/*/evaluation_results/*.csv"
    
    files = glob.glob(pattern)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not files:
        print("No evaluation results found!")
        return []
    
    print(f"Found {len(files)} evaluation result files:")
    print("=" * 100)
    
    for i, file_path in enumerate(files):
        experiment_name = file_path.split('/')[1]
        filename = os.path.basename(file_path)
        params = parse_evaluation_filename(filename)
        
        file_size = os.path.getsize(file_path) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        print(f"{i+1:2d}. {experiment_name.upper()}")
        print(f"    File: {filename}")
        print(f"    Size: {file_size:.1f} KB | Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if show_details and params:
            print(f"    Parameters:")
            print(f"      - Shift: {params['shift_type']} | Distribution: {params['distribution']}")
            print(f"      - Alpha: {params['alpha_steps']} steps ({params['alpha_min']:.1f} to {params['alpha_max']:.1f})")
            print(f"      - Noise: {params['noise_steps']} steps ({params['noise_min']:.1f} to {params['noise_max']:.1f})")
            print(f"      - Trials: {params['trials']} | Zero mean: {params['zero_mean']}")
        
        print()
    
    return files

def load_results(experiment='slc', shift_type=None, distribution=None, 
                alpha_steps=None, alpha_min=None, alpha_max=None,
                noise_steps=None, noise_min=None, noise_max=None,
                trials=None, zero_mean=None, latest=True):
    """
    Load evaluation results by specifying any combination of parameters.
    
    Args:
        experiment (str): Experiment name
        shift_type (str): 'additive' or 'multiplicative'
        distribution (str): 'gaussian', 'exponential', 'student-t'
        alpha_steps (int): Number of alpha steps
        alpha_min (float): Minimum alpha value
        alpha_max (float): Maximum alpha value
        noise_steps (int): Number of noise steps
        noise_min (float): Minimum noise value
        noise_max (float): Maximum noise value
        trials (int): Number of trials
        zero_mean (bool): Whether zero mean was used
        latest (bool): If True, load most recent matching file
    
    Returns:
        pd.DataFrame: The evaluation results
    """
    # Find all files for the experiment
    pattern = f"data/{experiment}/evaluation_results/*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found for experiment: {experiment}")
        return None
    
    # Filter files by parameters
    matching_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        params = parse_evaluation_filename(filename)
        
        if not params:
            continue
        
        # Check each parameter
        if shift_type and params['shift_type'] != shift_type:
            continue
        if distribution and params['distribution'] != distribution:
            continue
        if alpha_steps and params['alpha_steps'] != alpha_steps:
            continue
        if alpha_min is not None and abs(params['alpha_min'] - alpha_min) > 0.01:
            continue
        if alpha_max is not None and abs(params['alpha_max'] - alpha_max) > 0.01:
            continue
        if noise_steps and params['noise_steps'] != noise_steps:
            continue
        if noise_min is not None and abs(params['noise_min'] - noise_min) > 0.01:
            continue
        if noise_max is not None and abs(params['noise_max'] - noise_max) > 0.01:
            continue
        if trials and params['trials'] != trials:
            continue
        if zero_mean is not None and params['zero_mean'] != zero_mean:
            continue
        
        matching_files.append((file_path, params))
    
    if not matching_files:
        print(f"No files found matching the specified parameters")
        print("Available parameters:")
        list_available_results(experiment, show_details=False)
        return None
    
    # Sort by modification time
    matching_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    
    # Select file
    if latest:
        file_path, params = matching_files[0]
    else:
        file_path, params = matching_files[-1]
    
    filename = os.path.basename(file_path)
    print(f"Loading: {filename}")
    print(f"Parameters: {params['shift_type']} {params['distribution']}, "
          f"α: {params['alpha_steps']} steps ({params['alpha_min']:.1f}-{params['alpha_max']:.1f}), "
          f"noise: {params['noise_steps']} steps ({params['noise_min']:.1f}-{params['noise_max']:.1f}), "
          f"trials: {params['trials']}, zero_mean: {params['zero_mean']}")
    
    return pd.read_csv(file_path)

# Convenience functions
def load_latest(experiment='slc'):
    """Load the most recent results for an experiment."""
    return load_results(experiment=experiment)

def load_slc():
    """Load latest SLC results."""
    return load_latest('slc')

def load_lilucas():
    """Load latest LILUCAS results."""
    return load_latest('lilucas')

def load_additive_gaussian(experiment='slc'):
    """Load latest additive gaussian results."""
    return load_results(experiment=experiment, shift_type='additive', distribution='gaussian')

def load_multiplicative_gaussian(experiment='slc'):
    """Load latest multiplicative gaussian results."""
    return load_results(experiment=experiment, shift_type='multiplicative', distribution='gaussian')

def load_exponential(experiment='slc'):
    """Load latest exponential distribution results."""
    return load_results(experiment=experiment, distribution='exponential')

def load_student_t(experiment='slc'):
    """Load latest student-t distribution results."""
    return load_results(experiment=experiment, distribution='student-t')

def load_full_eval(experiment='slc'):
    """Load results from a full evaluation (many steps/trials)."""
    return load_results(experiment=experiment, alpha_steps=10, noise_steps=20, trials=20)

def load_quick_eval(experiment='slc'):
    """Load results from a quick evaluation (few steps/trials)."""
    return load_results(experiment=experiment, alpha_steps=3, noise_steps=4, trials=2)

def load_zero_mean(experiment='slc'):
    """Load results with zero mean contamination."""
    return load_results(experiment=experiment, zero_mean=True)

def load_non_zero_mean(experiment='slc'):
    """Load results with non-zero mean contamination."""
    return load_results(experiment=experiment, zero_mean=False)

if __name__ == "__main__":
    print("=== Evaluation Results Browser ===")
    print()
    list_available_results() 