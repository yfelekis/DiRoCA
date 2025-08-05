#!/usr/bin/env python3
"""
Helper to load evaluation results.
"""

import pandas as pd
import glob
import os
import re
from datetime import datetime

def parse_evaluation_filename(filename):
    """
    Parse detailed evaluation filename to extract all parameters.
    
    Examples: 
    - evaluation_additive_gaussian_alpha10-0.0-1.0_noise20-0.0-10.0_trials20_zero_meanTrue_20241201_143022.csv
    - empirical_evaluation_additive_gaussian_alpha5-0.0-1.0_noise5-0.0-10.0_trials2_zero_meanTrue_20241201_143022.csv
    """
    # Pattern for both regular and empirical evaluations
    pattern = r'(empirical_)?evaluation_(\w+)_(\w+)_alpha(\d+)-([\d.]+)-([\d.]+)_noise(\d+)-([\d.]+)-([\d.]+)_trials(\d+)_zero_mean(\w+)_(\d{8})_(\d{6})\.csv'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'evaluation_type': 'empirical' if match.group(1) else 'gaussian',
            'shift_type': match.group(2),
            'distribution': match.group(3),
            'alpha_steps': int(match.group(4)),
            'alpha_min': float(match.group(5)),
            'alpha_max': float(match.group(6)),
            'noise_steps': int(match.group(7)),
            'noise_min': float(match.group(8)),
            'noise_max': float(match.group(9)),
            'trials': int(match.group(10)),
            'zero_mean': match.group(11) == 'True',
            'date': match.group(12),
            'time': match.group(13),
            'timestamp': f"{match.group(12)}_{match.group(13)}"
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
            print(f"      - Type: {params['evaluation_type']} | Shift: {params['shift_type']} | Distribution: {params['distribution']}")
            print(f"      - Alpha: {params['alpha_steps']} steps ({params['alpha_min']:.1f} to {params['alpha_max']:.1f})")
            print(f"      - Noise: {params['noise_steps']} steps ({params['noise_min']:.1f} to {params['noise_max']:.1f})")
            print(f"      - Trials: {params['trials']} | Zero mean: {params['zero_mean']}")
        
        print()
    
    return files

def load_results(experiment='slc', evaluation_type=None, shift_type=None, distribution=None, 
                alpha_steps=None, alpha_min=None, alpha_max=None,
                noise_steps=None, noise_min=None, noise_max=None,
                trials=None, zero_mean=None, latest=True, timestamp=None):
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
        latest (bool): If True, load most recent matching file (ignored if timestamp is specified)
        timestamp (str): Specific timestamp to load (format: YYYYMMDD_HHMMSS)
    
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
        if evaluation_type and params['evaluation_type'] != evaluation_type:
            continue
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
    if timestamp:
        # Find file with specific timestamp
        target_file = None
        for file_path, params in matching_files:
            if params['timestamp'] == timestamp:
                target_file = (file_path, params)
                break
        
        if target_file is None:
            print(f"No file found with timestamp: {timestamp}")
            print("Available timestamps:")
            for file_path, params in matching_files:
                print(f"  - {params['timestamp']} ({os.path.basename(file_path)})")
            return None
        
        file_path, params = target_file
    elif latest:
        file_path, params = matching_files[0]
    else:
        file_path, params = matching_files[-1]
    
    filename = os.path.basename(file_path)
    print(f"Loading: {filename}")
    print(f"Parameters: {params['evaluation_type']} {params['shift_type']} {params['distribution']}, "
          f"α: {params['alpha_steps']} steps ({params['alpha_min']:.1f}-{params['alpha_max']:.1f}), "
          f"noise: {params['noise_steps']} steps ({params['noise_min']:.1f}-{params['noise_max']:.1f}), "
          f"trials: {params['trials']}, zero_mean: {params['zero_mean']}")
    
    return pd.read_csv(file_path)


def list_timestamps(experiment='slc', **kwargs):
    """
    List all available timestamps for a specific configuration.
    
    Args:
        experiment (str): Experiment name
        **kwargs: Any other parameters to filter by (shift_type, distribution, etc.)
    
    Returns:
        list: List of available timestamps
    """
    # Find all files for the experiment
    pattern = f"data/{experiment}/evaluation_results/*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found for experiment: {experiment}")
        return []
    
    # Filter files by parameters
    matching_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        params = parse_evaluation_filename(filename)
        
        if not params:
            continue
        
        # Check each parameter
        for key, value in kwargs.items():
            if key in params and params[key] != value:
                break
        else:
            matching_files.append((file_path, params))
    
    if not matching_files:
        print(f"No files found matching the specified parameters")
        return []
    
    # Sort by modification time
    matching_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    
    print(f"Available timestamps for {experiment} with parameters: {kwargs}")
    print("=" * 80)
    
    timestamps = []
    for i, (file_path, params) in enumerate(matching_files):
        timestamp = params['timestamp']
        timestamps.append(timestamp)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"{i+1:2d}. {timestamp} - {mod_time.strftime('%Y-%m-%d %H:%M:%S')} - {os.path.basename(file_path)}")
    
    return timestamps

if __name__ == "__main__":
    print("=== Evaluation Results Browser ===")
    print()
    list_available_results() 