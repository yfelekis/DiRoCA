"""
Examples of how to substitute dictionary key names using a label map
"""

# Example dictionary (similar to your results_to_evaluate)
example_dict = {
    'GradCA': {'fold_0': 'data1', 'fold_1': 'data2'},
    'BARYCA': {'fold_0': 'data3', 'fold_1': 'data4'},
    'DIROCA (eps_delta_4)': {'fold_0': 'data5', 'fold_1': 'data6'},
    'DIROCA (eps_delta_8)': {'fold_0': 'data7', 'fold_1': 'data8'},
    'DIROCA (eps_delta_1)': {'fold_0': 'data9', 'fold_1': 'data10'},
    'DIROCA (eps_delta_2)': {'fold_0': 'data11', 'fold_1': 'data12'},
    'DIROCA (eps_delta_0.111)': {'fold_0': 'data13', 'fold_1': 'data14'}
}

# Label map (similar to your label_map_gaussian)
label_map = {
    'DIROCA (eps_delta_0.111)': 'DiRoCA_star',
    'DIROCA (eps_delta_1)': 'DIROCA_1',
    'DIROCA (eps_delta_2)': 'DIROCA_2',
    'DIROCA (eps_delta_4)': 'DIROCA_4',
    'DIROCA (eps_delta_8)': 'DIROCA_8',
    'GradCA': 'GradCA',
    'BARYCA': 'BARYCA'
}

print("Original dictionary keys:")
print(list(example_dict.keys()))
print()

# Method 1: Dictionary comprehension (most Pythonic)
print("Method 1: Dictionary comprehension")
results_renamed = {label_map.get(key, key): value for key, value in example_dict.items()}
print("Renamed keys:", list(results_renamed.keys()))
print()

# Method 2: Using a function
def rename_dict_keys(dictionary, key_map):
    """
    Rename dictionary keys using a mapping dictionary.
    
    Args:
        dictionary (dict): The dictionary to modify
        key_map (dict): Mapping from old keys to new keys
    
    Returns:
        dict: New dictionary with renamed keys
    """
    return {key_map.get(key, key): value for key, value in dictionary.items()}

print("Method 2: Using function")
results_renamed_func = rename_dict_keys(example_dict, label_map)
print("Renamed keys:", list(results_renamed_func.keys()))
print()

# Method 3: Using dict() constructor with zip
print("Method 3: Using dict(zip())")
old_keys = list(example_dict.keys())
new_keys = [label_map.get(key, key) for key in old_keys]
results_renamed_zip = dict(zip(new_keys, example_dict.values()))
print("Renamed keys:", list(results_renamed_zip.keys()))
print()

# Method 4: In-place modification (modifies original dictionary)
print("Method 4: In-place modification")
def rename_dict_keys_inplace(dictionary, key_map):
    """
    Rename dictionary keys in-place.
    
    Args:
        dictionary (dict): The dictionary to modify (will be changed)
        key_map (dict): Mapping from old keys to new keys
    """
    keys_to_rename = [(old_key, new_key) for old_key, new_key in key_map.items() 
                      if old_key in dictionary]
    
    for old_key, new_key in keys_to_rename:
        dictionary[new_key] = dictionary.pop(old_key)

# Create a copy to avoid modifying the original
example_dict_copy = example_dict.copy()
rename_dict_keys_inplace(example_dict_copy, label_map)
print("Renamed keys (in-place):", list(example_dict_copy.keys()))
print()

# Method 5: For pandas DataFrames
import pandas as pd

print("Method 5: For pandas DataFrames")
# Create a sample DataFrame
df = pd.DataFrame({
    'GradCA': [1, 2, 3],
    'BARYCA': [4, 5, 6],
    'DIROCA (eps_delta_4)': [7, 8, 9]
})

print("Original DataFrame columns:")
print(df.columns.tolist())

# Rename columns using the label map
df_renamed = df.rename(columns=label_map)
print("Renamed DataFrame columns:")
print(df_renamed.columns.tolist())
print()

# Method 6: Handle missing keys gracefully
print("Method 6: Handle missing keys gracefully")
def safe_rename_dict_keys(dictionary, key_map, default_suffix="_renamed"):
    """
    Safely rename dictionary keys, handling missing keys in the mapping.
    
    Args:
        dictionary (dict): The dictionary to modify
        key_map (dict): Mapping from old keys to new keys
        default_suffix (str): Suffix to add to unmapped keys
    
    Returns:
        dict: New dictionary with renamed keys
    """
    result = {}
    for key, value in dictionary.items():
        if key in key_map:
            new_key = key_map[key]
        else:
            new_key = f"{key}{default_suffix}"
        result[new_key] = value
    return result

# Test with a dictionary that has some unmapped keys
test_dict = {
    'GradCA': 'data1',
    'BARYCA': 'data2', 
    'Unknown_Method': 'data3'  # This key is not in label_map
}

results_safe = safe_rename_dict_keys(test_dict, label_map)
print("Original keys:", list(test_dict.keys()))
print("Safely renamed keys:", list(results_safe.keys())) 