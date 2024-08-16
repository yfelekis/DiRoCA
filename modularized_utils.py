import numpy as np
import itertools
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from IPython.utils import io
from pgmpy import inference
from src.examples import smokingmodels as sm
import classes as cls

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def get_distribution(samples, variables=None):
    
    if variables is None:
        raise ValueError("variables must be specified")
        
    df = (samples.groupby(variables).size() / samples.shape[0]).to_frame()
    return df

def df_to_dict(df):
    d = {}
    for mi, row in df.iterrows():
        d[mi] = row.iloc[0]
    return d

def fillout(dict_, model):

    # Determine the length of the binary tuples
    n = len(list(dict_.keys())[0])
    
    # Generate all possible binary tuples of length n
    #keys = list(itertools.product([0, 1], repeat=n))
    cards = (np.arange(model.get_cardinality(n)) for n in model.nodes)
    keys = list(itertools.product(*cards))
    # Create a new dictionary with all keys from keys list and 0 as the default value
    # If the key exists in the original dictionary, use its value instead of the default
    
    return {key: dict_.get(key, 0) for key in keys}


def create_pairs(n_samples, I_relevant, omega, M_base, M_abst):

    pairs = []
    for iota in I_relevant:
        with io.capture_output() as captured:
            if iota != I_relevant[0]:
                D_base = iota.model.simulate(n_samples = n_samples, evidence = iota.intervention) 
                D_abst = omega[iota].model.simulate(n_samples = n_samples, evidence = omega[iota].intervention)  

            else:
                D_base = iota.model.simulate(n_samples = n_samples) 
                D_abst = omega[iota].model.simulate(n_samples = n_samples)  

            df_base = get_distribution(D_base, variables = list(M_base.nodes))  
            df_abst = get_distribution(D_abst, variables = list(M_abst.nodes))            

            p_base = fillout(df_to_dict(df_base), M_base)
            p_abst = fillout(df_to_dict(df_abst), M_abst)
            
            pairs.append(cls.Pair(p_base, p_abst, iota, omega[iota]))
    return pairs

def compute_medoids(data): #for continuous
    data_array = np.array(data)

    # Get unique first elements in the data --> make sure to have data for every intervention
    unique_first_elements = np.unique(data_array[:, 0])

    # Dictionary to store the count of each subgroup's representation
    subgroup_counts = {}

    # Calculate the count of each subgroup's representation
    for first_element in unique_first_elements:
        subset_data = data_array[data_array[:, 0] == first_element]
        subgroup_counts[first_element] = len(subset_data)
    

    # Set the number of representatives you want for each group as a percentage (e.g., 50%)
    num_representatives_per_group = 0.5

    # Set the desired percentage representation for each subgroup
    percentage_representatives = {key: value / len(data_array) for key, value in subgroup_counts.items()}

    # Calculate the total number of representatives based on the percentages
    total_representatives = {key: max(1, int(value * len(data_array) * num_representatives_per_group))
                             for key, value in percentage_representatives.items()}

    # List to store representatives for each subgroup
    representatives = []

    # Perform k-medoids for each unique first element
    for first_element in unique_first_elements:
        # Get the subset with the current first element
        subset_data = data_array[data_array[:, 0] == first_element][:, 1:]

        # Calculate the number of representatives required for this subgroup
        num_representatives = total_representatives[first_element]

        # Perform k-medoids if there are enough data points for representatives
        if len(subset_data) >= num_representatives:
            # Create the KMedoids object and fit it to the subset data
            kmedoids = KMedoids(n_clusters=num_representatives, random_state=0)
            kmedoids.fit(subset_data)

            # Get the cluster indices for the subset
            cluster_indices = kmedoids.labels_

            # Get the data points corresponding to each cluster index
            for cluster_index in range(num_representatives):
                cluster_points = subset_data[cluster_indices == cluster_index]

                # Choose the representative from each cluster based on your criteria
                representative = cluster_points.mean(axis=0)  # For example, mean of points in the cluster
                representatives.append(np.hstack(([first_element], representative)))
        else:
            # If there are fewer data points than requested representatives, append all data points as representatives
            for point in subset_data:
                representatives.append(np.hstack(([first_element], point)))

    # Convert the representatives list to a numpy array
    #representatives_array = np.array(representatives)
    representatives_lst = [tuple(arr) for arr in representatives]

    return representatives_lst 



def find_positions(lst, subset):
    positions = []
    for item in subset:
        if item in lst:
            positions.append(lst.index(item))
    return positions

def visualize_tree(node):
    G = nx.DiGraph()

    def add_edges(curr_node):
        for child in curr_node.children:
            G.add_edge(curr_node.data, child.data)
            add_edges(child)

    add_edges(node)

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=100, font_size=5)
    plt.show()
    
    return 

def to_chains(pairs, tree_structure):
    
    plans = [cls.Poset(pair) for pair in pairs]
    
    # Initialize parent-child relations
    for parent_index, child_index in tree_structure:
        plans[parent_index].add_child(plans[child_index])
    
    # Initialize variables
    paths = []
    current_path = []

    def traverse_paths(node):
        nonlocal current_path

        current_path.append(node)

        if not node.children:
            paths.append(current_path[:])

        for i, child in enumerate(node.children):
            traverse_paths(child)

        current_path.pop()

    # Traverse paths starting from each root node
    for plan in plans:
        if not any(node for node in plans if plan in node.children):
            traverse_paths(plan)

    return paths


def drop1omega(omega, key):
    modified_omega = omega.copy()
    if key in modified_omega:
        del modified_omega[key]
    return modified_omega

def interventional_order(iota_i, iota_j):
    i_variables = set(iota_i.intervention.keys())
    j_variables = set(iota_j.intervention.keys())

    if None in i_variables and not None in j_variables:
        return True
    elif i_variables <= j_variables and all(iota_i.intervention[var] == iota_j.intervention[var] for var in i_variables):
        return True
    else:
        return False
    
def build_poset(Iota):
    
    edges = []
    tree = {}

    for i in Iota:
        tree[i] = []
    
    for i in range(len(Iota)):
        for j in range(i + 1, len(Iota)):
            if interventional_order(Iota[i], Iota[j]):
                edges.append((i, j))
                tree[Iota[i]].append(Iota[j])
    
    return edges, tree

def visualize_tree(tree):
    
    G = nx.DiGraph()
    for node in tree:
        G.add_node(node)
    
    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent, child)
    
    labeldict = {}
    for node in list(G.nodes):
        labeldict[node] = node.intervention 
   
    pos = nx.spring_layout(G)
    nx.draw(G, pos, labels=labeldict, with_labels=True, arrows=True, node_size=80, font_size=7, arrowsize=10)
    nx.draw(G,pos,node_size=60,font_size=8) 
    plt.show()
    
    return

def load_pairs(experiment):
    return joblib.load(f'data/{experiment}/pairs.pkl')

def load_shufpairs(experiment):
    return joblib.load(f'data/{experiment}/shufpairs.pkl')

def load_omega(experiment):
    return joblib.load(f'data/{experiment}/omega.pkl')

def load_mask(experiment, mask, order):
    
    if order == 'perfect':
        return joblib.load(f'data/{experiment}/perf/{mask}.pkl')
    
    elif order == 'shuff':
        return joblib.load(f'data/{experiment}/shuff/{mask}.pkl')

def load_results(mode, experiment, dropped_pair):
    
    res = {}
    
    if mode == 'all':
        cota_plans_res  = joblib.load(f'results/{experiment}/{dropped_pair}/cota_plans.pkl')
        cota_maps_res   = joblib.load(f'results/{experiment}/{dropped_pair}/cota_maps.pkl')

        pwise_plans_res = joblib.load(f'results/{experiment}/{dropped_pair}/pwise_plans.pkl')
        pwise_maps_res  = joblib.load(f'results/{experiment}/{dropped_pair}/pwise_maps.pkl')

        bary_plans_res  = joblib.load(f'results/{experiment}/{dropped_pair}/bary_plans.pkl')
        bary_maps_res   = joblib.load(f'results/{experiment}/{dropped_pair}/bary_maps.pkl')

        agg_plans_res   = joblib.load(f'results/{experiment}/{dropped_pair}/agg_plans.pkl')
        agg_maps_res    = joblib.load(f'results/{experiment}/{dropped_pair}/agg_maps.pkl')
        
        res['cota']  = cota_maps_res
        res['pwise'] = pwise_maps_res
        res['bary']  = bary_maps_res
        res['agg']   = agg_maps_res
    
    else:
        _plans_res = joblib.load(f'results/{experiment}/{dropped_pair}/{mode}_plans.pkl')
        _maps_res  = joblib.load(f'results/{experiment}/{dropped_pair}/{mode}_maps.pkl')
        
        res[mode]    = _maps_res
    
    return res



def load_grid_results(experiment, combination):
    
    grid_plans_res = joblib.load(f'results/{experiment}/{combination}/cota_plans_grid.pkl')
    grid_maps_res  = joblib.load(f'results/{experiment}/{combination}/cota_maps_grid.pkl')
    
    return grid_maps_res

def load_grid_results_looo(experiment, combination, dropped_pair):
    
    grid_plans_res = joblib.load(f'results/{experiment}/{combination}/{dropped_pair}/cota_plans_grid.pkl')
    grid_maps_res  = joblib.load(f'results/{experiment}/{combination}/{dropped_pair}/cota_maps_grid.pkl')
    
    return grid_maps_res

def load_grid_results_looo_aggregated(experiment, combination, dropped_pair):
    
    grid_plans_res = joblib.load(f'results/{experiment}/{combination}/{dropped_pair}/cota_plans_grid_agg.pkl')
    grid_maps_res  = joblib.load(f'results/{experiment}/{combination}/{dropped_pair}/cota_maps_grid_agg.pkl')
    
    return grid_maps_res

"""def load_grid_results_looo_aggregated_ne(experiment, combination, dropped_pair):
    
    grid_plans_res = joblib.load(f'results/{experiment}/{combination}/{dropped_pair}/cota_plans_grid_agg_ne.pkl')
    grid_maps_res  = joblib.load(f'results/{experiment}/{combination}/{dropped_pair}/cota_maps_grid_agg_ne.pkl')
    
    return grid_maps_res"""