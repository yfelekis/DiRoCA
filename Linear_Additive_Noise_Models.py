import random
import itertools
import joblib
import pickle 
import os

from itertools import chain, combinations
from typing import Optional
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt

from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon

from src.CBN import CausalBayesianNetwork as CBN

class LinearAddSCM:
    def __init__(self, causal_graph, edge_weights, intervention = None):
        """
        Initialize the Linear Additive Noise SCM model.

        :param causal_graph: A CausalBayesianNetwork object representing the causal graph.
        :param en_coefficients: List of coefficients corresponding to the edges in the causal graph.
        :param intervention:
        """

        if intervention != None:
            self.intervention = intervention.vv()
            self.causal_graph = causal_graph.do(self.intervention.keys())
        else:
            self.intervention = intervention
            self.causal_graph = causal_graph
            
        self.edges           = list(self.causal_graph.edges())
        self.endogenous_vars = list(self.causal_graph.nodes())   
        self.var_index       = {node: idx for idx, node in enumerate(sorted(self.endogenous_vars))}
        self.num_vars        = len(self.endogenous_vars)

        self.edge_weights = edge_weights.copy()
        
        if intervention != None:
            
            for edge in list(self.edge_weights.keys()):
                
                if edge[1] in self.intervention.keys():
                    self.edge_weights[edge] = 0

        if self.intervention != None:
            self.interv = {self.var_index[key]: value for key, value in self.intervention.items()}

        node_index = {node: idx for idx, node in enumerate(self.endogenous_vars)}

        #n_nodes = len(self.endogenous_vars)
        weights = np.zeros((self.num_vars, self.num_vars))
    
        for edge, coeff in self.edge_weights.items():
            
            weights[node_index[edge[0]], node_index[edge[1]]] = coeff

        self.adjacency_matrix = weights #weights.T
        
    def compute_mechanism(self):
        adj_matrix = self.adjacency_matrix
        """Reduced form of the linear ANM."""
        n_nodes = adj_matrix.shape[0]
        mechanism = np.zeros((n_nodes, n_nodes))
        cumulator = np.eye(n_nodes)
        mechanism += cumulator
        for _ in range(n_nodes):
            cumulator = cumulator @ adj_matrix
            mechanism += cumulator
        
        return mechanism
    
    def sample_settings(self, exogenous):
        
        endogenous = np.zeros_like(exogenous)
        n_nodes    = self.adjacency_matrix.shape[0]
        
        assert self.adjacency_matrix.shape == (n_nodes, n_nodes)
        
        for _ in range(n_nodes):
            # update endogenous
            endogenous = endogenous @ self.adjacency_matrix + exogenous

            # apply intervention
            if self.intervention is not None:

                for target, value in self.interv.items():
                    endogenous[:, target] = value
        return endogenous
    
    def print_adjacency_matrix(self, show_labels = True):
        
        plt.figure(figsize=(3, 2))
        
        sns.heatmap(self.adjacency_matrix, annot=True, cmap='Greys', cbar=False, 
                xticklabels=self.endogenous_vars if show_labels else False, 
                yticklabels=self.endogenous_vars if show_labels else False,
                linewidths=0.5, linecolor='black', square=True)
        
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)

        plt.show()
        
    def return_adjacency_matrix(self):
        return self.adjacency_matrix
    
    def print_mechanisms(self, exogenous_coeff_dict):
        
        nodes = set()
        for (source, target) in self.edge_weights.keys():
            nodes.add(source)
            nodes.add(target)        

        equations = []
        for i, var in enumerate(nodes):

            equation = f"{var} = "
            parent_coeffs = []
            for (parent, child), coeff in zip(self.edge_weights.keys(), self.edge_weights.values()):
                if child == var:

                    parent_coeffs.append((parent, coeff))

            if parent_coeffs:
                equation += " + ".join([f"{coeff}* {parent}" for parent, coeff in parent_coeffs]) + " + "
            
            # Add the noise term
            if self.intervention == None or var not in list(self.intervention.keys()):
                noise_var  = f"U_{var.lower()}"
                noise_dist = f"N({exogenous_coeff_dict[var][0]}, {exogenous_coeff_dict[var][1]})"
                equation   += f"{noise_var},  {noise_var} ~ {noise_dist}"
            else:
                val_iota = self.intervention[var]
                equation   += f"{val_iota}"

            equations.append(f"{equation}")

        for eq in equations:
            print(eq)
            
        return
    
    def print_dag(self):
        """
        Generate the underlying DAG of the SCM.
        """
        G = nx.DiGraph()

        # Add nodes and edges for the causal graph
        for parent, child in self.causal_graph.edges():
            G.add_edge(parent, child, color='black', style='solid')

        # Add nodes for noise variables and edges connecting them to their corresponding endogenous variables
        for i, var in enumerate(self.causal_graph.nodes):
            noise_var = f'$U_{var}$'
            G.add_node(noise_var, color='lightblue')
            G.add_edge(noise_var, var, color='lightblue', style='dotted')

        # Define node positions
        pos = nx.spring_layout(G)  # or any other layout you prefer

        # Draw nodes with custom colors
        node_colors = [G.nodes[node].get('color', 'teal') for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)

        # Draw labels with custom font size
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Draw solid edges with arrows
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'solid']
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color='black', style='solid', arrows=True, arrowstyle='-|>', arrowsize=23)

        # Draw dotted edges for noise variables
        dotted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dotted']
        nx.draw_networkx_edges(G, pos, edgelist=dotted_edges, style='dotted', edge_color='lightblue', arrows=True, arrowstyle='-|>', arrowsize=23)

        plt.show()
        
        return   