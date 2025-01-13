import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from src.CBN import CausalBayesianNetwork as CBN

import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

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
        
        #self.causal_graph = self.causal_graph.copy()  
        self.edge_weights = edge_weights.copy() 

        if self.intervention is not None:
            for edge in list(self.edge_weights.keys()):
                if edge[1] in self.intervention.keys():
                    self.edge_weights[edge] = 0
        
        self.variables    = list(nx.topological_sort(self.causal_graph))
        self.var_index    = {var: i for i, var in enumerate(self.variables)}  # Map variables to indices
        self.dim          = len(self.variables)
        self.W            = self._compute_weight_matrix()
        self.I            = np.eye(self.dim)  # Identity matrix of size d
        self.F            = self._compute_reduced_form()

        if self.intervention is not None:
            self.interv = {self.var_index[key]: value for key, value in self.intervention.items()}

    def _compute_weight_matrix(self):
        """
        Compute the weight/adjacency matrix W based on the endogenous coefficient dictionary.
        """
        W = np.zeros((self.dim, self.dim))
        for (parent, child), coeff in self.edge_weights.items():
            parent_idx = self.var_index[parent]
            child_idx = self.var_index[child]
            W[parent_idx, child_idx] = coeff  # W[row=parent, col=child]
        return W

    def _compute_reduced_form(self):
        """
        Compute the reduced form transformation F = (I - W)^(-1).
        """
        try:
            return np.linalg.inv(self.I - self.W)
        
        except np.linalg.LinAlgError:
            print("Direct inversion for the reduced form failed.")
            F      = np.eye(self.dim)
            runsum = np.eye(self.dim)

            for _ in range(self.dim):
                runsum = runsum @ self.W  
                F     += runsum  

            return F
    
    def simulate(self, exogenous, intervention):
        """
        Returns:
            np.ndarray: Simulated endogenous variables (n_samples x n_nodes).
        """
        endogenous = np.zeros_like(exogenous)

        # Iteratively propagate effects using the adjacency matrix
        for _ in range(self.dim):
            endogenous = endogenous @ self.W + exogenous

        if intervention is not None:
            for target, value in intervention.vv().items():
                target_idx = self.var_index[target]  # Get index of the intervened variable
                endogenous[:, target_idx] = value  # Set value for the intervened variable
        return endogenous
    
    def simulate_reduced_form(self, exogenous):
        return exogenous @ self.F

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