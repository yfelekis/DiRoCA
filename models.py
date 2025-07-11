import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork

class CausalBayesianNetwork(BayesianNetwork):
    """
    Extends pgmpy's BayesianNetwork with a 'do' operator method.
    """
    def __init__(self, ebunch=None, latents=set()):
        super().__init__(ebunch=ebunch, latents=latents)

    def do(self, variables):
        """
        Performs the 'do' operation on the network, returning a new 
        graph with incoming edges to the specified variables removed.
        """
        model_copy = self.copy()
        for var in variables:
            if var in model_copy.nodes():
                # Get all incoming edges to the variable
                in_edges = list(model_copy.in_edges(var))
                # Remove them
                model_copy.remove_edges_from(in_edges)
        return model_copy

# class Intervention:
#     """
#     Represents an intervention in a causal model.
#     """
#     def __init__(self, intervention_dict):
#         if not isinstance(intervention_dict, dict):
#             raise TypeError("Intervention must be a dictionary.")
#         self.intervention = intervention_dict
    
#     def __repr__(self):
#         return f"Intervention({self.intervention})"
    
#     def vv(self):
#         """Returns the intervention dictionary."""
#         return self.intervention

class Intervention:
    
    def __init__(self, intervention):
        self.intervention = intervention
        
    def phi(self):
        return list(self.intervention.values())
        
    def Phi(self):
        return list(self.intervention.keys())
    
    def vv(self):
        return self.intervention
    
    def __eq__(self, other):
        if isinstance(other, Intervention):
            return self.intervention == other.intervention
        return False

    def __hash__(self):
        return hash(frozenset(self.intervention.items()))

class LinearAddSCM:
    def __init__(self, causal_graph, edge_weights, intervention=None):
        """
        Initialize the Linear Additive Noise SCM model.
        """
        self.edge_weights = edge_weights.copy()
        self.intervention_dict = intervention.vv() if intervention else {}

        # The .do() operation modifies the graph structure
        if self.intervention_dict:
            self.causal_graph = causal_graph.do(list(self.intervention_dict.keys()))
        else:
            self.causal_graph = causal_graph
        
        self.variables = list(nx.topological_sort(self.causal_graph))
        self.var_index = {var: i for i, var in enumerate(self.variables)}
        self.dim = len(self.variables)
        self.W = self._compute_weight_matrix()

        # Restore the calculation for the reduced-form matrix F
        self.I = np.eye(self.dim)
        self.F = self._compute_reduced_form()

    def _compute_weight_matrix(self):
        """
        Compute the weight matrix W where W[i, j] is the coefficient for i -> j.
        This is consistent with the simulation formula E = E @ W + U.
        """
        W = np.zeros((self.dim, self.dim))
        for (parent, child), coeff in self.edge_weights.items():
            if self.causal_graph.has_edge(parent, child):
                parent_idx = self.var_index.get(parent)
                child_idx = self.var_index.get(child)
                # Ensure parent and child are in the current graph after potential interventions
                if parent_idx is not None and child_idx is not None:
                    W[parent_idx, child_idx] = coeff
        return W

    def _compute_reduced_form(self):
        """
        Compute the reduced form transformation F = (I - W)⁻¹ for the
        system E = E @ W + U.
        """
        try:
            # The correct form is (I - W)⁻¹, with no transpose.
            return np.linalg.inv(self.I - self.W)
        
        except np.linalg.LinAlgError:
            print("Warning: Direct inversion for the reduced form failed. Using power series.")
            F = np.eye(self.dim)
            runsum = np.eye(self.dim)
            
            # The power series should also be based on W, not W.T
            W_current = self.W 
            for _ in range(self.dim * 2): # Iterate more for stability
                runsum = runsum @ W_current
                F += runsum
            return F

    def simulate(self, exogenous_noise):
        """
        Simulates data from the SCM using a topological sort, correctly handling interventions.
        """
        n_samples = exogenous_noise.shape[0]
        endogenous = np.zeros((n_samples, self.dim))

        # Iterate through variables in topological order to compute their values
        for var_name in self.variables:
            var_idx = self.var_index[var_name]

            if var_name in self.intervention_dict:
                # If the variable is intervened on, set its value directly
                endogenous[:, var_idx] = self.intervention_dict[var_name]
            else:
                # Otherwise, calculate its value from its parents and its own noise
                parents = list(self.causal_graph.predecessors(var_name))
                parent_effect = 0
                if parents:
                    parent_indices = [self.var_index[p] for p in parents]
                    weights = self.W[parent_indices, var_idx]
                    parent_effect = endogenous[:, parent_indices] @ weights

                endogenous[:, var_idx] = parent_effect + exogenous_noise[:, var_idx]
                
        return endogenous


class NonlinearAddSCM:
    """
    Represents a continuous Structural Causal Model with non-linear, additive noise assignments.
    """
    def __init__(self, causal_graph, functions, intervention=None):
        self.functions = functions
        self.intervention_dict = intervention.vv() if intervention else {}

        # The .do() operation correctly modifies the graph structure
        if self.intervention_dict:
            self.causal_graph = causal_graph.do(list(self.intervention_dict.keys()))
        else:
            self.causal_graph = causal_graph
        
        self.variables = list(nx.topological_sort(self.causal_graph))
        self.var_index = {var: i for i, var in enumerate(self.variables)}
        self.dim = len(self.variables)

    def simulate(self, exogenous_noise):
        """
        Simulates data from the SCM by executing the functions in topological order.
        """
        n_samples = exogenous_noise.shape[0]
        endogenous = np.zeros_like(exogenous_noise)

        # Iterate through variables in topological order to compute their values
        for var_name in self.variables:
            var_idx = self.var_index[var_name]

            if var_name in self.intervention_dict:
                # If the variable is intervened on, set its value directly
                endogenous[:, var_idx] = self.intervention_dict[var_name]
            else:
                # Otherwise, calculate its value from its parents using its function
                parents = list(self.causal_graph.predecessors(var_name))
                parent_values = {p: endogenous[:, self.var_index[p]] for p in parents}
                
                # Get the callable function for the current node
                func = self.functions[var_name]
                
                # Calculate the value from parents and add this node's exogenous noise
                endogenous[:, var_idx] = func(**parent_values) + exogenous_noise[:, var_idx]
                
        return endogenous