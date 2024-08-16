import numpy as np
import networkx as nx
import itertools
from scipy.spatial import distance

from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD as cpd
from pgmpy.inference import VariableElimination
            
class Abstraction():
    def __init__(self,M0,M1,R,a,alphas,
                 prettyprint=lambda x : x):
                       
        self.M0 = M0
        self.M1 = M1
        
        self.R = R
        self.a = a
        self.alphas = alphas
        
        self.nR = len(R)
        self.prettyprint = prettyprint
        
    def copy(self):
        Acopy = pgmpy_Abstraction(self.M0, self.M1, self.nR, 
                            R=self.R.copy(), a=self.a.copy(), alphas=self.alphas.copy())
        return Acopy
    
    def invert_a(self,v):
        #return list(np.array(list(self.a.keys()))[np.where(np.array(list(self.a.values()))==v)[0]])
        return list(np.array(list(self.a.keys()))[np.where(np.in1d(np.array(list(self.a.values())),v))[0]])
    
    def is_varlevel_complete(self):
        return self.M0.number_of_nodes() == self.nR
    
    def is_varlevel_isomorphic(self):
        return self.nR == self.M1.number_of_nodes()
    
    def is_domlevel_isomorphic(self):
        for k in self.alphas.keys():
            print("Mapping alpha_{0}: {1}".format(k, alphas[k].shape[0]==alphas[k].shape[1]))
    
    def print_M0_cardinalites(self):
        for n in self.M0.nodes():
            print('M0: cardinality of {0}: {1}'.format(n,self.M0.get_cardinality(n)))
    
    def print_R_cardinalites(self):
        for n in self.R:
            print('R: cardinality of {0}: {1}'.format(n,self.M0.get_cardinality(n)))
            
    def print_M1_cardinalites(self):
        for n in self.M1.nodes():
            print('M1: cardinality of {0}: {1}'.format(n,self.M1.get_cardinality(n)))
    
    def print_relevant_vars(self):
        print(self.R)
    
    def print_mapping_a(self):
        print('** The mapping a is indexed by R/M0 **')
        print(self.a)
        
    def print_mappings_alphas(self):
        print('** The mappings alpha are indexed by M1 **')
        for k in alphas.keys():
            domain = self.invert_a(k)
            print('Mapping alpha_{0}: {1} -> {2}'.format(k, domain ,k))
    
    def print_mappings_alphas_cardinalities(self):
        print('** The mappings alpha are indexed by M1 **')
        for k in alphas.keys():
            domain = self.invert_a(k)
            card_domain = 1
            for d in domain:
                card_domain = card_domain * self.M0.get_cardinality(d)
            card_codomain = self.M1.get_cardinality(k)   
            print('Mapping alpha_{0}: {1} -> {2}'.format(k, card_domain, card_codomain))

    def plot_variable_level_mapping(self):
        G = self.M0.copy()
        relabel_map = {}
        for n in G.nodes():
            relabel_map[n] = 'M0_'+str(n)
        G0 = nx.relabel.relabel_nodes(G,relabel_map)

        G = self.M1.copy()
        relabel_map = {}
        for n in G.nodes():
            relabel_map[n] = 'M1_'+str(n)
        G1 = nx.relabel.relabel_nodes(G,relabel_map)

        U = nx.union(G0,G1)

        edge_list = [('M0_'+str(k), 'M1_'+str(self.a[k])) for k in self.a.keys()]
        U.add_edges_from(edge_list)

        pos = nx.shell_layout(U)

        for k in pos.keys():
            if 'M1' in k:
                pos[k] = pos[k] + [10,0]
                
        R_list = np.array(['M0_'+n for n in self.R])
        nR = list(set(self.M0.nodes()) - set(self.R))
        nR_list = np.array(['M0_'+n for n in nR])

        nx.draw_networkx_nodes(U,pos,nodelist=R_list,node_color='b',alpha=.5)
        nx.draw_networkx_nodes(U,pos,nodelist=nR_list,node_color='b',alpha=.2)
        nx.draw_networkx_labels(U,pos)
        nx.draw_networkx_edges(U,pos,edgelist=G0.edges(),edge_color='k')

        nx.draw_networkx_nodes(U,pos,nodelist=G1.nodes(),node_color='g',alpha=.5)
        nx.draw_networkx_labels(U,pos)
        nx.draw_networkx_edges(U,pos,edgelist=G1.edges(),edge_color='k')

        nx.draw_networkx_edges(U,pos,edgelist=edge_list,edge_color='r',style='dashed')
    
    def plot_DAG_M0(self):
        nx.draw(self.M0,with_labels='True')
        
    def plot_DAG_M1(self):
        nx.draw(self.M1,with_labels='True')
    
    def list_DAG_nodes(self):
        print("M0 - Nodes: {0}".format(self.M0.nodes))
        print("M1 - Nodes: {0}".format(self.M1.nodes))
        print("R  - Nodes: {0}".format(self.R))
        
    def list_DAG_edges(self):
        print("M0 - Edges: {0}".format(self.M0.edges))
        print("M1 - Edges: {0}".format(self.M1.edges))
              
        a_edges = []
        for k in self.a.keys():
            print
            a_edges.append((k, self.a[k]))                      
        print("a  - Edges: {0}".format(a_edges))
        
         
    def list_FinStoch_objects_M0(self):
        print("Objects (sets) in FinStoch picked by M0:")
        for n in self.M0.nodes():
            print("{0}: {1}".format(n, np.arange(self.M0.get_cardinality(n))))
        print("** Some sets may be repeated. FinStoch contains also all products. **")
        
    def list_FinStoch_objects_M1(self):
        print("Objects (sets) in FinStoch picked by M1:")
        for n in self.M1.nodes():
            print("{0}: {1}".format(n, np.arange(self.M1.get_cardinality(n))))
        print("** Some sets may be repeated. FinStoch contains also all products. **")
        
    def list_FinStoch_objects_R(self):
        print("Objects (sets) in FinStoch picked by R:")
        for n in self.R:
            print("{0}: {1}".format(n, np.arange(self.M0.get_cardinality(n))))
        print("** Some sets may be repeated. FinStoch contains also all products. **")
        
    def list_FinStoch_morphisms_M0(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by M0:")        
        for n in self.M0.nodes():
            print("phi_{0}: {1}  ->  {2}".format(n, self.M0.get_cpds(n).get_values().shape[1], self.M0.get_cpds(n).get_values().shape[0]))   
        
    def list_FinStoch_morphisms_M1(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by M1:")
        for n in self.M1.nodes():
            print("phi_{0}: {1}  ->  {2}".format(n, self.M1.get_cpds(n).get_values().shape[1], self.M1.get_cpds(n).get_values().shape[0]))   
        
    def list_FinStoch_morphisms_alphas(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by alphas:")
        for k in self.alphas.keys():
            print("alpha_{0}: {1}  ->  {2}".format(k, self.alphas[k].shape[1], self.alphas[k].shape[0]))
    
    
    def _check_path_between_sets(self,G,sources,targets):
        augmentedG = G.copy()

        augmented_s = 'augmented_s_'+str(np.random.randint(10**6))
        augmented_t = 'augmented_t_'+str(np.random.randint(10**6))
        augmentedG.add_node(augmented_s)
        augmentedG.add_node(augmented_t)

        [augmentedG.add_edge(augmented_s,s) for s in sources]
        [augmentedG.add_edge(t,augmented_t) for t in targets]

        return nx.has_path(augmentedG,augmented_s,augmented_t)
    
    def _powerset(self,iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    
    def _get_pairs_in_M1_with_directed_path_in_M1(self):
        J = []
        sources = list(self.M1.nodes())
        targets = list(sefl.M1.nodes())
        for s in sources:
            for t in list(set(targets)-{s}):
                if nx.has_path(self.M1,s,t):
                    J.append((s,t))
        return J
    
    def _get_all_pairs_in_M1(self):
        J = list(itertools.permutations(self.M1.nodes(),2))
        return J
    
    def _get_pairs_in_M1_with_directed_path_in_M1_or_M0(self):
        J = []
        sources = list(self.M1.nodes())
        targets = list(self.M1.nodes())
        for s in sources:
            for t in list(set(targets)-{s}):
                if nx.has_path(self.M1,s,t):
                    J.append((s,t))
                else:
                    M0_sources = self.invert_a(s)
                    M0_targets = self.invert_a(t)
                    if self._check_path_between_sets(self.M0,M0_sources,M0_targets):
                        J.append((s,t))
        return J
    
    def _get_sets_in_M1_with_directed_path_in_M1_or_M0(self,verbose=False):
        J = []
        sets = list(self._powerset(self.M1.nodes()))
        sets.remove(())

        for i in sets:
            for j in sets:
                M1_sources = list(i)
                M1_targets = list(j)
                if not(any(x in M1_sources for x in M1_targets)):            
                    if self._check_path_between_sets(self.M1,M1_sources,M1_targets):
                        if verbose: print('- Checking {0} -> {1}: True'.format(M1_sources,M1_targets))
                        J.append([M1_sources,M1_targets])
                    else:
                        print('- Checking {0} -> {1}: False'.format(M1_sources,M1_targets))
                        M0_sources = self.invert_a(M1_sources)
                        M0_targets = self.invert_a(M1_targets)
                        if self._check_path_between_sets(self.M0,M0_sources,M0_targets):
                            if verbose: print('---- Checking {0} -> {1}: True'.format(M0_sources,M0_targets))
                            J.append([M1_sources,M1_targets])
                        else:
                            if verbose: print('---- Checking {0} -> {1}: False'.format(M0_sources,M0_targets))
        if verbose: print('\n {0} legitimate pairs of sets out of {1} possbile pairs of sets'.format(len(J),len(sets)**2))  
        
        return J
    
    
    def _tensorize_list(self,tensor,l):
        if tensor is None:
            if len(l)>1:
                tensor = np.einsum('ij,kl->ikjl',l[0],l[1])
                tensor = tensor.reshape((tensor.shape[0]*tensor.shape[1],tensor.shape[2]*tensor.shape[3]))
                return self._tensorize_list(tensor,l[2:])
            else:
                return l[0]
        else:
            if len(l)>0:
                tensor = np.einsum('ij,kl->ikjl',tensor,l[0])
                tensor = tensor.reshape((tensor.shape[0]*tensor.shape[1],tensor.shape[2]*tensor.shape[3]))
                return self._tensorize_list(tensor,l[1:])
            else:
                return tensor
            
    def _tensorize_mechanisms(self,inference,sources,targets,cardinalities):
        joint_TS = inference.query(targets+sources,show_progress=False)
        marginal_S = inference.query(sources,show_progress=False)
        cond_TS = joint_TS / marginal_S

        cond_TS_val = cond_TS.values

        old_indexes = range(len(targets+sources))
        new_indexes = [(targets+sources).index(i) for i in joint_TS.variables]
        cond_TS_val = np.moveaxis(cond_TS_val, old_indexes, new_indexes)

        target_cards=[cardinalities[t] for t in targets]
        target_card = np.prod(target_cards)
        source_cards=[cardinalities[s] for s in sources]
        source_card = np.prod(source_cards)
        cond_TS_val = cond_TS_val.reshape(target_card,source_card)

        return cond_TS_val
    
    def evaluate_abstraction_error(self, metric=None, J_algorithm=None, verbose=False):
        if J_algorithm is None:
            J = self._get_sets_in_M1_with_directed_path_in_M1_or_M0(verbose=verbose)
        else:
            J = J_algorithm(A)
            
        if metric is None:
            metric = distance.jensenshannon
            
        abstraction_errors = []

        for pair in J:
            # Get nodes in the abstracted model
            M1_sources = pair[0]
            M1_targets = pair[1]
            if verbose: print('\nM1: {0} -> {1}'.format(M1_sources,M1_targets))

            # Get nodes in the base model
            M0_sources = self.invert_a(M1_sources)
            M0_targets = self.invert_a(M1_targets)
            if verbose: print('M0: {0} -> {1}'.format(M0_sources,M0_targets))

            # Perform interventions in the abstracted model and setup the inference engine
            M1do = self.M1.do(M1_sources)
            inferM1 = VariableElimination(M1do)

            # Perform interventions in the base model and setup the inference engine
            M0do = self.M0.do(M0_sources)
            inferM0 = VariableElimination(M0do)

            # Compute the high-level mechanisms
            M1_cond_TS_val = self._tensorize_mechanisms(inferM1,M1_sources,M1_targets,self.M1.get_cardinality())
            if verbose: print('M1 mechanism shape: {}'.format(M1_cond_TS_val.shape))

            # Compute the low-level mechanisms
            M0_cond_TS_val = self._tensorize_mechanisms(inferM0,M0_sources,M0_targets,self.M0.get_cardinality())
            if verbose: print('M0 mechanism shape: {}'.format(M0_cond_TS_val.shape))

            # Compute the alpha for sources
            alphas_S = [self.alphas[i] for i in M1_sources]
            alpha_S = self._tensorize_list(None,alphas_S)
            if verbose: print('Alpha_s shape: {}'.format(alpha_S.shape))

            # Compute the alpha for targers
            alphas_T = [self.alphas[i] for i in M1_targets]
            alpha_T = self._tensorize_list(None,alphas_T)
            if verbose: print('Alpha_t shape: {}'.format(alpha_T.shape))

            # Evaluate the paths on the diagram
            lowerpath = np.dot(M1_cond_TS_val,alpha_S)
            upperpath = np.dot(alpha_T,M0_cond_TS_val)

            # Compute abstraction error for every possible intervention
            distances = []
            for c in range(lowerpath.shape[1]):
                distances.append( distance.jensenshannon(lowerpath[:,c],upperpath[:,c]) )
            if verbose: print('All JS distances: {0}'.format(distances))

            # Select the greatest distance over all interventions
            if verbose: print('\nAbstraction error: {0}'.format(np.max(distances)))
            abstraction_errors.append(np.max(distances))

        # Select the greatest distance over all pairs considered
        if verbose: print('\n\nOVERALL ABSTRACTION ERROR: {0}'.format(np.max(abstraction_errors)))
            
        return abstraction_errors


