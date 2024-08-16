import numpy as np
import networkx as nx
import itertools
from scipy.spatial import distance

import src.utils as ut
import src.evaluationsets as es
import src.MechMappings as mm

from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD as cpd
from pgmpy.inference import VariableElimination



class SCMMapping():
    def __init__(self,M0,M1):
        self.M0 = M0
        self.M1 = M1
        
    def print_M0_cardinalites(self):
        for n in self.M0.nodes():
            print('M0: cardinality of {0}: {1}'.format(n,self.M0.get_cardinality(n)))
            
    def print_M1_cardinalites(self):
        for n in self.M1.nodes():
            print('M1: cardinality of {0}: {1}'.format(n,self.M1.get_cardinality(n)))
            
    def list_DAG_nodes(self):
        print("M0 - Nodes: {0}".format(self.M0.nodes))
        print("M1 - Nodes: {0}".format(self.M1.nodes))
        
    def list_DAG_edges(self):
        print("M0 - Edges: {0}".format(self.M0.edges))
        print("M1 - Edges: {0}".format(self.M1.edges))
              
        a_edges = []
        for k in self.a.keys():
            print
            a_edges.append((k, self.a[k]))                      
        print("a  - Edges: {0}".format(a_edges))
        
    def plot_DAG_M0(self):
        nx.draw(self.M0,with_labels='True')
        
    def plot_DAG_M1(self):
        nx.draw(self.M1,with_labels='True')
            
            
class Abstraction(SCMMapping):
    
    ### INITIALIZER
    
    def __init__(self,M0,M1,R,a,alphas,deterministic=False):
        super().__init__(M0,M1)             
        
        self.deterministic = deterministic
        
        if not ut.is_list_contained_in_list(R,M0.nodes): raise ValueError("R contains illegal nodes")
        if not ut.is_list_contents_unique(R): raise ValueError("R contains duplicated nodes")
        self.R = R
        self.nR = len(R)
        
        if not ut.is_list_contained_in_list(list(a.keys()),M0.nodes): raise ValueError("Domain of a contains illegal nodes")
        if not ut.is_list_contained_in_list(list(a.values()),M1.nodes): raise ValueError("Codomain of a contains illegal nodes")
        if not ut.is_surjective(list(a.values()),M1.nodes): raise ValueError("a is not surjective")
        self.a = a
        
        if not ut.is_list_contained_in_list(list(alphas.keys()),M1.nodes): raise ValueError("Alphas contains functions defined on illegal nodes")
        if not ut.is_surjective(list(alphas.keys()),M1.nodes): raise ValueError("Alphas does not contain all the functions")
        self._are_alphas_cardinalities_correct(alphas)
        self._are_alphas_stochastic_and_deterministic(alphas)
        self.alphas = alphas
        
            
    ### INITIALIZATION VERIFICATION FUNCTIONS    
    def _are_alphas_cardinalities_correct(self,alphas):
        for k in alphas.keys():
            card_domain,card_codomain = self.get_cardinalities_alpha(k)
            if card_domain != alphas[k].shape[1]: raise ValueError("Alpha_{0} domain is misspecified".format(k))
            if card_codomain != alphas[k].shape[0]: raise ValueError("Alpha_{0} codomain is misspecified".format(k))
                
    def _are_alphas_stochastic_and_deterministic(self,alphas):
        for k in alphas.keys():
            if not np.allclose(np.sum(alphas[k],axis=0),1): raise ValueError("Alpha_{0} is not stochastic".format(k))
            if self.deterministic:
                if not len(np.where(alphas[k]==1)[0])==alphas[k].shape[1]: raise ValueError("Alpha_{0} is not deterministic".format(k))

    
    ### UTILS    
    def copy(self):
        Acopy = Abstraction(self.M0, self.M1, 
                            R=self.R.copy(), a=self.a.copy(), alphas=self.alphas.copy())
        return Acopy
    
    def invert_a(self,v):
        return ut.inverse_fx(self.a,v)
    
    def get_cardinalities_alpha(self,alphakey):
        return ut.get_cardinalities_Falpha(self.a,alphakey,self.M0.get_cardinality(),self.M1.get_cardinality())

    
    ### ABSTRACTION PROPERTIES
    def is_varlevel_complete(self):
        return self.M0.number_of_nodes() == self.nR
    
    def is_varlevel_isomorphic(self):
        return self.nR == self.M1.number_of_nodes()
    
    def is_domlevel_isomorphic(self):
        for k in self.alphas.keys():
            print("Mapping alpha_{0}: {1}".format(k, alphas[k].shape[0]==alphas[k].shape[1]))
            
    def is_domlevel_deterministic(self):
        return self.deterministic
    
    
    ### PRINTING AND LISTING FUNCTIONS    
    def print_R_cardinalites(self):
        for n in self.R:
            print('R: cardinality of {0}: {1}'.format(n,self.M0.get_cardinality(n)))
    
    def print_relevant_vars(self):
        print(self.R)
    
    def print_mapping_a(self):
        print('** The mapping a is indexed by R/M0 **')
        print(self.a)
        
    def print_mappings_alphas(self):
        print('** The mappings alpha are indexed by M1 **')
        for k in self.alphas.keys():
            domain = self.invert_a(k)
            print('Mapping alpha_{0}: {1} -> {2}'.format(k, domain ,k))
    
    def print_mappings_alphas_cardinalities(self):
        print('** The mappings alpha are indexed by M1 **')
        for k in self.alphas.keys():
            card_domain,card_codomain = self.get_cardinalities_alpha(k)
            print('Mapping alpha_{0}: {1} -> {2}'.format(k, card_domain, card_codomain))
            
    def list_DAG_nodes(self):
        print("M0 - Nodes: {0}".format(self.M0.nodes))
        print("M1 - Nodes: {0}".format(self.M1.nodes))
        print("R  - Nodes: {0}".format(self.R))
                
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
    
    
    ### PLOTTING FUNCTIONS
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
           
    
    ### ABSTRACTION ERROR HELPER FUNCTIONS
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
    
    
    ### ABSTRACTION ERROR FUNCTION    
    def evaluate_abstraction_error(self, metric=None, J=None,J_algorithm=None, verbose=False, debug=False):
        if J is None and J_algorithm is None:
            J = es.get_sets_in_M1_with_directed_path_in_M1_or_M0(self.M0,self.M1,self.a,verbose=verbose)
        elif J is None:
            J = J_algorithm(self)
        
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
            
            print(alpha_S)
            # Evaluate the paths on the diagram
            lowerpath = np.dot(M1_cond_TS_val,alpha_S)
            upperpath = np.dot(alpha_T,M0_cond_TS_val)

            # Compute abstraction error for every possible intervention
            distances = []
            if debug: print('{0} \n\n {1}'.format(lowerpath,upperpath))
            for c in range(lowerpath.shape[1]):
                distances.append( metric(lowerpath[:,c],upperpath[:,c]) )
            if verbose: print('All JS distances: {0}'.format(distances))

            # Select the greatest distance over all interventions
            if verbose: print('\nAbstraction error: {0}'.format(np.max(distances)))
            abstraction_errors.append(np.max(distances))

        # Select the greatest distance over all pairs considered
        if verbose: print('\n\nOVERALL ABSTRACTION ERROR: {0}'.format(np.max(abstraction_errors)))
            
        return abstraction_errors
    
    def is_exact(self, metric=None, J_algorithm=None, verbose=False):
        print ('Abstraction approximation error is: {0}'.format(np.max(self.evaluate_abstraction_error(metric,J_algorithm=J_algorithm,verbose=verbose))))

        
    ### INFO LOSS FUNCTION    
    def compute_joints_and_invalpha(self, invalpha_algorithm=None, verbose=False):
        if invalpha_algorithm is None:
            invalpha, orderingM0, orderingM1 = self.invert_alpha_max_entropy()
        else:
            invalpha, orderingM0, orderingM1 = invalpha_algorithm(self)
        if verbose: print('Alpha^-1: {0}'.format(invalpha))
           
        inferM0 = VariableElimination(self.M0)
        joint_M0 = inferM0.query(orderingM0,show_progress=False)
        
        old_indexes = range(len(orderingM0))
        new_indexes = [(orderingM0).index(i) for i in joint_M0.variables]
        joint_M0 = np.moveaxis(joint_M0.values, old_indexes, new_indexes)
        joint_M0 = joint_M0.reshape((np.prod(joint_M0.shape),1))
        if verbose: print('M0 joint: {0}'.format(joint_M0))
        
        inferM1 = VariableElimination(self.M1)
        joint_M1 = inferM1.query(orderingM1,show_progress=False)
        
        old_indexes = range(len(orderingM1))
        new_indexes = [(orderingM1).index(i) for i in joint_M1.variables]
        joint_M1 = np.moveaxis(joint_M1.values, old_indexes, new_indexes)
        joint_M1 = joint_M1.reshape((np.prod(joint_M1.shape),1))
        if verbose: print('M1 joint: {0}'.format(joint_M1))
        if verbose: print('Transformed M1 joint: {0}'.format(np.dot(invalpha,joint_M1)))
        
        return joint_M0,joint_M1,invalpha
    
    def evaluate_info_loss(self, metric=None, invalpha_algorithm=None, verbose=False):
        if metric is None:
            metric = distance.jensenshannon
        
        joint_M0,joint_M1,invalpha = self.compute_joints_and_invalpha(invalpha_algorithm=invalpha_algorithm, verbose=verbose)
        info_loss = metric( joint_M0, np.dot(invalpha,joint_M1) )
        
        return info_loss
    
    def compute_global_alpha(self):
        orderingM1 = list(self.M1.nodes)

        Alpha = self.alphas[orderingM1[0]]
        for i in range(1,len(orderingM1)):
            Alpha = np.einsum('ij,kl->ikjl',Alpha,self.alphas[orderingM1[i]])
            Alpha = Alpha.reshape(Alpha.shape[0]*Alpha.shape[1],Alpha.shape[2]*Alpha.shape[3])

        notR = list(set(self.M0.nodes)-set(self.R))
        for nr in notR:
            Alpha = np.tile(Alpha,(1,self.M0.get_cardinality(nr)))

        orderingM0 = [self.invert_a(x) for x in orderingM1]
        orderingM0 = list(itertools.chain.from_iterable(orderingM0))
        orderingM0 = notR + orderingM0

        return Alpha, orderingM0, orderingM1
        
    def invert_alpha_max_entropy(self):
        Alpha, orderingM0, orderingM1 = self.compute_global_alpha()
        invalpha = np.transpose(Alpha)
        invalpha = invalpha / np.sum(invalpha,axis=0)

        return invalpha, orderingM0, orderingM1
    
    def invert_alpha_pinv(self):
        Alpha, orderingM0, orderingM1 = self.compute_global_alpha()
        invalpha = np.linalg.pinv(Alpha)

        return invalpha, orderingM0, orderingM1
        
    
    
    ### EFFECTIVE INFORMATION FUNCTION
    def evaluate_EIs(self, J_algorithm=None, base=2, verbose=False, debug=False):
        if J_algorithm is None:
            J = es.get_sets_in_M1_with_directed_path_in_M1_or_M0(self.M0,self.M1,self.a,verbose=verbose)
        else:
            J = J_algorithm(self)
            
        EIs_low = []; EIs_high = []

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

            # Compute the EI for the mechanisms
            _,EI_low = mm.EI(M0_cond_TS_val)
            _,EI_high = mm.EI(M1_cond_TS_val)
            EIs_low.append(EI_low)
            EIs_high.append(EI_high)

        # Output all the EIs computed
        if verbose: print('All EIs in low-level model: {0}'.format(EIs_low))
        if verbose: print('All EIs in high-level model: {0}'.format(EIs_high))
        if verbose: print('\n\nMIN EI IN LOW MODEL: {0}'.format(np.min(EIs_low)))
        if verbose: print('\n\nMIN EI IN HIGH MODEL: {0}'.format(np.min(EIs_high)))
            
        return EIs_low,EIs_high
    
    
    

class TauOmegaAbstraction(SCMMapping):
    def __init__(self,M0,M1,I0,I1,tau,omega):
        super().__init__(M0,M1)
        
        self.I0 = I0
        self.I1 = I1
        self.tau = tau
        
        if not ut.is_list_contained_in_list(list(omega.keys()),list(I0.keys())): raise ValueError("Domain of omega contains illegal elements")
        if not ut.is_list_contained_in_list(list(omega.values()),list(I1.keys())): raise ValueError("Codomain of omega contains illegal elements")
        if not ut.is_surjective(list(omega.values()),list(I1.keys())): raise ValueError("omega is not surjective")
        if not self._is_order_preserving(I0,I1,omega): raise ValueError("omega is not order-preserving")
        self.omega = omega
    
    def _is_order_preserving(self,I0,I1,omega):
        self.I0_poset = self._build_poset(I0)
        self.I1_poset = self._build_poset(I1)
        
        for e in self.I0_poset.edges():
            if not nx.has_path(self.I1_poset, omega[e[0]], omega[e[1]]): return False
        
        return True
    
    
    def _build_poset(self,I):        
        G0 = nx.DiGraph()
        G0.add_nodes_from(list(I.keys()))
        
        for k,v in I.items():    
            if len(v)==0:
                for k1 in I.keys():
                    if k1!=k: G0.add_edge(k,k1)

            else:
                for k1,v1 in I.items():
                    if k1!=k:
                        intervenednodeset_0 = list(I[k].keys())
                        intervenednodeset_1 = list(I[k1].keys())
                        
                        if ut.is_list_contained_in_list(intervenednodeset_0,intervenednodeset_1):
                            areintervenedvaluesequal = True
                            for i in intervenednodeset_0:
                                if I[k][i] != I[k1][i]:
                                    areintervenedvaluesequal = False

                            if areintervenedvaluesequal: 
                                G0.add_edge(k,k1)
        
        return G0