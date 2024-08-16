
### Legacy code used in the first notebooks

import numpy as np
import networkx as nx
import itertools

class FinStochSCM():
    def __init__(self,n_endogenous, 
                 sets=None, links=None, stochmatrices=None,
                 f_sets=None, f_links=None, f_stochmatrices=None):
                
        self.X = np.arange(n_endogenous)
        self.nX = n_endogenous
        
        self.MX = self.set_sets(sets,f_sets)
        self.links = self.set_links(links,f_links)    
        self.G = self.build_DAG()
        
        self.MphiX = self.set_stochmatrices(stochmatrices,f_stochmatrices)
                    
        
    def set_sets(self,sets,f_sets):
        if sets is None:
            # TODO:check instantiate_random_sets is not null
            MX = f_sets(self.nX)
        else:
            # TODO:check sets is a list/array of length n_endogenous containing arrays
            MX = sets
        return MX
    
    def set_links(self,links,f_links):
        if links is None:
            # TODO:check instantiate_random_sets is not null
            L = f_links(self.X)
        else:
            # TODO:check sets is a list/array of length n_endogenous containing arrays
            L = links
        return L
            
    def set_stochmatrices(self,stochmatrices,f_stochmatrices):
        if stochmatrices is None:
            # TODO:check instantiate_random_stochmatrices is not null
            MphiX = f_stochmatrices(self.G,self.MX)
        else:
            # TODO:check
            MphiX = stochmatrices
        return MphiX
    
    def build_DAG(self):
        G = nx.DiGraph()
        G.graph['dpi'] = 120

        nodes = [str(i) for i in list(self.X)]
        edges = self.links
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        return G
    
    def copy(self):
        Mcopy = FinStochSCM(self.nX, sets=self.MX.copy(), links=self.links.copy(), stochmatrices=self.MphiX.copy())
        return Mcopy
    
    def plot_DAG(self):
        _ = nx.draw_networkx(self.G)          
        
    def add_labels(self,labels):
        self.labels = labels
        
        
class Abstraction():
    def __init__(self,M0,M1, nR, 
                 R=None, a=None, alphas=None,
                 f_R=None, f_a=None, f_alphas=None,
                 prettyprint=lambda x : x):
                       
        self.M0 = M0
        self.M1 = M1
        self.nR = nR
        
        self.R = self.set_R(R,f_R)
        self.a = self.set_a(a,f_a)
        self.alphas = self.set_alphas(alphas,f_alphas)
        
        self.prettyprint = prettyprint
        
    def set_R(self,R,f_R):
        if R is None:
            # TODO:check nR is less or equal to length of M0.X
            R = f_R(self.M0,self.nR)
        
        # TODO:check length of R matches nR
        return R

    def set_a(self,a,f_a):
        if a is None:
            # TODO:check 
            a = f_a(self.R,self.M1)
        
        # TODO:check dimensions of a
        return a

    def set_alphas(self,alphas,f_alphas):
        if alphas is None:
            # TODO:check
            alphas = f_alphas(self.M0,self.M1,self.R,self.a)
        
        # TODO:check dimensions of alpha
        return alphas
        
    def copy(self):
        Acopy = Abstraction(self.M0, self.M1, self.nR, 
                            R=self.R.copy(), a=self.a.copy(), alphas=self.alphas.copy())
        return Acopy
    
    def is_varlevel_complete(self):
        return self.M0.nX == self.nR
    
    def is_varlevel_isomorphic(self):
        return self.a.shape[0] == self.a.shape[1]
    
    def is_domlevel_isomorphic(self):
        for i,alpha in enumerate(self.alphas.values()):
            print("Mapping alpha_X'{0}: {1}".format(i, alpha.shape[0]==alpha.shape[1]))
    
    def print_M0_cardinalites(self):
        for i in range(self.M0.nX):
            print('M0: cardinality of X{0}: {1}'.format(i,len(self.M0.MX[i])))
    
    def print_R_cardinalites(self):
        for i in range(self.nR):
            print('R: cardinality of X{0}: {1}'.format(self.R[i],len(self.M0.MX[self.R[i]])))
            
    def print_M1_cardinalites(self):
        for i in range(self.M1.nX):
            print("M1: cardinality of X'{0}: {1}".format(i,len(self.M1.MX[i])))
    
    def print_relevant_vars(self):
        print('R = {0}'.format(', '.join(['X_'+str(r) for r in self.R])))
    
    def print_mapping_a(self):
        print(self.a)
        print('\n')
        print('Row indexes: {0}'.format(', '.join(["X"+str(r) for r in self.R])))
        print('Col indexes: {0}'.format(', '.join(["X'"+str(x) for x in self.M1.X])))
        
    def print_mappings_alphas(self):
        for i in range(self.a.shape[1]):
            domain = self.R[np.where(self.a[:,i]==1)[0]]
            print("Mapping alpha_X'{0}: {1} -> {2}".format(i, ', '.join(["X_"+str(d) for d in domain]), "X'_"+str(i)))
            
    def print_mappings_alphas_cardinalities(self):
        for i in range(self.a.shape[1]):
            domain = self.R[np.where(self.a[:,i]==1)[0]]
            card_domain = 1
            for d in domain:
                card_domain = card_domain * len(self.M0.MX[d])
            card_codomain = len(self.M1.MX[i])

            print("Cardinalities in mapping alpha_X'{0}: {1} -> {2}".format(i, card_domain, card_codomain))
       
    def plot_variable_level_mapping(self):
        G = self.M0.G.copy()
        relabel_map = {}
        for n in G.nodes():
            relabel_map[n] = 'M0_'+str(n)
        G0 = nx.relabel.relabel_nodes(G,relabel_map)

        G = self.M1.G.copy()
        relabel_map = {}
        for n in G.nodes():
            relabel_map[n] = 'M1_'+str(n)
        G1 = nx.relabel.relabel_nodes(G,relabel_map)

        U = nx.union(G0,G1)

        edge_list = [('M0_'+str(self.R[i]), 'M1_'+str(np.where(self.a[i,:]==1)[0][0])) for i in range(self.a.shape[0])]
        U.add_edges_from(edge_list)

        pos = nx.shell_layout(U)

        for k in pos.keys():
            if 'M1' in k:
                pos[k] = pos[k] + [10,0]

        R_list = np.array(['M0_'+str(self.R[i]) for i in range(len(self.R))])
        nR = np.array(list(set(self.M0.X)-set(self.R)))
        nR_list = np.array(['M0_'+str(nR[i]) for i in range(len(nR))])
        
        nx.draw_networkx_nodes(U,pos,nodelist=R_list,node_color='b',alpha=.5)
        nx.draw_networkx_nodes(U,pos,nodelist=nR_list,node_color='b',alpha=.2)
        nx.draw_networkx_labels(U,pos)
        nx.draw_networkx_edges(U,pos,edgelist=G0.edges(),edge_color='k')

        nx.draw_networkx_nodes(U,pos,nodelist=G1.nodes(),node_color='g',alpha=.5)
        nx.draw_networkx_labels(U,pos)
        nx.draw_networkx_edges(U,pos,edgelist=G1.edges(),edge_color='k')

        nx.draw_networkx_edges(U,pos,edgelist=edge_list,edge_color='r',style='dashed')
    
    def plot_DAG_M0(self):
        self.M0.plot_DAG()
        
    def plot_DAG_M1(self):
        self.M1.plot_DAG()
    
    def list_DAG_nodes(self):
        print("M0 - Nodes: {0}".format(', '.join(["X"+str(n) for n in self.M0.G.nodes()])))
        print("M1 - Nodes: {0}".format(', '.join(["X"+str(n) for n in self.M1.G.nodes()])))
        print("R  - Nodes: {0}".format(', '.join(["X"+str(n) for n in self.R])))
        
    def list_DAG_edges(self):
        print("M0 - Edges: {0}".format(';  '.join(["X"+str(u)+" -> X"+str(v) for u,v in self.M0.G.edges()])))
        print("M1 - Edges: {0}".format(';  '.join(["X'"+str(u)+" -> X'"+str(v) for u,v in self.M1.G.edges()])))
        alpha_edges = []
        for i in range(self.a.shape[1]):
            domain = self.R[np.where(self.a[:,i]==1)[0]]
            for d in domain:
                alpha_edges.append(("X"+str(d)+" -> X'"+str(i)))      
        print("a  - Edges: {0}".format(';  '.join(alpha_edges)))
        
         
    def list_FinStoch_objects_M0(self):
        print("Objects (sets) in FinStoch picked by M0:")
        for n in self.M0.G.nodes():
            print("X{0}: {1}".format(n, list(self.M0.MX[int(n)])))
        print("(Some sets may be repeated. FinStoch contains also all products.)")
        
    def list_FinStoch_objects_M1(self):
        print("Objects (sets) in FinStoch picked by M1:")
        for n in self.M1.G.nodes():
            print("X'{0}: {1}".format(n, list(self.M1.MX[int(n)])))
        print("(Some sets may be repeated. FinStoch contains also all products.)")
        
    def list_FinStoch_objects_R(self):
        print("Objects (sets) in FinStoch picked by R:")
        for n in self.M0.G.nodes():
            if (int(n) in self.R):
                print("X{0}: {1}".format(n, list(self.M0.MX[int(n)])))
        print("(Some sets may be repeated. FinStoch contains also all products.)")
        
    def list_FinStoch_morphisms_M0(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by M0:")
        for n in self.M0.G.nodes():
            dims = []
            for u,_ in self.M0.G.in_edges(n):
                dims.append(str(len(self.M0.MX[int(u)])))
            if dims:
                print("phi_X{0}: {1}  ->  {2}".format(n, ' x '.join(dims), len(self.M0.MX[int(n)])))
            else:
                print("phi_X{0}: *  ->  {1}".format(n, len(self.M0.MX[int(n)])))      
        
    def list_FinStoch_morphisms_M1(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by M1:")
        for n in self.M1.G.nodes():
            dims = []
            for u,_ in self.M1.G.in_edges(n):
                dims.append(str(len(self.M1.MX[int(u)])))
            if dims:
                print("phi_X'{0}: {1}  ->  {2}".format(n, ' x '.join(dims), len(self.M1.MX[int(n)])))
            else:
                print("phi_X'{0}: *  ->  {1}".format(n, len(self.M1.MX[int(n)])))
        
    def list_FinStoch_morphisms_alphas(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by alphas:")
        for i in range(self.a.shape[1]):
            dims = []
            doms = np.where(self.a[:,i]==1)[0]
            for d in doms:
                dims.append(str(len(self.M0.MX[self.R[d]])))
            print("alpha_X'{0}: {1}  ->  {2}".format(i, ' x '.join(dims), len(self.M1.MX[i])))
            