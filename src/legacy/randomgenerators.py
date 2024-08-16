import numpy as np
import networkx as nx
import itertools


def set_seed(seed):
     np.random.seed(seed)

def instantiate_random_sets_1(n_sets):
    cardinalities = np.random.randint(1,10,n_sets)
    return [np.arange(cardinalities[i]) for i in range(len(cardinalities))]

def instantiate_random_links_1(X):
    n_links = np.max([2,int(np.floor(len(X)/1.5))])
    links = []
    
    for i in range(n_links):
        link = np.random.choice(X,2,replace=False)
        links.append((str(link[0]), str(link[1])))
    
    return links

def instantiate_random_sets_large_1(n_sets):
    cardinalities = np.random.randint(1,12,n_sets)
    return [np.arange(cardinalities[i]) for i in range(len(cardinalities))]

def instantiate_random_sets_small_1(n_sets):
    cardinalities = np.random.randint(1,3,n_sets)
    return [np.arange(cardinalities[i]) for i in range(len(cardinalities))]

def instantiate_random_stochmatrices_1(G,MX):
    stochmatrices = {}
    
    for n in G.nodes():
        in_edges = (G.in_edges(n))

        card_domains = 1
        for ie in in_edges:
            card_domains = card_domains * len(MX[int(ie[0])])
            
        card_codomain = len(MX[int(n)])
        
        matrix = np.random.rand(card_domains,card_codomain)
        stochmatrix = matrix / np.sum(matrix,axis=1)[:,None]
        stochmatrices[n] = stochmatrix
    
    return stochmatrices



def generate_random_R_1(M0,nR):
    # Check nR is less or equal to the variables in M0
    return np.sort(np.random.choice(M0.X,nR,replace=False))

def generate_random_a_1(R,M1):
    # Check R is greater or equal to the variables in M1
    diff = len(R) - M1.nX
    codomain = list(M1.X.copy()) + list(np.random.choice(M1.X,diff))
    np.random.shuffle(codomain)
    
    a = np.zeros((len(R),M1.nX))
    a[np.arange(len(R)),codomain] = 1
    
    return a

def generate_random_alphas_1(M0,M1,R,a):
    alphas = {}
    for n in M1.X:
        incoming_vars = np.where(a[:,n]==1)[0]

        domain = M0.MX[R[incoming_vars[0]]]

        for v in incoming_vars[1:]:
            domain = [x for x in itertools.product(domain,M0.MX[R[int(v)]])]

        if len(domain) < len(M1.MX[n]):
            raise Exception('Can not define a surjective alpha with this a!')
            
        else:
            diff = len(domain) - len(M1.MX[n])
            codomain = list(M1.MX[n].copy()) + list(np.random.choice(M1.MX[n],diff))
            np.random.shuffle(codomain)
        
            alpha = np.zeros((len(domain),len(M1.MX[n])))
            alpha[np.arange(len(domain)),codomain] = 1

            alphas[n] = alpha

    return alphas
