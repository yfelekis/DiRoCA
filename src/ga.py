
import numpy as np
import src.utils as ut


def generate_random_population(size,matrices):
    """
    Instantiate a new vector population

    Args:
        size: an integer denoting the size of population
        matrices: a list with the size of the matrices

    Returns:
        List of individuals. Every individual is a list of numpy arrays.
        
    Example: 
        size = 10;
        matrices = [[3,4],[2,2]];
        res = [[array([1, 1, 2, 0]), array([0, 0])],
               [array([2, 2, 2, 1]), array([1, 0])]....
    """
    new_population = []

    for _ in range(size):
        new_individual = []
        for i in range(len(matrices)):
            new_vector = np.random.randint(matrices[i][0],size=matrices[i][1])
            new_individual.append(new_vector)
        new_population.append(new_individual)
    
    return new_population

def convert_individual_vector_to_alphas(individual,matrices,alpha_labels):
    """
    Translate an individual vector into an alpha

    Args:
        individual: a list of numpy arrays 
        matrices: a list with the size of the matrices associated with each numpy array
        alpha_labels: a list of the labels associated with each matrix

    Returns:
        Dictionary of alpha maps

    Example: 
        individual = [[array([1, 1, 2, 0]), array([0, 1])]];
        matrices = [[3,4],[2,2]];
        alpha_labels = ['A','B'];
        res = {'A': [[0,0,0,1],[1,1,0,0],[0,0,1,0]]; 'B': [[1,0],[0,1]]}
        
    """
    alphas = {}
    for i in range(len(individual)):
        alphas[alpha_labels[i]] = ut.map_vect2matrix(individual[i],matrices[i][0])
    return alphas