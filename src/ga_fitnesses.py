
import numpy as np
import src.utils as ut
import src.ga as ga

def fitness_addittive_surjective_penalty(individual,matrices):
    penalty = 0
    for i in range(len(individual)):
        individualmatrix = ut.map_vect2matrix(individual[i],matrices[i][0])
        if not ut.is_matrix_surjective(individualmatrix): penalty+=1    
    return penalty

def fitness_jsd(individual,Aev,matrices,alpha_labels, J=None):
    new_alphas = ga.convert_individual_vector_to_alphas(individual,matrices,alpha_labels)
    Aev.A.alphas = new_alphas
    Aev.aggoverall = np.sum
    return Aev.compute_overall_error(J=J)