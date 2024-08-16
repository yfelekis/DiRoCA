
import numpy as np
import torch
import itertools
from pgmpy.inference import VariableElimination


def get_alpha_dims(M0, M1, a):
    M0_card = M0.get_cardinality()
    M1_card = M1.get_cardinality()
    res = {}
    for k, v in a.items():
        if v not in res: res[v] = (M1_card[v], M0_card[k])
        else: res[v] = (M1_card[v], res[v][1]*M0_card[k])
    return res


def get_cards(variables,model):
    cards = {}
    for var in variables:
        variable_card = model.get_cardinality()[var]
        cards[var] = variable_card
    return cards


def get_variable_dimension(variables,model):
    cards = get_cards(variables,model)
    dimension = 1
    for card in cards.values():
        dimension *= card
    return dimension


def get_all_evidence_values(evidence_cards):
    evidence_permutations = {}
    for evidence in evidence_cards:
        evidence_permutations[evidence] = list(range(evidence_cards[evidence]))
    all_evidence_values = []
    for evidence_permutation in itertools.product(*list(evidence_permutations.values())):
        all_evidence_values.append(dict(zip(evidence_permutations.keys(), evidence_permutation)))
    return all_evidence_values


def construct_mechanism(infer, variables, variable_dimension,all_evidence_values):
    mechanism = np.zeros((variable_dimension, len(all_evidence_values)), dtype=np.float32)
    for i in range(len(all_evidence_values)):
        tmp_inference = infer.query(variables, evidence=all_evidence_values[i], show_progress=False)
        flatten_inf = np.array(tmp_inference.values, dtype=np.float32).flatten()
        mechanism[:, i] = flatten_inf
    return torch.from_numpy(mechanism)


def get_mechanisms_from_diagram(diagrams, model):
    infer = VariableElimination(model)
    mechanisms = {}
    for diagram in diagrams:
        evidences = diagram[0]
        variables = diagram[1]
        variable_dimension = get_variable_dimension(variables, model)
        evidence_cards = get_cards(evidences, model)
        all_evidence_values = get_all_evidence_values(evidence_cards)
        mechanism = construct_mechanism(infer, variables, variable_dimension, all_evidence_values)
        mechanisms[str(diagram)] = mechanism
    return mechanisms


def create_subscript(n):
    curr = '`'
    script = ''
    script_coma = ''
    for _ in range(n):
        script += chr(ord(curr) + 1)+chr(ord(curr) + 2)
        script_coma += chr(ord(curr) + 1)+chr(ord(curr) + 2)
        curr = script[-1]
        if _ != n-1: script_coma += ','
    mid = '->'
    odd = "".join([chr(ord(script[i])) for i in range(1,len(script),2)])
    even = "".join([chr(ord(script[i])) for i in range(0,len(script),2)])
    return script_coma + mid + even + odd


def one_hot_encoding(data, num_classes):
    res = np.zeros((data.shape[0],num_classes),dtype=np.float32)
    for i in range(data.shape[0]):
        tmp = np.zeros((num_classes),dtype=np.float32)
        tmp[data[i]] = 1
        res[i] = tmp
    return res


def one_hot_vector(mat2d, dims):
    res = np.zeros((mat2d.shape[0], np.prod(dims)), dtype=np.float32)
    for i in range(mat2d.shape[0]):
        tmp = np.zeros(dims, dtype=np.float32)
        tmp[mat2d[i]] = 1
        vec = tmp.flatten()
        res[i] = vec
    return res


def generate_data(M0_diagrams, M0, n_samples, r_seed):
    diagrams = set([tuple(diagram[0]) for diagram in M0_diagrams])
    diagrams = [list(dia) for dia in diagrams]
    torch_data_list = []
    data_index = {}
    for diagram in diagrams:
        do = M0.do(diagram)
        simulated_data = do.simulate(n_samples=n_samples, show_progress=False, seed=r_seed)
        if len(diagram) > 1:
            seq = [np.expand_dims(simulated_data[variable].to_numpy(dtype=np.float32),axis=1) for variable in diagram]
            selected_data = np.concatenate(seq, axis=1)
            card_dic = get_cards(diagram, M0)
            one_hot_shape = tuple([card_dic[variable] for variable in diagram])
            one_hot_data = one_hot_vector(selected_data.astype(int), one_hot_shape)
            torch_data_list.append(torch.from_numpy(one_hot_data))
            data_index[str(diagram)] = len(torch_data_list)-1
        else:
            selected_data = np.expand_dims(simulated_data[diagram].to_numpy(dtype=np.float32),axis=1)
            one_hot_shape = M0.get_cardinality()[diagram[0]]
            one_hot_data = one_hot_encoding(selected_data.astype(int),one_hot_shape)
            torch_data_list.append(torch.from_numpy(one_hot_data))
            data_index[str(diagram)] = len(torch_data_list)-1
            
    return torch_data_list, data_index