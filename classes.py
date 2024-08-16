import cvxpy as cp
import numpy as np
from IPython.utils import io
from pgmpy import inference
from src.examples import smokingmodels as sm


class Intervention:
    
    def __init__(self, model, intervention):
        
        if intervention != {None:None}:
            self.model = model.do(intervention.keys())
        else: 
            self.model = model
            
        base_vars = list(model.nodes)
        abst_vars = list(model.nodes)
        
        self.base_structure = {}
        for i, var in enumerate(base_vars):
            self.base_structure[var] = i

        self.abst_structure = {}
        for i, var in enumerate(abst_vars):
            self.abst_structure[var] = i
            
        self.intervention = intervention
        self.intervention_var = list(self.intervention.keys())
        self.intervention_val = list(self.intervention.values())
        
    def get_value(self):
        return list(self.intervention.values())
        
    def get_variable(self):
        return list(self.intervention.keys())
        
    def get_base_criteria(self):
        return [(self.base_structure[var], value) for var, value in zip(self.get_variable(), self.get_value())]
    
    def get_abst_criteria(self):
        return [(self.abst_structure[var], value) for var, value in zip(self.get_variable(), self.get_value())]
    

class Pair:
    
    def __init__(self, base_dict, abst_dict, iota_base, iota_abst):
        self.base_dict         = base_dict
        self.abst_dict         = abst_dict
        self.iota_base         = iota_base
        self.iota_abst         = iota_abst
        self.base_distribution = list(self.base_dict.values())
        self.abst_distribution = list(self.abst_dict.values())
        self.base_labels       = list(self.base_dict.keys())
        self.abst_labels       = list(self.abst_dict.keys())
        

    def get_domain(self, model):
        dom = []
        if model == 'base':
            if self.iota_base.get_variable() == [None]:
                return self.base_labels
            for label in self.base_labels:
                if all(label[var] == val for var, val in self.iota_base.get_base_criteria()):
                    dom.append(label)
                    
        elif model == 'abst':
            if self.iota_abst.get_variable() == [None]:
                return self.abst_labels
            for label in self.abst_labels:
                if all(label[var] == val for var, val in self.iota_abst.get_abst_criteria()):
                    dom.append(label)
        
        return dom


class Poset: 
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def to_path(self, current_path):
        path = current_path[:]
        path.append(self)
        return path
    
    def get_child(self, current_path):
        for i in range(len(current_path) - 1):
            if current_path[i] == self:
                return current_path[i + 1]
        return None