import numpy as np
from pgmpy.models import BayesianNetwork as BN
from pgmpy.factors.discrete import TabularCPD as cpd


M0_mech = {
    "Battery_Size": np.array([[0.3],[0.35],[0.15],[0.2]], dtype=np.float32),
    "Weight": np.array([[0.6,0.5,0.3,0.1],[0.3,0.45,0.55,0.2],[0.1,0.05,0.15,0.7]], dtype=np.float32),
    "Efficiency": np.array([[0.5,0.3,0.1],[0.3,0.3,0.15],[0.1,0.3,0.15],[0.1,0.1,0.6]], dtype=np.float32),
    "Range": np.array([
              [0.5,0.4,0.3,0.1,0.55,0.43,0.35,0.1,0.65,0.5,0.2,0.15,0.7,0.55,0.35,0.18],
              [0.35,0.35,0.35,0.15,0.35,0.32,0.4,0.2,0.3,0.35,0.45,0.25,0.25,0.35,0.3,0.22],
              [0.10,0.15,0.2,0.35,0.08,0.15,0.2,0.4,0.04,0.1,0.25,0.35,0.04,0.07,0.2,0.4],
              [0.05,0.1,0.15,0.4,0.02,0.1,0.05,0.3,0.01,0.05,0.1,0.25,0.01,0.03,0.15,0.2]
            ], dtype=np.float32)
}

M1_mech = {
    "Weight_": np.array([[0.2],[0.2],[0.6]], dtype=np.float32),
    "Efficiency_": np.array([[0.5,0.3,0.1],[0.3,0.3,0.15],[0.1,0.3,0.15],[0.1,0.1,0.6]], dtype=np.float32),
    "Range_": np.array([[0.05,0.15,0.1,0.25],[0.15,0.15,0.25,0.45],[0.3,0.35,0.4,0.25],[0.5,0.35,0.25,0.05]], dtype=np.float32)
}


M0 = BN([('Battery_Size','Weight'),('Battery_Size','Range'),('Weight','Efficiency'),('Efficiency','Range')])
cpdB = cpd(variable='Battery_Size',
          variable_card=4,
          values=M0_mech['Battery_Size'],
          evidence=None,
          evidence_card=None)
cpdS = cpd(variable='Weight',
          variable_card=3,
          values=M0_mech['Weight'],
          evidence=['Battery_Size'],
          evidence_card=[4])
cpdT = cpd(variable='Efficiency',
          variable_card=4,
          values=M0_mech['Efficiency'],
          evidence=['Weight'],
          evidence_card=[3])
cpdC = cpd(variable='Range',
          variable_card=4,
          values = M0_mech['Range'],
          evidence=['Efficiency','Battery_Size'],
          evidence_card=[4,4])
M0.add_cpds(cpdB,cpdS,cpdT,cpdC)
M0.check_model()


M1 = BN([('Weight_','Efficiency_'),('Efficiency_','Range_')])
cpdS = cpd(variable='Weight_',
          variable_card=3,
          values=M1_mech['Weight_'],
          evidence=None,
          evidence_card=None)
cpdT = cpd(variable='Efficiency_',
          variable_card=4,
          values=M1_mech['Efficiency_'],
          evidence=['Weight_'],
          evidence_card=[3])
cpdC = cpd(variable='Range_',
          variable_card=4,
          values=M1_mech['Range_'],
          evidence=['Efficiency_'],
          evidence_card=[4])
M1.add_cpds(cpdS,cpdT,cpdC)
M1.check_model()


R = ['Weight','Efficiency','Range','Battery_Size']


a = {'Battery_Size': 'Weight_',
     'Weight': 'Weight_',
     'Efficiency': 'Efficiency_',
     'Range': 'Range_'}


alphas = {"Weight_": np.array([[0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1.],
                  [1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]]),
         "Efficiency_": np.array([[0., 0., 0., 1.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [1., 0., 0., 0.]]),
         "Range_": np.array([[1., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., 1., 0.],
                  [0., 1., 0., 0.]])}


def load_example():
    return M0, M0_mech, M1, M1_mech, R, a, alphas