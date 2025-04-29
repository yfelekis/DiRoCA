import numpy as np

class CMNISTLinearSCM:
    def __init__(self, F):
        self.F = F
    
    def _compute_reduced_form(self):
        return self.F
    
    def simulate(self, exogenous, intervention=None):
        """
        Simulates the SCM using the reduced form matrix F.
        
        Args:
            exogenous: Input noise/exogenous variables
            intervention: Optional intervention values
            
        Returns:
            Simulated endogenous variables
        """
        endogenous = exogenous @ self.F
        
        if intervention is not None:
            for target, value in intervention.vv().items():
                target_idx = self.var_index[target]  # Get index of the intervened variable
                endogenous[:, target_idx] = value  # Set value for the intervened variable
        
        return endogenous 