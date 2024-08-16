
class SCMEnv():
    def __init__(self,scm,actions,target,optimal_action):
        self.scm = scm
        self.actions = actions
        self.target = target
        self.optimal_action = optimal_action
        
    def get_optimal_action(self):
        return optimal_action   
        
    def reset(self):
        return None, 0, False, None
          
    def step(self,a):
        sample = self.scm.simulate(n_samples=1,do=self.actions[a],show_progress=False)
        return None, sample[self.target][0], True, None
    
    def multistep(self,a,n_steps):
        sample = self.scm.simulate(n_samples=n_steps,do=self.actions[a],show_progress=False)
        return None, sample[self.target][0], True, None 