import numpy as np

def initialize_Q_zeros(n):
    return np.zeros(n)      

def initialize_Q_ones(n):
    return np.ones(n)

def initialize_Q_c(n,c):
    return np.ones(n)*c

def initialize_Q_map(Q, map_Q):
    return np.array([np.mean(Q[map_Q[k]]) for k in map_Q.keys()])

class Agent():
    def __init__(self,env,k, 
                 log_actions=True,log_rewards=True,log_Qs=True):
        self.env = env
        self.n_actions = k
        self.action_counter = np.zeros(self.n_actions)
        self.reward_sum = np.zeros(self.n_actions)
        
        self.log_actions = log_actions; self.log_rewards = log_rewards; self.log_Qs = log_Qs
        self.history_actions = []; self.history_rewards = []; self.history_Qs = []
        
    
    def initialize_Q(self,Q):
        self.Q = Q
    
    def get_cumulative_reward(self):
        return np.cumsum(np.array(self.history_rewards))
    
    def get_cumulative_regret(self):
        return range(1,len(self.history_rewards)+1) - np.cumsum(np.array(self.history_rewards))
    
    def get_optimality(self):
        return np.cumsum(np.array(self.history_actions)==self.env.optimal_action)
        
    def run(self,n_steps=1):
        for _ in range(n_steps):
            action = self._select_action()
            self.step(action)
    
    def imitate(self,D_actions):
        for action in D_actions:
            self.step(action)
            
    def replay(self,D_actions,D_rewards):
        for action,reward in zip(D_actions,D_rewards):
            self.step(action,fixedreward=reward)
        
    def step(self,action,fixedreward=None):
        if fixedreward is not None: reward = fixedreward
        else: _,reward,_,_ = self.env.step(action)
                
        if self.log_actions: self.history_actions.append(action)
        if self.log_rewards: self.history_rewards.append(reward)
        if self.log_Qs: self.history_Qs.append(self.Q.copy())
        self.reward_sum[action] = self.reward_sum[action] + reward
        self.action_counter[action] = self.action_counter[action]+1
        
        self._update_agent(action,reward)
        return reward            
    
class Agent_epsilon(Agent):
    
    def __init__(self,env,n_actions,eps):
        super().__init__(env,n_actions)
        self.eps = eps
           
    def _select_action(self):
        if(np.random.random() < self.eps):
            return np.random.randint(0,self.n_actions)
        else:
            return np.argmax(self.Q)
        
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + (1./self.action_counter[action])*(reward-self.Q[action])