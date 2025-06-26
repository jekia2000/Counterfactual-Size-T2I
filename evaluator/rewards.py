import yaml

class SizeReward:
    def __init__(self, reward_config):
        opt = {}
        with open(reward_config, 'r') as file:
            opt = yaml.safe_load(file)

        
        self.tau = opt['tau']
        self.alpha = opt['alpha']
        self.g = opt['g']

    def get_reward(self, big_area, small_area):
        reward = 0.0
        
        if not(small_area == float('-inf') or big_area == float('inf')):
            if small_area > big_area:
                reward = min( self.tau, (1.0*small_area/big_area)**self.alpha)
                
            else:
                reward = max(-1*self.tau, -1*(1.0*big_area/small_area)**self.alpha)
            
        elif (small_area == float('-inf') and big_area == float('inf')):
            return -1.0*self.tau*(1+self.g)**2, 0
        
        else:
            return -1.0*self.tau*(1 + self.g), 0

        return reward, int(reward == 1.5)
    
    def get_advantage(self, big_area, small_area):
        reward = 0.0
        
        if not(small_area == float('-inf') or big_area == float('inf')):
            if small_area > big_area:
                reward = 1.0*small_area/big_area
                
            else:
                reward = -1*(1.0*big_area/small_area)
            
        elif (small_area == float('-inf') and big_area == float('inf')):
            return -100.0, 0
        
        else:
            return -200.0, 0

        return reward, int(reward > 0)
    
    
    