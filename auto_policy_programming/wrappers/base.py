class BaseWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)

    # any attribute fallback
    def __getattr__(self, name):
        return getattr(self.env, name)

    # def get_available_actions(self):
    #     return self.env.get_available_actions()
