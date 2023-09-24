from typing import Dict
from auto_policy_programming.fsm.action import Action
from auto_policy_programming.fsm.state import State


class BaseWrapper:
    def __init__(self, env):
        self.env = env
        self.cur_raw_obs = None
        self.cur_state: State = None
        self.cur_state_name: str = None
        self.env_state_description: dict = None

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
