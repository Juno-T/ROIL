
class State:
    def __init__(self, state_name, state_observations: dict, state_actions: dict, description: str = '', raw_state: str=''):
        self.state_name = state_name
        self.state_observations = state_observations
        self.state_actions = state_actions
        self.description = description
        self.raw_state = raw_state