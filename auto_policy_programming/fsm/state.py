
class State:
    def __init__(self, name, observation: dict, actions: dict, description: str = '', raw_state: str=''):
        self.name = name
        self.observation = observation
        self.actions = actions
        self.description = description
        self.raw_state = raw_state