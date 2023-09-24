
class State:
    def __init__(self, name, observations: dict, actions: dict, description: str = '', raw_state: str=''):
        self.name = name
        self.observations = observations
        self.actions = actions
        self.description = description
        self.raw_state = raw_state