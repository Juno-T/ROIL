from enum import Enum
from typing import Any, Dict, List

class ActionType(Enum):
    FIXED = 0
    OPTIONS = 1
    TEXT = 2

class Action:
    def __init__(self, action_name, action_type: ActionType, options: Dict[str, List[Any]]=None, description: str = ''):
        self.action_name = action_name
        self.action_type = action_type
        self.options = options
        self.description = description