from enum import Enum
import gym
import gym.spaces as spaces
from pandas import describe_option

from auto_policy_programming.wrappers.base import BaseWrapper
from auto_policy_programming.fsm.action import Action, ActionType
from auto_policy_programming.fsm.state import State


webshop_action_keywords = {
    "search": Action("search", ActionType.TEXT),
    "back to search": Action("back to search", ActionType.FIXED),
    "< prev": Action("< prev", ActionType.FIXED),
    "next >": Action("next >", ActionType.FIXED),
    "description": Action("description", ActionType.FIXED),
    "features": Action("features", ActionType.FIXED),
    "reviews": Action("reviews", ActionType.FIXED),
    "buy now": Action("buy now", ActionType.FIXED),
}

webshop_state_description = {
    "search": {
        "_description": "",
        "observations": {
            "instruction": "",
        },
        "actions": {
            "search": "",
        },
    },
    "results": {
        "_description": "",
        "observations": {
            "instruction": "",
            "items": "A list of triplet (item_code, item_name, item_price).",
        },
        "actions": {
            "back to search": "",
            "< prev": "",
            "next >": "",
            "options": "",
        },
    },
    "item": {
        "_description": "",
        "observations": {},
        "actions": {
            "< prev": "back to result page",
            "description": "",
            "features": "",
            "reviews": "",
            "buy now": "",
            "options": "",
        },
    },
    "item_description": {
        "_description": "",
        "observations": {},
        "actions": {
            "< prev": "back to item page",
            "back to search": "",
        },
    },
}


def wrap_text_env(env):
    return WebAgentTextEnvTypedState(env)


class WebAgentTextEnvTypedState(BaseWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.current_raw_obs = None
        self.all_states_actions = {
            k: set(v['actions'].keys()) for k, v in webshop_state_description.items()
        }

    def step(self, *args, **kwargs):
        ret = list(self.env.step(*args, **kwargs))
        self.current_raw_obs = ret[0]
        ret[0] = self._typed_state(ret[0])
        return ret

    def reset(self, *args, **kwargs):
        ret = list(self.env.reset(*args, **kwargs))
        ret[0] = self._typed_state(ret[0])
        return ret

    def get_typed_available_actions(self, raw_state):
        raw_actions = self.env.get_available_actions()["clickables"]
        action_dict = {}
        option_lists = []
        for action in raw_actions:
            if action.lower() in webshop_action_keywords:
                action_dict[action.lower()] = webshop_action_keywords[action.lower()]
            else:
                option_lists.append(action)
        option_dict = {}
        if len(option_lists) > 0:
            elements = raw_state.split(" [SEP] ")
            cur_set = []
            for element in elements[::-1]:
                if element in option_lists:
                    cur_set.append(element)
                elif len(cur_set) > 0:
                    option_dict[element] = cur_set
                    cur_set = []
        if len(option_dict) > 0:
            action_dict["options"] = Action(
                "options", ActionType.OPTIONS, options=option_dict
            )
        return action_dict

    def _typed_state(self, raw_state):
        action_dict = self.get_typed_available_actions(raw_state)

        # find state with most actions in common
        actions_set = set(action_dict.keys())
        max_intersection = 0
        max_intersection_state = None
        for state, state_actions in self.all_states_actions.items():
            intersection = len(actions_set.intersection(state_actions))
            if intersection > max_intersection:
                max_intersection = intersection
                max_intersection_state = state

        # add description to action
        state_desctiptions = webshop_state_description[max_intersection_state]
        for action_name, action in action_dict.items():
            if action_name in state_desctiptions['actions']:
                action.description = state_desctiptions['actions'][action_name]

        return State(
            max_intersection_state,
            {},
            action_dict,
            description=state_desctiptions["_description"],
            raw_state=raw_state,
        )
