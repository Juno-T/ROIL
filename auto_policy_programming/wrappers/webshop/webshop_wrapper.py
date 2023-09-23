import gym
from auto_policy_programming.wrappers.base import BaseWrapper
from auto_policy_programming.fsm.action import Action, ActionType
from auto_policy_programming.fsm.state import State
from auto_policy_programming.wrappers.webshop.observation_extractions import extract_observation


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

remove_from_clickables = ["features", "reviews"]

webshop_state_description = {
    "search": {
        "_description": "Search page contains a search input which can be use to search for desired items.",
        "observations": {
            "instruction": "Instruction to find the desired item.",
        },
        "actions": {
            "search": "Search for the desired item given the search input.",
        },
    },
    "results": {
        "_description": "Results page contains a paginated list of items that match the search query.",
        "observations": {
            "instruction": "Instruction to find the desired item.",
            "options": "Result items of the search query."
        },
        "actions": {
            "back to search": "Go back to search page",
            "< prev": "Go to previous page",
            "next >": "Go to next page",
            "options": "Select an item and navigate to the item page.",
        },
    },
    "item": {
        "_description": "Item page contains overview of the item, including item's options and buy now button.",
        "observations": {
            "instruction": "Instruction to find the desired item.",
            "item_name": "Item name.",
            "price": "Item price in string.",
            "options": "A dictionary of categorized options for the item.",
        },
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
        "observations": {
            "instruction": "Instruction to find the desired item.",
            "item_description": "",
        },
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
        raw_actions = [a for a in raw_actions if not a in remove_from_clickables]
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
                elif len(cur_set) > 0 and (not element in webshop_action_keywords): # ignore result page
                    option_dict[element] = cur_set
                    cur_set = []
                else:
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
        matched_state_name = None
        for state, state_actions in self.all_states_actions.items():
            intersection = len(actions_set.intersection(state_actions))
            if intersection > max_intersection:
                max_intersection = intersection
                matched_state_name = state

        # add description to action
        state_desctiptions = webshop_state_description[matched_state_name]
        for action_name, action in action_dict.items():
            if action_name in state_desctiptions['actions']:
                action.description = state_desctiptions['actions'][action_name]
        

        observation = extract_observation(raw_state, matched_state_name)
        if matched_state_name == "results":
            # Result page
            action_dict['options'] = Action(
                "options", ActionType.OPTIONS, options=observation["options"]['value']
            )

        elif matched_state_name == "item":
            # Item page
            if not "options" in action_dict:
                action_dict['options'] = Action(
                    "options", ActionType.OPTIONS, options={}
                )
            observation['options'] = {
                "dtype": "Dict[category_name: str, List[option: str]], or an empty dict \{\} if no options",
                "value": action_dict["options"].options
            }

        return State(
            matched_state_name,
            observation,
            action_dict,
            description=state_desctiptions["_description"],
            raw_state=raw_state,
        )
