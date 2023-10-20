from curses import raw
import re
from typing import List, Union
import gym
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from auto_policy_programming.wrappers.base import BaseWrapper
from auto_policy_programming.fsm.action import Action, ActionType
from auto_policy_programming.fsm.state import State
from auto_policy_programming.wrappers.webshop.observation_extractions import extract_observation

def state_name_transition(prev_state_name, action, cur_state_unprocessed_actions):
    try:
        action_type = action.split("[")[0]
        action_value = action.split("[")[1][:-1]
        if action_type == "search":
            return "results"
        elif action_type == "click":
            if action_value == "back to search":
                return "search"
            elif action_value == "description":
                return "item_description"
            elif action_value == "< prev":
                if 'click[description]' in cur_state_unprocessed_actions:
                    return "item"
                if prev_state_name == "item":
                    return "results"
                elif prev_state_name == "item_description":
                    return "item"
            elif action_value == "next >" and prev_state_name == "results":
                return "results"
            elif prev_state_name == "results":
                return "item"
            elif prev_state_name == "item" and action_value in ["features", "reviews"]:
                return "item"
        return prev_state_name
    except:
        return prev_state_name

fixed_actions = [
    'features',
    '< prev',
    'reviews',
    'next >',
    'description',
    'back to search',
    'buy now',
]

def clean_available_action(cur_state_name, available_action, clicked_options=[]):
    clicked_options = [c.lower() for c in clicked_options]
    processed_actions = []
    if cur_state_name=="search":
        processed_actions.append("search[<search query>]")
    elif cur_state_name=="results":
        processed_actions.append("click[<item_code>]")
    # elif cur_state_name=="item":
    #     processed_actions.append("click[<item_option>]")
    
    for a in available_action:
        if a.lower() in ["features", "reviews"]:
            continue
        if a.lower() in fixed_actions:
            processed_actions.append(f"click[{a}]")
        # if a.lower() in fixed_actions:
        if a.lower() in clicked_options:
            continue
        if cur_state_name=="item":
            processed_actions.append(f"click[{a}]")
    return processed_actions

class WebAgentTextEnvReActStyle(WebAgentTextEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_state_name = None
        self.instruction = None
    
    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.cur_state_name = "search"
        raw_obs = ret[0]
        matches = re.findall('Instruction[\n\s]*:[\n\s]*(.+?)\n', raw_obs, re.IGNORECASE)
        if len(matches) > 0:
            self.instruction = "Instruction: " + matches[-1].strip()
        else:
            self.instruction = raw_obs.split("[button]")[-1].strip()
        try:
            raw_obs = "[button]".join(raw_obs.split("[button]")[1:]).strip()
            raw_obs = "[button] " + raw_obs
        except:
            raw_obs = ""
        # raw_obs = raw_obs.replace("[button] ", "[")
        # raw_obs = raw_obs.replace("[button]", "[")
        # raw_obs = raw_obs.replace(" [button_]", "]")
        # raw_obs = raw_obs.replace("[button_]", "]")
        raw_obs = "search[<search query>]"
        return (raw_obs, ret[1])

    def get_instruction(self):
        return self.instruction

    def step(self, *args, **kwargs):
        ret = super().step(*args, **kwargs)
        self.cur_state_name = self.state_name_transition(args[0])
        matches = re.findall("^(You have clicked.*)[\s\n]*Instruction", ret[0], re.IGNORECASE)
        raw_obs = ret[0]
        if len(matches) > 0:
            raw_obs = matches[0]
        elif self.cur_state_name == "search":
            raw_obs = "search[<search query>]"
        else:
            try:
                if self.cur_state_name == "results":
                    raw_obs = "[button]".join(raw_obs.split("[button]")[1:6]).strip()
                raw_obs = "[button]".join(raw_obs.split("[button]")[1:]).strip()
                raw_obs = "[button] " + raw_obs
            except:
                raw_obs = ""
            raw_obs = raw_obs.replace("[button] ", "[")
            raw_obs = raw_obs.replace("[button]", "[")
            raw_obs = raw_obs.replace(" [button_]", "]")
            raw_obs = raw_obs.replace("[button_]", "]")
        return (raw_obs, ret[1], ret[2], ret[3])

    def state_name_transition(self, action: str):
        return state_name_transition(self.cur_state_name, action, self.get_available_actions()["clickables"])

    def get_cleaned_available_actions(self):
        return clean_available_action(self.cur_state_name, self.get_available_actions()["clickables"])

class WebAgentTextEnvWithStateName(WebAgentTextEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_state_name = None
        self.cur_raw_obs = None
    
    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.cur_state_name = "search"
        self.cur_raw_obs = ret[0]
        return ret
    
    def step(self, *args, **kwargs):
        ret = super().step(*args, **kwargs)
        self.cur_state_name = self.state_name_transition(args[0])
        self.cur_raw_obs = ret[0]
        return ret

    def state_name_transition(self, action: str):
        return state_name_transition(self.cur_state_name, action, self.get_available_actions()["clickables"])

    def get_cleaned_available_actions(self):
        clicked_options = []
        if self.cur_state_name == "item":
            matches = re.findall("You have clicked (.*)\.", self.cur_raw_obs, re.IGNORECASE)
            if len(matches) > 0:
                clicked_options = [m.strip().lower() for m in matches]
                # print(matches[0], clicked_options)
        return clean_available_action(self.cur_state_name, self.get_available_actions()["clickables"], clicked_options)
        

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
            "instruction": "Description of the desired item.",
        },
        "actions": {
            "search": "Search for the desired item given the search input.",
        },
    },
    "results": {
        "_description": "Results page contains a paginated list of items that match the search query.",
        "observations": {
            "instruction": "Description of the desired item.",
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
            "instruction": "Description of the desired item.",
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
            "instruction": "Description of the desired item.",
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
        self.cur_raw_obs = None
        self.all_states_actions = {
            k: set(v['actions'].keys()) for k, v in webshop_state_description.items()
        }
        self.cur_state: State = None
        self.cur_state_name = None
        self.env_state_description = webshop_state_description
    
    def state_name_transition(self, action):
        action_type = action.split("[")[0]
        action_value = action.split("[")[1][:-1]
        if action_type == "search":
            return "results"
        elif action_type == "click":
            if action_value == "back to search":
                return "search"
            elif action_value == "description":
                return "item_description"
            elif action_value == "< prev":
                if self.cur_state_name == "item":
                    return "results"
                elif self.cur_state_name == "item_description":
                    return "item"
            elif not action_value in webshop_state_description[self.cur_state_name]['actions'] and self.cur_state_name == "results":
                return "item"
        return self.cur_state_name

    def format_action(self, state: State, action: Action, aux: Union[str, dict]=None) -> List[str]:
        if action.type == ActionType.TEXT:
            return [f"{action.name}[{aux}]"]

        if action.type == ActionType.FIXED:
            return [f"click[{action.name}]"]

        if action.type == ActionType.OPTIONS:
            if state.name == "results":
                return [f"click[{state.observations['options']['value']['items'][aux['items']][0]}]"]
            elif state.name == "item":
                actions = []
                for category in aux:
                    if category in state.observations['options']['value']:
                        actions.append(f"click[{state.observations['options']['value'][category][aux[category]]}]")
                # Auto buy after selecting option
                actions.append(f"click[buy now]")
                return actions

        raise Exception(f"Unknown action {action}")

    def step(self, *args, **kwargs):
        if isinstance(args[0], str):
            ret = list(self.env.step(*args, **kwargs))
            self.cur_state_name = self.state_name_transition(args[0])
            self.cur_raw_obs = ret[0]
            ret[0] = self._typed_state(ret[0])
            self.cur_state = ret[0]
            return ret
        else:
            action = args[0]
            aux = None
            if len(args) > 1:
                aux = args[1]
            actions = self.format_action(self.cur_state, action, aux)
            for action in actions:
                assert isinstance(action, str)
                ret = list(self.step(action, **kwargs))
            return ret

    def reset(self, *args, **kwargs):
        ret = list(self.env.reset(*args, **kwargs))
        self.cur_state_name = "search"
        ret[0] = self._typed_state(ret[0])
        self.cur_state = ret[0]
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

    def _typed_state(self, raw_state) -> State:
        action_dict = self.get_typed_available_actions(raw_state)

        # add description to action
        state_desctiptions = webshop_state_description[self.cur_state_name]
        for action_name, action in action_dict.items():
            if action_name in state_desctiptions['actions']:
                action.description = state_desctiptions['actions'][action_name]


        observation = extract_observation(raw_state, self.cur_state_name)
        if self.cur_state_name == "results":
            # Result page
            action_dict['options'] = Action(
                "options", ActionType.OPTIONS, options=observation["options"]['value']
            )

        elif self.cur_state_name == "item":
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
            self.cur_state_name,
            observation,
            action_dict,
            description=state_desctiptions["_description"],
            raw_state=raw_state,
        )
