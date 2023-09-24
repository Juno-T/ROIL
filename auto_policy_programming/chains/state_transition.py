from ast import Dict
from calendar import c
from typing import Any, Dict, List
from typing_extensions import override
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.schema.language_model import BaseLanguageModel
from auto_policy_programming.chains import ParsingChain
from auto_policy_programming.fsm.state import State


class StateTransition(ParsingChain):
    env_state_description: Dict[str, Any]
    def __init__(self, llm: BaseLanguageModel, env_state_description: dict):
        super().__init__(prompt=prompt_template, llm=llm, env_state_description=env_state_description)

    @property
    @override
    def output_keys(self) -> List[str]:
        return ["state_transition"]
    
    @override
    def input_pre_format(self, inputs: Dict) -> Dict[str, Any]:
        state: State = inputs["state"]
        cur_state_name = state.name
        state_list = list(self.env_state_description.keys())
        state_list_description = "\n".join([f"{k}: {v['_description']}" for k, v in self.env_state_description.items()])
        action_list = list(state.actions.keys())
        action_list_description = "\n".join([f"{k}: {self.env_state_description[cur_state_name]['actions'][k]}" for k in action_list])
        state_observation_arg_list = ", ".join([f"{k}: \"{v['dtype']}\" = None" for k, v in state.observations.items()])
        state_observation_description = "\n".join([f"{k}: {v['dtype']} # {self.env_state_description[cur_state_name]['observations'][k]}" for k, v in state.observations.items()])
        parsed_input = {
            "webshop_goal": "Search and buy an item that match with the user's description by browsing web shopping interface.",
            "cur_state_name": str(cur_state_name),
            "state_len": str(len(state_list)),
            "state_list": str(state_list),
            "state_list_description": str(state_list_description),
            "action_len": str(len(action_list)),
            "action_list": str(action_list),
            "action_list_description": str(action_list_description),
            "state_observation_arg_list": str(state_observation_arg_list),
            "state_observation_description": str(state_observation_description),
            "state_observation_len": str(len(state.observations)),
        }
        return parsed_input

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        # TODO: parse code block from output
        # raise NotImplementedError()
        return {"state_transition": output}

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a Python programmer that will always write a complete function or program when asked."
        ),
        HumanMessagePromptTemplate.from_template(
            """You are assigned to build a finite state machine with python. This fsm is designed for the task of "Buy an item that match with the user's description by browsing web shopping interface.". It should be able to run automatically without relying on additional user's instructions. 
Everything is linked except for one state.

List of all states:
{state_list_description}

Your task is to complete {cur_state_name} state's transition funciton.

In {cur_state_name} state, there are {action_len} possible actions:
{action_list_description}

In {cur_state_name} state, you will see {state_observation_len} observations:
{state_observation_description}

When response, please follows these steps.

Step 1. Re-iterate the request.

Step 2. Now, for each actions, try forming a sentence like "Since I'm alone with a human expert, based on rational reasoning, I should execute <action_name> action when <something conditioned on short_description, options, prev_state, visited items, and user_input>".

You can ask the human expert by calling `ask_expert` function:
``` python
def ask_expert(question: str, return_entity: str, return_entity_type: Literal[bool, int, float]):
    # The implementation is omitted but can be summarize as:
    entity = expert.response(question)
    assert isinstance(return_entity, return_entity_type)
    return entity
```
The human expert is capable of reading text or description, make comparison, reasoning or evaluating. However, you will need to form a complete question for the expert and also explicitly provide what's your expected response and what is the type of the expected response. You don't need to always ask the expert, but you should do it if the condition involves reading or come up with text or descriptions or using user_input, unless it is trivial. Ask yourself whether the condition for executing an action involves text reading/comparing or evaluating.

The catch is, the user left you to do the task alone with a human expert and you have no way knowing what's the user desire. You MUST ACTIVELY come up with a rational conditions to execute each action that will ultimately achieve the goal of "{webshop_goal}" without relying on the user.

For example with a hypothetical action "ACTION_DRINK" your response for step 2 should contains 2 sentence.
1. Since I'm alone with my sidekick, based on rational reasoning, I should execute ACTION_DRINK when variable `thirsty_level` reached 0.9 threshold and the liquid is drinkable.
2. Does the condition involves text reading/comparing/evaluating/etc?: Yes, so I should call ask_expert giving the variable `liquid_description` and expecting is_drikable: bool as the response.

Step 3. Based on step2, write ONE COMPLETED python function name get_action that look like this:
``` python
def get_action({state_observation_arg_list}) -> str:
    \"\"\" code here\"\"\"
```
Your response MUST contain only get_action function with no other helper functions. Inside the get_action function, there should be a COMPLETED implementation for every cases.

"""
        ),
        AIMessagePromptTemplate.from_template(
            """
Assistant:

Step 1: 
I'm building a finite state machine for the task of "Buy an item that matches the user's description through a web shopping interface." The state machine has {state_len} states: {state_list} I'm currently focused on the {cur_state_name} state, which has {action_len} possible actions in the action_list. I have access to various variables and previous states to help you determine the appropriate action to take. I'm also accompanied with a human expert who is capable of reading, comparing, evaluating texts which I can ask through `ask_expert` function.

Step 2:
"""
        ),
    ]
)
