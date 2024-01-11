from ast import Dict
from calendar import c
from email.utils import parsedate
from math import e
import re
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


class EvaluationStep(ParsingChain):
    react_style: bool = False
    past_observations: List[str] = []
    past_selected_actions: List[str] = []
    past_thoughts: List[str] = []
    def __init__(self, llm: BaseLanguageModel, react_style=False):
        if react_style:
            super().__init__(prompt=react_style_prompt_template, llm=llm, react_style=react_style)
        else:
            super().__init__(prompt=prompt_template, llm=llm, react_style=react_style)
        self.react_style = react_style
        self.reset()

    @property
    @override
    def output_keys(self) -> List[str]:
        return ["valid", "raw_output", "thoughts", "selected_rule_number", "selected_action"]
    
    @override
    def input_pre_format(self, inputs: Dict) -> Dict[str, Any]:
        """
        inputs: {
            "observation": ,
            "state_name": ,
            "taken_actions":,
            "available_actions": ,
            "rule_list": List[str],
            (optional) "instruction": str instruction,
        }
        """
        available_actions = ", ".join(inputs["available_actions"])
        based_on_rulelist = " Use applicable rules from the provided rule_list as a source for reasoning."
        rule_list = []
        for i, rule in enumerate(inputs["rule_list"]):
            rule_list.append(f"{i+1}. {rule}")
        if len(rule_list) == 0:
            rule_list = ""
            based_on_rulelist = ""
        else:
            rule_list = "\n".join(rule_list)
            rule_list = f"\nrule_list:\n{rule_list}\n"
        if not self.react_style:
            taken_actions = ""
            if len(inputs['taken_actions']) > 0:
                taken_actions = ", ".join(inputs["taken_actions"])
            return {
                "observation": inputs["observation"],
                "available_actions": available_actions,
                "rule_list": rule_list,
                "taken_actions": taken_actions,
                "based_on_rulelist": based_on_rulelist,
            }
        else:
            observation = f"Instruction: {inputs['instruction']}\n"
            observation += "Observation history:\n"
            observation_history = ""
            for obs, thought, action in zip(self.past_observations[::-1], self.past_thoughts[::-1], self.past_selected_actions[::-1]):
                to_add = f"past_observation:\n{str(obs)}\npast_thoughts: {str(thought)}\npast_action: {str(action)}\n\n"
                if len(observation_history) + len(to_add) < 3000:
                    observation_history = to_add + observation_history
            observation += observation_history[-3000:] + "\n"
            observation += f"Current observation:\n{inputs['observation']}\n"
            
            self.past_observations.append(inputs["observation"])
            return {
                "observation": observation,
                "available_actions": available_actions,
                "rule_list": rule_list,
                "based_on_rulelist": based_on_rulelist,
            }

    @override
    def reset(self):
        self.past_observations = []
        self.past_selected_actions = []
        self.past_thoughts = []

    def save_history(self, parsed_output):
        self.past_thoughts.append(parsed_output["thoughts"])
        self.past_selected_actions += parsed_output["selected_action"]

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        try:
            parsed_output = {"raw_output": output}
            parsed_output["thoughts"] = extract_line(output, "thoughts")
            selected_action = extract_line(output, "selected_one_best_action")
            selected_action = [action.strip() for action in selected_action.split('AND')][0]
            parsed_output["selected_action"] = selected_action
            selected_rule_number = extract_rule_number(output)
            try:
                parsed_output["selected_rule_number"] = int(selected_rule_number)
            except:
                parsed_output["selected_rule_number"] = None
            if selected_action is None or parsed_output["thoughts"] is None:
                parsed_output["valid"] = False
            else:
                parsed_output["valid"] = True
            if self.react_style:
                self.save_history(parsed_output)
            return parsed_output
        except:
            return {"valid": False, "raw_output": output, 'selected_action': None, 'selected_rule_number': None}


def extract_line(output, keyword):
    # using regex to match "keyword: <content>", keyword is case insensitive
    match = re.findall(f"{keyword}:[\n\s]*(.*)[>)]*\n", output+"\n\n", re.IGNORECASE)
    if len(match) > 0:
        # return last match
        return match[-1].strip()
    else:
        return None

def extract_rule_number(output, keyword = "selected_rule"):
    match = re.findall(f"{keyword}[\d\s]*:[\n\s]*(\d*).*[\n$]", output+"\n\n", re.IGNORECASE)
    if len(match) > 0:
        # return last match
        try:
            return int(match[0])-1
        except:
            return None
    else:
        return None

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are an assistant capable of reasoning. You will response in the following response format, based on the observation, your past actions.{based_on_rulelist}
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """
You will strictly follow the response format down to the characters.

Response format:
(begin format)
(answer these 2 questions in the following format)
(Fill in the {{}} brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)

1. Based on the observation, what's your thought on selecting an action?{based_on_rulelist} Think step by step.
thoughts: {{your thought process}}

2. From your thoughts, what is the best action to take? Select 1 best action from the available_action. The action must be in the format of {{action_name}}[{{action_input}}], some action may have pre-defined {{action_input}}. Make sure that the action is in the provided available_action list.
selected_one_best_action: {{action_name}}[{{action_input}}]
(if any) selected_second_best_action: ...
(end format)

observation:
{observation}

Assistant's previously taken actions:
{taken_actions}

available_action list to select from:
{available_actions}
{rule_list}
Assistant:
Based on the given information, here is my response in the instructed response format,
(begin format)
(answer these 2 questions in the following format)
(Fill in the {{}} brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
Here are the answers to the 2 questions,
""")
])

react_style_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are an assistant capable of reasoning. You will response in the following response format, based on the observation, your past actions.{based_on_rulelist}
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """
You will strictly follow the response format down to the characters.

Response format:
(begin format)
(Fill in the {{}} brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
thoughts: {{your step-by-step thought process.{based_on_rulelist}}}
selected_one_best_action: {{action_name}}[{{action_input}}]
(end format)
(example)
observation history:
instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
past_thoughts: I need to search for the instructed item.
past_action: search[3 ounce bright citrus deodorant sensitive skin]
past_observation:
[Back to Search] 
Page 1 (Total results: 50) 
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95
past_thoughts: B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J
past_action: click[B078GWRC1J]
past_observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]
past_thoughts: For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
past_action: click[bright citrus]
past_observation: You have clicked bright citrus. 
past_thoughts: I have already selected 'bright citrus', next is '3 ounce (pack of 1)'.
past_action: click[3 ounce (pack of 1)]
(current observation)
Current observation:
You have clicked 3 ounce (pack of 1). You have clicked bright citrus.
(Your answers for thoughts and selected_one_best_action,)
thoughts: I selected all the required criteria, I can click buy now.
selected_one_best_action: click[buy now]
(end example)
(end format)

{observation}

available_action list to select from:
{available_actions}
{rule_list}
Assistant:
Based on the given information, here is my response in the instructed response format,
(begin format)
(Fill in the {{}} brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
Here are the answers to for thoughts and selected_one_best_action,
""")
])