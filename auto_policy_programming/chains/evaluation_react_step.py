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


class EvaluationReactStep(ParsingChain):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(prompt=prompt_template, llm=llm)

    @property
    @override
    def output_keys(self) -> List[str]:
        return ["valid", "raw_output", "selected_rule_number", "selected_action", "thought"]
    
    @override
    def input_pre_format(self, inputs: Dict) -> Dict[str, Any]:
        """
        inputs: {
            "instruction": ,
            "observation": ,
            "state_name": ,
            "available_actions": ,
            "rule_list": List[str],
            "react_history": List[(observation, thought, action)]
        }
        """
        rule_list = []
        for i, rule in enumerate(inputs["rule_list"]):
            rule_list.append(f"{i+1}. {rule}")
        if len(inputs["react_history"]) == 0:
            react_history = "\n"
        else:
            react_history = ""
            for i, (observation, thought, action) in enumerate(inputs["react_history"][-5:]):
                react_history += f"thought: {str(thought)}\naction: {str(action)}\n"
                # react_history += f"observation: {str(observation)}\nthought: {str(thought)}\naction: {str(action)}\n"
            react_history += "\n"

        return {
            "instruction": inputs["instruction"],
            "observation": inputs["observation"],
            "available_actions": ", ".join(inputs["available_actions"]),
            "rule_list": "\n".join(rule_list),
            "react_history": react_history,
        }

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        parsed_output = {"raw_output": output}
        selected_action = extract_line(output, "action")
        selected_action = [action.strip() for action in selected_action.split('AND')]
        selected_rule_number = extract_rule_number(output, keyword="relevant_rule_number")
        try:
            parsed_output["selected_rule_number"] = int(selected_rule_number)
        except:
            parsed_output["selected_rule_number"] = None
        parsed_output["selected_action"] = selected_action
        thought = extract_line(output, "key_informations")
        parsed_output["thought"] = thought
        if selected_action is None:
            parsed_output["valid"] = False
        else:
            parsed_output["valid"] = True
        return parsed_output

    def test_parse_output(self, output: str) -> Dict[str, str]:
        return self.output_parser(output)

    def test_format_input(self, inputs: Dict[str, Any]) -> str:
        inputs = self.input_pre_format(inputs)
        return self.prompt.format_prompt(**inputs)


def extract_line(output, keyword):
    # using regex to match "keyword: <content>", keyword is case insensitive
    match = re.findall(f"{keyword}:[\n\s]*(.*)[>)]*\n", output+"\n\n", re.IGNORECASE)
    if len(match) > 0:
        # return last match
        return match[-1].strip()
    else:
        return None

def extract_rule_number(output, keyword = "selected_rule"):
    match = re.findall(f"{keyword}[\d\s]*:[\n\s]*(\d*).*\n", output+"\n\n", re.IGNORECASE)
    if len(match) > 0:
        # return last match
        return match[0]
    else:
        return None

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are an assistant capable of reasoning. You will response in the following format. 
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """

You will strictly follow this response format down to the characters.
Response format:
(begin format)
key_informations: <key_informations and your thought process in one line>
relevant_rule_number: <rule_number from rule_list>
action: <action_name[action_input]: select only 1 best action from available_actions list in the format according to the rule and key_informations>
(end format)

available_actions list to select from:
{available_actions}

rule_list to select from:
{rule_list}
# Past timesteps
past_instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
key_informations: I must search for products that fullfill the requirements.
action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
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
key_informations: B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J
action: click[B078GWRC1J]
Observation: 
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
key_informations: For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
action: click[bright citrus]
Observation: You have clicked bright citrus. 
key_informations: I have already selected 'bright citrus', next is '3 ounce (pack of 1)'.
action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
key_informations: I selected all the required criteria, I can click buy now.
action: click[buy now]
thought: I have finished the task. I can reset the environment.
action: reset
{instruction}
{react_history}
# Current timestep
observation:
{observation}
"""),
        AIMessagePromptTemplate.from_template(
            """
Assistant:
Based on the given information, available_actions and rule_list, here is my response in the provided format,
(begin format)
"""
        ),
    ]
)