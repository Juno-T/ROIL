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
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(prompt=prompt_template, llm=llm)

    @property
    @override
    def output_keys(self) -> List[str]:
        return ["valid", "raw_output", "selected_rule_number", "selected_action"]
    
    @override
    def input_pre_format(self, inputs: Dict) -> Dict[str, Any]:
        """
        inputs: {
            "observation": ,
            "state_name": ,
            "taken_actions":,
            "available_actions": ,
            "rule_list": List[str]
        }
        """
        rule_list = []
        for i, rule in enumerate(inputs["rule_list"]):
            rule_list.append(f"{i+1}. {rule}")
        taken_actions = ""
        if len(inputs['taken_actions']) > 0:
            taken_actions = ", ".join(inputs["taken_actions"])
        return {
            "observation": inputs["observation"],
            "available_actions": ", ".join(inputs["available_actions"]),
            "rule_list": "\n".join(rule_list),
            "taken_actions": taken_actions
        }

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        parsed_output = {"raw_output": output}
        selected_action = extract_line(output, "selected_one_best_action")
        selected_action = [action.strip() for action in selected_action.split('AND')]
        selected_rule_number = extract_rule_number(output)
        try:
            parsed_output["selected_rule_number"] = int(selected_rule_number)
        except:
            parsed_output["selected_rule_number"] = None
        parsed_output["selected_action"] = selected_action
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
            """You are an assistant capable of reasoning. You will response in the following format, based on the observation. 
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """
You will strictly follow the response format down to the characters.

Response format:
(begin format)
(answer these 3 questions in the following format)
(Fill in the <> brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
1. Based on the observation, what are the relevant rules? Select one or more rules from the rule_list.
* selected_rule1: <rule_number>, <rule detail>
* selected_rule2: <rule_number>, <rule detail>
...

2. According to the observation, what are the key information?
* key_information1: <Key information1>
* key_information2: <Key information2>(if any more)
...

3. Based on the selected rules and key information in 1 and 2, what is the best action to take? Select 1 best action from the available_action. The action must be in the format of <action_name>[<action_input>], some action may have pre-defined <action_input>. Make sure that the action is in the provided available_action list.
selected_one_best_action: <action_name>[<action_input>]
(if any) selected_second_best_action: ...
(end format)

observation:
{observation}

previously taken actions:
{taken_actions}

available_action list to select from:
{available_actions}

rule_list to select from:
{rule_list}
        """),
        AIMessagePromptTemplate.from_template(
            """
Assistant:
Based on the observation and give rule_list, here is my response in the provided format,
(begin format)
(Fill in the <> brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
(answer these 3 questions in the following format)
1. Based on the observation, what are the relevant rules? Select one or more rules from the rule_list.
"""
        ),
    ]
)