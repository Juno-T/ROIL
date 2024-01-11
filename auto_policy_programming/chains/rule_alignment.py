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


class RuleAlignmentChain(ParsingChain):
    def __init__(self, llm: BaseLanguageModel, react_style=False):
        super().__init__(prompt=prompt_template, llm=llm)

    @property
    @override
    def output_keys(self) -> List[str]:
        return [
            "valid",
            "raw_output",
            "thoughts",
            "wrong_selected_rule_number",
            "wrong_selected_rule_update",
            "should_be_selected_rule_number",
            "should_be_selected_rule_update"
        ]
    
    @override
    def input_pre_format(self, inputs: Dict) -> Dict[str, Any]:
        """
        inputs: {
            "observation": ,
            "taken_actions":,
            "available_actions": ,
            "rule_list": List[str],
            "wrong_thoughts": str,
            "wrong_action": str,
            "correct_action": str,
        }
        """
        available_actions = ", ".join(inputs["available_actions"])
        rule_list = []
        for i, rule in enumerate(inputs["rule_list"]):
            rule_list.append(f"{i+1}. {rule}")
        if len(rule_list) == 0:
            rule_list = ""
        else:
            rule_list = "\n".join(rule_list)
            rule_list = f"\nrule_list:\n{rule_list}\n"

        taken_actions = ""
        if len(inputs['taken_actions']) > 0:
            taken_actions = ", ".join(inputs["taken_actions"])
        return {
            "observation": inputs["observation"],
            "available_actions": available_actions,
            "rule_list": rule_list,
            "taken_actions": taken_actions,
            "wrong_thoughts": inputs["wrong_thoughts"],
            "wrong_action": inputs["wrong_action"],
            "correct_action": inputs["correct_action"],
        }

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        try:
            parsed_output = {"raw_output": output}
            parsed_output["valid"] = True
            thoughts = extract_line(output, "thoughts")
            parsed_output["thoughts"] = thoughts
            parsed_output["wrong_selected_rule_number"] = extract_rule_number(output, "wrong_selected_rule")
            parsed_output["wrong_selected_rule_update"] = extract_line(output, "amend_wrong_selected_rule")
            parsed_output["should_be_selected_rule_number"] = extract_rule_number(output, "should_be_selected_rule")
            parsed_output["should_be_selected_rule_update"] = extract_line(output, "amend_should_be_selected_rule")
            
            if parsed_output["wrong_selected_rule_number"] is not None:
                try:
                    parsed_output["wrong_selected_rule_number"] = int(parsed_output["wrong_selected_rule_number"])
                except:
                    parsed_output["wrong_selected_rule_number"] = None
            if parsed_output["should_be_selected_rule_number"] is not None:
                try:
                    parsed_output["should_be_selected_rule_number"] = int(parsed_output["should_be_selected_rule_number"])
                except:
                    parsed_output["should_be_selected_rule_number"] = None
            if not self.validate_output(parsed_output):
                parsed_output["valid"] = False
            return parsed_output
        except:
            return {"valid": False, "raw_output": output,}
    
    def validate_output(self, parsed_output: Dict[str, str]) -> bool:
        if parsed_output["wrong_selected_rule_number"] is None:
            return False
        if parsed_output["wrong_selected_rule_update"] is None:
            return False
        if parsed_output["should_be_selected_rule_number"] is None:
            return False
        if parsed_output["should_be_selected_rule_update"] is None:
            return False
        return True

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
            """You are an assistant capable of reasoning. You will response in the following format, based on a record of your activity.
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """
You will strictly follow the response format down to the characters. When amending rules, try to write it in details while using adjectives and adverbs and conjunctions to make it more precise. The rule can be many sentences but it must be generalized, not specific to this example.

Response format:
(begin format)
(Fill in the <> brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
(Answer these 4 questions)

1. According to the record\'s observation, past actions, available actions, what are the key information? And, what is the reason that made {{correct_action}} the best action to take?
* key_information1: {{Key information1}}
* thought: {{thought about this key information. Could it be the reason for the correct_action?}}
* key_information2: {{Key information2}}
* thought: {{thought about this key information. Could it be the reason for the correct_action?}}
...
(repeat until the assistant found a sounding reason.)
* conclusion: {{Summarize the main reason why this action was chosen.}}

2. Thoughts on why the wrong action shouldn't be selected and why the correct action is {{correct_action}}. Your thought must be based on the Recorded Observation and the recorded incorrect thoughts. Think step-by-step.
thoughts: {{thoughts, reasons}}

3. What is the rule that's most relevant to the wrong action?
wrong_selected_rule: {{rule's number}}, {{rule_detail}}

3.1 Amend the wrong selected rule by making it taking this case into an account. Try to retain the original detail while adding exception or more precise condition. Remember that the rule must be generalized.
amend_wrong_selected_rule: {{updated rule detail}}

4. What is the rule that should have been selected instead?
should_be_selected_rule: {{rule's number}}, {{rule detail}}

4.1 Amend the should_be_selected_rule so that it's more likely to be selected in this case. Try to retain the original detail while adding exception or more precise condition. Remember that the rule must be generalized.
amend_should_be_selected_rule: {{updated rule detail}}
(end format)

Recorded Observation:
{observation}
Available Actions:
{available_actions}
Past Actions:
{taken_actions}

Wrong thoughts:
{wrong_thoughts}

Wrong action selection:
{wrong_action}

The correct action:
{correct_action}

rule_list:
{rule_list}

Assistant:
(Fill in the <> brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
Here are the answers for the 4 questions,
""")
])