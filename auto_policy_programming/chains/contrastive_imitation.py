from ast import Dict
from calendar import c
from email.utils import parsedate
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


class ContrastiveImitation(ParsingChain):
    def __init__(self, llm: BaseLanguageModel):
        super().__init__(prompt=prompt_template, llm=llm)

    @property
    @override
    def output_keys(self) -> List[str]:
        return ["raw_output", "related_rule_number", "found_rule", "related_rule_classification", "updated_rule", "no_existing_rule", "found_rule_better"]
    
    @override
    def input_pre_format(self, traj_sa: Dict) -> Dict[str, Any]:
        """
            traj_sa: {
                "records": [],
                "existing_rules": []
            }
        """
        traj_records = traj_sa["records"]
        existing_rules = traj_sa["existing_rules"]
        parsed_input = {}
        activity_records = []
        for i, record in enumerate(traj_records):
            activity_records.append(format_activity_record(i+1, record["state"], record["available_action"], record["action"]))
        parsed_input["activity_records"] = "\n".join(activity_records)
        if len(existing_rules) == 0:
            parsed_input["numbered_existing_rules"] = "No existing rules"
        else:
            numbered_existing_rules = []
            for i, rule in enumerate(existing_rules):
                numbered_existing_rules.append(f"{i+1}. {rule}")
            parsed_input["numbered_existing_rules"] = "\n".join(numbered_existing_rules)

        return parsed_input

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        no_existing_rule = False
        if "NO_EXISTING_RULE" in output.upper():
            no_existing_rule = True

        related_rule_number = extract_line(output, "related_rule_number")
        if related_rule_number is not None:
            try:
                related_rule_number = int(related_rule_number.strip().replace(".", ""))-1
            except:
                related_rule_number = None

        found_rule = extract_line(output, "found_rule")

        related_rule_classification = None
        for c in ["same_intention", "similar_intention", "different_intention"]:
            if c in output:
                related_rule_classification = c
                break

        found_rule_better = None
        if "found_rule_better" in output:
            found_rule_better = True
        elif "selected_rule_better" in output:
            found_rule_better = False

        updated_rule = extract_line(output, "updated_rule")

        parsed_output = {
            "raw_output": output,
            "related_rule_number": related_rule_number,
            "found_rule": found_rule,
            "related_rule_classification": related_rule_classification,
            "updated_rule": updated_rule,
            "found_rule_better": found_rule_better,
            "no_existing_rule": no_existing_rule, # bool
        }
        parsed_output["valid"] = validate_output(parsed_output)
        return parsed_output

    def test_parse_output(self, output: str) -> Dict[str, str]:
        return self.output_parser(output)

    def test_format_input(self, inputs: Dict[str, Any]) -> str:
        inputs = self.input_pre_format(traj_sa=inputs)
        return self.prompt.format_prompt(**inputs)


def validate_output(output) -> bool:
    if output["no_existing_rule"] and output["found_rule"] is not None:
        return True
    for k in ["related_rule_number", "found_rule", "related_rule_classification"]:
        if k not in output:
            return False
        if output[k] is None:
            return False
    if output["related_rule_classification"] == "similar_intention" and output["updated_rule"] is None:
        return False
    if output["related_rule_classification"] == "same_intention" and output["found_rule_better"] is None:
        return False
    return True

def format_activity_record(index: int, observation, available_actions, selected_action):
    return f"""
\t# Record{index}
## Recorded Observation
{observation}
## Available Actions
{available_actions}
## Recorded selected action
{selected_action}
"""

def extract_line(output, keyword):
    # using regex to match "keyword: <content>", keyword is case insensitive
    match = re.findall(f"{keyword}:[\n\s]*(.*)[>)]*\n", output+"\n\n", re.IGNORECASE)
    if len(match) > 0:
        # return last match
        return match[-1]
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
You will strictly follow the response format down to the characters.

Response format:
(begin format)
(Fill in the <> brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
There are two or more recorded activities in the same state.
Record 1:
According to the record's observation, what are the key information that made you choose <recorded selected action> and why?
* key_information1: <Key information1>
* key_information2: <Key information2>(if any more)
...
* reason: Based on the listed key information, <Why choosing this action?>

Record 2:
...
(finally answer these 5 questions)

1. Why the two scenarios resulted in difference different actions?
differences: <selected action1> was chosen instead of <selected action2> because <some reasons>

2. Extract one key idea or rule behind the reason why <record1's selected action> were chosen. Key idea or rule must be generalized and not specific to one particular record. It must be one detailed sentence. Prefix your answer with `found_rule: `.
found_rule: When <detailed condition(s) based on some key information>, the best action to take is <action (with action_input guidelines if any)>.

3. From the list of existing rule, select the most related rule.
related_rule_number: <rule's number>
related_rule_content: <rule's content>
If there is no existing rules to select from, print "NO_EXISITING_RULE" and skip question 4 and 5.

4. Compare the selected rule with the founded key idea/rule.
rules_comparisons: <some reasons>

5. What's best describe the selected rule, select one of the 3 choices?
<same_intention: The selected rule is applicable in this scenario and resulted in the same action and action_input.>
<similar_intention: This rule should be updated.>
<different_intention: This rule is for a different kind of scenario.>

5.1 Follow up question.
* If you chose same_intention, select one of the 2 choices.
<found_rule_better: The found rule is better than the selected rule>
<selected_rule_better: The selected rule is better than the found rule>
* If you chose similar_intention, write down the updated rule.
updated_rule: <When ..., the best action to take is ...>
* If you chose different_intention, just print N/A for this question.
(end format)

Here are the recorded activities:
{activity_records}

Here are the list of existing rules you can choose from:
{numbered_existing_rules}
        """),
        AIMessagePromptTemplate.from_template(
            """
Based on the recorded activities, here is my response in the provided format:
(begin format)
(Fill in the <> brackets without altering any character outside the brackets. Repeat every words that's not in the brackets. Always use the given snake_case as is.)
There are two or more recorded activities in the same state.
(For each records,)
"""
        ),
    ]
)