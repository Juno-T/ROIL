from ast import Dict
from typing import Any, Dict, List
from typing_extensions import override
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from auto_policy_programming.chains.base import ParsingChain


class StateTransition(ParsingChain):
    def __init__(self, llm):
        super().__init__(prompt=prompt_template, llm=llm)
    
    @override
    def output_keys(self) -> List[str]:
        return ["state_transition"]
    
    @override
    def input_pre_format(self, inputs: Dict) -> Dict[str, Any]:
        extra_inputs = {}
        extra_inputs["state"] = None # TODO: get state from inputs
        inputs.update(extra_inputs)
        return inputs

    @override
    def output_parser(self, output: str) -> Dict[str, str]:
        # TODO: parse code block from output
        return {"state_transition": output}

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a programmer. You are writing a program to help you write programs.
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """You are writing a program to help you write programs.
"""
        ),
    ]
)
