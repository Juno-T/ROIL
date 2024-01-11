
from abc import abstractmethod
from ast import Dict
from math import log
from typing import Any, Dict, List, Optional, Union
from venv import logger
from langchain import BasePromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.base import Chain

import logging

logger = logging.getLogger(__name__)
# format with date time, line number, and function name

class ParsingChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables
    
    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def output_parser(self, output: str) -> Dict[str, str]:
        raise NotImplementedError()

    def reset(self):
        pass

    def input_pre_format(self, inputs: Dict) -> Dict[str, any]:
        return inputs
    
    def _call(
        self,
        inputs: Dict[str, any],
        run_manager,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)
        logger.info(f"\nPrompt value: {prompt_value}")
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None)
        if run_manager:
            run_manager.on_text("Calling llm")
        return self.output_parser(response.generations[0][0].text)

    async def _acall(
        self,
        inputs: Dict[str, any],
        run_manager,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)
        logger.debug(f"\nPrompt value: {prompt_value}")
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None)
        if run_manager:
            await run_manager.on_text("Calling llm")
        return self.output_parser(response.generations[0][0].text)
    
    def __call__(self, inputs: Union[Dict[str, Any], Any], *args, return_only_outputs=True, **kwargs):
        if isinstance(inputs, dict):
            inputs = self.input_pre_format(inputs)
        response = super(ParsingChain, self).__call__(inputs, *args, return_only_outputs=True,  **kwargs)
        return response
    
    async def acall(self, inputs: Union[Dict[str, Any], Any], *args, **kwargs):
        if isinstance(inputs, dict):
            inputs = self.input_pre_format(inputs)
        return await super().acall(inputs, *args, **kwargs)

    def test_parse_output(self, output: str) -> Dict[str, str]:
        return self.output_parser(output)

    def test_format_input(self, inputs: Dict[str, Any]) -> str:
        inputs = self.input_pre_format(inputs)
        return self.prompt.format_prompt(**inputs)
    @property
    def _chain_type(self) -> str:
        return super()._chain_type + "_ParsingChain"
    