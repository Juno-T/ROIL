

import os
from langchain.chat_models import ChatOpenAI
from langchain.llms.fake import FakeListLLM
from auto_policy_programming.chains import StateTransition
from auto_policy_programming.fsm.state import State
from auto_policy_programming.fsm.transition import Transition
from auto_policy_programming.wrappers.base import BaseWrapper


class AutoPolicyProgrammingFSM:
    def __init__(self, env: BaseWrapper):
        self.env = env
        self.env_state_description = self.env.env_state_description
        self.transition_functions = {
            k: None for k in self.env_state_description.keys()
        }

        fsm_llm = FakeListLLM(responses=["fake response"])
        # fsm_llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.7, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
        # llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))

        self.transition_function_generator =  StateTransition(llm=fsm_llm, env_state_description=self.env_state_description)
    
    def reset(self):
        return self.env.reset()
    
    def step(self):
        cur_state = self.env.cur_state
        if self.transition_functions[cur_state.name] is None:
            self.generate_transition_function(cur_state)
        action = self.transition_functions[cur_state.name](cur_state.observations)
        # parse action here
        print(action)
        raise NotImplementedError()
        return self.env.step(action)

    def generate_transition_function(self, state: State):
        transition_function_code = self.transition_function_generator(inputs={"state": state})["state_transition"]
        transition_function = Transition(state.name, transition_function_code, extra_namespace={"ask_expert": ask_expert})
        self.transition_functions[state.name] = transition_function
        

def ask_expert():
    raise NotImplementedError()