import sys
import pytest
from auto_policy_programming.fsm.transition import Transition
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def test_transition_compile():
    transition = Transition("state1", """
def transition(state: str) -> str:
    return "state2"
""")
    assert transition.transition_function is not None
    assert transition("state1") == "state2"


def test_transition_compile_error():
    with pytest.raises(Exception):
        transition = Transition("state1", """
def whatever(state):
return "state2"
""")


def test_transition_compile_error2():
    with pytest.raises(Exception):
        transition = Transition("state1", """x = 1""")


from auto_policy_programming.common.llm_helper import test_calling_llm
def test_transition_function_calling():
    transition = Transition("state1", """
def whatever(state):
    return test_calling_llm()
""", {"test_calling_llm": test_calling_llm})
    assert transition.transition_function is not None
    assert transition("state1") == "test"