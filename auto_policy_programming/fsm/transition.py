import traceback
import types
import logging

logger = logging.getLogger(__name__)

class Transition:
    # state_from
    # transition_function
    # score etc.
    def __init__(self, state_name_from, transition_function_code, extra_namespace={}):
        logger.debug(f"\nCreating transition from {state_name_from}\n{transition_function_code}\n"+"-"*80+"\n")
        self.state_name_from = state_name_from
        self.transition_function_code = transition_function_code.strip()
        self.transition_function = None
        self.score = None
        self.extra_namespace = extra_namespace
        self._compile_transition_function()
    
    def _compile_transition_function(self):
        try:
            self.compiled_functions = [f for f in extract_functions(compile(self.transition_function_code, "<string>", "exec"))]
            if len(self.compiled_functions) != 1:
                raise Exception(f"Expected 1 function, found {len(self.compiled_functions)}")
            # self.compiled_functions = [(fname, code_obj), ...]
            global_namespace = globals().copy()
            global_namespace.update(self.extra_namespace)
            print("Transition:")
            print("test_calling_llm" in global_namespace)

            self.transition_function = types.FunctionType(self.compiled_functions[0][1], global_namespace)
        except Exception as e:
            logger.exception(e)
            raise Exception(f"Error compiling transition function:\n {e}")


    def __call__(self, state):
        if self.transition_function is None:
            raise Exception("Transition function not compiled")
        # TODO: validate & format state
        raise NotImplementedError()
        return self.transition_function(state)


import dis
from itertools import islice

# old itertools example to create a sliding window over a generator
def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def extract_functions(codeobject):
    codetype = type(codeobject)
    signature = ('LOAD_CONST', 'LOAD_CONST', 'MAKE_FUNCTION', 'STORE_NAME')
    for op1, op2, op3, op4 in window(dis.get_instructions(codeobject), 4):
        if (op1.opname, op2.opname, op3.opname, op4.opname) == signature:
            # Function loaded
            fname = op2.argval
            assert isinstance(op1.argval, codetype)
            yield fname, op1.argval