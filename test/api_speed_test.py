import os
import sys
import time

sys.path.append(os.getcwd())

import asyncio
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import openai

from auto_policy_programming.chains.evaluation_step import prompt_template
from auto_policy_programming import chains
openai_prompt = prompt_template.format(
    observation="",
    state_name="",
    available_actions=[],
    rule_list=[],
    taken_actions=[],
)

openai.api_key = os.getenv("OPENAI_APIKEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0613",
    max_tokens=1024, temperature=0, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
# test_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate(
#         input_variables=["topic"],
#         template="Write 200 words paragraph about {topic}."
#     ),
#     output_key="joke",
# )
test_chain = chains.EvaluationStep(llm=llm)

async def async_chain_run(fn):
    return await fn()

async def run_test(fn, count=5):
    tasks = [async_chain_run(fn) for _ in range(count)]
    ret = await asyncio.gather(*tasks)
    return ret

async def langchain_chain_fn():
    inputs = {
        "observation": "",
        "state_name": "",
        "available_actions": [],
        "rule_list": [],
        "taken_actions": [],
    }
    with get_openai_callback() as cb:
        output = await test_chain.acall(inputs)
    output["token_count"] = int(cb.total_tokens)
    return output

async def openai_fn():
    output = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": str(openai_prompt)}],
        temperature=0,
        max_tokens=1024,
    )
    output["token_count"] = int(output["usage"]["total_tokens"])
    return output

if __name__=="__main__":
    count = 3
    start = time.time()
    ret = asyncio.run(run_test(langchain_chain_fn, count=count))
    end = time.time()
    total_tokens = sum([r["token_count"] for r in ret])
    print(f"langchain: {end - start}s, total_tokens: {total_tokens}, speed: {total_tokens / (end - start)} tokens/s")
    start = time.time()
    ret = asyncio.run(run_test(openai_fn, count=count))
    end = time.time()
    total_tokens = sum([r["token_count"] for r in ret])
    print(f"openai: {end - start}s, total_tokens: {total_tokens}, speed: {total_tokens / (end - start)} tokens/s")