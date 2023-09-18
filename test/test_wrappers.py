import gym

from web_agent_site.envs import WebAgentTextEnv
from auto_policy_programming.wrappers import BaseWrapper
from auto_policy_programming.wrappers.webshop import webshop_wrapper
from auto_policy_programming.wrappers.webshop.observation_extractions import extract_observation

def test_base():
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = BaseWrapper(env)
    env.reset()
    assert env.get_available_actions() == env.env.get_available_actions()

def test_webshop():
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    obs = env.reset()[0]
    assert isinstance(obs, webshop_wrapper.State)


def test_webshop_observation():
    searches = ["something here", "other query", "red tall hat"]
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    for s in searches:
        obs = env.reset()[0]
        obs = env.step(f"search[{s}]")[0]
        results_obs = extract_observation(obs.raw_state, "results")
        assert "instruction" in results_obs
        assert "items" in results_obs
        assert len(results_obs["items"]) > 0
    
def test_webshop_item_observation():
    s = "something here"
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    obs = env.reset()[0]
    obs = env.step(f"search[{s}]")[0]
    results_obs = extract_observation(obs.raw_state, "results")
    obs = env.step(f"click[{results_obs['items'][0][0]}]")[0]
    item_obs = extract_observation(obs.raw_state, "item")
    assert "instruction" in item_obs
    assert "item_name" in item_obs
    assert item_obs["item_name"].strip() == results_obs["items"][0][1].strip()
    assert "price" in item_obs