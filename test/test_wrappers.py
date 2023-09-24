import gym

from web_agent_site.envs import WebAgentTextEnv
from auto_policy_programming.wrappers import BaseWrapper
from auto_policy_programming.wrappers.webshop import webshop_wrapper

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
    assert obs.name == "search"


def test_webshop_result_observation():
    searches = ["something here", "other query", "red tall hat"]
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    for s in searches:
        obs = env.reset()[0]
        obs = env.step(f"search[{s}]")[0]
        assert "instruction" in obs.observations
        assert "options" in obs.observations
        assert len(obs.observations["options"]['value']['items']) > 0
        assert "options" in obs.actions
        assert "items" in obs.actions['options'].options
        assert obs.actions['options'].options['items'] == obs.observations['options']['value']['items']
    
def test_webshop_item_observation():
    s = "something here"
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    obs = env.reset()[0]
    obs = env.step(f"search[{s}]")[0]
    item = obs.observations['options']['value']['items'][0]
    obs = env.step(f"click[{item[0]}]")[0]
    assert "instruction" in obs.observations
    assert "item_name" in obs.observations
    assert obs.observations["item_name"]['value'].strip() == item[1]
    assert "price" in obs.observations
    assert "options" in obs.actions
    assert isinstance(obs.actions['options'].options, dict)

def test_webshop_item_description():
    s = "something here"
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    obs = env.reset()[0]
    obs = env.step(f"search[{s}]")[0]
    obs = env.step(f"click[{obs.observations['options']['value']['items'][0][0]}]")[0]
    obs = env.step(f"click[description]")[0]
    assert "instruction" in obs.observations
    assert "item_description" in obs.observations


def test_webshop_action_execution():
    s = "something here"
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)
    obs = env.reset()[0]
    obs = env.step(obs.actions['search'], s)[0]
    item = obs.observations['options']['value']['items'][5]
    obs = env.step(obs.actions['options'], {'items': 5})[0]
    assert obs.observations["item_name"]['value'].strip() == item[1]
    obs = env.step(obs.actions['description'])[0]
    assert "instruction" in obs.observations
    assert "item_description" in obs.observations