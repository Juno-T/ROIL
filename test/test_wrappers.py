import gym

from web_agent_site.envs import WebAgentTextEnv
from auto_policy_programming.wrappers import BaseWrapper, webshop_wrapper

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


    