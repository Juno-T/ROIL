import os
import sys
sys.path.append(os.getcwd())

import gym

from web_agent_site.envs import WebAgentTextEnv
from auto_policy_programming.fsm import AutoPolicyProgrammingFSM
from auto_policy_programming.wrappers.webshop import webshop_wrapper


def main():
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)

    fsm = AutoPolicyProgrammingFSM(env=env)
    obs = fsm.reset()[0]
    for i in range(2):
        fsm.step()

if __name__ == "__main__":
    main()
    