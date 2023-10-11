import os
import sys
sys.path.append(os.getcwd())

import logging
from pathlib import Path
from datetime import datetime

import gym
from web_agent_site.envs import WebAgentTextEnv
from auto_policy_programming.fsm import AutoPolicyProgrammingFSM
from auto_policy_programming.wrappers.webshop import webshop_wrapper

# set root logger
# logger = logging.getLogger("auto_policy_programming")


# create log folder at log/yyyy-mm-dd/
log_folder = Path("log") / Path(datetime.now().strftime("%Y-%m-%d"))
log_folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_folder / str(datetime.now().strftime("%H-%M-%S") + ".log")), 
    filemode='w',
    format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
    level=logging.DEBUG)

def main():
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
    env = webshop_wrapper.wrap_text_env(env)

    fsm = AutoPolicyProgrammingFSM(env=env)
    obs = fsm.reset()[0]
    fsm.env.step(obs.actions['search'], "Red dress")
    for i in range(2):
        fsm.step()

if __name__ == "__main__":
    main()
    