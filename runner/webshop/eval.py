from math import e
import os
from re import L
import sys
import time
from typing import List

from regex import D
sys.path.append(os.getcwd())

import asyncio
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.llms.fake import FakeListLLM
from langchain.callbacks import get_openai_callback
from auto_policy_programming import chains
from auto_policy_programming.rule import Rule, RuleSet
from auto_policy_programming.wrappers.timelimit import LimitTimeStepWrapper
from auto_policy_programming.wrappers.webshop.webshop_wrapper import WebAgentTextEnvWithStateName, WebShopEnv

DEBUG = False
MAX_PARALLEL = 20
MAX_TOKEN_PER_MINUTE = 90000

log_folder = Path("log") / Path(datetime.now().strftime("%Y-%m-%d")) / str(datetime.now().strftime("%H-%M-%S"))
log_folder.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_folder / "test.log"), 
    filemode='w',
    format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
    level=logging.DEBUG if DEBUG else logging.INFO)

async def async_chain_run(chain, query, timeout=30, call_delay=0.0):
    try:
        with get_openai_callback() as cb:
            try:
                time.sleep(call_delay)
                output = await asyncio.wait_for(chain.acall(query), timeout=timeout)
                output["token_count"] = int(cb.total_tokens)
            except Exception as e:
                print(f"Exception: {e}")
                output = {
                    "valid": False,
                    "selected_action": None,
                    "token_count": int(cb.total_tokens),
                    "_reason": "Timeout"
                }
        return output
    except:
        return {
            "valid": False,
            "selected_action": None,
            "token_count": 0,
            "_reason": "unknown"
        }
    
async def async_chains_run(chains, queries, timeout=50, max_retry=0):
    outputs = []
    for i in range(len(queries)):
        outputs.append({
            "valid": False,
        })
    
    while max_retry>=0:
        # tasks = [async_chain_run(chain, *query) for chain, query in zip(chains, queries) if query is not None]
        to_call_indices = [i for i in range(len(queries)) if (not outputs[i]["valid"]) and (queries[i] is not None)]
        random_delay = np.arange(len(to_call_indices))*0.1
        np.random.shuffle(random_delay)
        if len(to_call_indices)==0:
            break
        logging.info(f"Calling {len(to_call_indices)} chains.")
        tasks = [async_chain_run(chains[idx], queries[idx], timeout=timeout, call_delay=random_delay[i]) for i, idx in enumerate(to_call_indices)]
        ret = await asyncio.gather(*tasks)
        # update outputs:
        for i in range(len(to_call_indices)):
            outputs[to_call_indices[i]] = ret[i]
        max_retry-=1

    return outputs

req_timestamp = []
req_tokens = []
def token_rate_limit_handler(req_time, total_tokens):
    cur_time = time.time()
    min_tokens = MAX_PARALLEL * 2000
    req_timestamp.append(req_time)
    req_tokens.append(total_tokens)

    # prune old requests
    while len(req_timestamp)>0 and cur_time-req_timestamp[0]>60:
        req_timestamp.pop(0)
        req_tokens.pop(0)

    used_tokens = sum(req_tokens)
    available_tokens = MAX_TOKEN_PER_MINUTE - used_tokens

    if available_tokens<min_tokens:
        if len(req_timestamp)==0:
            return None
        if len(req_timestamp)==1:
            wait_time = req_timestamp[0] + 60 - cur_time
            # return False
        else:
            wait_time = req_timestamp[1] + 60 - cur_time
        if wait_time<0:
            return None
        print(f"Rate limit reached: {used_tokens} tokens in {cur_time-req_timestamp[0]}s,\n\n waiting {wait_time}s\n")
        print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(wait_time+1)

def eval(
    envs: List[WebShopEnv],
    eval_record_path: str,
    ruleset_path: str = None,
    train_config_path: str = None,
    seed=0,
    num_eval=500,
    max_ep_len=15,
    low_timestep_threshold=-1,
    num_rule_per_step=10,
    react_style=False
):
    assert (ruleset_path is not None) or (train_config_path is not None)

    # Overriding default eval config with train config
    if train_config_path is not None:
        with open(train_config_path, "r") as f:
            train_config = json.load(f)
        if ruleset_path is None:
            ruleset_path = Path(train_config["final_ruleset_path"])
            ruleset_path = Path(train_config_path).parent / ruleset_path
        seed = train_config.get("seed", seed)
        react_style = train_config.get("react_style", react_style)
        # Eval specific configs, no inheritance
        # low_timestep_threshold = train_config.get("low_timestep_threshold", -1)
        num_rule_per_step = min(num_rule_per_step, train_config.get("num_rule_per_step", num_rule_per_step))

    assert (ruleset_path is not None) and os.path.exists(ruleset_path), f"ruleset_path {ruleset_path} does not exist"
    assert (low_timestep_threshold==-1) or (low_timestep_threshold>0 and low_timestep_threshold<=max_ep_len)
    np.random.seed(seed)
    start_time = datetime.now()

    # Wrap env if timestep threshold is set
    if react_style:
        for env in envs:
            env.set_react_style(react_style)

    if low_timestep_threshold>0:
        envs = [LimitTimeStepWrapper(env, max_episode_steps=max_ep_len, low_episode_timestep=low_timestep_threshold) for env in envs]


    # Load test indices, load ruleset
    test_indices = np.arange(num_eval)
    test_indices = test_indices.tolist()
    test_indices.append(-1) # mark the end for final eval loop
    ruleset = RuleSet.json_load(ruleset_path)

    # Initiations
    if os.path.exists(eval_record_path):
        with open(eval_record_path, "r") as f:
            eval_record = json.load(f)
            assert eval_record["seed"]==seed, f"seed mismatch: {eval_record['seed']} vs {seed}"
    else:
        Path(eval_record_path).parent.mkdir(parents=True, exist_ok=True)
        eval_record = {
            "seed": seed,
            "num_eval": num_eval,
            "split": split,
            "max_ep_len": max_ep_len,
            "low_timestep_threshold": low_timestep_threshold,
            "ruleset_path": str(Path(ruleset_path).absolute()),
            "train_config_path": str(Path(train_config_path).absolute()),
            "react_style": react_style,
            "num_rule_per_step": num_rule_per_step,
            "average_reward": 0.0,
            "eval_history": {}
        }

    best_rules = {state: ruleset.get_best_rules(state, num_rules=num_rule_per_step) for state in ruleset.states}
    if DEBUG:
        llm = FakeListLLM(responses=["thoughts: Some thoughts here.\nselected_one_best_action: search[red dress]\n\n"])
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    eval_action_chains = [chains.EvaluationStep(llm=llm, react_style=react_style) for _ in range(MAX_PARALLEL)]
    
    env_status = [0] * len(envs)
    env_ep_len = [0] * len(envs)
    env_test_idx = [-1] * len(envs)
    env_cur_ret = [None] * len(envs)
    env_instruction = [""] * len(envs)
    tmp_records = {
        idx: {
            'state_name': [],
            'action': [],
            'reward': [],
            'info': [],
            'selected_rule_id': [],
            'selected_rule_number': [],
        } for idx in test_indices if idx!=-1
    }

    elapsed_time_s = (datetime.now() - start_time).total_seconds()
    est_total_cost = 0.0
    call_count = 0

    # eval loop
    for eval_step, test_idx in enumerate(test_indices):
        if str(test_idx) in eval_record.get("eval_history", {}).keys():
            print(f"Skip Eval {eval_step}/{num_eval}, already done")
            continue
        
        # Find & assign untested index
        step_elapsed_time = (datetime.now() - start_time).total_seconds() - elapsed_time_s
        elapsed_time_s += step_elapsed_time
        print(f"Eval {eval_step}/{num_eval} elapsed_time {elapsed_time_s} step_elapsed_time {step_elapsed_time}")
        assigned_env = False
        if test_idx!=-1:
            for env_i in range(len(envs)):
                if env_status[env_i]==0:
                    env_status[env_i]=1
                    env_ep_len[env_i]=0
                    env_test_idx[env_i]=test_idx
                    env_cur_ret[env_i] = list(envs[env_i].reset(test_idx))+[False, None]
                    obs, reward, done, info = env_cur_ret[env_i]
                    if react_style:
                        env_instruction[env_i] = envs[env_i].get_instruction()
                    assigned_env = True
                    print(f"\n\nAssigned env {env_i} to test_idx {test_idx}\n\n")
                    break
            if assigned_env:
                continue

        some_done = False
        while True:
            if DEBUG:
                print(f"\n\n\nWARNING: USING DEBUG MODE!.\n\n\n")
            log = "\n"+("-"*10)+"\n"
            # Format queries
            queries = []
            query_rulelist = [None]*len(envs)
            for env_i in range(len(envs)):
                if env_status[env_i]==1:
                    env = envs[env_i]
                    obs, reward, done, info = env_cur_ret[env_i]
                    state_name = env.cur_state_name
                    instruction = env_instruction[env_i]
                    taken_actions = tmp_records[env_test_idx[env_i]]['action'][-10:]
                    available_actions = env.get_cleaned_available_actions()
                    # random rule order
                    query_rulelist[env_i] = np.random.permutation(best_rules[state_name].copy()).tolist()
                    queries.append({
                        "observation": obs,
                        "state_name": state_name,
                        "available_actions": available_actions,
                        "rule_list": [r.content for r in query_rulelist[env_i]],
                        "taken_actions": taken_actions,
                        "instruction": instruction,
                    })
                    call_count+=1
                else:
                    queries.append(None)
            
            if queries[0] is not None:
                prompt = eval_action_chains[0].test_format_input(queries[0])
            # Get actions
            s = time.time()
            outputs = asyncio.run(async_chains_run(eval_action_chains, queries))
            e = time.time()
            token_count = sum([int(output.get("token_count", 0)) for output in outputs if output is not None])
            est_step_cost = token_count * 0.0017/1000
            est_total_cost += est_step_cost
            call_time_s = e-s
            token_rate_limit_handler(s, token_count)

            eval_record["est_total_cost"] = eval_record.get("est_total_cost", 0.0) + float(est_step_cost)
            eval_record["total_token_count"] = eval_record.get("total_token_count", 0) + token_count
            log = log+f"Eval {eval_step}/{num_eval} average_reward {eval_record.get('average_reward', 'N/A')} success_rate {eval_record.get('success_rate', 'N/A')} num_active_env {sum(env_status)} call_count {call_count} call_time {call_time_s} token_count {token_count} est_total_cost {est_total_cost}\n"
            print(log)
            logging.info(log)
            log = ""

            # Step Envs
            for env_i in range(len(envs)):
                if env_status[env_i]==0:
                    continue
                env_ep_len[env_i]+=1
                if not outputs[env_i].get("valid", False):
                    print(f"env{env_i}({env_test_idx[env_i]}, {env_ep_len[env_i]} steps) invalid output")
                    outputs[env_i]["selected_action"] = "N/A"
                env = envs[env_i]
                obs, reward, done, info = env_cur_ret[env_i]
                reward = reward if reward is not None else 0.0
                done = done if done is not None else False

                actions = outputs[env_i].get("selected_action", "N/A")
                if isinstance(actions, str):
                    actions = [actions]
                for action in actions:
                    selected_rule_number = outputs[env_i].get("selected_rule_number", None)
                    selected_rule_id = None
                    if not selected_rule_number is None:
                        selected_rule: Rule = query_rulelist[env_i][selected_rule_number-1]
                        selected_rule_id = (selected_rule.rule_type, selected_rule.rule_id)
                    tmp_records[env_test_idx[env_i]]['selected_rule_number'].append(selected_rule_number)
                    tmp_records[env_test_idx[env_i]]['selected_rule_id'].append(selected_rule_id)
                    tmp_records[env_test_idx[env_i]]['state_name'].append(env.cur_state_name)
                    tmp_records[env_test_idx[env_i]]['action'].append(action)
                    tmp_records[env_test_idx[env_i]]['reward'].append(float(reward))
                    tmp_records[env_test_idx[env_i]]['info'].append(info)
                    log = log+f"\nenv{env_i}({env_test_idx[env_i]}, {env_ep_len[env_i]} steps) {env.cur_state_name} {action} "

                    env_cur_ret[env_i] = env.step(action)
                    obs, reward, done, info = env_cur_ret[env_i]
                    log = log+f"{env.cur_state_name} {reward} {done} "
                    if done or env_ep_len[env_i]>=max_ep_len:
                        log = log+f"DONE {reward} "
                        tmp_records[env_test_idx[env_i]]['state_name'].append("N/A")
                        tmp_records[env_test_idx[env_i]]['action'].append("N/A")
                        tmp_records[env_test_idx[env_i]]['reward'].append(float(reward))
                        tmp_records[env_test_idx[env_i]]['info'].append(info)
                        # Save record
                        eval_record["eval_history"][str(env_test_idx[env_i])] = {
                            "history": tmp_records[env_test_idx[env_i]].copy(),
                            "total_reward": sum(tmp_records[env_test_idx[env_i]]['reward'])
                        }
                        average_reward = np.mean([eval_record["eval_history"][str(idx)]["total_reward"] for idx in eval_record["eval_history"].keys()])
                        success_rate = np.mean([int(eval_record["eval_history"][str(idx)]["total_reward"]>=1-1e-6) for idx in eval_record["eval_history"].keys()])
                        print(f"\nAverage reward {eval_step}/{num_eval}: {average_reward}, success_rate {success_rate}\n")
                        logging.info(f"Average reward {eval_step}/{num_eval}: {average_reward}, success_rate {success_rate}\n")
                        eval_record["average_reward"] = float(average_reward)
                        eval_record["success_rate"] = float(success_rate)
                        some_done = True

                        with open(str(eval_record_path), "w") as f:
                            json.dump(eval_record, f)
                        with open(str(Path(log_folder) / "eval_record.json"), "w") as f:
                            json.dump(eval_record, f)

                        # RESET / Assign new idx
                        eval_action_chains[env_i].reset()
                        # assign new test to done env.
                        if test_idx!=-1 and not assigned_env:
                            env_status[env_i]=1
                            env_ep_len[env_i]=0
                            env_test_idx[env_i]=test_idx
                            env_cur_ret[env_i] = list(envs[env_i].reset(test_idx))+[False, None]
                            obs, reward, done, info = env_cur_ret[env_i]
                            if react_style:
                                env_instruction[env_i] = envs[env_i].get_instruction()
                            assigned_env = True
                            print(f"\n\nAssigned env {env_i} to test_idx {test_idx}\n\n")
                        else:
                            env_status[env_i]=0
                            env_ep_len[env_i]=0
                            env_test_idx[env_i]=-1
                            env_cur_ret[env_i]=None
                        break
            if len(log)>0:
                print(log+"\n")
                logging.info(log)
            if some_done and test_idx!=-1:
                # more test_idx to eval
                break
            elif sum(env_status)==0:
                # all done
                break

if __name__ == "__main__":
    split = "test"
    max_results = 3
    num_products=None
    if DEBUG:
        print(f"\n\n\nWARNING: USING DEBUG MODE!.\n\n\n")
        split="test"
        num_products=1000
    if not num_products is None:
        assert split=="test", "If using 1000 products, split must be test"
        print(f"\n\n\nWARNING: num_products is {num_products}. Set it to None for real evaluation.\n\n\n")
    save_file_prefix = f"eval_{split}_{'1M' if num_products is None else str(num_products)}"
    envs = []
    for i in range(MAX_PARALLEL):
        attempt = 5
        while attempt>0:
            try:
                env = WebShopEnv(observation_mode='text_rich', max_result_items=max_results, split=split, human_goals=True, num_products=num_products)
                break
            except:
                attempt-=1
                time.sleep(0.5)
                continue
        envs.append(env)
    print(f"Created {len(envs)} envs")

    train_config_args = []
    # rulesets_paths = Path("rulesets/contrastive_imitation")
    # for child in rulesets_paths.iterdir():
    #     if child.is_dir():
    #         ruleset_dir = child
    #         train_config_path = ruleset_dir / "train_config.json"
    #         eval_record_path = ruleset_dir / "eval" / "eval_record_500_w_act_clean.json"
    #         train_config_args.append((train_config_path, eval_record_path, 0))


    # Hand written #TODO
    for seed in range(3):
        train_config_args.append((
            "rulesets/contrastive_imitation/w_timerule/hand_written/train_config.json",
            f"rulesets/contrastive_imitation/w_timerule/hand_written/eval/{save_file_prefix}_seed_{seed}.json",
            seed
        ))
    
    # No rule
        # react_style
    # train_config_args.append((
    #     "rulesets/contrastive_imitation/no_rule/react_config.json",
    #     f"rulesets/contrastive_imitation/no_rule/eval/{save_file_prefix}_NORULE_REACT_seed_1.json",
    #     1
    # ))
    # train_config_args.append((
    #     "rulesets/contrastive_imitation/no_rule/react_config.json",
    #     f"rulesets/contrastive_imitation/no_rule/eval/{save_file_prefix}_NORULE_REACT_seed_2.json",
    #     2
    # ))
        # non-react style #TODO
    for seed in range(3):
        train_config_args.append((
            f"rulesets/contrastive_imitation/no_rule/non_react_config.json",
            f"rulesets/contrastive_imitation/no_rule/eval/{save_file_prefix}_NORULE_NONREACT_seed_{seed}.json",
            seed
        ))

    # Filtered with time
    # train_config_args.append(
    # (
    #     "rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/0/train_config.json",
    #     f"rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/0/eval/{save_file_prefix}.json",
    #     0
    # ),)
    # train_config_args.append((
    #     "rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/1/train_config.json",
    #     f"rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/1/eval/{save_file_prefix}.json",
    #     1
    # ))
    # train_config_args.append((
    #     "rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/2/train_config.json",
    #     f"rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/2/eval/{save_file_prefix}.json",
    #     2
    # ))
    
    # All with time
    # train_config_args.append(
    # (
    #     "rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/0/train_config.json",
    #     f"rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/0/eval/{save_file_prefix}.json",
    #     0
    # ),)
    # train_config_args.append((
    #     "rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/1/train_config.json",
    #     f"rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/1/eval/{save_file_prefix}.json",
    #     1
    # ))
    # train_config_args.append((
    #     "rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/2/train_config.json",
    #     f"rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/2/eval/{save_file_prefix}.json",
    #     2
    # ))

    # Iteration vs Score
    # for seed in range(3):
    #     for ckp_timestep in [20,60]:
    #         ruleset_ckp_rel_path = f"checkpoint/ruleset_checkpoint_timestep_{ckp_timestep}.json"
    #         train_config_path = f"rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/{seed}/train_config.json"
    #         ruleset_path = str(Path(train_config_path).parent / ruleset_ckp_rel_path)
    #         assert os.path.exists(ruleset_path), f"ruleset_path {ruleset_path} does not exist"
    #         train_config_args.append((
    #             train_config_path,
    #             ruleset_path,
    #             f"rulesets/contrastive_imitation/filtered_traj/w_time_obs_timelimit_2/{seed}/eval/{save_file_prefix}_ckp_{ckp_timestep}.json",
    #             seed
    #         ))


    for arg in train_config_args:
        ruleset_path = None
        if len(arg) == 3:
            train_config_path, eval_record_path, seed = arg
        elif len(arg) == 4:
            train_config_path, ruleset_path, eval_record_path, seed = arg
        eval(
            envs,
            seed=seed,
            train_config_path=str(train_config_path),
            ruleset_path=ruleset_path,
            eval_record_path=str(eval_record_path),
            low_timestep_threshold=5,
            num_rule_per_step=10,
            num_eval=50,
        )
