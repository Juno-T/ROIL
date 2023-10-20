from math import e
import os
import sys
import time

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
from auto_policy_programming.wrappers.webshop.webshop_wrapper import WebAgentTextEnvReActStyle, WebAgentTextEnvWithStateName

DEBUG = True
MAX_PARALLEL = 1

log_folder = Path("log") / Path(datetime.now().strftime("%Y-%m-%d")) / str(datetime.now().strftime("%H-%M-%S"))
log_folder.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_folder / "test.log"), 
    filemode='w',
    format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
    level=logging.DEBUG if DEBUG else logging.INFO)

def get_rules(ruleset, state, num_rules=5):
    best_rules = ruleset.best_rules[state].copy()
    sorted_rules = sorted(best_rules.values(), key=lambda x: x.score, reverse=True)
    return sorted_rules[:num_rules]


async def async_chain_run(chain, instruction, obs, state_name, available_actions, rules, react_history):
    try:
        inputs = {
            "instruction": instruction,
            "observation": obs,
            "state_name": state_name,
            "available_actions": available_actions,
            "rule_list": rules,
            "react_history": react_history
        }
        # print(chain.test_format_input(inputs))
        with get_openai_callback() as cb:
            output = await chain.acall(inputs)
        output["token_count"] = int(cb.total_tokens)
        return output
    except:
        return {
            "valid": False,
            "selected_action": None,
        }
    
async def get_actions(chain, queries):
    tasks = [async_chain_run(chain, *query) for query in queries if query is not None]
    ret = await asyncio.gather(*tasks)
    cur_ret = 0
    outputs = []
    for query in queries:
        if query is None:
            outputs.append({
                "valid": False,
                "selected_action": None,
            })
        else:
            outputs.append(ret[cur_ret])
            cur_ret+=1
    return outputs

def eval(
    envs,
    ruleset_path: str,
    eval_record_path: str,
    seed=0,
    perc_eval=1.0,
    max_ep_len=15,
):
    np.random.seed(seed)
    start_time = datetime.now()
    if perc_eval==1.0:
        num_eval = 500
        test_indices = np.arange(500).tolist()
    else:
        num_eval = int(perc_eval * 500)
        test_indices = np.arange(num_eval).tolist()
    test_indices.append(-1) # mark the end for final eval loop
    ruleset = RuleSet.json_load(ruleset_path)
    if os.path.exists(eval_record_path):
        with open(eval_record_path, "r") as f:
            eval_record = json.load(f)
    else:
        Path(eval_record_path).parent.mkdir(parents=True, exist_ok=True)
        eval_record = {
            "ruleset_path": str(Path(ruleset_path).absolute()),
            "average_reward": 0.0, "eval_history": {}
        }

    best_rules = {state: get_rules(ruleset, state, 5) for state in ruleset.states}
    # llm = FakeListLLM(responses=["thought: my thought\nrelevant_rule: 1\naction: search[asdfg]\n\n"])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    eval_action_chain = chains.EvaluationReactStep(llm=llm)
    
    env_status = [0] * len(envs)
    env_ep_len = [0] * len(envs)
    env_test_idx = [-1] * len(envs)
    env_cur_ret = [None] * len(envs)
    tmp_records = {
        idx: {
            'state_name': [],
            'action': [],
            'reward': [],
            'info': [],
            'selected_rule_id': [],
            'selected_rule_number': [],
            'react_history': [],
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
                    tmp_records[test_idx]["instruction"] = envs[env_i].instruction
                    obs, reward, done, info = env_cur_ret[env_i]
                    assigned_env = True
                    print(f"\n\nAssigned env {env_i} to test_idx {test_idx}\n\n")
                    break
            if assigned_env:
                continue

        some_done = False
        while True:
            queries = []
            log = "\n"+("-"*10)+"\n"
            query_rulelist = [None]*len(envs)
            for env_i in range(len(envs)):
                if env_status[env_i]==1:
                    env = envs[env_i]
                    obs, reward, done, info = env_cur_ret[env_i]
                    state_name = env.cur_state_name
                    available_actions = env.get_cleaned_available_actions()
                    # random rule order
                    query_rulelist[env_i] = np.random.permutation(best_rules[state_name].copy()).tolist()
                    react_history = tmp_records[env_test_idx[env_i]]['react_history']
                    instruction = tmp_records[env_test_idx[env_i]]["instruction"]
                    queries.append((instruction, obs, state_name, available_actions, [r.content for r in query_rulelist[env_i]], react_history))
                else:
                    queries.append(None)
            
            s = time.time()
            outputs = asyncio.run(get_actions(eval_action_chain, queries))
            e = time.time()
            call_count+=1

            token_count = sum([int(output.get("token_count", 0)) for output in outputs if output is not None])
            est_step_cost = token_count * 0.0017/1000
            est_total_cost += est_step_cost
            eval_record["est_total_cost"] = eval_record.get("est_total_cost", 0.0) + float(est_step_cost)
            eval_record["total_token_count"] = eval_record.get("total_token_count", 0) + token_count
            log = log+f"Eval {eval_step}/{num_eval} average_reward {eval_record.get('average_reward', 0.0)} num_active_env {sum(env_status)} call_count {call_count} call_time {e-s} token_count {token_count} est_total_cost {est_total_cost}\n"
            print(log)
            logging.info(log)
            log = ""
            for env_i in range(len(envs)):
                if env_status[env_i]==0:
                    continue
                env_ep_len[env_i]+=1
                if not outputs[env_i].get("valid", False):
                    print(f"env{env_i}({env_test_idx[env_i]}) invalid output")
                    continue


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
                    tmp_records[env_test_idx[env_i]]['react_history'].append((obs, outputs[env_i]["thought"], action))
                    log = log+f"\nenv{env_i}({env_test_idx[env_i]}) {env.cur_state_name} {action} "

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
                        print(f"\nAverage reward {eval_step}/{num_eval}: {average_reward}\n")
                        logging.info(f"Average reward {eval_step}/{num_eval}: {average_reward}")
                        eval_record["average_reward"] = float(average_reward)
                        some_done = True

                        with open(str(eval_record_path), "w") as f:
                            json.dump(eval_record, f)
                        with open(str(Path(log_folder) / "eval_record.json"), "w") as f:
                            json.dump(eval_record, f)

                        # assign new test to done env.
                        if test_idx!=-1 and not assigned_env:
                            env_status[env_i]=1
                            env_ep_len[env_i]=0
                            env_test_idx[env_i]=test_idx
                            env_cur_ret[env_i] = list(envs[env_i].reset(test_idx))+[False, None]
                            tmp_records[test_idx]["instruction"] = envs[env_i].instruction
                            obs, reward, done, info = env_cur_ret[env_i]
                            obs, reward, done, info = env_cur_ret[env_i]
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
    envs = []
    for i in range(MAX_PARALLEL):
        attempt = 5
        while attempt>0:
            try:
                env = WebAgentTextEnvReActStyle(observation_mode='text_rich')
                break
            except:
                attempt-=1
                time.sleep(0.5)
                continue
        envs.append(env)
    print(f"Created {len(envs)} envs")

    rulesets_path = Path("rulesets/contrastive_imitation")
    for child in rulesets_path.iterdir():
        if child.is_dir():
            ruleset_dir = child
            ruleset_path = ruleset_dir / "checkpoint/ruleset_checkpoint_timestep_500.json"
            eval_record_path = ruleset_dir / "eval" / "eval_record_500_react_style.json"
        eval(
            envs,
            ruleset_path=str(ruleset_path),
            eval_record_path=str(eval_record_path),
            perc_eval=0.01,
        )
