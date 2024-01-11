import asyncio
from copy import deepcopy
import os
import sys
from typing import List


sys.path.append(os.getcwd())

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
from auto_policy_programming.common.utils import timeout
from auto_policy_programming.chains.base import ParsingChain
from auto_policy_programming.wrappers.timelimit import low_timestep_obs_aug
from runner.webshop.eval import async_chains_run

DEBUG = True
MAX_PARALLEL = 5

log_folder = Path("log") / Path(datetime.now().strftime("%Y-%m-%d")) / str(datetime.now().strftime("%H-%M-%S"))
log_folder.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_folder / "train.log"), 
    filemode='w',
    format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
    level=logging.DEBUG if DEBUG else logging.INFO)


def select_learning_rules(ruleset: RuleSet, state: str, num_rules: int = 10, timestep: int = 1):
    best_rules = ruleset.best_rules[state].copy()
    sorted_rules = sorted(best_rules.values(), key=lambda x: x.calc_ucb(timestep), reverse=True)
    return sorted_rules[:num_rules]

@timeout(seconds=30)
def timeout_chain_call(chain, inputs):
    return chain(inputs=inputs)

def evaluate(chains: List[ParsingChain], rules: List[Rule], test_trajs: List[dict], low_timestep_threshold):
    results = []

    queries = []
    for traj in test_trajs:
        instruction = None
        obs = traj['state']
        state_name = traj['state_name']
        taken_actions = [a for i, a in traj['action_history']][-5:]
        available_actions = traj['available_action']
        rule_list = [r.content for r in rules]
        if low_timestep_threshold > 0:
            remaining_timesteps = traj[i]['original_length'] - traj[i]['timestep']
            if remaining_timesteps <= low_timestep_threshold:
                obs = low_timestep_obs_aug(obs)
        queries.append({
            "instruction": instruction,
            "observation": obs,
            "state_name": state_name,
            "taken_actions": taken_actions,
            "available_actions": available_actions,
            "rule_list": rule_list,
        })

    prompt = chains[0].test_format_input(queries[0])
    # outputs = async_chains_run(chains, queries, max_retry=1)
    outputs = []
    for i in range(0, len(queries), MAX_PARALLEL):
        outputs += asyncio.run(async_chains_run(chains, queries[i:i+MAX_PARALLEL], max_retry=1))

    total_tokens = sum([output.get("token_count", 0) for output in outputs])

    score = 0
    for traj, output in zip(test_trajs, outputs):
        result = {
            "traj": traj,
            "output": output,
            "correct": 0,
            "token_count": output.get("token_count", 0)
        }
        if output['valid']:
            result["correct"] = int(output['selected_action'].lower() == traj['action'].lower())
        score += result["correct"]
        results.append(result)
    return score, results, total_tokens

def re_align(chains, rules: List[Rule], result, low_timestep_threshold, num_populate=3):
    if not result["correct"]==False:
        logging.warning("No need to re-align")
        return None
    if not result["output"]["valid"]:
        logging.warning("Invalid output")
        return None
    traj = result["traj"]

    if low_timestep_threshold > 0:
        remaining_timesteps = traj['original_length'] - traj['timestep']
        if remaining_timesteps <= low_timestep_threshold:
            obs = low_timestep_obs_aug(obs)
    query = {
        "observation": traj["state"],
        "taken_actions": [a for i, a in traj['action_history']][-5:],
        "available_actions": traj["available_action"],
        "rule_list": [r.content for r in rules],
        "wrong_thoughts": result["output"]["thoughts"],
        "wrong_action": result["output"]["selected_action"],
        "correct_action": traj["action"],
    }
    queries = [query for _ in range(num_populate)]
    prompt = chains[0].test_format_input(queries[0])
    outputs = asyncio.run(async_chains_run(chains, queries, max_retry=1))
    total_tokens = sum([output.get("token_count", 0) for output in outputs])
    amendments = []
    for output in outputs:
        if not output["valid"]:
            amendments.append(None)
            continue
        to_amend_rules = [
            {
                "rule_number": output["should_be_selected_rule_number"],
                "updated_content": output["should_be_selected_rule_update"],
            },
            {
                "rule_number": output["wrong_selected_rule_number"],
                "updated_content": output["wrong_selected_rule_update"],
            },
        ]
        amendments.append(to_amend_rules)
    return amendments, total_tokens

# Populate 3 with re-align, evaluate with test_traj, then throw it back into the pool. Then randomly select one to re-align using score as prob.
# End if found perfect score or reach max_iter.
    # Because we have limited evaluation budget, we can only evaluate a small number of trajectories.

def align(
    save_dir: str,
    train_config_path: str,
    seed=0,
    num_rule_per_step = 10,
    max_align_iter = 10,
    min_align_iter = 7,
    checkpoint_interval = 2,
    low_timestep_threshold = -1,
    max_stale_iter = 5,
    num_populate = 3,
    num_test_traj = 10,
    filtered_traj = False,
):
    start_time = datetime.now()
    np.random.seed(seed)

    assert train_config_path is not None, "train_config_path is None"
    with open(train_config_path, "r") as f:
        align_config = json.load(f)
    ruleset_path = Path(align_config["final_ruleset_path"])
    ruleset_path = Path(train_config_path).parent / ruleset_path
    seed = align_config.get("seed", seed)
    low_timestep_threshold = align_config.get("low_timestep_threshold", low_timestep_threshold)
    
    align_config = {
        "seed": seed,
        "ruleset_path": str(ruleset_path),
        "num_rule_per_step": num_rule_per_step,
        "max_align_iter": max_align_iter,
        "num_test_traj": num_test_traj,
        "num_populate": num_populate,
        "filtered_traj": filtered_traj,
        "low_timestep_threshold": low_timestep_threshold,
        "checkpoint_interval": checkpoint_interval,
        "checkpoints": {},
        # "llm": "fakellm",
        "llm": "gpt-3.5-turbo-0613",
        # "llm": "gpt-4-0613",
    }


    assert (ruleset_path is not None) and os.path.exists(ruleset_path), f"ruleset_path {ruleset_path} does not exist"
    base_ruleset = RuleSet.json_load(ruleset_path)
    best_rules = {state: base_ruleset.get_best_rules(state, num_rules=align_config["num_rule_per_step"]) for state in base_ruleset.states}
    # TODO: Add progress log with scores, num amendment, time elapsed, tokens used.
    progress_log = {
        state_name: {
            "skipped": False,
            "num_amendments": [],
            "time_elapsed_s": [],
            "tokens_used": [],
        }
        for state_name in base_ruleset.states
    }


    logging.info(f"ALIGN CONFIG:\n{json.dumps(align_config, indent=4)}\n")

    save_dir = Path(save_dir)
    checkpoint_folder = save_dir / "align_checkpoint"
    align_config_save_path = save_dir / "align_config.json"
    final_save_path = checkpoint_folder / f"final_ruleset.json"
    assert not os.path.exists(align_config_save_path)
    assert not os.path.exists(checkpoint_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    if align_config["llm"] == "fakellm":
        llm = FakeListLLM(responses=["thoughts: My thoughts\nselected_one_best_action: click[something]\nwrong_selected_rule: 1, content\namend_wrong_selected_rule: My updated rule 0\nshould_be_selected_rule: 2, content\namend_should_be_selected_rule: My updated rule 1\n"])
    else:
        llm = ChatOpenAI(model=align_config["llm"], temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    evaluate_steps = [chains.EvaluationStep(llm=llm) for _ in range(MAX_PARALLEL)]
    align_steps = [chains.RuleAlignmentChain(llm=llm) for _ in range(MAX_PARALLEL)]


    traj_path = "./data/webshop/human_traj_sa_by_state_filtered.json" if align_config['filtered_traj'] \
        else "./data/webshop/human_traj_sa_by_state.json"
    with open(traj_path, "r") as f:
        traj = json.load(f)

    sa_lengths = [len(traj[k]) for k in traj.keys()]
    total_sa_length = sum(sa_lengths)

    state_test_trajs = {state_name: np.random.choice(traj[state_name], size=align_config["num_test_traj"], replace=False) for state_name in base_ruleset.states}
    final_ruleset = RuleSet(base_ruleset.states)
    checkpoint_rulesets = {i: RuleSet(base_ruleset.states) for i in range(1,align_config["max_align_iter"]+1) if i % align_config["checkpoint_interval"] == 0}
    total_tokens = 0
    total_iter = 0
    for state_name in base_ruleset.states:
        assert state_name in traj.keys(), f"state {state_name} not in traj"
        if len(best_rules[state_name])<=1:
            logging.info(f"State {state_name} has only one rule, saving to final ruleset without alignment.")
            progress_log[state_name]["skipped"] = True
            for checkpoint_iter in checkpoint_rulesets.keys():
                checkpoint_save_path = checkpoint_folder / f"checkpoint_{checkpoint_iter}.json"
                checkpoint_rulesets[checkpoint_iter].add_rules(state_name, best_rules[state_name])
                checkpoint_rulesets[checkpoint_iter].json_save(checkpoint_save_path)
                
                if state_name == base_ruleset.states[-1]:
                    align_config["checkpoints"].update({
                        str(checkpoint_iter): str(Path.resolve(checkpoint_save_path).relative_to(Path.resolve(save_dir)))
                    })
            final_ruleset.add_rules(state_name, best_rules[state_name])
            continue
        # if state_name!="item":
        #     continue
        log = "="*20
        log += f"\n\nState {state_name} has {len(best_rules[state_name])} rules, starting alignment\n\n"
        log += "="*20
        print(log)
        logging.info(log)
        test_trajs = state_test_trajs[state_name]
        rulelist_pool = [deepcopy(best_rules[state_name])]
        score, results, token_count = evaluate(evaluate_steps, rulelist_pool[0], test_trajs, align_config["low_timestep_threshold"])
        print(f"\teval score {score} token_count {token_count}")

        stale_iter_count = 0
        total_tokens += token_count
        max_score = score
        max_score_rulelist = rulelist_pool[0]
        max_score_rulelist_results = results
        rulelist_pool_score = [score]
        rulelist_pool_results = [results]
        rulelist_pool_amend_depth = [0]
        for align_iter in range(align_config['max_align_iter']):
            logging.info(f"State {state_name} align iter {align_iter}")
            total_iter += 1
            stale_iter_count += 1
            iter_token_count = 0
            iter_start_time = datetime.now()
            num_amendments = 0
            if max_score == align_config["num_test_traj"]:
                # If max score achieved, skip alignment
                log = f"State {state_name} found perfect score, skipping align iter {align_iter}"
                print(log)
                logging.info(log)
            elif stale_iter_count > max_stale_iter and align_iter >= min_align_iter:
                # If stale iter count reached max, skip alignment
                log = f"State {state_name} stale iter count reached max, skipping align iter {align_iter}"
                print(log)
                logging.info(log)
            else:
                # Random select rulelist from pool by softmax of score
                prob = np.array(rulelist_pool_score)
                prob = np.exp(prob)
                prob = prob / prob.sum()
                rulelist_pool_index = np.random.choice(len(rulelist_pool), p=prob)
                to_align_rulelist = rulelist_pool[rulelist_pool_index]
                to_align_results = rulelist_pool_results[rulelist_pool_index]
                aligned_rulelist_amend_depth = rulelist_pool_amend_depth[rulelist_pool_index] + 1

                # Populate by re_align with incorrect results

                ## random incorrect result
                incorrect_results = [r for r in to_align_results if r["correct"]==False]
                if len(incorrect_results) == 0:
                    logging.info(f"State {state_name} align iter {align_iter} found no incorrect result")
                    break
                incorrect_result = np.random.choice(incorrect_results)
                ## Amend
                amendments, token_count = re_align(
                    align_steps,
                    to_align_rulelist, 
                    incorrect_result, 
                    align_config["low_timestep_threshold"], 
                    num_populate=align_config["num_populate"]
                )
                total_tokens += token_count
                iter_token_count += token_count
                num_amendments = len([r for r in amendments if r is not None])
                print(f"Re_aligned, {num_amendments} amendments, token_count {token_count}")
                for amends in amendments:
                    if amends is None:
                        continue
                    aligned_rulelist = deepcopy(to_align_rulelist)
                    for amend in amends:
                        aligned_rulelist[amend["rule_number"]].content = amend["updated_content"]

                    ## Eval
                    score, results, token_count = evaluate(evaluate_steps, aligned_rulelist, test_trajs, align_config["low_timestep_threshold"])
                    total_tokens += token_count
                    iter_token_count += token_count
                    print(f"\teval score {score} token_count {token_count}")
                    rulelist_pool.append(aligned_rulelist)
                    rulelist_pool_score.append(score)
                    rulelist_pool_results.append(results)
                    rulelist_pool_amend_depth.append(aligned_rulelist_amend_depth)
                    if score >= max_score:
                        if score > max_score:
                            stale_iter_count = 0
                        max_score = score
                        max_score_rulelist = aligned_rulelist
                        max_score_rulelist_results = results
            
            # Exp Records
            progress_log[state_name]["score"] = rulelist_pool_score
            progress_log[state_name]["amend_depth"] = rulelist_pool_amend_depth
            progress_log[state_name]["num_amendments"].append(num_amendments)
            progress_log[state_name]["time_elapsed_s"].append((datetime.now() - iter_start_time).total_seconds())
            progress_log[state_name]["tokens_used"].append(iter_token_count)
            elapsed_time_s = (datetime.now() - start_time).total_seconds()
            est_cost = total_tokens*0.0017/1000*(25 if "gpt-4" in align_config["llm"] else 1)
            log = f"ELAPSED_TIME_S: {elapsed_time_s} State: {state_name} Iter: {align_iter}/{align_config['max_align_iter']} EST_TOTAL_COST($): {est_cost}"
            log += f"\tMAX_SCORE: {max_score}\n\tMAX_SCORE_RULELIST: {[r.content for r in max_score_rulelist]}\n\n" + ("-"*20) + "\n\n"
            print(log)
            logging.info(log)
            if (align_iter+1) % align_config["checkpoint_interval"] == 0:
                checkpoint_save_path = checkpoint_folder / f"checkpoint_{align_iter+1}.json"
                checkpoint_rulesets[align_iter+1].add_rules(state_name, max_score_rulelist)
                checkpoint_rulesets[align_iter+1].json_save(checkpoint_save_path)
                
                align_config.update({
                    "total_tokens": total_tokens,
                    "progress_log": progress_log,
                })
                if state_name == base_ruleset.states[-1]:
                    align_config["checkpoints"].update({
                        checkpoint_iter: str(Path.resolve(checkpoint_save_path).relative_to(Path.resolve(save_dir)))
                    })
                with open(str(align_config_save_path), "w") as f:
                    json.dump(align_config, f, indent=4)

        final_ruleset.add_rules(state_name, max_score_rulelist)
        final_ruleset.json_save(final_save_path)

    # Save final_ruleset
    final_ruleset.json_save(final_save_path)
    align_config["final_ruleset_path"] = str(Path.resolve(final_save_path).relative_to(Path.resolve(save_dir)))
    align_config.update({
        "total_tokens": total_tokens,
        "progress_log": progress_log,
    })
    with open(str(align_config_save_path), "w") as f:
        json.dump(align_config, f, indent=4)


if __name__ == "__main__":
    seed = 0
    train_config = Path(f"./rulesets/contrastive_imitation/filtered_traj/{seed}/train_config.json")
    save_dir = train_config.parent

    align(save_dir, train_config, seed=seed, filtered_traj=True,
          num_test_traj=10, num_populate=3, max_align_iter=20, min_align_iter=10, num_rule_per_step=5)