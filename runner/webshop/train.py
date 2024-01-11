import os
import sys


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
from auto_policy_programming.wrappers.timelimit import low_timestep_obs_aug

DEBUG = True

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

def increment_selection_count(ruleset: RuleSet, state: str, rule_type: str, rule_id: int):
    ruleset.rules[state][rule_type][rule_id].selection_count += 1
    ruleset.best_rules[state][rule_type].selection_count += 1

@timeout(seconds=30)
def timeout_chain_call(chain, inputs):
    return chain(inputs=inputs)

def train(
    save_dir: str,
    seed=0,
    num_training_steps = 100,
    num_rule_per_step = 10,
    learning_step_max_retry = 2,
    low_timestep_threshold = -1,
    checkpoint_interval = 10,
    filtered_traj = False,
):
    start_time = datetime.now()
    np.random.seed(seed)
    train_config = {
        "seed": seed,
        "num_training_steps": num_training_steps,
        "num_rule_per_step": num_rule_per_step,
        "learning_step_max_retry": learning_step_max_retry,
        "checkpoint_interval": checkpoint_interval,
        "learning_policy": "contrastive_imitation",
        "rule_selection_method": "ucb",
        "low_timestep_threshold": low_timestep_threshold,
        "filtered_traj": filtered_traj,
        "llm": "gpt-3.5-turbo-0613",
        # "llm": "gpt-4-0613",
    }
    logging.info(f"TRAIN CONFIG:\n{json.dumps(train_config, indent=4)}\n")

    assert not os.path.exists(save_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_folder = save_dir / "checkpoint"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    # llm = FakeListLLM(responses=[
    #     f"related_rule_number: 1\nfound_rule: My found rule{i}\nrelated_rule_content: rule1\nsame_intention\nfound_rule_better\n"
    #     + "\nNO_EXISTING_RULE" if i==0 else "" for i in range(20)
    # ])
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    llm = ChatOpenAI(model=train_config["llm"], temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    # llm = ChatOpenAI(model="gpt-4-0613", temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))


    traj_path = "./data/webshop/human_traj_sa_by_state_filtered.json" if filtered_traj \
        else "./data/webshop/human_traj_sa_by_state.json"
    with open(traj_path, "r") as f:
        traj = json.load(f)

    ruleset = RuleSet(list(traj.keys()))

    contrastive_step = chains.ContrastiveImitation(llm=llm)
    progress_log = {
        "token_counts": [],
        "valid": [],
        "rule_type_counts": {
            k: [] for k in ruleset.states
        },
        "time_elapsed_s": [],
    }

    sa_lengths = [len(traj[k]) for k in traj.keys()]
    total_sa_length = sum(sa_lengths)

    for training_step in range(num_training_steps):
        sa_index = np.random.randint(total_sa_length)
        for k in traj.keys():
            if sa_index < len(traj[k]):
                selected_sa = traj[k][sa_index]
                break
            sa_index -= len(traj[k])
        selected_state_name = selected_sa["state_name"]
        selected_state_sa_list = traj[selected_state_name]
        for _ in range(10):
            contrast_sa_index = np.random.randint(len(selected_state_sa_list))
            if selected_state_sa_list[contrast_sa_index]["action"] != selected_sa["action"]:
                break
        contrast_sa = selected_state_sa_list[contrast_sa_index]
        records = [selected_sa, contrast_sa]
        if low_timestep_threshold > 0:
            for i in range(len(records)):
                remaining_timesteps = records[i]['original_length'] - records[i]['timestep']
                if remaining_timesteps <= low_timestep_threshold:
                    records[i]["state"] = low_timestep_obs_aug(records[i]["state"])

        existing_rules = select_learning_rules(ruleset, selected_state_name, num_rule_per_step, training_step)
        existing_rules_str = [r.content for r in existing_rules]
        
        prompt = contrastive_step.test_format_input(inputs={"records": records, "existing_rules": existing_rules_str})
        logging.info(f"STEP: {training_step}\n")
        
        token_count = 0
        for __ in range(learning_step_max_retry):
            try:
                with get_openai_callback() as cb:
                    output = timeout_chain_call(contrastive_step, inputs={"records": records, "existing_rules": existing_rules_str})
                    token_count += int(cb.total_tokens)
            except Exception as e:
                # log full traceback
                logging.error(f"ERROR: {e}")
                output = {
                    "valid": False,
                    "raw_output": "ERROR",
                }
                break
            logging.debug(f"PROMPT:\n{prompt}\nRAW OUTPUT:\n{output['raw_output']}\n")
            if output["valid"]:
                break
        
        # Output parsing
        parse_successfully = True
        if not output["valid"]:
            parse_successfully = False
            logging.info(f"INVALID OUTPUT")
        else:
            selected_rule = None
            if output["no_existing_rule"]:
                selected_rule = None
            elif isinstance(output["related_rule_number"], int) and output["related_rule_number"] < len(existing_rules_str):
                selected_rule = existing_rules[output["related_rule_number"]]
            else:
                parse_successfully = False
            new_rule = output["found_rule"]
            if output["related_rule_classification"] == "similar_intention":
                new_rule = output["updated_rule"]
            if new_rule is None:
                parse_successfully = False
            else:
                new_rule = new_rule.replace("\n", " ")
                new_rule = new_rule.replace("{", "")
                new_rule = new_rule.replace("}", "")

        if parse_successfully:
            for i, rule in enumerate(existing_rules):
                increment_selection_count(ruleset, selected_state_name, rule.rule_type, rule.rule_id)
            ruleset.update(selected_state_name, selected_rule, new_rule, output["related_rule_classification"])

        
        progress_log["token_counts"].append(token_count)
        progress_log["valid"].append(output["valid"])
        elapsed_time_s = (datetime.now() - start_time).total_seconds()
        progress_log["time_elapsed_s"].append(elapsed_time_s)
        for state in ruleset.states:
            progress_log["rule_type_counts"][state].append(len(ruleset.rules[state]))
        est_cost = sum(progress_log["token_counts"])*0.0017/1000*(25 if "gpt-4" in train_config["llm"] else 1)
        log = f"ELAPSED_TIME_S: {elapsed_time_s} STEP: {training_step+1}/{num_training_steps} TOKENS: {token_count} EST_TOTAL_COST($): {est_cost} VALID: {output['valid']}"
        print(log)
        logging.info(log)

        train_config["progress_log"] = progress_log
        if (training_step + 1) % checkpoint_interval == 0:
            ruleset.json_save(checkpoint_folder / f"ruleset_checkpoint_timestep_{training_step + 1}.json")
            with open(save_dir / "train_config.json", "w") as f:
                json.dump(train_config, f, indent=4)

    
    final_save_path = checkpoint_folder / f"ruleset_checkpoint_final_timestep_{num_training_steps}.json"
    ruleset.json_save(final_save_path)
    train_config["final_ruleset_path"] = str(Path.resolve(final_save_path).relative_to(Path.resolve(save_dir)))
    with open(save_dir / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)


if __name__ == "__main__":
    # seed = 0
    for seed in range(3):
        # if seed==0:
        #     continue
        save_dir = f"./rulesets/contrastive_imitation/all_traj/w_time_obs_timelimit_2/{str(seed)}"
        train(save_dir, seed=seed,
            filtered_traj=False, low_timestep_threshold=2)