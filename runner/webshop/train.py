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

def train(
    seed=0,
    num_training_steps = 500,
    num_rule_per_step = 10,
    learning_step_max_retry = 2,
    checkpoint_interval = 25,
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
    }
    logging.info(f"TRAIN CONFIG:\n{json.dumps(train_config, indent=4)}\n")

    checkpoint_folder = log_folder / "checkpoint"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    progress_log_path = log_folder / "progress_log.json"

    # llm = FakeListLLM(responses=[
    #     f"related_rule_number: 1\nfound_rule: My found rule{i}\nrelated_rule_content: rule1\nsame_intention\nfound_rule_better\n"
    #     + "\nNO_EXISTING_RULE" if i==0 else "" for i in range(20)
    # ])
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))
    # llm = ChatOpenAI(model="gpt-4-0613", temperature=0.6, max_tokens=1024, openai_api_key=os.environ.get("OPENAI_APIKEY", None))

    with open("./data/human_traj_sa_by_state.json", "r") as f:
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

        existing_rules = select_learning_rules(ruleset, selected_state_name, num_rule_per_step, training_step)
        existing_rules_str = [r.content for r in existing_rules]
        
        prompt = contrastive_step.test_format_input(inputs={"records": records, "existing_rules": existing_rules_str})
        logging.info(f"STEP: {training_step}\n")
        
        token_count = 0
        for __ in range(learning_step_max_retry):
            try:
                with get_openai_callback() as cb:
                    output = contrastive_step(inputs={"records": records, "existing_rules": existing_rules_str})
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
        if not output["valid"]:
            logging.info(f"INVALID OUTPUT")
        else:
            selected_rule = None
            if output["related_rule_number"] is not None and output["related_rule_number"] < len(existing_rules_str):
                selected_rule = existing_rules[output["related_rule_number"]]
            ruleset.update(selected_state_name, output, selected_rule)
        
        progress_log["token_counts"].append(token_count)
        progress_log["valid"].append(output["valid"])
        elapsed_time_s = (datetime.now() - start_time).total_seconds()
        progress_log["time_elapsed_s"].append(elapsed_time_s)
        for state in ruleset.states:
            progress_log["rule_type_counts"][state].append(len(ruleset.rules[state]))
        est_cost = sum(progress_log["token_counts"])*0.0017/1000
        print(f"ELAPSED_TIME_S: {elapsed_time_s} STEP: {training_step+1}/{num_training_steps} TOKENS: {token_count} EST_TOTAL_COST($): {est_cost} VALID: {output['valid']}")
    
        if (training_step + 1) % checkpoint_interval == 0:
            ruleset.json_save(checkpoint_folder / f"ruleset_checkpoint_timestep_{training_step + 1}.json")
            with open(progress_log_path, "w") as f:
                json.dump(progress_log, f, indent=4)

    
    final_save_path = checkpoint_folder / f"ruleset_checkpoint_final_timestep_{num_training_steps}.json"
    ruleset.json_save(final_save_path)
    train_config["final_ruleset_path"] = str(Path.resolve(final_save_path))
    with open(log_folder / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=4)


if __name__ == "__main__":
    train(seed=2)