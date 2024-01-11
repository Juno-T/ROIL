from copy import deepcopy
from uuid import uuid4
from typing import List, Literal, Tuple
import json
import numpy as np
from pathlib import Path


class Rule:
    def __init__(self, rule_type: str, rule_id: Tuple[str, int], content: str, init_score: int = 1, selected_count: int = 1):
        self.rule_type = rule_type
        self.rule_id = rule_id
        self.content = content
        self.score = init_score
        self.selected_count = selected_count
    
    def calc_ucb(self, timestep: int):
        return self.score / (self.selected_count+1e6) + np.sqrt(2 * np.log(timestep) / (self.selected_count+1e6))

    def to_dict(self):
        return {
            "rule_type": self.rule_type,
            "rule_id": self.rule_id,
            "content": self.content,
            "score": self.score,
            "selected_count": self.selected_count,
        }
    
    @classmethod
    def from_dict(cls, rule_dict: dict):
        return cls(
            rule_type=rule_dict["rule_type"],
            rule_id=rule_dict["rule_id"],
            content=rule_dict["content"],
            init_score=rule_dict["score"],
            selected_count=rule_dict["selected_count"],
        )

class RuleSet:
    def __init__(self, states: List[str]):
        self.states = states
        self.rules = {state: {} for state in states}
        self.best_rules = {state: {} for state in states}

    def add_rules(self, state: str, rules: List[Rule]):
        rules = deepcopy(rules)
        if state not in self.states:
            raise ValueError(f"State {state} not in states")
        for rule in rules:
            if rule.rule_type not in self.rules[state]:
                self.rules[state][rule.rule_type] = []
            self.rules[state][rule.rule_type].append(rule)
            if rule.rule_type not in self.best_rules[state]:
                self.best_rules[state][rule.rule_type] = rule
            else:
                if rule.score > self.best_rules[state][rule.rule_type].score:
                    self.best_rules[state][rule.rule_type] = rule
    
    def update(
        self,
        state: str,
        selected_rule: Rule,
        new_rule: str,
        update_method: Literal["found_rule_better", "selected_rule_better", "similar_intention", "different_intention"]
    ):
        if state not in self.states:
            raise ValueError(f"State {state} not in states")
        # if not output["valid"]:
        #     return
        # if output["no_existing_rule"]:
        #     rule_type = str(uuid4())
        #     self.rules[state][rule_type] = [Rule(rule_type, 0, output["found_rule"])]
        #     self.best_rules[state][rule_type] = self.rules[state][rule_type][0]
        #     return
        if selected_rule is None:
            rule_type = str(uuid4())
            self.rules[state][rule_type] = [Rule(rule_type, 0, new_rule)]
            self.best_rules[state][rule_type] = self.rules[state][rule_type][0]
            return
        rule_type = selected_rule.rule_type
        rule_id = selected_rule.rule_id
        if update_method == "found_rule_better":
            self.rules[state][rule_type].append(Rule(rule_type, len(self.rules[state][rule_type]), new_rule, selected_rule.score + 1))
            self.best_rules[state][rule_type] = self.rules[state][rule_type][-1]
            return
        if update_method == "selected_rule_better":
            self.rules[state][rule_type][rule_id].score += 1
            self.rules[state][rule_type].append(Rule(rule_type, len(self.rules[state][rule_type]), new_rule))
            self.best_rules[state][rule_type] = self.rules[state][rule_type][rule_id]
            return
        if update_method == "similar_intention":
            self.rules[state][rule_type].append(Rule(rule_type, len(self.rules[state][rule_type]), new_rule, selected_rule.score + 1))
            self.best_rules[state][rule_type] = self.rules[state][rule_type][-1]
            return
        if update_method == "different_intention":
            rule_type = str(uuid4())
            self.rules[state][rule_type] = [Rule(rule_type, 0, new_rule)]
            self.best_rules[state][rule_type] = self.rules[state][rule_type][-1]
            return

    def old_update(self, state: str, output: str, selected_rule: Rule = None):
        if state not in self.states:
            raise ValueError(f"State {state} not in states")
        if not output["valid"]:
            return
        if output["no_existing_rule"]:
            rule_type = str(uuid4())
            self.rules[state][rule_type] = [Rule(rule_type, 0, output["found_rule"])]
            self.best_rules[state][rule_type] = self.rules[state][rule_type][0]
            return
        rule_type = selected_rule.rule_type
        rule_id = selected_rule.rule_id
        self.rules[state][rule_type][rule_id].selected_count += 1
        if output["related_rule_classification"] == "same_intention":
            if output["found_rule_better"]:
                self.rules[state][rule_type].append(Rule(rule_type, len(self.rules[state][rule_type]), output["found_rule"], selected_rule.score + 1))
                self.best_rules[state][rule_type] = self.rules[state][rule_type][-1]
            else:
                self.rules[state][rule_type][rule_id].score += 1
                self.rules[state][rule_type].append(Rule(rule_type, len(self.rules[state][rule_type]), output["found_rule"]))
                self.best_rules[state][rule_type] = self.rules[state][rule_type][rule_id]
            return
        if output["related_rule_classification"] == "similar_intention":
            self.rules[state][rule_type].append(Rule(rule_type, len(self.rules[state][rule_type]), output["updated_rule"], selected_rule.score + 1))
            self.best_rules[state][rule_type] = self.rules[state][rule_type][-1]
            return
        if output["related_rule_classification"] == "different_intention":
            rule_type = str(uuid4())
            self.rules[state][rule_type] = [Rule(rule_type, 0, output["found_rule"])]
            self.best_rules[state][rule_type] = self.rules[state][rule_type][-1]
            return

    def get_best_rules(self, state: str, num_rules: int = 10) -> List[Rule]:
        best_rules = self.best_rules[state].copy()
        sorted_rules = sorted(best_rules.values(), key=lambda x: x.score, reverse=True)
        return sorted_rules[:num_rules]
    
    def to_dict(self):
        return {
            "states": self.states,
            "rules": {
                state: {
                    rule_type: 
                        [rule.to_dict() for rule in rules] for rule_type, rules in state_rules.items()
                } for state, state_rules in self.rules.items()},
            "best_rules": {
                state: {
                    rule_type: rule.to_dict() for rule_type, rule in state_rules.items()
                } for state, state_rules in self.best_rules.items()},
        }

    @classmethod
    def from_dict(cls, ruleset_dict: dict):
        ruleset = cls(ruleset_dict["states"])
        for state, state_rules in ruleset_dict["rules"].items():
            for rule_type, rules in state_rules.items():
                ruleset.rules[state][rule_type] = [Rule.from_dict(rule) for rule in rules]
        for state, state_rules in ruleset_dict["best_rules"].items():
            for rule_type, rule in state_rules.items():
                ruleset.best_rules[state][rule_type] = Rule.from_dict(rule)
                best_rule_id = ruleset.best_rules[state][rule_type].rule_id
                assert ruleset.best_rules[state][rule_type].content == ruleset.rules[state][rule_type][best_rule_id].content
        return ruleset

    def json_save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def json_load(cls, path: str):
        with open(str(path), "r") as f:
            ruleset_dict = json.load(f)
        return cls.from_dict(ruleset_dict)


def select_learning_rules(ruleset: RuleSet, state: str, num_rules: int = 10, timestep: int = 1):
    best_rules = ruleset.best_rules[state].copy()
    sorted_rules = sorted(best_rules.values(), key=lambda x: x.calc_ucb(timestep), reverse=True)
    return sorted_rules[:num_rules]

if __name__ == "__main__":
    ruleset = RuleSet(["stateA", "stateB"])
    ruleset.update("stateA", {
        "valid": True,
        "no_existing_rule": True,
        "found_rule": "rule1",
    })
    selected_rule = select_learning_rules(ruleset, "stateA")
    ruleset.update("stateA", {
        "valid": True,
        "no_existing_rule": False,
        "related_rule_classification": "same_intention",
        "found_rule_better": True,
        "found_rule": "rule2",
    }, selected_rule = selected_rule[0])
    ruleset.json_save("test.json")
    ruleset = RuleSet.json_load("test.json")
    print(ruleset.to_dict())