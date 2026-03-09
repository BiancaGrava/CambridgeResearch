"""
Adapter that biases LLEGO's mutation to use accepted rules as candidate terminals.
Import and wrap the original mutation operator, injecting the RulePool when choosing splits.
"""
from typing import List
from ..rule_pool import Rule, RulePool

class RuleAwareMutation:
    def __init__(self, base_mutation, rule_pool: RulePool, p_rule=0.6):
        self.base = base_mutation
        self.rule_pool = rule_pool
        self.p_rule = p_rule

    def mutate(self, tree):

        rules = self.rule_pool.all()
        has_rules = bool(rules)

        if (not has_rules) or (self.rng.random() > self.p_rule):
            return self.base.mutate(tree)

        rule = self.rng.choice(rules)
        predicate = getattr(rule, "expr", str(rule))

        if hasattr(tree, "replace_random_split"):
            tree.replace_random_split(predicate)
            return tree

        self._replace_random_split_fallback(tree, predicate)
        return tree
