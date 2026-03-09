class RuleAwareCrossover:
    def __init__(self, base_crossover, rule_pool, bias_prob=0.3):
        self.base = base_crossover
        self.rule_pool = rule_pool
        self.bias_prob = bias_prob

    def crossover(self, parent_a, parent_b):
        child = self.base.crossover(parent_a, parent_b)
        return child