from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Rule:
    expr: str
    meta: Dict[str, Any] = field(default_factory=dict)

class RulePool:
    def __init__(self):
        self._rules: List[Rule] = []

    def add_many(self, rules: List[Rule]):
        seen = {r.expr for r in self._rules}
        for r in rules:
            if r.expr not in seen:
                self._rules.append(r); seen.add(r.expr)

    def top_k(self, k=10) -> List[Rule]:
        return sorted(self._rules, key=lambda r: r.meta.get("util", 0.0), reverse=True)[:k]

    def all(self) -> List[Rule]:
        return list(self._rules)