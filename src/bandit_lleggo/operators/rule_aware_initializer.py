from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import random

class RuleAwareInitializer:
    """
    Wraps a base initializer and seeds a fraction of the population with trees
    constructed from accepted rules (Pittsburgh-style rule-set individuals).

    Contract/assumptions:
      - base_initializer (if provided) supports:
            initialize(pop_size, algo) -> List[individual]
        OR   initialize(pop_size) -> List[individual]
        OR   random_individual() -> individual  (used repeatedly)
      - algo (if provided) exposes at least one of:
            make_tree_from_rule(expr) -> individual
            create_tree_from_rule(expr) -> individual
            tree_from_rule(expr) -> individual
        OR   algo.tree_factory.from_rule(expr) -> individual
      - If none of the above exist, we fall back to base init and mark seeds as None.
    """

    def __init__(
        self,
        base_initializer: Optional[Any] = None,
        seeded_ratio: float = 0.25,
        rng_seed: int = 42,
        max_rule_literals: int = 4,
    ):
        self.base = base_initializer
        self.seeded_ratio = float(max(0.0, min(1.0, seeded_ratio)))
        self.rng = random.Random(rng_seed)
        self.max_rule_literals = int(max_rule_literals)
        self.rule_pool: List[Dict[str, Any]] = []
        self.feature_names: List[str] = []

        self._from_rule_fns: List[Callable[[str], Any]] = []


    def set_feature_names(self, names: List[str]):
        self.feature_names = list(names or [])

    def set_rule_pool(self, rules: List[Dict[str, Any]]):
        uniq, seen = [], set()
        for r in rules or []:
            expr = str(r.get("expr", "")).strip()
            if expr and expr not in seen:
                uniq.append({"expr": expr, "meta": r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}})
                seen.add(expr)
        self.rule_pool = uniq

    def initialize(self, pop_size: int, algo: Optional[Any] = None):
        base_pop = self._call_base_initialize(pop_size, algo)

        if not self.rule_pool or self.seeded_ratio <= 0.0:
            return base_pop

        n_seed = max(1, int(round(pop_size * self.seeded_ratio)))
        seeds = self._build_seeds(n_seed, algo)

        if seeds:
            if base_pop and len(base_pop) >= len(seeds):
                base_pop[:len(seeds)] = seeds
            else:
                base_pop = (seeds + base_pop)[:pop_size]

        print(f"[INIT] Pittsburgh-style seeding: seeded={len(seeds)}/{pop_size} | pool={len(self.rule_pool)}")
        return base_pop


    def _call_base_initialize(self, pop_size: int, algo: Optional[Any]):
        if self.base is not None:
            fn = getattr(self.base, "initialize", None)
            if callable(fn):
                try:
                    return fn(pop_size, algo)
                except TypeError:
                    try:
                        return fn(pop_size)
                    except Exception:
                        pass
            rnd = getattr(self.base, "random_individual", None)
            if callable(rnd):
                return [rnd() for _ in range(pop_size)]

        if algo is not None:
            fn = getattr(algo, "initialize_population", None)
            if callable(fn):
                try:
                    return fn(pop_size)
                except Exception:
                    pass
            rnd = getattr(algo, "random_individual", None)
            if callable(rnd):
                return [rnd() for _ in range(pop_size)]

        return [None for _ in range(pop_size)]

    def _discover_from_rule_fns(self, algo: Optional[Any]):
        self._from_rule_fns = []
        if algo is None:
            return
        for name in ("make_tree_from_rule", "create_tree_from_rule", "tree_from_rule"):
            fn = getattr(algo, name, None)
            if callable(fn):
                self._from_rule_fns.append(fn)
        tf = getattr(algo, "tree_factory", None)
        if tf is not None:
            fn = getattr(tf, "from_rule", None)
            if callable(fn):
                self._from_rule_fns.append(fn)

    def _from_rule(self, expr: str, algo: Optional[Any]):
        if not self._from_rule_fns:
            self._discover_from_rule_fns(algo)
        for fn in self._from_rule_fns:
            try:
                ind = fn(expr)
                if ind is not None:
                    return ind
            except Exception:
                continue
        return None

    def _pick_rule(self) -> Optional[str]:
        if not self.rule_pool:
            return None
        r = self.rng.choice(self.rule_pool)
        expr = str(r.get("expr", "")).strip()
        if not expr:
            return None
        parts = [p.strip() for p in expr.replace("(", " ").replace(")", " ").split(" and ")]
        if len(parts) > self.max_rule_literals and self.max_rule_literals > 0:
            parts = parts[: self.max_rule_literals]
        return " and ".join(parts)

    def _build_seeds(self, n_seed: int, algo: Optional[Any]):
        seeds = []
        for _ in range(n_seed):
            expr = self._pick_rule()
            if not expr:
                continue
            ind = self._from_rule(expr, algo)
            if ind is None:
                break
            seeds.append(ind)
        return seeds
