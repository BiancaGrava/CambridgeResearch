from __future__ import annotations
from typing import Dict, Any, List, Optional
from .bandit import DuelingBandit
from .rule_pool import Rule, RulePool
from .rule_proposer import propose_rules_sketch, propose_rules_rich, parse_rules_robust
from .rule_refiner import refine_rules_llm
from .rule_filter import filter_rules


def _data_seed_rules(X, feature_names: List[str], k_per_feat: int = 2) -> List[Rule]:
    """Lightweight data-based seeds when the LLM returns nothing."""
    import numpy as np
    import pandas as pd
    rules: List[Rule] = []
    if not hasattr(X, "quantile"):
        return rules
    for col in feature_names or []:
        try:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                qs = np.linspace(0.3, 0.7, k_per_feat)
                for q in qs:
                    th = float(s.quantile(q))
                    rules.append(Rule(expr=f"{col} >= {th:.6g}", meta={"seed": "quantile"}))
        except Exception:
            continue
    return rules


class BanditScheduler:
    """
    Two-phase aware scheduler:
      - mode="auto"       : original behavior (bandit + stagnation)
      - mode="rules_only" : always pick an LLM arm: new_rules then refine_rules
      - mode="gp_only"    : never call LLM (pure GP)
    """

    def __init__(self, cfg: Dict[str, Any], dataset_meta: Dict[str, Any], llm_cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.meta = dataset_meta or {}
        self.llm_cfg = llm_cfg or {}

        self.rule_pool = RulePool()
        self.bandit = DuelingBandit(self.cfg.get("arms", ["new_rules", "refine_rules", "gp_only"]))
        self.window = int(self.cfg.get("window", 5))
        self.tol = float(self.cfg.get("tol", 1e-3))
        self.history: List[float] = []
        self._first_arm_used = False
        self._sketch_max = int(self.cfg.get("prompt", {}).get("sketch", {}).get("max_tokens", 600))
        self._rich_max = int(self.cfg.get("prompt", {}).get("rich", {}).get("max_tokens", 1200))
        self.meta.setdefault("feature_names", [])
        self.mode = "auto"

        mut = self.llm_cfg.get("mut_llm", {})
        xo = self.llm_cfg.get("xo_llm", {})
        print(f"[LLM-CFG] mut_llm.api_base={mut.get('api_base')} model={mut.get('model')}")
        print(f"[LLM-CFG]  xo_llm.api_base={xo.get('api_base')} model={xo.get('model')}")


    def stagnates(self) -> bool:
        if len(self.history) < self.window + 1:
            return False
        delta = self.history[-1] - self.history[-1 - self.window]
        return delta < self.tol

    def _improvement(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return max(0.0, self.history[-1] - self.history[-2])

    def step(self, gp, X, y, task_type: str, groups: Optional[Any] = None):
        inner = int(self.cfg.get("inner_generations", 3))
        for _ in range(inner):
            gp.step()
            if hasattr(gp, "best_fitness"):
                self.history.append(gp.best_fitness())

        if self.mode == "gp_only":
            arm = "gp_only"
        elif self.mode == "rules_only":
            arm = "new_rules" if (len(self.rule_pool.all()) == 0 or not self._first_arm_used) else "refine_rules"
            self._first_arm_used = True
        else:
            force = (not self._first_arm_used) or (len(self.history) % (self.window + 1) == 0)
            if not (self.stagnates() or force):
                return
            arm = self.bandit.select()
            if not self._first_arm_used:
                arm = "new_rules"
                self._first_arm_used = True

        reward, tokens = 0.0, 0
        accepted: List[Rule] = []

        if arm == "new_rules":
            lines = gp.summarize_lineages(k=self.cfg.get("k", 8)) if hasattr(gp, "summarize_lineages") else []
            prompt_vars = {"max_literals": 3, "feature_names": self.meta.get("feature_names", [])}

            raw = propose_rules_sketch(
                lines, self.meta, prompt_vars, self.llm_cfg.get("mut_llm", {}), return_raw=True
            )
            rules = parse_rules_robust(raw)
            accepted, _ = filter_rules(rules, X, y, task_type, self.cfg.get("thresholds", {}), groups)
            tokens = self._sketch_max

            if not accepted:
                raw = propose_rules_rich(
                    lines, self.meta, prompt_vars, self.llm_cfg.get("mut_llm", {}), return_raw=True
                )
                rules = parse_rules_robust(raw)
                accepted, _ = filter_rules(rules, X, y, task_type, self.cfg.get("thresholds", {}), groups)
                tokens = self._rich_max

            if not accepted:
                seeds = _data_seed_rules(X, self.meta.get("feature_names", []), k_per_feat=2)
                accepted, _ = filter_rules(seeds, X, y, task_type, self.cfg.get("thresholds", {}), groups)

            self.rule_pool.add_many(accepted)
            gp.update_rule_pool(self.rule_pool.all())
            if accepted and hasattr(gp, "ensure_seeded_population"):
                gp.ensure_seeded_population()
            reward = self._improvement()

        elif arm == "refine_rules":
            top = self.rule_pool.top_k(self.cfg.get("m", 10))
            if top:
                prompt_vars = {"max_literals": 4, "feature_names": self.meta.get("feature_names", [])}
                raw = refine_rules_llm(top, self.meta, prompt_vars, self.llm_cfg.get("xo_llm", {}), return_raw=True)
                rules = parse_rules_robust(raw)
                accepted, _ = filter_rules(rules, X, y, task_type, self.cfg.get("thresholds", {}), groups)
                self.rule_pool.add_many(accepted)
                gp.update_rule_pool(self.rule_pool.all())
                reward = self._improvement()
                tokens = self._rich_max
            else:
                reward, tokens = self._improvement(), 0

        else:
            reward, tokens = self._improvement(), 0

        self.bandit.update(arm, reward=reward, tokens=tokens)
        print(f"[SCHED] arm={arm} pool={len(self.rule_pool.all())} reward={reward:.5f} tokens+={tokens}")