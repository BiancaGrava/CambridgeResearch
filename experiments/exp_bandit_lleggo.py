from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from lleggo_utils.data_utils import get_data
from bandit_lleggo.scheduler import BanditScheduler
from bandit_lleggo_core.gp_wrapper import GPWrapper


def select_top_k_rules(rules, k: int = 16):
    """
    Pick top-K rules by a simple composite score; robust to missing fields.
    Each rule is a dict like {"expr": "...", "meta": {...}} or a dataclass with .expr/.meta.
    Score = util + 0.2*support - 0.5*ece  (higher is better).
    """
    scored = []
    for r in rules or []:
        if isinstance(r, dict):
            meta = r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}
        else:
            meta = getattr(r, "meta", {})
            if not isinstance(meta, dict):
                meta = {}
        util = float(meta.get("util", meta.get("ig", 0.0)))
        ece = float(meta.get("ece", 0.0))
        sup = float(meta.get("support", meta.get("sup", 0.0)))
        score = util + 0.2 * sup - 0.5 * ece
        scored.append((score, {"expr": str(r.get("expr") if isinstance(r, dict) else getattr(r, "expr", "")), "meta": meta}))
    scored.sort(key=lambda z: z[0], reverse=True)
    return [r for _, r in scored[:max(1, int(k))]]


@hydra.main(config_path="configs", config_name="bandit_lleggo", version_base=None)
def main(cfg):
    """
    Phase switch:
      - alg.phase=rules  → collect rules, freeze S* (no tree evolution)
      - alg.phase=tree   → load S*, run rule-aware tree GP (no LLM)
      - alg.phase=joint  → (default) original joint evolution
    """
    # 1) Load data + meta
    dataset_name = cfg.alg.dataset.name
    task_type = cfg.alg.dataset.task
    X, Y, meta = get_data(dataset_name, dataset_details={})

    gp = GPWrapper(cfg.alg)
    if hasattr(gp, "set_feature_names"):
        gp.set_feature_names(meta.get("feature_names", []))

    sched = BanditScheduler(
        cfg=OmegaConf.to_container(cfg.alg.scheduler, resolve=True),
        dataset_meta={"feature_names": meta.get("feature_names", [])},
        llm_cfg=OmegaConf.to_container(cfg.alg.endpoint, resolve=True),
    )

    exp_name = cfg.alg.exp_name
    sstar_path = Path("results") / f"{exp_name}_Sstar.json"
    sstar_path.parent.mkdir(parents=True, exist_ok=True)

    phase = getattr(cfg.alg, "phase", "joint")

    if phase == "rules":
        original_inner = sched.cfg.get("inner_generations", 3)
        sched.cfg["inner_generations"] = 0
        sched.mode = "rules_only"

        rounds = int(getattr(cfg.alg.training, "outer_rounds_rules", 4))
        for _ in range(rounds):
            sched.step(gp, X, Y, task_type=task_type, groups=meta.get("groups"))

        top_k = int(getattr(cfg.alg.rules, "top_k", 16))
        Sstar = select_top_k_rules(sched.rule_pool.all(), k=top_k)
        with open(sstar_path, "w") as f:
            json.dump(Sstar, f, indent=2)
        print(f"[PHASE-1] Saved S* ({len(Sstar)} rules) → {sstar_path}")

        sched.cfg["inner_generations"] = original_inner
        return

    if phase == "tree":
        if not sstar_path.exists():
            raise FileNotFoundError(
                f"Missing S*: {sstar_path}. Run phase=rules first to create the frozen rule set."
            )
        Sstar = json.load(open(sstar_path))
        if hasattr(gp, "lock_rule_pool"):
            gp.lock_rule_pool(Sstar)
        else:
            gp.update_rule_pool(Sstar)
        if hasattr(gp, "ensure_seeded_population"):
            gp.ensure_seeded_population()

        sched.mode = "gp_only"
        rounds = int(getattr(cfg.alg.training, "outer_rounds_tree", 5))
        for _ in range(rounds):
            sched.step(gp, X, Y, task_type=task_type, groups=meta.get("groups"))

        metrics = gp.compute_metrics(X, Y)
        metrics.update({"dataset": dataset_name, "task": task_type, "exp_name": exp_name, "llm_calls": 0})
        gp.save_metrics(exp_name, metrics)
        print(f"[PHASE-2] {dataset_name}: {metrics}")
        return

    rounds = int(getattr(cfg.alg.training, "outer_rounds", 5))
    for _ in range(rounds):
        sched.step(gp, X, Y, task_type=task_type, groups=meta.get("groups"))
    metrics = gp.compute_metrics(X, Y)
    metrics.update({"dataset": dataset_name, "task": task_type, "exp_name": exp_name})
    gp.save_metrics(exp_name, metrics)


if __name__ == "__main__":
    main()