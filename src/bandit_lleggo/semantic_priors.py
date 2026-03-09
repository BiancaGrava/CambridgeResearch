from typing import List, Dict, Any
from .rule_pool import Rule
from .utils_llm import call_llm

def score_with_semantic_priors(rules: List[Rule], dataset_meta: Dict[str, Any],
                               llm_cfg: Dict[str, Any]) -> List[Rule]:
    """
    Optional: ask LLM to rank rules by semantic plausibility (few tokens).
    This is called rarely (bandit) to avoid cost; store `semantic_score` in meta.
    """
    if not rules: return rules
    prompt = _make_scoring_prompt([r.expr for r in rules], dataset_meta)
    resp = call_llm(prompt, llm_cfg)
    scores = _parse_scores(resp)
    for r in rules:
        r.meta["semantic_score"] = float(scores.get(r.expr, 0.0))
    return rules

def _make_scoring_prompt(rule_exprs, meta):
    return (
        "Rank these rule predicates for plausibility given feature semantics. "
        "Return JSON {\"scores\": [{\"expr\": ..., \"score\": ...} ...]}.\n"
        f"Rules: {rule_exprs}\nMeta: {meta}\n"
    )

def _parse_scores(text):
    import json, re
    blob = re.search(r"\{.*\}", text, re.S).group(0)
    js = json.loads(blob)
    return {x["expr"]: x["score"] for x in js.get("scores", [])}