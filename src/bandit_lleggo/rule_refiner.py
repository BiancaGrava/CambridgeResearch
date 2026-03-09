from __future__ import annotations
from typing import List, Dict, Any
from jinja2 import Template
import pathlib

from .rule_pool import Rule
from .utils_llm import call_llm
from .rule_proposer import parse_rules_robust

def _load_template_text(name: str) -> str:
    p = pathlib.Path(__file__).parent / "prompts" / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return 'Return STRICTLY JSON: {"rules": []}'

def _load(name: str) -> Template:
    return Template(_load_template_text(name))

TPL_RICH = _load("rich.jinja")

def refine_rules_llm(top_rules: List[Rule], dataset_meta: Dict[str, Any],
                     prompt_vars: Dict[str, Any], llm_cfg: Dict[str, Any],
                     return_raw: bool=False):
    summaries = [{"rule": r.expr, "meta": getattr(r, "meta", {})} for r in top_rules]
    prompt = TPL_RICH.render(summaries=summaries, meta=dataset_meta, mode="refine", **(prompt_vars or {}))
    out = call_llm(prompt, llm_cfg)
    return out if return_raw else parse_rules_robust(out)