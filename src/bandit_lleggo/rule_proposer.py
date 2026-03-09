from __future__ import annotations
from typing import List, Dict, Any, Optional
from jinja2 import Template
import pathlib, re, json, html

from .rule_pool import Rule
from .utils_llm import call_llm

def _load_template_text(name: str) -> str:
    p = pathlib.Path(__file__).parent / "prompts" / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    if name.lower().startswith("sketch"):
        return 'Return STRICTLY JSON: {"rules":[{"expr":"feature_1 > 0.5"}]}'
    if name.lower().startswith("rich"):
        return 'Return STRICTLY JSON: {"rules":[{"expr":"feature_2 <= 1.2"}]}'
    return 'Return STRICTLY JSON: {"rules": []}'

def _load(name: str) -> Template:
    return Template(_load_template_text(name))

TPL_SKETCH = _load("sketch.jinja")
TPL_RICH   = _load("rich.jinja")


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)
_JSON_RE       = re.compile(r"(\{.*\}|\[.*\])", re.S)

def _json_fixup(text: str) -> str:
    s = text
    s = re.sub(r"'", '"', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def parse_rules_robust(llm_text: Optional[str]) -> List[Rule]:
    """
    Extract JSON from LLM string and build Rule objects.
    Accepts {"rules":[...]} or [...].
    Also unescapes HTML entities (&gt;= to >=) that some providers return.
    """
    if not llm_text:
        return []
    blob = None
    m = _CODE_BLOCK_RE.search(llm_text) or _JSON_RE.search(llm_text)
    if m:
        blob = m.group(1)
    if not blob:
        return []
    try:
        js = json.loads(_json_fixup(blob))
    except Exception:
        return []

    arr = js.get("rules", []) if isinstance(js, dict) else (js if isinstance(js, list) else [])
    out: List[Rule] = []
    for item in arr:
        if isinstance(item, dict) and "expr" in item:
            expr = html.unescape(str(item["expr"]).strip())
            notes = item.get("notes", {})
            if expr:
                out.append(Rule(expr=expr, meta=notes if isinstance(notes, dict) else {}))
        elif isinstance(item, str):
            expr = html.unescape(item.strip())
            if expr:
                out.append(Rule(expr=expr, meta={}))
    return out


def propose_rules_sketch(lineages, dataset_meta, prompt_vars, llm_cfg, return_raw: bool=False):
    prompt = TPL_SKETCH.render(summaries=lineages, meta=dataset_meta, **(prompt_vars or {}))
    out = call_llm(prompt, llm_cfg)
    return out if return_raw else parse_rules_robust(out)

def propose_rules_rich(lineages, dataset_meta, prompt_vars, llm_cfg, return_raw: bool=False):
    prompt = TPL_RICH.render(summaries=lineages, meta=dataset_meta, **(prompt_vars or {}))
    out = call_llm(prompt, llm_cfg)
    return out if return_raw else parse_rules_robust(out)