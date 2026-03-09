# tools/test_gemini_llm.py
import os
import json
import requests
import re

# ---- 1) CONFIG ----
API_BASE = os.getenv("GEMINI_OPENAI_API_BASE",
                     "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY  = os.getenv("GEMINI_API_KEY")  # <-- set this in your env

# Minimal prompt (whitelists features and requests strictly JSON)
FEATURES = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "worst_texture"]
PROMPT = f"""
You are given the exact list of valid feature names. Use ONLY these in predicates.

VALID_FEATURES:
{json.dumps(FEATURES)}

Return STRICTLY a single JSON object (no prose, no code fences):
{{"rules":[{{"expr":"<expr using ONLY VALID_FEATURES>", "notes":{{"reason":"..."}}}}]}}

Constraints:
- Allowed operators: >, >=, <, <=, ==, and, or, parentheses.
- ≤ 3 literals per rule.
- Keep expressions parsable by pandas.eval without quoting column names.
"""

# ---- 2) CALL GEMINI (OpenAI-compatible endpoint) ----
def call_gemini(prompt: str) -> str:
    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY env var first.")
    url = API_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 400,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ---- 3) ROBUST PARSER (same logic we use in bandit_lleggo.rule_proposer) ----
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)
_JSON_RE       = re.compile(r"(\{.*\}|\[.*\])", re.S)

def _json_fixup(text: str) -> str:
    s = text
    s = re.sub(r"'", '"', s)                          # single → double quotes
    s = re.sub(r",\s*([}\]])", r"\1", s)              # trailing commas
    return s

def parse_rules_robust(text: str):
    if not text:
        return []

    blob = None
    m = _CODE_BLOCK_RE.search(text)
    if m:
        blob = m.group(1)
    else:
        m = _JSON_RE.search(text)
        if m:
            blob = m.group(1)
    if not blob:
        return []

    try:
        js = json.loads(_json_fixup(blob))
    except Exception:
        return []

    if isinstance(js, dict):
        arr = js.get("rules", [])
    elif isinstance(js, list):
        arr = js
    else:
        arr = []

    rules = []
    for item in arr:
        if isinstance(item, dict) and "expr" in item:
            expr = str(item["expr"]).strip()
            notes = item.get("notes", {})
            if expr:
                rules.append({"expr": expr, "notes": notes if isinstance(notes, dict) else {}})
        elif isinstance(item, str):
            expr = item.strip()
            if expr:
                rules.append({"expr": expr, "notes": {}})
    return rules

# ---- 4) RUN ----
if __name__ == "__main__":
    print("[Test] Calling Gemini …")
    raw = call_gemini(PROMPT)
    print("\n=== RAW (first 500 chars) ===")
    print((raw[:500] + "...") if raw else "<empty>")

    rules = parse_rules_robust(raw)
    print("\n=== PARSED RULES ===")
    print(json.dumps(rules, indent=2))
    print(f"\nParsed {len(rules)} rule(s).")