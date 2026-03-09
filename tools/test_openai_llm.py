# tools/test_openai_llm.py
import os, json, time, requests, re

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")  # OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

FEATURES = ["mean_radius","mean_texture","mean_perimeter","mean_area","worst_texture"]
PROMPT = f"""
You are given VALID_FEATURES:
{json.dumps(FEATURES)}

Return STRICTLY a single JSON object (no prose, no code fences):
{{"rules":[{{"expr":"<expr using ONLY VALID_FEATURES>", "notes":{{"reason":"..."}}}}]}}

Constraints: only >, >=, <, <=, ==, and, or, parentheses; ≤ 3 literals; parsable by pandas.eval.
"""

# robust JSON parser (same as pipeline)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)
_JSON_RE       = re.compile(r"(\{.*\}|\[.*\])", re.S)
def _json_fixup(s: str) -> str:
    s = re.sub(r"'", '"', s)                 # single → double
    s = re.sub(r",\s*([}\]])", r"\1", s)     # trailing comma
    return s
def parse_rules_robust(text: str):
    if not text: return []
    m = _CODE_BLOCK_RE.search(text) or _JSON_RE.search(text)
    if not m: return []
    try:
        js = json.loads(_json_fixup(m.group(1)))
    except Exception:
        return []
    arr = js.get("rules", []) if isinstance(js, dict) else js if isinstance(js, list) else []
    out = []
    for x in arr:
        if isinstance(x, dict) and "expr" in x:
            out.append({"expr": str(x["expr"]).strip(), "notes": x.get("notes", {}) if isinstance(x.get("notes", {}), dict) else {}})
        elif isinstance(x, str):
            out.append({"expr": x.strip(), "notes": {}})
    return out

def call_openai(prompt: str, retries=6, max_tokens=200):
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY first.")
    url = OPENAI_API_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    for i in range(retries):
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        print(f"[OpenAI HTTP {r.status_code}] {r.text[:400]}...")
        if r.status_code == 429:
            wait = (2 ** i) + 0.25 * i
            print(f"[Backoff] Sleeping {wait:.1f}s...")
            time.sleep(wait); continue
        r.raise_for_status()
    raise RuntimeError("Exhausted retries for OpenAI.")

if __name__ == "__main__":
    print(f"[Test] Provider=openai  URL={OPENAI_API_BASE}/chat/completions  MODEL={OPENAI_MODEL}")
    raw = call_openai(PROMPT)
    print("\n=== RAW (first 600 chars) ===")
    print((raw[:600] + "...") if raw else "<empty>")
    rules = parse_rules_robust(raw)
    print("\n=== PARSED RULES ===")
    print(json.dumps(rules, indent=2))
    print(f"\nParsed {len(rules)} rule(s).")