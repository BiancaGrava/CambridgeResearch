import os, json, time, re, requests

# ---------- Config via env ----------
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" or "openai"

# Common prompt (asks for STRICT JSON rules)
FEATURES = ["mean_radius","mean_texture","mean_perimeter","mean_area","worst_texture"]
PROMPT = f"""
You are given VALID_FEATURES:
{json.dumps(FEATURES)}

Return STRICTLY a single JSON object (no prose, no code fences):
{{"rules":[{{"expr":"<expr using ONLY VALID_FEATURES>", "notes":{{"reason":"..."}}}}]}}

Constraints: only >, >=, <, <=, ==, and, or, parentheses; ≤ 3 literals; parsable by pandas.eval.
"""

# ---------- Provider-specific endpoints & keys ----------
if PROVIDER == "gemini":
    API_BASE = os.getenv("GEMINI_OPENAI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    API_KEY  = os.getenv("GEMINI_API_KEY")
    URL      = API_BASE.rstrip("/") + "/chat/completions"  # per Google docs
elif PROVIDER == "openai":
    API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
    API_KEY  = os.getenv("OPENAI_API_KEY")
    URL      = API_BASE.rstrip("/") + "/chat/completions"
else:
    raise SystemExit("LLM_PROVIDER must be 'gemini' or 'openai'")

# ---------- Robust JSON parser (same as in your pipeline) ----------
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S | re.I)
_JSON_RE       = re.compile(r"(\{.*\}|\[.*\])", re.S)

def _json_fixup(text: str) -> str:
    s = re.sub(r"'", '"', text)               # single -> double quotes
    s = re.sub(r",\s*([}\]])", r"\1", s)      # trailing commas
    return s

def parse_rules_robust(text: str):
    if not text: return []
    m = _CODE_BLOCK_RE.search(text) or _JSON_RE.search(text)
    if not m: return []
    blob = _json_fixup(m.group(1))
    try:
        js = json.loads(blob)
    except Exception:
        return []
    arr = js.get("rules", []) if isinstance(js, dict) else js if isinstance(js, list) else []
    rules = []
    for item in arr:
        if isinstance(item, dict) and "expr" in item:
            rules.append({"expr": str(item["expr"]).strip(),
                          "notes": item.get("notes", {}) if isinstance(item.get("notes", {}), dict) else {}})
        elif isinstance(item, str):
            rules.append({"expr": item.strip(), "notes": {}})
    return rules

# ---------- Backoff caller ----------
def call_with_backoff(url, api_key, model, prompt, retries=6, max_tokens=200):
    if not api_key:
        raise RuntimeError(f"Missing API key for provider '{PROVIDER}'.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2, "top_p": 0.9, "max_tokens": max_tokens
    }
    for i in range(retries):
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        # Diagnostics + backoff for 429
        print(f"[HTTP {resp.status_code}] {resp.text[:500]}...")
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            sleep_s = float(ra) if ra else (2 ** i) + 0.25 * i
            print(f"[Backoff] Sleeping {sleep_s:.1f}s (attempt {i+1}/{retries})")
            time.sleep(sleep_s)
            continue
        resp.raise_for_status()
    raise RuntimeError("Exhausted retries; still receiving non-200 responses.")

if __name__ == "__main__":
    print(f"[Test] Provider={PROVIDER}  URL={URL}  MODEL={MODEL}")
    raw = call_with_backoff(URL, API_KEY, MODEL, PROMPT, retries=6, max_tokens=120)
    print("\n=== RAW (first 600 chars) ===")
    print((raw[:600] + "...") if raw else "<empty>")
    rules = parse_rules_robust(raw)
    print("\n=== PARSED RULES ===")
    print(json.dumps(rules, indent=2))
    print(f"\nParsed {len(rules)} rule(s).")