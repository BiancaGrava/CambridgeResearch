import os, json, requests, time

def call_llm(prompt: str, llm_cfg: dict) -> str:
    """
    OpenAI Chat Completions call with 429 backoff + dummy fallback.
    """
    api_base = (llm_cfg or {}).get("api_base", "https://api.openai.com/v1")
    model    = (llm_cfg or {}).get("model", "gpt-4o")
    api_key  = (llm_cfg or {}).get("api_key") or os.getenv("OPENAI_API_KEY")
    temperature = (llm_cfg or {}).get("temperature", 0.2)
    top_p = (llm_cfg or {}).get("top_p", 0.9)
    max_tokens = (llm_cfg or {}).get("max_tokens", 800)

    def _dummy():
        return json.dumps({"rules":[
            {"expr":"mean_radius >= 14.0","notes":{"reason":"size prior"}},
            {"expr":"worst_texture >= 20.0","notes":{"reason":"texture prior"}}
        ]})

    if not api_key:
        print("[LLM] OPENAI_API_KEY missing; using dummy rules.")
        return _dummy()

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    for i in range(5):
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        txt = r.text[:400]
        if r.status_code == 429:
            wait = (2 ** i) + 0.25 * i
            print(f"[OpenAI 429] {txt}  (backoff {wait:.1f}s)")
            time.sleep(wait); continue
        if r.status_code == 400 and "Unsupported parameter: 'max_tokens'" in txt:
            print("[OpenAI] This model does not accept max_tokens via Chat Completions. "
                  "Set model=gpt-4o or switch to Responses API (max_output_tokens). Using dummy rules for now.")
            return _dummy()
        print(f"[OpenAI {r.status_code}] {txt}")
        return _dummy()
    return _dummy()