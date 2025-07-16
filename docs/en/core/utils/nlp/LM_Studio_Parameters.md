Below is a **concise “LLM-setup cheat-sheet”** (Markdown) meant to sit **next to** your full module documentation.
It keeps only the information a newcomer must absorb *before* running the pipeline: **what LLM is used, why LM Studio is chosen, and the exact knobs that have to be configured**.
Feel free to drop it into `docs/llm_quickstart.md` and cross-link from the larger manual.

---

## 🚀 Quick-start: configuring the LLM that powers `text_transformer`

> **TL;DR** – we run a *local* Quantized GGUF model through **LM Studio’s WebSocket API**.
> That gives us GUI convenience *without* giving up llama.cpp speed.
> Only five things must be in sync: **model file · LM Studio preset · JSON config · prompt template · generation-params**.

---

### 1 · Pick & download the model

| We recommend                                         | Why                                                                                 |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `gemma-2-9b-it-russian-function-calling.Q5_K_S.gguf` | ✅ Russian instruction-tuned ✅ 4 GB VRAM (Q5) ✅ precise follow for re-phrasing tasks |

Place the file under `~/.lmstudio/models/` (LM Studio will detect it).

---

### 2 · Create an **LM Studio preset**

1. **Context → System Prompt**

   ```text
   You are a helpful assistant.
   ```
2. **Inference**

   * Temperature `0.30`
   * Top-P `0.95`
   * Top-K `50`
   * Repeat Penalty `1.10`
   * Min-P `0.05`
   * **Limit Response Length** : off
3. **Save as** `anonymization_stable`.

> *Why these numbers?* Low temperature ensures deterministic rewrites; Top-K 50 is a safe widening; other values are defaults that do not hurt the task.

---

### 3 · Wire the module to LM Studio

```jsonc
// t_3LLM2.json  (excerpt)
"llm": {
  "server_ip": "127.0.0.1",
  "server_port": 1234,        // set in LM Studio ▸ API ▸ Port
  "model_name": "gemma-2-9b-it-russian-function-calling",
  "ttl": 86400,
  "max_workers": 1,
  "debug_logging": true,

  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 50,
  "max_tokens": 512,
  "stop_sequences": ["</text>", "\n\n"]
}
```

*The JSON values **override** the GUI; you can tune them from CI without touching LM Studio.*

---

### 4 · Prompt template used by the code

```txt
<start_of_turn>user
Задача: анонимизируй текст опыта работы,
убери личные данные, названия компаний и ВУЗов.
Сохрани смысл и навыки.

<text>
{text}
</text>
<end_of_turn>
<start_of_turn>model
```

`{text}` is injected per-row; `stop_sequences` ensure generation stops right after the answer.

---

### 5 · Generation parameters applied at runtime

```python
generation_params = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 512,
    "stop": ["</text>", "\n\n"],
    "stream": False          # single-frame answer
}
```

The helper `_call_llm()` already merges these with the WebSocket call.

---

### 6 · Smoke test

```bash
# run five rows with clean cache
python t_3LLM2_Experience.py --max-records 5 --no-cache --debug-llm
```

Output should **not** contain lines like

> “Пожалуйста, предоставьте мне исходный текст…”

---

## What if I skip LM Studio?

| Direct llama.cpp (`llama-cpp-python`) | LM Studio WebSocket                         |
| ------------------------------------- | ------------------------------------------- |
| + 2–3 ms faster per request (no JSON) | Built-in GUI, model browser, preset manager |
| Manual build / install / reload       | One-click model switch                      |
| No extra RAM footprint (–60 MB)       | Zero code changes to swap quantisation      |

> **Rule of thumb** – stick to LM Studio unless you must serve **>10 k short requests per minute** or you need custom kv-cache tricks.

---

**That’s it!** With these five aligned layers – model, preset, JSON config, prompt, and generation params – the anonymization module runs reproducibly on any workstation.
