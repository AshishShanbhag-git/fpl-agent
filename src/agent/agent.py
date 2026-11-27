"""The FPL agent: an LLM that answers by calling tools (a "tool-calling loop").

How it works, per user question:
  1. Send the conversation + the tool descriptions to the LLM.
  2. If the LLM asks for tools, run them (FplTools) and send the results back. Repeat (at most MAX_STEPS times).
  3. When the LLM answers in plain text, that's the reply.
The LLM decides *which* data to fetch and explains it; the numbers come from the tools (xPts model, optimiser).

`llm(messages, tools) -> assistant message dict` is injected: `databricks_llm(endpoint)` calls a Databricks
Foundation Model endpoint; tests use a scripted fake.
"""

import json
from datetime import datetime, timezone

from tools import TOOL_SPECS

DEFAULT_ENDPOINT = "databricks-gpt-oss-120b"
MAX_STEPS = 8

SYSTEM_PROMPT = """You are an FPL (Fantasy Premier League) assistant for one manager, who does not watch matches.
Today is {today}. The next gameweek is GW{next_gw} (deadline {deadline} UTC).

Rules:
- Every number you state (xPts, prices, points, fixtures, chances of playing) must come from a tool result in this
  conversation. Never guess or use your own football knowledge for facts; it is out of date.
- Transfer, chip, captain and lineup advice must come from get_recommendation or run_optimizer. Do not invent
  your own picks. For what-ifs ("what if I keep X?"), call run_optimizer with constraints and compare horizon_score
  with the unconstrained plan.
- Only mention an injury, suspension or doubt if a tool reports it (status, news, chance_of_playing).
- xPts = expected FPL points from our model (an average outcome, so even top players are ~5-7 per gameweek).
  Differences under ~0.3 xPts are effectively a coin flip: say so instead of overstating them.
- Copy players, positions, teams, prices, fixtures and xPts exactly as tool results give them. Do not compute your
  own per-player gains or describe fixtures, opponents or form that no tool returned. If a detail is missing, say
  you don't have it or call another tool.
- Explain the "why" briefly with those numbers (xPts, fixtures, price, availability). Keep answers short: bullet
  points, no big tables unless asked.
- If a tool returns an error, explain it plainly or try a corrected call. If you lack the user's team id, ask for it.
- Recommendations only: the manager makes transfers themselves in the FPL app."""


def _jsonable(o):
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    return str(o)


def text_of(message):
    """Assistant text from a chat message (some models return a list of content blocks incl. reasoning)."""
    c = message.get("content")
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if b.get("type") == "text")
    return c or ""


class FplAgent:
    def __init__(self, llm, tools, max_steps=MAX_STEPS):
        self.llm, self.tools, self.max_steps = llm, tools, max_steps

    def system_message(self):
        d = self.tools.data
        return {"role": "system", "content": SYSTEM_PROMPT.format(
            today=datetime.now(timezone.utc).strftime("%Y-%m-%d"), next_gw=d.next_gw,
            deadline=str(d.next_deadline)[:16])}

    def run(self, history):
        """history: [{"role": "user"|"assistant", "content": str}, ...] ending with the user's question.

        Returns {"answer": str, "steps": [{"tool", "arguments", "result"}...], "messages": full transcript}.
        """
        messages = [self.system_message()] + list(history)
        steps = []
        for _ in range(self.max_steps):
            msg = self.llm(messages, TOOL_SPECS)
            calls = msg.get("tool_calls") or []
            if not calls and not text_of(msg).strip():
                # some models occasionally stop after reasoning with no text: ask once more for the answer
                messages.append({"role": "user", "content": "Please write your answer now, using the tool results."})
                msg = self.llm(messages, TOOL_SPECS)
                calls = msg.get("tool_calls") or []
            if not calls:
                return {"answer": text_of(msg), "steps": steps, "messages": messages + [msg]}
            messages.append({"role": "assistant", "content": text_of(msg) or None, "tool_calls": calls})
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or "{}"
                result = self.tools.call(name, args)
                steps.append({"tool": name, "arguments": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(result, default=_jsonable)})
        return {"answer": "Sorry, I couldn't finish that within the tool-call limit. Try a narrower question.",
                "steps": steps, "messages": messages}


def databricks_llm(endpoint=DEFAULT_ENDPOINT, client=None, max_tokens=1500, temperature=0.0):
    """llm() that calls a Databricks model serving endpoint (OpenAI-compatible chat format)."""
    from databricks.sdk import WorkspaceClient
    w = client or WorkspaceClient()

    def llm(messages, tools):
        body = {"messages": messages, "tools": tools, "max_tokens": max_tokens, "temperature": temperature}
        r = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
        return r["choices"][0]["message"]
    return llm
