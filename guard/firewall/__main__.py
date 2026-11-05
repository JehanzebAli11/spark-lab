"""python -m guard.firewall --out runs/firewall/rules [--agent scripted|chat] [--guard none|rules|systemone|llm]

  --per-task 40            runs per task (a quarter are clean)
  --agent-url/--agent-model/--agent-key-env    for --agent chat (e.g. the vLLM teacher on localhost:8000/v1)
  --guard-url/--guard-model/--guard-key-env    for --guard systemone (POST {url}/systemone) or llm
  --threshold 0.5          allow when P(action serves the request) >= threshold

Writes OUT/actions.jsonl (every proposed action, labelled), OUT/results.json and OUT/report.md.
Reruns skip finished runs. Frontier models through OpenRouter may be agents or judges here, for
evaluation only; their outputs must not become training data (CLAUDE.md rule 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import ChatAgent, ScriptedAgent
from .guards import AllowAll, LLMJudgeGuard, RuleGuard, SystemOneGuard
from .harness import make_variants, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m guard.firewall")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", choices=["scripted", "chat"], default="scripted")
    ap.add_argument("--obey", type=float, default=1.0, help="scripted agent: chance it follows an injection")
    ap.add_argument("--agent-url", default="http://localhost:8000/v1")
    ap.add_argument("--agent-model", default="google/gemma-4-31B-it")
    ap.add_argument("--agent-key-env")
    ap.add_argument("--guard", choices=["none", "rules", "systemone", "llm"], default="rules")
    ap.add_argument("--guard-url", default="http://localhost:8100/v1")
    ap.add_argument("--guard-model", default="spark-1")
    ap.add_argument("--guard-key-env")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args(argv)

    agent = ScriptedAgent(a.obey, a.seed) if a.agent == "scripted" else ChatAgent(
        a.agent_url, a.agent_model, a.agent_key_env)
    guard = {"none": AllowAll, "rules": RuleGuard}.get(a.guard)
    guard = guard() if guard else (SystemOneGuard if a.guard == "systemone" else LLMJudgeGuard)(
        a.guard_url, a.guard_model, a.guard_key_env, a.threshold)
    s = run(make_variants(a.per_task, a.seed), agent, guard, a.out)
    print((a.out / "report.md").read_text())
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
