"""Action-firewall harness: how often do injected pages hijack an agent, with and without a guard?

Three tasks (summarise an article, buy the cheapest cable, reply to Alice) run on small simulated
sites, clean or poisoned with one injection from guard.screen_injection. Every proposed action is
logged with the user's request and a ground-truth label (attack or benign), which is also the
training data shape for the guard model: (request, action) -> does it serve the request?
"""

from .agents import ChatAgent, ScriptedAgent
from .attacks import Action, is_attack, read_instructions
from .guards import AllowAll, LLMJudgeGuard, RuleGuard, SystemOneGuard, Verdict
from .harness import Variant, make_variants, run, run_one, score
from .site import TASKS, Env, build_site, page_text

__all__ = [
    "TASKS", "Action", "AllowAll", "ChatAgent", "Env", "LLMJudgeGuard", "RuleGuard", "ScriptedAgent",
    "SystemOneGuard", "Variant", "Verdict", "build_site", "is_attack", "make_variants", "page_text",
    "read_instructions", "run", "run_one", "score",
]
