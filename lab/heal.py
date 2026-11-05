"""Decide what to do after a failed attempt, apply the fix, and write incident files.

| class       | action             | what happens                                                   |
| ----------- | ------------------ | -------------------------------------------------------------- |
| oom         | halve_batch        | halve `batch_env` (and LAB_BATCH_SCALE), resume from checkpoint |
| disk_full   | prune_retry        | keep the newest 2 checkpoints per run under runs/, then retry   |
| network     | backoff_retry      | wait 30 s, 60 s, 120 s, ... then retry                          |
| cuda_error  | cooldown_retry     | one retry from checkpoint after a cool-down, then incident      |
| flaky       | retry              | retry straight away                                             |
| bad_data    | incident           | stop: retrying cannot fix the data                              |
| code_bug    | incident           | stop: an agent should fix the code                              |
| unknown     | retry, then incident | plain `retries` as before, then escalate to the owner         |

Healing retries come from `max_heals` (per job, default 4) and per-class caps; plain
retries for unknown failures come from the job's `retries`. Every decision is appended to
`job.heal_log`, which the nightly report prints.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import checkpoints
from .triage import UNKNOWN, Diagnosis, Triage

if TYPE_CHECKING:
    from .paths import LabPaths
    from .queue import Job

CLASS_CAPS = {"oom": 3, "disk_full": 2, "network": 4, "cuda_error": 1, "flaky": 2}
ACTIONS = {"oom": "halve_batch", "disk_full": "prune_retry", "network": "backoff_retry",
           "cuda_error": "cooldown_retry", "flaky": "retry", "bad_data": "incident", "code_bug": "incident"}
OWNER = {"bad_data": "agent", "code_bug": "agent"}  # everything else escalates to the owner
NEXT_STEPS = {
    "bad_data": "Find the bad rows (the last lines of the log name the file), fix or drop them, "
                "add a validation check that fails fast, then re-enqueue the job.",
    "code_bug": "Reproduce with the job's cmd in its cwd, fix the bug with a test, open a pull request, "
                "then re-enqueue the job after the owner merges.",
    "oom": "Batch size is already at its floor or the halving budget is spent; reduce sequence length, "
           "enable gradient checkpointing, or move the job to a quieter time.",
    "disk_full": "Pruning freed too little. Look at `du -sh runs/* data/*` and ask the owner what can go.",
    "network": "The network did not come back after several back-offs. Check the remote host and DNS.",
    "cuda_error": "The CUDA error came back after a cool-down. Check `nvidia-smi -q`, dmesg for Xid lines, "
                  "and whether the same step fails every time (that suggests a code bug, not hardware).",
    "flaky": "The job keeps failing on what looks like a transient error. Look for lock contention.",
    UNKNOWN: "The classifier was not sure what went wrong. Read the log tail below and label it; "
             "add the label to the chaos dataset if it is a new kind of failure.",
}


@dataclass
class Decision:
    action: str          # halve_batch | prune_retry | backoff_retry | cooldown_retry | retry | incident | give_up
    retry: bool
    detail: str = ""
    wait_s: float = 0.0


class Healer:
    def __init__(
        self,
        triage: Triage | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        backoff_base_s: float = 30.0,
        cooldown_s: float = 60.0,
        prune_keep: int = 2,
    ):
        self.triage = triage or Triage.default()
        self.sleep = sleep
        self.backoff_base_s = backoff_base_s
        self.cooldown_s = cooldown_s
        self.prune_keep = prune_keep

    # ---- decision -------------------------------------------------------------------
    def diagnose(self, log_text: str) -> Diagnosis:
        return self.triage.classify(log_text)

    def decide(self, job: Job, diag: Diagnosis, plain_retries_left: int) -> Decision:
        label = diag.label
        if label == UNKNOWN:
            if plain_retries_left > 0:
                return Decision("retry", True, f"unclassified; plain retry ({plain_retries_left} left)")
            return Decision("incident", False, "unclassified and out of retries")
        if label in ("bad_data", "code_bug"):
            return Decision("incident", False, f"{label}: retrying will not help")
        heals = [h for h in job.heal_log if h.get("retry")]
        same = sum(1 for h in heals if h["class"] == label)
        if len(heals) >= job.max_heals:
            return Decision("incident", False, f"heal budget spent ({job.max_heals})")
        if same >= CLASS_CAPS.get(label, 1):
            return Decision("incident", False, f"{label} fix already tried {same} time(s)")
        action = ACTIONS[label]
        if action == "halve_batch":
            return Decision(action, True)
        if action == "prune_retry":
            return Decision(action, True)
        if action == "backoff_retry":
            return Decision(action, True, wait_s=self.backoff_base_s * 2**same)
        if action == "cooldown_retry":
            return Decision(action, True, wait_s=self.cooldown_s)
        return Decision("retry", True)

    # ---- fixes ----------------------------------------------------------------------
    def apply(self, job: Job, decision: Decision, paths: LabPaths, cwd: Path) -> Decision:
        """Carry out the fix. May downgrade the decision to an incident if the fix is impossible."""
        if decision.action == "halve_batch":
            scale = float(job.env.get("LAB_BATCH_SCALE", "1")) / 2
            job.env["LAB_BATCH_SCALE"] = f"{scale:g}"
            msg = [f"LAB_BATCH_SCALE={scale:g}"]
            if job.batch_env:
                try:
                    old = int(job.env.get(job.batch_env, "0"))
                except ValueError:
                    old = 0
                if old <= 1 and job.batch_env in job.env:
                    return Decision("incident", False, f"{job.batch_env} already {old}; cannot halve")
                if old > 1:
                    job.env[job.batch_env] = str(old // 2)
                    msg.insert(0, f"{job.batch_env} {old}->{old // 2}")
            msg.append("resume from checkpoint" if job.resume_dir else "restart (no resume_dir)")
            decision.detail = ", ".join(msg)
        elif decision.action == "prune_retry":
            removed = self.prune(job, paths, cwd)
            if not removed:
                return Decision("incident", False, "disk full and nothing under runs/ to prune")
            decision.detail = f"pruned {len(removed)} old checkpoint(s), kept newest {self.prune_keep}"
        elif decision.action in ("backoff_retry", "cooldown_retry"):
            decision.detail = f"waited {decision.wait_s:g} s"
            self.sleep(decision.wait_s)
        return decision

    def prune(self, job: Job, paths: LabPaths, cwd: Path) -> list[Path]:
        """Only ever deletes old checkpoints inside a directory named runs/ (CLAUDE.md rule 4)."""
        roots = {paths.runs.resolve()}
        if job.resume_dir:
            run_dir = (cwd / job.resume_dir).resolve()
            for parent in [run_dir, *run_dir.parents]:
                if parent.name == "runs":
                    roots.add(parent)
                    break
        removed: list[Path] = []
        for root in sorted(roots):
            if root.is_dir() and "eval/locked" not in root.as_posix():
                removed += checkpoints.prune(root, keep=self.prune_keep)
        return removed

    # ---- incidents ------------------------------------------------------------------
    def write_incident(self, job: Job, diag: Diagnosis, decision: Decision, paths: LabPaths,
                       log_tail: str, stamp: str) -> Path:
        label = diag.label
        who = OWNER.get(label, "owner")
        paths.incidents.mkdir(parents=True, exist_ok=True)
        out = paths.incidents / f"{stamp}-{job.name}.md"
        lines = [
            f"# Incident: {job.name}",
            "",
            f"- **For:** {who}",
            f"- **Class:** {label} (confidence {diag.confidence:.2f}, {diag.source})",
            f"- **Why stopped:** {decision.detail}",
            f"- **Exit code:** {job.returncode} after {job.attempts} attempt(s)",
            f"- **Log:** `{job.log}`",
            f"- **Evidence:** `{diag.evidence}`" if diag.evidence else "- **Evidence:** -",
            "",
            "## Next steps",
            "",
            NEXT_STEPS.get(label, NEXT_STEPS[UNKNOWN]),
            "",
            "## Healing so far",
            "",
        ]
        lines += [f"- attempt {h['attempt']}: {h['class']} ({h['confidence']:.2f}) -> {h['action']}"
                  f"{': ' + h['detail'] if h.get('detail') else ''}" for h in job.heal_log] or ["- none"]
        lines += ["", "## Log tail", "", "```", log_tail, "```", "",
                  "## Job", "", "```json", _job_json(job), "```", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        return out


def _job_json(job: Job) -> str:
    import json

    keep = ("name", "cmd", "cwd", "priority", "timeout_min", "retries", "resume_dir", "env", "batch_env", "report")
    return json.dumps({k: getattr(job, k) for k in keep}, indent=2)
