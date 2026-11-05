# Handover: guard/ security tooling

State as of 2026-11-05, for any agent picking this up. Read `CLAUDE.md` first; its rules win.
PR: [JehanzebAli11/spark-lab#3](https://github.com/JehanzebAli11/spark-lab/pull/3), branch
`claude/dgx-spark-bootstrap-73xdyl`. Open, not merged, and independent of the bootstrap PR #1.

## Goal

Lightweight guard models that protect people and AI agents. The core question is always the
same: **does what is about to happen match what the person intended?** It applies to agent
actions, web pages and wallet signatures.

The plan has four stages:

1. Generate exact-labelled data in code.
2. Train a small calibrated "System One" model on the Spark: Gemma 4 E2B with Choice, Noul, Score
   and Box answers.
3. Serve it locally.
4. Plug it into the harness, the extension and the wallet.

Background docs, written by the owner with Claude:
- [Building our own System One models](https://claude.ai/artifact/2oYdKqZiLcvq3ocPCLTJKP)
- [Lightweight security models](https://claude.ai/artifact/PfFFDiPmkAvAcaNQakUquN)
- [Small models on the Spark](https://claude.ai/artifact/9k9XAMogMy3VrSBkdSWEBQ)

## What exists (all tested, 79 tests, no model needed)

| Part | Entry point | Status |
| --- | --- | --- |
| Screen-injection dataset generator | `python -m guard.screen_injection --n N --out DIR` | Done. 8 techniques, 6 goals, decoys, exact boxes, resumable. ~8 pages/s on 4 CPU cores. |
| Action-firewall harness | `python -m guard.firewall --out DIR --agent scripted\|chat --guard none\|rules\|systemone\|llm` | Done. Scripted and chat agents; rules, Jev-format and LLM-judge guards; scoring by technique and goal. |
| Chrome extension shell | `extension/` (load unpacked) + `python -m guard.extension_stub` | Done against the stub. Verified headless only, not yet in desktop Chrome. |
| Wallet signature decoder | `python -m guard.wallet tx\|typed\|eth-sign\|personal-sign` | Done for decoding and flags. No intent check and no fork simulation yet. |

Job files are ready to queue in `jobs/examples/`:

- `screen-injection-v1.json`
- `firewall-rules.json`
- `firewall-teacher-agent.json`

Each writes a `report.md` that the nightly report quotes.

## What the local runs showed (synthetic; numbers belong in reports, not READMEs)

- **Text agents miss canvas text.** A fully obedient, text-reading scripted agent is hijacked by
  every technique except `canvas_text`, which text-only agents can't see. This is the case for a
  screenshot guard.
- **Rules have a gap.** The rule guard stops tool-based attacks (email, open URL, pay, wallet,
  run command) with no false alarms on these tasks. Leaks back to the user through `reply`
  mostly get through, and that is the gap the model must close.
- **The scripted agent is a worst case, not a real model.** Real hijack rates need the
  `firewall-teacher-agent` job (Gemma 4 31B as the agent) and, for evaluation only, frontier
  models through `--agent chat` with an OpenRouter key.

## Next steps, in order

1. **Get PR #3 green.** CI now installs Chromium, so check the run.
2. **On the Spark:**
   - `pip install -e '.[security]' && python -m playwright install chromium` (ARM64 untested).
   - Queue the three example jobs.
3. **Real baseline.** Read `runs/firewall/teacher-none/report.md`: how often does Gemma 4 31B
   obey injections? This is the first honest headline number.
4. **Training data:**
   - Rows of user request, action and label from `runs/firewall/*/actions.jsonl`.
   - Screenshots plus boxes from `runs/screen-injection/v1`.
   - Teacher soft labels via vLLM logprobs only. Never API models (rule 1).
5. **Train the guard.** Gemma 4 E2B with LoRA in Unsloth (`nvcr.io/nvidia/pytorch:25.11-py3`),
   resumable through `LAB_RESUME_FROM`, then calibrated with temperature scaling. Keep a locked
   eval in `eval/locked/`, which is human-only.
6. **Serve it.** Expose `POST /v1/systemone` (Jev's request and response shape; see
   `guard/firewall/guards.py`) and run `--guard systemone --guard-url http://localhost:8100/v1`.
   Compare it with rules, frontier LLM judges and Jev (evaluation only).
7. **Replace the stub.** Swap `guard/extension_stub` for a server that keeps the same contract
   (documented at the top of `server.py`) but answers with the model.
8. **Wallet intent check.** Add simulation on a local fork (`anvil --fork-url`, testnets only),
   then one guard question: does the decoded effect match the user's stated intent?
9. **Release, by the owner only.** Make the dataset card and model card. Hugging Face uploads
   go to private repos only; `publish/` is owner-only (rule 3).

## Rules that matter here

- **Training data** comes only from code generators, the Spark teacher, or real data collected
  with consent. Claude, OpenAI, Gemini, OpenRouter and Jev may be **evaluated**, never used to
  label training data.
- **Attack material:**
  - Attacker addresses stay on reserved names (`.example`, `.test`).
  - Agent tools stay simulated.
  - Crawl live phishing only inside an isolated sandbox with no credentials.
  - Keep raw attack corpora private.
- **The extension** must stay localhost-only and must never read form values.
- **No hand-typed numbers in READMEs.** Results come from job reports.

## Known gaps and caveats

- The dataset is synthetic. Before claiming real-world performance, validate on real pages.
- The extension's candidate finder is heuristic (class names, fixed position). Native ads
  without ad-like markup are what the model is for.
- The wallet decoder recognises a fixed set of functions and typed-data shapes. Anything else is
  reported as "unrecognised", by design.
- The `firewall` tasks are small (3). Add tasks before trusting attack-success numbers.
