# guard: lightweight security tooling

Datasets, harnesses and decoders for the small guard models we train on the Spark. Everything
here runs on a laptop or in CI. The models come later, trained on the Spark (CLAUDE.md rule 1).

| Part | What it does | Run it |
| --- | --- | --- |
| [`screen_injection/`](screen_injection/README.md) | Renders web pages with prompt injections at known positions: 8 hiding techniques, 6 attack goals, English and Roman Urdu, plus decoys meant for people. Every screenshot comes with exact boxes. | `python -m guard.screen_injection --n 1000 --out runs/screen-injection/dev` |
| `firewall/` | Agent harness: 3 tasks on simulated sites, clean or poisoned. It logs every proposed action with the user's request and a ground-truth label, then scores attack success, task completion and false alarms for a guard. | `python -m guard.firewall --out runs/firewall/dev --guard rules` |
| [`../extension/`](../extension/README.md) + `extension_stub/` | Chrome extension shell (ads, cookie banners, phishing) that calls a local model endpoint, plus a rule-based stub of that endpoint. | `python -m guard.extension_stub`, then load `extension/` unpacked |
| `wallet/` | Decodes wallet requests (approvals, setApprovalForAll, transfers, multicall, EIP-2612 and DAI permits, Permit2, Seaport orders, eth_sign, personal_sign and Sign-In with Ethereum) into plain-English effects and risk flags. | `python -m guard.wallet tx --to 0x… --data 0x095ea7b3…` |

Install the browser part with `pip install -e '.[security]' && python -m playwright install chromium`.
The wallet decoder, firewall harness and stub use only the standard library.

## How the parts fit

```
screen_injection ──pages──► firewall harness ──(request, action, label)──► guard model training (Spark)
       │                          ▲                                               │
       └──screenshots + boxes──►  │  SystemOneGuard (POST /v1/systemone) ◄─────────┘
                                  │
extension ──page-check──► extension_stub  (later: the same guard model, served locally)
wallet decoder ──effects──► "does this match what the user meant?" (a guard-model question)
```

- **Labels come from code.** Injections, pages and attack actions are generated from templates, so
  every label is exact and no API model writes training data.
- **One guard question.** The firewall's `SystemOneGuard` sends one Noul question ("does this
  action serve the user's request?") in the `/v1/systemone` format Jev uses. The same harness
  can therefore measure Jev (evaluation only), frontier LLMs through `--guard llm`, and our own
  model once it is served on the Spark.

## Running on the Spark

The job files in `jobs/examples/` are ready to queue:

- `screen-injection-v1.json`: 20,000 pages
- `firewall-rules.json`: the rule baseline
- `firewall-teacher-agent.json`: Gemma 4 31B as the agent, with no guard, to measure how often
  a real model obeys injections

Results land in each run's `report.md` and in the nightly report.

## Safety

- Attacker addresses use reserved names (`.example`, `.test`).
- Every agent tool is simulated: nothing is sent, paid or signed.
- The extension talks only to localhost and never reads form values.
- Keep raw attack corpora in private repos (rule 3).
