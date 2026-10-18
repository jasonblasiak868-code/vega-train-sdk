# vega-train-sdk

![vega-train-sdk](assets/tile.svg)

Governed model training: environment, data and plan cards, a human-committed run, DAG-frozen checkpoints, and a freeze card that ties the weights to everything that made them. The back end of the Vega OS training tab.

> **Licensing.** Source-available, not open source. Free for personal use; free for registered small businesses (under US$1M revenue and valuation); everyone else needs a commercial license (see LICENSE; for a commercial license write to jasonblasiak.ai.mail@gmail.com). Patent pending. Draft licence text, not yet reviewed by an attorney.

Package: `vega_train` (src layout). Depends on: vega-custody-sdk, vega-governance-jacket-sdk, vega-harness-sdk, vega-pareah-sdk, vega-dkg-lattice-sdk, vega-superproof-sdk.

## Modules

| Module | What it does |
|---|---|
| `env_card` | The machine and software a run happened on: Python, packages, GPU as reported, a short allow-list of determinism settings. No other environment variable is ever recorded. |
| `data_card` | Exactly which bytes were trained on (hashes, never content), where they came from, license class and taint. A file holding a key or token is refused. |
| `trainers` | Back ends with canonical, pickle-free checkpoints: a small numpy network (deterministic) and a torch template using safetensors. |
| `run` | The governed run: plan, single-use human commitment, hashed checkpoints, hard stop, resume under a new commitment, freeze, and an independent verifier with optional replay. |
| `cards` | The shared card shape: hash for integrity, never mistaken for authenticity. |
| `lattice_data` | Lattice-governed training data: pick what to train on from DKG cards, hold out a clean test set by a fixed hash rule, keep one copy of identical content, apply the licence and taint rules, write the lattice node ids onto the DATA card, plan leave-one-repo-out ablations and write TRAINED_ON / HELD_OUT links back. |
| `bundle` | Seal a run's cards into one superproof and open them again with every card re-checked. |
| `goal` | The goal request bar: the person's words kept verbatim with who asked and the identity level they reached, testable success tests, constraints and the conduct code the training aims at; a model may draft the structure but never approves it; a goal is ready only when the requester and a human signing the conduct code have both signed. |
| `conduct` | A signed conduct code (rules with their statute, precedent, policy or owner source, severity tier and requirement), the reward arithmetic (a serious violation erases the episode's task reward; in a team one member's erasing violation erases everyone's and the others share the penalty) and a run-time meta-audit before an action. |
| `moral_data` | Training examples built from precedents the owner supplies: refusals that name the rule, the cost and who bears it, preference pairs, allowed-behaviour examples and held-out probes; the prohibited-act side is flagged private and its data card can never be public. |
| `behavior_eval` | Measure what a model actually does on held-out probes: violation rate with a 95% Wilson upper bound, over-refusal, by severity tier; a crash counts as the worst answer. |
| `__main__` | One command line (JSON lines) used by the Vega OS training tab. |

## What it does NOT do (read this)

- A freeze proves which bytes produced which weights under which plan and that a human approved that plan. It does not prove the model is good, safe, or free of memorised data.
- A replay proves reproducibility only for a deterministic trainer on the same setup; GPU runs are not claimed reproducible.
- The trainer runs in this process with this process's rights: containment is not an operating-system sandbox.
- This program never signs or mints a commitment; the human does that in their own terminal.
- Training on a conduct code changes how likely a model is to act a certain way: it does not make it understand or feel anything and gives no guarantee; the external gates stay in force. A reward for no violations can teach hiding them, so violations must be detected by outside audit. Law differs by place; nothing here is legal advice.

## Status

See STATUS.md. Tests: 115 tests in 7 files, all passing.  See SECURITY.md for the threat model.

## Run the tests

Put `src` of this repo and of the repos it depends on on PYTHONPATH, then run each file in `tests/`.
