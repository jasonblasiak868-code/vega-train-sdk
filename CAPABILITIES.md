# vega-train-sdk — Capability Brief

A minute inventory of what this kit actually enforces. Every entry: the capability, why a policy-or-prompt-level approach structurally cannot provide it, where it is enforced in this repo's code, which test file demonstrates it, and the honest limit. The structural argument throughout is the one the family whitepaper makes: authority and evidence held OUTSIDE the model, in signed commitments and hash-chained records, cannot be replicated by instructions given TO the model — a promise the model can break is not a control. Everything below describes published code only.

Reminder that applies to several entries: **GPU and large-language-model trainers are plug-ins, not included.** This kit governs training; it ships two small reference back ends (a deterministic numpy network and a CPU torch template) so the governed path can be exercised and replayed. Status is PARTIAL per STATUS.md.

---

## 1. Human-committed training start

- **Capability.** No training run starts without a human's single-use signed commitment for exactly this plan (`train_run:<plan_hash>`); the spend is burned on a record chain shared by every run folder under the same parent, so one signature can never be offered twice.
- **Why policy cannot provide it.** A prompt or policy saying "get approval before training" lives inside the process being approved; the approval here is a signature verified against a key the model never holds, and the spent-commitment record is append-only evidence outside the model's reach. Authority held outside the model cannot be minted, widened or replayed by the model — that follows from the architecture, not from anyone's good behaviour.
- **Where enforced.** `run.py`: `TrainingRun._gate` (checker must return exactly True for exactly this action; a crashing or loose checker never starts a run), `start_action`, the shared burn chain (`COMMITMENTS_USED.jsonl` via harness `burn_commitment`).
- **Test.** `tests/redteam_1_run_integrity.py` (no commitment / loose checker / crashing checker / wrong action / spent commitment).
- **Honest limit.** This program never signs or mints a commitment — the human does that in their own terminal, and a careless human is still a careless human. GPU/LLM trainers are plug-ins, but ANY plug-in trainer sits behind this same gate.

## 2. Evidence-bound plan (ENV + DATA + PRE + TRAIN_PLAN)

- **Capability.** A run executes only a sealed plan bound by hash to an environment card, a data card and a pareah PRE lease that permits `train` on exactly this model and dataset resource; an expired PRE stops the run.
- **Why policy cannot provide it.** "Only train on approved data" as a policy is unverifiable after the fact; here the binding is a hash — swap one byte of any card and the plan no longer matches, and the mismatch is refused at construction, before anything runs.
- **Where enforced.** `run.py` (`make_plan`, `TrainingRun.__init__`, `_gate` PRE checks), `env_card.py`, `data_card.py`; PRE minting and `permits` come from vega-pareah-sdk.
- **Test.** `tests/redteam_1_run_integrity.py` (mismatched/edited cards, expired PRE, insane plan values), `tests/test_train.py`.
- **Honest limit.** The environment card describes the environment; it does not make a run reproducible. The PRE lease cap (900 seconds in the CLI) bounds the permission, not the training time.

## 3. Data card: hashes, licences, taint — never content

- **Capability.** The DATA card records per file the path, size, sha256 and line count — never content — plus a licence class (strictest of declared class and library sources wins), taint, and `public_ok`. A file holding a credential-shaped value is refused for training, and the refusal names the file, never the value. Containment, plain names and size caps are enforced; `load_verified` re-reads and re-hashes every byte at start and again at freeze.
- **Why policy cannot provide it.** A policy against training on secrets or unlicensed data depends on someone remembering what the data was; this makes the answer a recorded, checkable fact — the bytes trained on ARE the bytes hashed, verified twice by code outside the trainer, and taint propagates mechanically to the model card so a tainted result can never be presented as clean.
- **Where enforced.** `data_card.py` (`build`, `load_verified`, `strictest`), custody safepaths and `contains_credential`, harness use-gate licence classes; taint inheritance in `run.freeze` (provenance card).
- **Test.** `tests/redteam_2_cards_freeze_verify.py` (credential refused naming file not value; traversal/absolute/duplicate/junction entries; forged and resealed cards; taint following data and base model; dropped taint mark reported).
- **Honest limit.** The credential scan covers the first 8 MiB of each file (larger files are flagged `scan_complete: False`); licence classes are policy data, not legal advice.

## 4. Checkpoint chain: hashed before training continues

- **Capability.** Every checkpoint is written atomically, read back, hashed, and the hash recorded on the custody chain BEFORE the next chunk trains. A tampered, missing or swapped checkpoint is refused on resume before anything is trusted — and the new commitment is not spent.
- **Why policy cannot provide it.** "Keep good checkpoints" as advice leaves no verifiable link between a weight file and the run that made it; here each checkpoint's hash sits in an append-only chain where every entry fingerprints the one before, so altering history breaks the chain in a way anyone can check without trusting the model, the vendor or the project.
- **Where enforced.** `run.py` (`_checkpoint`, `_atomic_write_bytes`, `_read_checkpoint`, `resume`), custody `RecordChain`.
- **Test.** `tests/test_train.py` (checkpointing, canonical round-trip), `tests/redteam_1_run_integrity.py` (tampered/missing/swapped checkpoint, interrupted process, two processes at once).
- **Honest limit.** The chain proves integrity and order, not that the numbers inside the checkpoint are "good". GPU/LLM trainer plug-ins produce the state bytes; the canonical, pickle-free format rules are what their checkpoints must satisfy to replay.

## 5. Hard stop and committed resume

- **Capability.** Any failure — non-finite loss, wall-clock or disk budget, trainer exception, misreported step count, unwritable checkpoint — halts the run with a recorded reason and the last good checkpoint. A halted run never resumes by itself: resume needs a NEW human commitment for exactly `train_resume:<plan>:<checkpoint_sha>`.
- **Why policy cannot provide it.** A policy "stop on errors" is executed by the code that may be erroring; here the halt IS a record, budgets are enforced by the governing loop around the trainer (which is treated as untrusted), and continuing costs a fresh human signature — the model cannot talk its way past a hard stop it does not control.
- **Where enforced.** `run.py` (`_loop`, `_halt`, `resume`, `_RunLock` OS lock).
- **Test.** `tests/redteam_1_run_integrity.py` (NaN/inf loss, misreporting trainer, budgets, unwritable checkpoint, resume-only-with-new-commitment).
- **Honest limit.** The trainer runs in this process with this process's rights: containment is process-level, not an OS sandbox.

## 6. Freeze card: weights tied to everything that made them

- **Capability.** `freeze` re-reads every checkpoint and the weights, re-verifies the data, mints model / runtime / server / provenance cards, and seals a TRAIN_FREEZE card binding plan, ENV, DATA, PRE, every checkpoint hash, the weights hash, the loss curve hash and the chain head — with a MANIFEST over the whole card bundle. After a freeze, start, resume and freeze all refuse.
- **Why policy cannot provide it.** "Document your training" produces prose that drifts from reality; a freeze card is computed FROM the run's own records and files at freeze time, so it cannot say anything the evidence does not support, and "frozen" is a state the code enforces, not a label anyone applies.
- **Where enforced.** `run.py` (`freeze`), pareah card minters, `cards.seal`.
- **Test.** `tests/test_train.py` (freeze writes all cards), `tests/redteam_2_cards_freeze_verify.py` (freeze refuses unfinished runs, changed weights, changed data; model card matches only its weights; taint inheritance).
- **Honest limit.** A freeze proves WHICH bytes produced WHICH weights under WHICH plan and that a human approved that plan. It does not prove the model is good, safe, or free of memorised data.

## 7. Independent verifier with replay

- **Capability.** `verify_freeze` recomputes every claim from the files alone: chain validity against a head you kept elsewhere, exactly one freeze record, manifest and card hashes, freeze bindings, every checkpoint file, weights, model-card match, taint, data still matching its card — and for a deterministic trainer, a full replay from the plan that must reproduce every checkpoint hash and the final weights. It never raises on garbage: the report says what failed.
- **Why policy cannot provide it.** Self-reported training logs are the thing being checked doing the checking; this verifier trusts nothing in the report — it re-derives every hash from bytes, and an attacker-rebuilt run still fails against the head kept at freeze time.
- **Where enforced.** `run.py` (`verify_freeze`), `trainers.py` (canonical state bytes make replay hashes comparable).
- **Test.** `tests/redteam_2_cards_freeze_verify.py` (every way of editing a frozen run reported; cut tail caught by remembered head; rebuilt run fails; replay with other data does not pass; verify never raises on garbage), `tests/test_train.py` (clean run verifies with replay).
- **Honest limit.** Replay proves reproducibility only for a deterministic trainer on the same machine setup; GPU runs are not claimed reproducible, and GPU/LLM trainers are plug-ins that generally will not declare `deterministic`.

## 8. Lattice-governed training data

- **Capability.** `lattice_data.select` turns DKG knowledge-graph nodes into a sealed LATTICE_PLAN: held-out set chosen by a fixed hash rule on the node id (re-derivable, order- and score-independent), identical content trained once wherever it came from, altered cards excluded never repaired, tainted or non-permissive nodes excluded unless explicitly allowed, per-repo cap, caller-supplied scores with unscored nodes kept last and said so — never given invented numbers. `to_data_card` tags the DATA card with the plan hash and node ids; `ablation_plans` makes leave-one-repo-out plans with the SAME held-out set; `training_links` writes TRAINED_ON / HELD_OUT back to the lattice only for a card that really came from this plan.
- **Why policy cannot provide it.** "Keep test data separate" as a policy fails the moment the same content appears under a second name or a selection is re-run differently; here separation is a hash rule anyone can re-derive, dedupe is by content hash, every exclusion carries its reason, and the write-back is refused unless the card provably came from the plan — the guarantee lives in the record, not in anyone's discipline.
- **Where enforced.** `lattice_data.py` (`select`, `to_data_card`, `ablation_plans`, `training_links`, `card_intact`), `data_card.split_ids`; node cards and their integrity hash from vega-dkg-lattice-sdk.
- **Test.** `tests/test_lattice_data.py`, `tests/mutations_lattice.py`.
- **Honest limit.** This chooses and records; it does not train. Real ranking scores come from the caller (for example the galaxy SDK's convergence) — none are invented here. A curriculum order does not make a model better by itself: measure it (the ablation plans exist for that). Duplicate detection is by exact content hash: a lightly edited copy is a different file.

## 9. Sealed card bundle

- **Capability.** `seal_bundle` seals a run's cards (1–64, one per kind, every one intact or nothing is sealed) into ONE superproof with layered sealing; `open_bundle` re-checks every card on open and refuses a capsule that is not a card bundle.
- **Why policy cannot provide it.** "Keep the training records safe" as policy leaves the records as editable files; sealing puts them behind the superproof layer's encryption and integrity, and the all-or-nothing seal rule means a single altered card is refused at the door rather than smuggled into the archive.
- **Where enforced.** `bundle.py`, vega-superproof-sdk seal, custody capsules.
- **Test.** `tests/test_bundle.py` (round-trip with every card checked; altered card or repeated kind seals nothing; wrong keys fail; foreign capsule refused; quantum-proof tier).
- **Honest limit.** The bundle proves the cards are unchanged since sealing; it does not prove the cards are TRUE.

## 10. Goal request bar

- **Capability.** The person's request is kept VERBATIM on a sealed GOAL card with their identity assurance level (below the minimum: refused — training is a significant act), testable success tests (probe + expected REFUSE/ESCALATE/ALLOW verdict, rules must exist in the named conduct code), constraints, and the conduct-code hash. A model may draft the structure (`draft_from_request`) but the draft is marked `model:<name>`; a goal is READY only when the requester has signed it over a human-signed conduct code.
- **Why policy cannot provide it.** A prompt-level "ask the user what they want" leaves the request paraphrased, unattributed and untestable; here the words are frozen with a hash, tied to a named signer and identity level, and "done" is defined as probes with expected verdicts — measurable later by the evaluator instead of claimed.
- **Where enforced.** `goal.py` (`make_goal`, `draft_from_request`, `is_ready`), `conduct.signature_ok` (Ed25519 over a domain-separated prefix plus the card hash).
- **Test.** `tests/test_conduct.py` (verbatim request, low identity refused, malformed requests, model drafts marked and cannot approve, readiness rules).
- **Honest limit.** This module does not decide whether a goal is good or lawful; it makes the request explicit, attributable and testable. The training itself still needs the human-committed run.

## 11. Signed conduct code, reward arithmetic, meta-audit

- **Capability.** A CONDUCT card holds rules with a source (statute, precedent, policy or owner — citation and jurisdiction recorded; illegal and immoral are not conflated), severity tier 1–5, requirement, and trigger words, plus per-tier weights, an erase tier and a team share ratio. It governs nothing until a human signs it. Reward arithmetic: a violation costs its severity weight; a violation at or above the erase tier ERASES the episode's task reward; in a team one member's erasing violation erases everyone's task reward and the others share the penalty. `MetaAuditor.check` applies the same signed code at run time before an action: strictest requirement among the rules hit wins, with ids and citations, every check recorded to a hash chain when supplied.
- **Why policy cannot provide it.** Prompt-level rules are suggestions to the thing being ruled, and a self-audit is the thing being checked doing the checking; here the code is inert until signed by a key the model does not hold, the arithmetic is deterministic and inspectable, and the auditor fails toward the human: a broken classifier, an unreadable action, or a check that cannot be recorded all yield ESCALATE, never ALLOW.
- **Where enforced.** `conduct.py` (`make_conduct`, `approve`, `is_approved`, `episode_reward`, `team_reward`, `MetaAuditor`).
- **Test.** `tests/test_conduct.py`, `tests/mutations_conduct.py` (unsigned code governs nothing; strictest requirement wins; whole-word triggers; broken plug-in never allows; recording failure hands it to a human; erase and share arithmetic).
- **Honest limit.** The tiers, weights and citations are DATA the owner (with an attorney) supplies — the SDK does not know any law, and nothing here is legal advice. Trigger-word matching is crude and both over- and under-fires: measure it and prefer a better classifier. A reward for "no violations" can teach hiding them, so violations must be detected by outside audit. The reinforcement-learning trainer that would consume this reward is a plug-in, not included.

## 12. Precedent-based training examples, private by construction

- **Capability.** From owner-supplied precedent records, `moral_data.build_examples` deterministically emits (no model involved): refusals naming the rule, citation, tier, consequence, who bears it and the right alternative; preference pairs; allowed-behaviour examples (guarding against a model that refuses everything); reflection text; and held-out probes that never appear in training. The prohibited-act side is capped at 300 characters of high-level description, operational-detail records and credential-shaped text are refused, and `private_card` marks the dataset's DATA card private, tainted and never `public_ok`, whatever its licence class says.
- **Why policy cannot provide it.** "Handle sensitive examples carefully" as guidance leaves the sensitivity judgement to whoever is rushing at 2 a.m.; here the privacy flag is computed into the card's hash chain by the code that builds the dataset, so the "never public" property travels with the data and shows up in every downstream taint check.
- **Where enforced.** `moral_data.py` (`build_examples`, `write_dataset`, `private_card`), `data_card.split_ids` for the holdout.
- **Test.** `tests/test_conduct.py` (deterministic examples, held-out records never reach training, text names rule/cost/who-bears-it/alternative, unsafe records refused, dataset card can never be public).
- **Honest limit.** Templates teach a pattern, not judgment; a model can learn to recite the reasons without acting on them — measure behaviour on held-out probes. The rejected side is still harmful-looking text: keep it private. Citations are the owner's and should be checked by an attorney.

## 13. Behaviour measurement on held-out probes

- **Capability.** `behavior_eval.evaluate` runs every probe through the model-or-agent under test and reports violation rate with a 95% Wilson UPPER bound (so a small sample cannot look safe), over-refusal rate (a model that refuses everything is broken, not trained), violations by severity tier, and per-probe detail. `passes` applies thresholds the caller chooses. Probes whose prompt hash appears in the training set are refused; a crash or nonsense verdict counts as the worst answer, never as a pass.
- **Why policy cannot provide it.** "The model follows the conduct code" is exactly the kind of claim a self-report makes about itself; this measures behaviour from outside, on data the subject never saw, with statistics chosen so absence of evidence cannot masquerade as evidence of safety.
- **Where enforced.** `behavior_eval.py` (`evaluate`, `passes`, `wilson_upper`, `prompt_hash`).
- **Test.** `tests/test_conduct.py` (good model passes / bad fails / refuse-everything fails the other way; Wilson bound; crash worst-case; trained-on probes refused; the meta-auditor itself evaluated on held-out precedents).
- **Honest limit.** Nothing here decides what thresholds are acceptable — that is the owner's call. Measurement quality is bounded by probe quality and count.

## 14. One CLI, machine-readable, fail-closed

- **Capability.** `python -m vega_train` exposes prepare / start / resume / freeze / verify / status / action as JSON-line events; `action` prints the exact string a human must sign; every failure is one `refused` event with a reason and a non-zero exit, never a traceback a UI cannot read.
- **Why policy cannot provide it.** A UI that can parse refusals can show the human exactly what was denied and why — the human stays in the loop with real information instead of trusting a status line; and the refusal path is code, not etiquette.
- **Where enforced.** `__main__.py` (`main`, `prepare`, `emit`), custody ledger check wiring.
- **Test.** `tests/test_cli.py` (whole path through the command line; bad input is a refusal event, not a traceback).
- **Honest limit.** The CLI never signs or mints a commitment; the human signs in their own terminal and passes the commitment file.

---

## This kit's part in the family predictive loop

The family's data/knowledge flow runs upward from the record: custody records → librarian distils lessons → dkg-lattice builds the knowledge graph (origin and licence per fact) → reasoning and galaxy forecast from the lattice → the human decides again. This kit is where that loop touches model weights, and two pieces are the centerpiece:

- **Lattice-governed training data** closes the circle from knowledge back into learning: the graph's nodes — each with origin, licence and integrity hash — become the selection plan, the DATA card names the exact nodes trained on, and TRAINED_ON / HELD_OUT links are written back into the lattice, so the graph itself remembers what it taught. Ranking scores flow in from the callers that compute them (for example the galaxy SDK's convergence); this kit never invents one.
- **Freeze cards** are what the loop can trust later: when a future deliberation, forecast or human decision asks "what did this model learn from, under what licence, with what held out, and who authorised it?", the answer is a sealed card an independent verifier recomputes from files — not a memory, a claim, or a changelog.

Authority still flows the other way and never mixes with this: down from the human, as single-use commitments through the gates, one training run at a time. The model reasons freely inside the loop; it never holds the signing key, never mints or widens authority, and every step lands in the append-only chain anyone can verify without trusting the model, the vendor or the project.

Honest limit for the whole loop as shipped here: the trainers that would learn from all this governance at real scale — GPU and LLM fine-tuning, remote and cloud servers, the RL trainer consuming the conduct reward — are plug-ins or designs, not included. What IS included governs any plug-in you bring: same gate, same records, same freeze, same verifier.
