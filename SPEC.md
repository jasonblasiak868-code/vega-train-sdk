# vega-train-sdk — Specification

Spec v1.0 — 2026-10-19 — describes the published reference implementation as tested; where this document and the source disagree, the source wins.

## 1. Purpose and scope

This specification covers governed model training: the cards that describe a run's environment, data and plan; the human-committed run itself; the freeze card and its independent verifier; lattice-governed data selection; the conduct layer (conduct code, goal card, precedent-based examples, behaviour measurement); and the card bundle. Everything specified here is BUILT and tested per STATUS.md, with the PARTIAL exceptions listed in §4.2 and §7.

Out of scope: the trainers themselves beyond the two small reference back ends (GPU and large-language-model trainers are plug-ins, not included); remote and cloud training servers (designed only); any ranking score for the lattice curriculum (the caller supplies real scores, none are invented here); the reinforcement-learning trainer that would consume the conduct reward (a plug-in, not included); and everything owned by the six dependency SDKs — this kit uses their owners instead of growing its own copies.

## 2. Vocabulary

- **card** — a plain-JSON object with a `kind` and a `card_hash` over everything else (`cards.py`). The hash is INTEGRITY, not authenticity: it shows a card was not edited after it was made; who made it and that a human approved comes from the commitment, never from the hash alone.
- **seal / intact** — `seal` computes and attaches `card_hash`; `intact(card, kind)` is True only when the hash matches the content (and the kind matches, when given). `intact` never raises.
- **ENV card** — the machine and software a run happened on (`env_card.py`): Python build, OS, CPU count, versions of training-relevant packages (read from metadata, nothing imported), GPUs exactly as `nvidia-smi` reports them, torch build flavour, and the values of a short allow-list of determinism settings. No other environment variable is ever recorded. Its **fingerprint** hashes the stable parts (not the capture time); **drift** lists exactly what differs from a fresh capture.
- **DATA card** — exactly which bytes a run learned from (`data_card.py`): per file the relative path, size, sha256 and line count — never content. Carries a **licence class** (strictest of declared class and library sources wins), **taint** (True when any source is a library repo or the class is not PERMISSIVE or OWNED) and **public_ok** (the negation of taint; the model card inherits it).
- **PRE card** — the pareah lease that permits the action `train` on the named `model/…` and `dataset/…` resources; minted and checked by vega-pareah-sdk, bound into the plan by hash.
- **plan (TRAIN_PLAN card)** — trainer name, hyper-parameters, seed, steps, checkpoint cadence, wall-clock and disk budgets, optional base model, bound by hash to the ENV, DATA and PRE cards (`run.make_plan`).
- **commitment** — a human's single-use signed authorisation for exactly one action string. Signed in the person's own terminal (governance-jacket); checked and **burned** (spent, on a shared record chain) by this kit through vega-harness records. Action strings: `train_run:<plan_hash>` and `train_resume:<plan_hash>:<checkpoint_sha>`.
- **checkpoint** — the trainer's canonical state bytes, written atomically, read back, hashed, and recorded on the run's chain BEFORE the next chunk trains.
- **TRAIN_HALT** — the record written when any failure stops a run (non-finite loss, budget spent, trainer exception, misreported step, unwritable checkpoint). A halted run never resumes by itself.
- **freeze card (TRAIN_FREEZE)** — the sealed card tying the final weights hash to plan, ENV, DATA, PRE, every checkpoint hash, the model card, and the provenance card; after it, the run folder is finished and start/resume/freeze all refuse.
- **manifest** — `cards/MANIFEST.json`: the hash of every card file in the bundle plus the freeze hash; what `verify_freeze` checks the card files against.
- **replay** — re-running the plan from scratch and requiring the same checkpoint hashes and the same final weights; only offered for a trainer that declares `deterministic`.
- **lattice plan (LATTICE_PLAN card)** — a selection of DKG lattice nodes: which to train on, which are held out, which were excluded and why (`lattice_data.select`).
- **held-out set** — nodes assigned to testing by a fixed hash rule on the node id (re-derivable, independent of order and of scores); never trained on.
- **ablation plan** — a leave-one-repo-out variant of a lattice plan with the SAME held-out set, so a difference in results is that repo's contribution.
- **conduct code (CONDUCT card)** — rules a human writes and signs: each with an id, text, a source (statute, precedent, policy or owner, with citation and jurisdiction), a severity tier 1–5, a requirement (REFUSE, ESCALATE, CHECK_THEN_ALLOW) and trigger words. Plus per-tier weights, an **erase tier** (a violation at or above it erases the episode's task reward) and a team **share ratio**.
- **meta-audit** — `MetaAuditor.check`: the same signed code applied at run time before an action, returning the strictest requirement among the rules hit, with rule ids and citations.
- **goal card (GOAL)** — the person's request kept verbatim, with requester reference and identity assurance level, testable success tests (probe + expected verdict), constraints, and the hash of the conduct code the training aims at. A model may draft the structure; it never approves it.
- **probe** — a held-out behaviour test: a prompt with an expected verdict (REFUSE, ESCALATE or ALLOW).
- **bundle** — a run's cards sealed into ONE superproof (`bundle.py`), openable only with the right keys and with every card re-checked.
- **taint inheritance** — taint flows from data and base model to the provenance and model cards; a tainted result can never claim `public_ok`.

## 3. Components

### 3.1 `cards.py`
- Role: the shared card shape — hash for integrity, never mistaken for authenticity.
- Interface: `CardRefused`, `card_hash(card)`, `seal(card)`, `intact(card, kind=None)`, `sha256_hex(data)`; canonical JSON (`canon`) re-exported from vega-harness records.
- Behavior: `card_hash` refuses non-objects and non-plain-JSON content; `intact` never raises — an altered or foreign card is simply False.
- Invariants: a sealed card's hash covers every field except itself; equal content gives equal hash (canonical JSON).

### 3.2 `env_card.py`
- Role: capture the machine and software a run happened on.
- Interface: `capture(...)`, `fingerprint(card)`, `drift(card, observed)`, `parse_gpus(text)`, constants `PACKAGES`, `ENV_ALLOWLIST`, `NVIDIA_SMI`, `STABLE_EXCLUDED`.
- Behavior: GPU query is a bounded `nvidia-smi` call; only lines matching the exact expected shape are parsed (an error banner or hostile string is ignored, not trusted); no GPU or no `nvidia-smi` is recorded as such, never invented. Settings are recorded only from the allow-list, only when the value is short, plain and not credential-shaped. `drift` against an altered card reports the alteration instead of comparing.
- Invariants: no environment variable outside the allow-list is ever recorded; the fingerprint excludes capture time, so two captures of one setup compare equal.
- Honest limit: a card describes the environment; it does not make a run reproducible.

### 3.3 `data_card.py`
- Role: record exactly which bytes a run learned from, and on what terms.
- Interface: `build(name, root, files, license_class, sources, matrix)`, `load_verified(card, root)`, `split_ids(ids, val_permille, seed)`, `strictest(classes)`, `DataRefused`, caps `MAX_FILE_BYTES`, `MAX_TOTAL_BYTES`, `MAX_FILES`, `SCAN_BYTES`, class `OWNED`.
- Behavior: every file must resolve inside the root (symlinks and junctions included, via custody safepaths), be a plain relative name, and sit under the size caps; a file holding a credential-shaped value is REFUSED and the refusal names the file, never the value. `load_verified` reads each file once and checks those bytes against the card — the bytes trained on are the bytes hashed. `split_ids` assigns ids by a fixed hash rule, so a split is re-derivable and order-independent.
- Invariants: the card holds hashes and counts, never content; licence resolution takes the strictest class (same rule as the harness use gate); taint implies `public_ok` False.
- Failure: any mismatch, missing file or altered card raises `DataRefused` (fail-closed).

### 3.4 `trainers.py`
- Role: the training back ends a governed run can drive, and the rules every back end must follow.
- Interface: `NumpyMLP`, `torch_trainer_class()`, `make(name)`, `REGISTRY`, `parse_csv(files, names)`, `pack_arrays(arrays, meta)`, `unpack_arrays(blob)`, `TrainerRefused`. A trainer is `init(config, files)` / `step(n)` / `state_bytes()` / `load_state(bytes)` / `weights_bytes()` and states `deterministic` honestly.
- Behavior: NO PICKLE anywhere — state is hand-serialised (numpy, `VTA1` canonical format) or safetensors (torch); the same state always gives the same bytes, so a checkpoint's hash means something and a replay can match it. `unpack_arrays` refuses damaged, truncated, over-long or non-numeric checkpoints; a checkpoint never executes code on load. Loss must stay finite; step counts must be reported honestly (the run checks both). `NumpyMLP` trains a one-hidden-layer network (or logistic regression when hidden=0) on CSV whose last column is a 0/1 label; `TorchTrainer` is full-batch CPU SGD, built lazily so importing the module never imports torch.
- Invariants: canonical serialisation; no code-executing checkpoint format; hyper-parameters validated at init.
- Honest limit: determinism is claimed only per-machine (numpy: float64, seeded; torch: full-batch CPU, deterministic algorithms, one thread) — never across machines or on a GPU. GPU/LLM trainers are plug-ins, not included.

### 3.5 `run.py`
- Role: the governed run — plan, committed start, hashed checkpoints, hard stop, resume, freeze, verify.
- Interface: `make_plan(...)`, `start_action(plan)`, `resume_action(plan, last_sha)`, `TrainingRun(run_dir, plan, env, data, pre_card, data_root, commitment_check, burn, trainer, clock, on_event)` with `start(commitment, observed_env, require_same_env)`, `resume(commitment)`, `freeze(runtime_license, model_id)`, `state()`, `rows()`, `checkpoints()`; `verify_freeze(run_dir, expected_head, min_length, data_root, replay, trainer_factory, clock)`; `TrainRefused`; budgets `MAX_STEPS`, `MAX_WALL`, `MAX_DISK`.
- Behavior (the chain of record kinds): `TRAIN_PLAN` → `COMMITMENT_USED` → `TRAIN_START` → `TRAIN_CHECKPOINT`* → (`TRAIN_HALT` → `TRAIN_RESUME`)* → `TRAIN_DONE` → `TRAIN_FREEZE`. Every step is one record on a custody chain in the run folder. Construction refuses altered cards or a plan bound to different cards, and a run folder that belongs to a different plan. `start` verifies the data against the card BEFORE the commitment is spent, requires the PRE to permit `train` on both the model and the dataset resource, then gates on a commitment that checks exactly True for exactly this action — a missing, crashing or loose checker never starts a run — and burns it on a chain shared by every run under the same parent folder, so a signature spent in one run folder cannot be offered to another. The loop halts on: wall-clock budget, disk budget, non-finite loss, trainer exception, misreported step count, unwritable checkpoint. `resume` works only on a halted or interrupted run, re-hashes the last checkpoint before trusting it, and needs a NEW commitment for exactly `train_resume:<plan>:<checkpoint_sha>`. A non-blocking OS lock prevents two processes training the same run at once. `freeze` re-reads every checkpoint and the weights, re-verifies the data, mints the pareah model/runtime/server cards, computes the provenance card (taint follows data and base model), seals the freeze card, writes the card bundle and manifest, and appends `TRAIN_FREEZE`. After a freeze, start/resume/freeze all refuse.
- `verify_freeze` recomputes every claim from the files: chain validity (optionally against a head you kept elsewhere and a minimum length), exactly one freeze record, manifest against the chain record, every card file against the manifest and its own hash, freeze bindings to each card, checkpoint list and per-checkpoint file hashes, weights hash, model card against the weights, taint inheritance, data still matching the card (when a data root is given), and — for a deterministic trainer with data — a full replay reproducing every checkpoint hash and the final weights. It never raises on a damaged run: the report says what failed, and `ok` is True only if every check passed.
- Invariants: no commitment, no run; one commitment, one use; checkpoints hashed before training continues; frozen means finished.

### 3.6 `lattice_data.py`
- Role: pick what a model learns from DKG lattice nodes, record exactly which nodes, keep the test set clean.
- Interface: `select(node_cards, scores, strategy, bands, max_per_repo, allow_tainted, holdout_permille, seed, kinds)`, `to_data_card(plan, name, root, license_class, sources, matrix)`, `ablation_plans(plan)`, `training_links(plan, data)`, `card_intact(card)`, `plan_intact(plan)`, `LatticeRefused`, `STRATEGIES`, `OK_CLASSES`.
- Behavior: a card whose dna_hash does not match its content is excluded, never repaired; identical content (same sha256) appears once wherever it came from, so a held-out file cannot also be a training file under another name; the held-out set follows the fixed hash rule of `data_card.split_ids`; tainted or non-permissive nodes are excluded unless the caller allows taint (then the plan says tainted); ordering uses caller-supplied scores (`foundation_first`, `frontier_first`, `flat` into bands) and a node with no score is kept, placed last, and marked "unscored" — never given an invented number; a per-repo cap stops one source dominating; every exclusion carries its reason. `to_data_card` produces a normal DATA card tagged with the plan hash and the lattice node ids; `training_links` emits TRAINED_ON / HELD_OUT records only for a card that really came from this plan. Selecting nothing left raises.
- Invariants: train and held-out never overlap; the selection is deterministic and re-derivable; nothing is read from disk by `select`.
- Honest limit: this chooses and records; it does not train. A curriculum order does not make a model better by itself — measure it (the ablation plans exist for that). Scores are only as good as whoever computed them. Duplicate detection is by exact content hash: a lightly edited copy is a different file.

### 3.7 `bundle.py`
- Role: seal a run's cards into one superproof and open them again with every card re-checked.
- Interface: `seal_bundle(card_list, key1, key2, signing_key, signing_public, dag, ledger, preset)`, `open_bundle(sealed, key1, key2)`, `BundleRefused`, `MAX_CARDS`.
- Behavior: sealing requires every card intact and one card per kind, or nothing is sealed; opening decrypts via superproof and re-checks every card, refusing a capsule that is not a card bundle. Any preset the superproof layer offers can be used.
- Honest limit: the bundle proves the cards are unchanged since sealing; it does not prove the cards are TRUE.

### 3.8 `goal.py`
- Role: the goal request bar — who is asking, in their own words, for what measurable result.
- Interface: `make_goal(request_text, requester_ref, assurance_level, conduct_card, success_tests, constraints, drafted_by, min_assurance)`, `draft_from_request(..., drafter, drafter_name)`, `is_ready(goal, goal_approval, conduct_card, conduct_approval, requester_public)`, `GoalRefused`, `MIN_ASSURANCE` (2), `VERDICTS`.
- Behavior: the request text is kept verbatim (plus its sha256) with the requester reference and their identity assurance level; a request below the minimum level is refused (training is a significant act). Success tests are concrete probes with an expected verdict and may name only rules the conduct card actually has. A drafting model may propose the structure; the result is marked `model:<name>` and the person must still sign it. `is_ready` returns the list of reasons it is NOT ready (empty = ready): requester signature by the named public key, human-signed conduct card whose hash matches, assurance level at or above the minimum.
- Invariants: a model never approves a goal; only a ready goal may be the aim of a training plan.

### 3.9 `conduct.py`
- Role: the signed conduct code, its reward arithmetic, and the run-time meta-audit.
- Interface: `make_conduct(name, rules, weights, erase_at, share_ratio)`, `Signer`, `approve(card, signer)`, `is_approved(card, approval)`, `signature_ok(public_hex, card, sig_hex)`, `episode_reward(conduct, task_rewards, violations)`, `team_reward(conduct, episodes)`, `MetaAuditor(conduct, approval, classifier, dag)` with `check(action_text, context)` returning a `Decision`, `ConductRefused`, `REQUIRED`, `STRICTNESS`, `SOURCE_KINDS`, `ID`, `DEFAULT_WEIGHTS`.
- Behavior: signatures are Ed25519 over a domain-separated prefix plus the card hash; only an intact card can be signed. Reward arithmetic: a violation costs its severity weight; a violation at or above `erase_at` erases the episode's task reward; in a team, if ANY member erases, every member's task reward is erased and each non-violating member carries `share_ratio` of the others' penalties. The meta-audit matches the action text against trigger words (whole words/phrases) or a plug-in classifier, returns the strictest requirement among the rules hit with ids and citations, and records every check to a hash chain when one is supplied.
- Fail-closed invariants: an unsigned or altered code governs nothing (`MetaAuditor` refuses to construct); a classifier that crashes or returns nonsense yields ESCALATE, never ALLOW; an unreadable action yields ESCALATE; a check that cannot be recorded hands the decision to a human (ESCALATE).
- Honest limits: the tiers and weights are DATA the owner (with an attorney) supplies — the SDK does not know any law; trigger-word matching is crude and both over- and under-fires; a reward for "no violations" can teach hiding them, so detection must be external.

### 3.10 `moral_data.py`
- Role: deterministic training examples from precedent records the owner supplies.
- Interface: `build_examples(records, conduct_card, goal_request, holdout_permille, seed)`, `write_dataset(examples, path)`, `private_card(data, dataset)`, `MoralDataRefused`, `MAX_ACT` (300).
- Behavior: from each record it emits template text — REFUSE examples naming the rule, citation, tier, consequence, who bears it, and the right alternative; PREFERENCE pairs (chosen = the reasoned refusal, rejected = the act stated at the same high level); ALLOW examples for legal acts (guarding against a model that refuses everything); and the reflection form of the meta-audit. Held-out records become probes and never appear in training examples. The module reads no law and scrapes nothing; output is deterministic, no model involved.
- Fail-closed rules: an illegal act without a severity tier or alternative is refused; records flagged `operational_detail` are refused; acts are capped at 300 characters of high-level description; credential-shaped text is refused anywhere in a record; duplicates (by id or by situation+act hash) are dropped.
- Invariants: the prohibited-act side is the sensitive side — preference-pair examples carry `private_negative`, and `private_card` marks the DATA card `private`, `tainted`, `public_ok` False whatever its licence class says: it can never be public.

### 3.11 `behavior_eval.py`
- Role: measure what a model actually does on held-out probes; never assume training worked.
- Interface: `evaluate(decide, probes, train_prompt_sha256)`, `passes(report, max_violation_upper, max_over_refusal, min_guard_probes)`, `wilson_upper(k, n, z)`, `prompt_hash(prompt)`, `EvalRefused`, `VERDICTS`.
- Behavior: every probe runs through `decide(prompt)` returning REFUSE, ESCALATE or ALLOW. Reports violation rate (expected guard, got ALLOW — or a crash) with a 95% Wilson UPPER bound so a small sample cannot look safe, over-refusal rate (expected ALLOW, got REFUSE), violations by severity tier, and per-probe failure detail. `passes` applies thresholds the caller sets and returns reasons it does NOT pass; nothing here decides what thresholds are acceptable.
- Fail-closed invariants: a probe whose prompt hash appears in the training set is REFUSED (a model that saw the answer proves nothing); a crash or nonsense verdict in `decide` counts as the worst answer, never as a pass.

### 3.12 `__main__.py`
- Role: one command line (JSON lines) for the whole governed path; what a UI calls.
- Interface: `python -m vega_train` with subcommands `prepare`, `start`, `resume`, `freeze`, `verify`, `status`, `action`; `main(argv)`, `emit(event, **kw)`.
- Behavior: `prepare` captures the environment, builds the data card, mints the PRE (lease capped at 900 seconds) and the plan, and writes `env.json data.json pre.json plan.json`. `start`/`resume` need `--ledger`, a `--commitment` file the human signed in their own terminal, and `--data-root`. `verify` accepts `--head`, `--min-length`, `--data-root`, `--replay`. `action` prints the exact action string a human must sign. Every command prints JSON-line events; every failure is one `refused` event with a reason and a non-zero exit, never a traceback a UI cannot read.
- Invariant: this program never signs and never mints a commitment.

## 4. Guarantees and non-guarantees

### 4.1 Guarantees (each demonstrated by the named module and test)

| Guarantee | Module | Test |
|---|---|---|
| No run starts without a human commitment that checks exactly True for exactly this plan's action | `run.py` | `redteam_1_run_integrity.py` |
| A spent commitment cannot be replayed — not on this run, not on any run under the same parent folder | `run.py` (shared burn chain) | `redteam_1_run_integrity.py` |
| Data is byte-verified against the DATA card before the commitment is spent; changed data stops the run without spending it | `data_card.py`, `run.py` | `redteam_1_run_integrity.py` |
| A file holding a credential-shaped value is refused for training; the refusal names the file, never the value | `data_card.py` | `redteam_2_cards_freeze_verify.py` |
| The card holds hashes and counts, never content, machine paths, secrets or user names | `data_card.py`, `env_card.py` | `redteam_2_cards_freeze_verify.py` |
| Every checkpoint is hashed and recorded before the next chunk trains; a tampered or missing checkpoint is refused before anything is trusted | `run.py` | `test_train.py`, `redteam_1_run_integrity.py` |
| Any failure halts with a recorded reason; a halted run resumes only under a NEW commitment for exactly that checkpoint | `run.py` | `redteam_1_run_integrity.py` |
| After a freeze, start, resume and freeze all refuse | `run.py` | `test_train.py` |
| `verify_freeze` recomputes every claim, never raises on garbage, and reports every way of editing a frozen run | `run.py` | `redteam_2_cards_freeze_verify.py` |
| An edited record breaks the chain; a cut tail or attacker-rebuilt run fails against a head kept elsewhere | `run.py` + custody chain | `redteam_2_cards_freeze_verify.py` |
| A replay from the plan reproduces checkpoints and weights — for a deterministic trainer, with the data; replay with other data does not pass | `run.py`, `trainers.py` | `test_train.py`, `redteam_2_cards_freeze_verify.py` |
| Taint follows the data and the base model to the model and provenance cards; dropping the taint mark is reported | `run.py`, `data_card.py` | `redteam_2_cards_freeze_verify.py` |
| A checkpoint in a pickle never executes and is never loaded; malformed checkpoint shapes are refused | `trainers.py` | `redteam_1_run_integrity.py` |
| Two processes cannot train the same run at once; a second plan cannot share a run folder | `run.py` | `redteam_1_run_integrity.py` |
| Train and held-out never overlap; identical content is trained once and cannot leak into the test set; the rule is re-derivable | `lattice_data.py` | `test_lattice_data.py` |
| Altered lattice cards are excluded, never repaired; unscored nodes are kept last and said so, never given invented numbers | `lattice_data.py` | `test_lattice_data.py` |
| An unsigned or altered conduct code governs nothing; a broken classifier or unrecordable check yields ESCALATE, never ALLOW | `conduct.py` | `test_conduct.py` |
| One serious violation erases the episode's task reward; in a team, everyone's — and the others share the penalty | `conduct.py` | `test_conduct.py` |
| A goal is ready only when the requester signed it over a human-signed conduct code at sufficient identity assurance; a model may draft but never approve | `goal.py` | `test_conduct.py` |
| Held-out records never reach training examples; the private-negative dataset's card can never be public | `moral_data.py` | `test_conduct.py` |
| Probes seen in training are refused; a crash or nonsense answer counts as the worst case; a small sample cannot look safe (Wilson upper bound) | `behavior_eval.py` | `test_conduct.py` |
| A bundle seals only intact cards, one per kind, and re-checks every card on open; wrong keys fail | `bundle.py` | `test_bundle.py` |
| The CLI reports every failure as one refusal event and a non-zero exit | `__main__.py` | `test_cli.py` |

### 4.2 Non-guarantees

| Not guaranteed | Why |
|---|---|
| That the model is good, safe, or free of memorised data | A freeze proves provenance and approval, nothing about quality |
| Reproducibility across machines or on GPUs | Determinism is per-machine; GPU kernels and thread scheduling differ; GPU/LLM trainers are plug-ins, not included |
| That the environment card makes a run reproducible | It describes the environment, that is all |
| That the cards are TRUE | Hashes and bundles prove integrity; truth comes from the run's checks and the human commitment |
| Any legal conclusion | Licence classes, taint and conduct tiers are owner-supplied policy data, not legal advice |
| That conduct training produces understanding | It shifts probabilities; external gates stay in force; violations must be detected by outside audit |
| Curriculum quality | Lattice scores come from the caller; this kit never invents them |
| OS-level containment | The trainer runs in this process with this process's rights |

## 5. Failure modes (from the tests)

- **Missing, loose or crashing commitment checker** — the run never starts (`redteam_1_run_integrity.py`).
- **Commitment for another action or plan; already-spent commitment** — refused; the spend is on the record (`redteam_1_run_integrity.py`).
- **PRE that does not permit training, or an expired PRE** — the run stops (`redteam_1_run_integrity.py`).
- **Cards edited or mismatched with the plan** — refused at construction (`redteam_1_run_integrity.py`).
- **Insane plan values, out-of-range hyper-parameters, non-numeric rows** — refused, not crashed (`redteam_1_run_integrity.py`).
- **NaN/infinite loss, misreported step, trainer exception, wall-clock or disk budget, unwritable checkpoint** — `TRAIN_HALT` with the reason; earlier checkpoints stay intact (`redteam_1_run_integrity.py`).
- **Tampered, missing or swapped checkpoint on resume** — refused before anything is trusted, and the commitment is NOT spent (`redteam_1_run_integrity.py`).
- **Pickle checkpoint** — never executes, never loads (`redteam_1_run_integrity.py`).
- **Interrupted process** — leaves a resumable run that needs a new commitment (`redteam_1_run_integrity.py`).
- **Two processes, one run** — the second is refused by the OS lock (`redteam_1_run_integrity.py`).
- **Data changed after the card** — the run stops without spending the commitment (`redteam_1_run_integrity.py`).
- **Credential in a data file; traversal, absolute, duplicate or junction-escaping entries; forged or resealed card; edited or swapped files at load** — refused by the data card (`redteam_2_cards_freeze_verify.py`).
- **Hostile `nvidia-smi` output, missing/failing/flooding `nvidia-smi`, secret in an allowed setting** — recorded safely or ignored, never fatal, never recorded (`redteam_2_cards_freeze_verify.py`).
- **Any edit to a frozen run — record, card, checkpoint, weights, manifest, dropped taint mark, whole-run rebuild** — reported by `verify_freeze`; a remembered head catches a cut tail (`redteam_2_cards_freeze_verify.py`).
- **Garbage or missing inputs to the verifier** — a report, never an exception (`redteam_2_cards_freeze_verify.py`).
- **Replay without data, with other data, or with a non-deterministic trainer** — fails or is declared not replayable (`redteam_2_cards_freeze_verify.py`).
- **Altered lattice card, wrong kind, repeated id, tainted node, repo cap, empty selection, altered plan, links for a foreign card** — excluded with a reason or refused (`test_lattice_data.py`, `mutations_lattice.py`).
- **Unsigned/altered conduct code, broken classifier, unrecordable check, malformed rules or weights** — ESCALATE or refusal, never ALLOW (`test_conduct.py`, `mutations_conduct.py`).
- **Low identity assurance, unsigned goal, mismatched conduct hash** — the goal is not ready (`test_conduct.py`).
- **Operational-detail record, credential-shaped text, duplicate records** — refused or dropped (`test_conduct.py`).
- **Probe seen in training; crashing `decide`; too-few guard probes** — refused, worst case, or not passing (`test_conduct.py`).
- **Altered card, repeated kind, wrong keys, foreign capsule in a bundle** — nothing sealed / refused (`test_bundle.py`).
- **Bad CLI input** — one refusal event, non-zero exit (`test_cli.py`).

## 6. Dependencies and layer rules

Registry entry (verbatim): layer **8 learn** — `vega-train-sdk`, owns "governed model training and freeze cards", requires `[vega-custody-sdk, vega-governance-jacket-sdk, vega-harness-sdk, vega-pareah-sdk, vega-dkg-lattice-sdk, vega-superproof-sdk]`; status PARTIAL — "GPU and LLM trainers are plug-ins, not included".

Dependencies point down layers only, never up or sideways:

| Dependency | Layer | What this kit uses it for (real imports) |
|---|---|---|
| vega-custody-sdk | 0 record | record chains and ledger (`vega_custody.ledger`), safe paths and capped reads (`safepaths`), bounded processes (`proc`), credential detection (`dag.dag.contains_credential`), capsules (`capsule`) |
| vega-governance-jacket-sdk | 1 authority | the human's single-use signed commitment (signed in the person's own terminal; checked and burned through harness records) |
| vega-superproof-sdk | 2 seal | the bundle seal (`vega_superproof.seal.superproof`) |
| vega-pareah-sdk | 3 adapt | PRE lease checks and the model / runtime / server cards (`vega_pareah.cards`) |
| vega-harness-sdk | 6 conduct | canonical JSON, record chains, commitment burn and ledger check (`vega_harness.records`), licence classes and the use-gate matrix (`vega_harness.use_gate`) |
| vega-dkg-lattice-sdk | 7 knowledge | lattice node cards and their integrity hash (`vega_dkg.cards`) |

No SDK in the public family requires this one; it is the family's most-connected consumer.

## 7. Conformance

A third party verifies this specification by running `tests/`: put this repo's `src/` and the six dependency repos' `src/` directories on `PYTHONPATH` and run each test file. STATUS.md records 115 tests in 7 files, all passing; REVIEW.md's automated run reports 67 tests ok (4 test files plus both red-team files) with no high, medium or low findings.

Passing means: the guarantees in §4.1 hold for the published reference implementation on the test inputs — commitments gate runs, records and hashes bind every claim, refusals are fail-closed, and the verifier recomputes everything from files.

Passing does NOT mean: the code is hardened for production; the cryptography has expert review (it has not); any model trained through this kit is good, safe or lawful; GPU or LLM training works out of the box (those trainers are plug-ins, not included); or that any threshold, licence class or conduct tier chosen by a caller is correct — those are owner-supplied policy data.
