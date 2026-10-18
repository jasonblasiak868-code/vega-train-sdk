"""run.py - a governed training run: plan, human-committed start, DAG-frozen checkpoints, hard stop, resume, freeze, and an independent verifier.

The life of a run (every step is one record on a custody chain in the run folder):
  TRAIN_PLAN        the plan card (trainer, hyper-parameters, seed, steps, budgets) bound to the environment card, the data card and the PRE card that permits training
  COMMITMENT_USED   a human's commitment for exactly THIS plan was spent (single use). No commitment, no run.
  TRAIN_START
  TRAIN_CHECKPOINT  every checkpoint_every steps: the state is written, read back, hashed, and the hash recorded BEFORE the next chunk runs
  TRAIN_HALT        any failure (loss not finite, a budget used up, a trainer that misreports its step, a checkpoint that cannot be written): the run stops and says why.
                    A halted run does not resume by itself: resume() needs a NEW commitment for exactly this plan and this last checkpoint, and the checkpoint is re-hashed first.
  TRAIN_DONE        the final weights file and its hash
  TRAIN_FREEZE      the freeze card: environment, data, plan, every checkpoint hash and the final weights hash, sealed, then the card bundle (MANIFEST) is written beside it.
After a freeze nothing in the run folder is changed by this code: start, resume and freeze all refuse.

verify_freeze() recomputes every claim from the files: chain valid (against a head you kept elsewhere, if you pass one), every card matches the manifest, every checkpoint matches
its record, the weights match the model card, the data still matches the data card, and, for a deterministic trainer, a REPLAY from the plan reproduces the same checkpoints and weights.
HONEST LIMITS: a freeze proves WHICH bytes produced WHICH weights under WHICH plan and that a human approved that plan. It does not prove the model is good, safe or free of memorised
data. A replay proves reproducibility only for a deterministic trainer on the same machine setup. The trainer runs in this process with this process's rights."""
import json
import math
import os
import time

from vega_custody import safepaths as sp
from vega_harness.records import Duplicate, RecordChain, RecordRefused, burn_commitment
from vega_pareah.cards import model_card, pre as pre_mod, runtime_card, server_card
from vega_train import cards, data_card, env_card, trainers

KIND = "TRAIN"
MAX_STEPS = 10_000_000
MAX_WALL = 7 * 86400
MAX_DISK = 1 << 40
PARAREAH_CLASS = {"PERMISSIVE": "permissive", "OWNED": "permissive", "WEAK_COPYLEFT": "weak_copyleft", "STRONG_COPYLEFT": "strong_copyleft",
                  "NON_COMMERCIAL_OR_RESTRICTED": "restricted", "NO_LICENSE_FILE": "no_license", "UNKNOWN": "no_license"}


class TrainRefused(ValueError):
    pass


def make_plan(name, env, data, pre_card, trainer, hyper=None, seed=0, steps=100, checkpoint_every=10, max_wall_seconds=3600, max_disk_bytes=1 << 30, base_model=None) -> dict:
    sp.safe_name(name)
    for label, v, lo, hi in (("steps", steps, 1, MAX_STEPS), ("checkpoint_every", checkpoint_every, 1, steps if isinstance(steps, int) else 1), ("max_wall_seconds", max_wall_seconds, 1, MAX_WALL),
                             ("max_disk_bytes", max_disk_bytes, 1, MAX_DISK)):
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise TrainRefused(f"{label} must be a whole number from {lo} to {hi}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise TrainRefused("seed is a whole number from 0 to 2^63-1")
    if not isinstance(trainer, str) or not isinstance(hyper or {}, dict):
        raise TrainRefused("trainer is a name and hyper is an object")
    for label, c, kind in (("environment", env, "ENV"), ("data", data, "DATA"), ("PRE", pre_card, "PRE")):
        if not cards.intact(c, kind):
            raise TrainRefused(f"the {label} card is altered or not a {kind} card")
    if base_model is not None and (not isinstance(base_model, dict) or not isinstance(base_model.get("id"), str)):
        raise TrainRefused("base_model is None or an object with an id (and its license_class)")
    card = {"kind": "TRAIN_PLAN", "name": name, "trainer": trainer, "hyper": dict(hyper or {}), "seed": seed, "steps": steps, "checkpoint_every": checkpoint_every,
            "max_wall_seconds": max_wall_seconds, "max_disk_bytes": max_disk_bytes, "base_model": base_model,
            "env_card_hash": env["card_hash"], "data_card_hash": data["card_hash"], "pre_hash": pre_card["card_hash"]}
    return cards.seal(card)


def start_action(plan):
    return "train_run:" + plan["card_hash"]


def resume_action(plan, last_sha):
    return "train_resume:%s:%s" % (plan["card_hash"], last_sha)


def _atomic_write_bytes(path, data):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        if fh.read() != data:
            raise OSError("read-back mismatch")


class _RunLock:
    """A non-blocking OS lock on the run folder for the whole of a start or resume, so two processes cannot train the same run at once (a double click, a second terminal).
    The OS releases it if the process dies, so a crash never leaves a stale lock that needs deleting."""

    def __init__(self, run_dir):
        self.path, self.fd = os.path.join(run_dir, "RUN.runlock"), None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self.fd)
            raise TrainRefused("another process is already running this training run") from exc
        return self

    def __exit__(self, *a):
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self.fd, 0, 0)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


class TrainingRun:
    def __init__(self, run_dir, plan, env, data, pre_card, data_root, commitment_check=None, burn=None, trainer=None, clock=time.time, on_event=None):
        for label, c, kind in (("plan", plan, "TRAIN_PLAN"), ("environment", env, "ENV"), ("data", data, "DATA"), ("PRE", pre_card, "PRE")):
            if not cards.intact(c, kind):
                raise TrainRefused(f"the {label} card is altered or not a {kind} card")
        if plan["env_card_hash"] != env["card_hash"] or plan["data_card_hash"] != data["card_hash"] or plan["pre_hash"] != pre_card["card_hash"]:
            raise TrainRefused("the plan was made for different environment, data or PRE cards")
        self.run_dir = os.path.abspath(run_dir)
        self.plan, self.env, self.data, self.pre, self.data_root = plan, env, data, pre_card, data_root
        self.commitment_check, self.clock, self.on_event = commitment_check, clock, on_event
        self.trainer = trainer if trainer is not None else trainers.make(plan["trainer"])
        os.makedirs(self.run_dir, exist_ok=True)
        self.chain = RecordChain(os.path.join(self.run_dir, "RUN.jsonl"))
        # one spent-commitment record shared by every run under the same parent folder, so a signature spent in one run folder cannot be offered to another
        self.burn = burn or RecordChain(os.path.join(os.path.dirname(self.run_dir), "COMMITMENTS_USED.jsonl"))
        rows = self.chain.read()
        planned = [r for r in rows if r.get("kind") == "TRAIN_PLAN"]
        if planned and planned[0].get("plan_hash") != plan["card_hash"]:
            raise TrainRefused("this run folder belongs to a different plan")
        if not planned:
            self.chain.append("TRAIN_PLAN", {"plan_hash": plan["card_hash"], "env_card_hash": env["card_hash"], "data_card_hash": data["card_hash"], "pre_hash": pre_card["card_hash"],
                                             "name": plan["name"]})

    def _emit(self, event, **kw):
        """Progress for a UI. A broken listener never stops or changes a run."""
        if self.on_event is not None:
            try:
                self.on_event({"event": event, **kw})
            except Exception:  # reviewed: a listener that fails must not be able to halt, alter or fake a training run
                pass

    # ------------------------------------------------------------------ state, derived from the record
    def rows(self):
        return self.chain.read()

    def checkpoints(self, rows=None):
        return [r for r in (rows or self.rows()) if r.get("kind") == "TRAIN_CHECKPOINT"]

    def state(self, rows=None):
        rows = rows or self.rows()
        kinds = [r.get("kind") for r in rows]
        if "TRAIN_FREEZE" in kinds:
            return "frozen"
        if "TRAIN_DONE" in kinds:
            return "done"
        last = [k for k in kinds if k in ("TRAIN_START", "TRAIN_RESUME", "TRAIN_HALT")]
        if not last:
            return "new"
        return "halted" if last[-1] == "TRAIN_HALT" else "running"

    def ckpt_path(self, step):
        return os.path.join(self.run_dir, "checkpoints", "ckpt_%08d.bin" % step)

    # ------------------------------------------------------------------ gates
    def _gate(self, commitment, action):
        for res in ("model/" + self.plan["name"], "dataset/" + self.data["name"]):
            if not pre_mod.permits(self.pre, "train", res):
                raise TrainRefused(f"the PRE does not permit 'train' on {res!r} (or it expired)")
        ok = False
        if self.commitment_check is not None and commitment is not None:
            try:
                ok = self.commitment_check(commitment, action) is True
            except Exception:  # reviewed: a checker that crashes does not accept the commitment
                ok = False
        if not ok:
            raise TrainRefused("training needs a human commitment for exactly this action: " + action)
        try:
            burn_commitment(self.burn, commitment, action)
        except Duplicate as exc:
            raise TrainRefused("this commitment was already spent") from exc
        except RecordRefused as exc:
            raise TrainRefused(f"the commitment could not be recorded as spent: {exc}") from exc

    # ------------------------------------------------------------------ the loop
    def _halt(self, reason, last):
        self.chain.append("TRAIN_HALT", {"reason": reason, "last_checkpoint_step": last["step"] if last else None, "last_checkpoint_sha256": last["sha256"] if last else None})
        self._emit("halted", reason=reason, last_checkpoint_step=last["step"] if last else None)
        return {"status": "halted", "reason": reason, "last_checkpoint_step": last["step"] if last else None}

    def _checkpoint(self, step, loss):
        blob = self.trainer.state_bytes()
        path = self.ckpt_path(step)
        _atomic_write_bytes(path, blob)
        rec = self.chain.append("TRAIN_CHECKPOINT", {"step": step, "sha256": cards.sha256_hex(blob), "bytes": len(blob), "loss": repr(float(loss))})
        self._emit("checkpoint", step=step, of=self.plan["steps"], sha256=rec["sha256"], loss=float(loss))
        return rec

    def _disk_used(self):
        return sum(r["bytes"] for r in self.checkpoints())

    def _loop(self, step, last, t0):
        p = self.plan
        losses = []
        while step < p["steps"]:
            if self.clock() - t0 > p["max_wall_seconds"]:
                return self._halt("wall-clock budget of %d seconds used up" % p["max_wall_seconds"], last)
            n = min(p["checkpoint_every"] - step % p["checkpoint_every"], p["steps"] - step)
            try:
                res = self.trainer.step(n)
                loss = float(res["loss"])
            except Exception as exc:  # reviewed: any trainer failure halts the run with the reason; it is never swallowed and never retried silently
                return self._halt("trainer raised %s" % type(exc).__name__, last)
            if not math.isfinite(loss):
                return self._halt("loss is not finite at step %s" % res.get("step"), last)
            if res.get("step") != step + n:
                return self._halt("trainer reported step %r, expected %d" % (res.get("step"), step + n), last)
            step += n
            losses.append(loss)
            try:
                rec = self._checkpoint(step, loss)
            except (OSError, RecordRefused) as exc:
                return self._halt("checkpoint could not be written and recorded: %s" % type(exc).__name__, last)
            last = {"step": step, "sha256": rec["sha256"]}
            if self._disk_used() > p["max_disk_bytes"]:
                return self._halt("disk budget of %d bytes used up by checkpoints" % p["max_disk_bytes"], last)
        weights = self.trainer.weights_bytes()
        wpath = os.path.join(self.run_dir, "weights.bin")
        try:
            _atomic_write_bytes(wpath, weights)
        except OSError as exc:
            return self._halt("final weights could not be written: %s" % type(exc).__name__, last)
        curve = [float(r["loss"]) for r in self.checkpoints()]
        self.chain.append("TRAIN_DONE", {"final_step": step, "weights_sha256": cards.sha256_hex(weights), "weights_bytes": len(weights), "final_loss": repr(curve[-1]),
                                         "metrics_sha256": cards.sha256_hex(repr(curve).encode("utf-8"))})
        self._emit("done", final_step=step, weights_sha256=cards.sha256_hex(weights), final_loss=curve[-1])
        return {"status": "done", "final_step": step, "weights_sha256": cards.sha256_hex(weights), "final_loss": curve[-1]}

    def _init_trainer(self, files):
        config = {**self.plan["hyper"], "seed": self.plan["seed"]}
        try:
            self.trainer.init(config, files)
        except (trainers.TrainerRefused, ValueError, TypeError, KeyError) as exc:
            raise TrainRefused(f"the trainer refused its configuration or data: {exc}") from exc

    def _verified_files(self):
        try:
            return data_card.load_verified(self.data, self.data_root)
        except data_card.DataRefused as exc:
            raise TrainRefused(f"the data is not what the data card says: {exc}") from exc

    def start(self, commitment, observed_env=None, require_same_env=False):
        with _RunLock(self.run_dir):
            rows = self.rows()
            if self.state(rows) != "new":
                raise TrainRefused("this run is %s: start only works on a new run" % self.state(rows))
            drift = env_card.drift(self.env, observed_env) if observed_env is not None else []
            if require_same_env and observed_env is not None and drift:
                raise TrainRefused("the environment differs from the one on the card: " + "; ".join(drift[:5]))
            files = self._verified_files()                                           # checked BEFORE the commitment is spent
            for res in ("model/" + self.plan["name"], "dataset/" + self.data["name"]):
                if not pre_mod.permits(self.pre, "train", res):
                    raise TrainRefused(f"the PRE does not permit 'train' on {res!r} (or it expired)")
            self._gate(commitment, start_action(self.plan))
            self._init_trainer(files)
            self.chain.append("TRAIN_START", {"plan_hash": self.plan["card_hash"], "env_drift": len(drift), "env_drift_sha256": cards.sha256_hex(repr(drift).encode("utf-8"))})
            return self._loop(0, None, self.clock())

    def resume(self, commitment):
        with _RunLock(self.run_dir):
            rows = self.rows()
            st = self.state(rows)
            if st not in ("halted", "running"):
                raise TrainRefused("this run is %s: only a halted run can resume" % st)
            cps = self.checkpoints(rows)
            if not cps:
                raise TrainRefused("there is no checkpoint to resume from: start a new run")
            last = cps[-1]
            blob = self._read_checkpoint(last)                                       # re-hashed before anything is trusted
            files = self._verified_files()
            self._gate(commitment, resume_action(self.plan, last["sha256"]))
            self._init_trainer(files)
            try:
                self.trainer.load_state(blob)
            except (trainers.TrainerRefused, ValueError, KeyError, TypeError) as exc:
                raise TrainRefused(f"the checkpoint does not fit the trainer: {exc}") from exc
            self.chain.append("TRAIN_RESUME", {"from_step": last["step"], "checkpoint_sha256": last["sha256"]})
            return self._loop(last["step"], {"step": last["step"], "sha256": last["sha256"]}, self.clock())

    def _read_checkpoint(self, rec):
        try:
            blob = sp.read_capped(sp.contained(self.run_dir, "checkpoints", "ckpt_%08d.bin" % rec["step"]), 1 << 36)
        except (OSError, sp.PathRefused) as exc:
            raise TrainRefused(f"checkpoint {rec['step']} cannot be read: {exc}") from exc
        if cards.sha256_hex(blob) != rec["sha256"]:
            raise TrainRefused(f"checkpoint {rec['step']} does not match its record: it was altered after it was written")
        return blob

    # ------------------------------------------------------------------ freeze
    def freeze(self, runtime_license="unknown", model_id=None) -> dict:
        rows = self.rows()
        st = self.state(rows)
        if st != "done":
            raise TrainRefused("only a finished run can be frozen; this run is %s" % st)
        cps = self.checkpoints(rows)
        for r in cps:
            self._read_checkpoint(r)                                                  # every checkpoint still matches its record
        done = [r for r in rows if r.get("kind") == "TRAIN_DONE"][-1]
        wpath = sp.contained(self.run_dir, "weights.bin")
        weights = sp.read_capped(wpath, 1 << 36)
        if cards.sha256_hex(weights) != done["weights_sha256"]:
            raise TrainRefused("weights.bin does not match the record of the finished run")
        data_card.load_verified(self.data, self.data_root)                            # the data still is what the card says
        cls = data_card.strictest([self.data["license_class"]] + ([self.plan["base_model"].get("license_class", "UNKNOWN")] if self.plan["base_model"] else []))
        tainted = bool(self.data["tainted"]) or cls not in ("PERMISSIVE", data_card.OWNED)
        mid = model_id or self.plan["name"]
        mcard = model_card.mint_model_card(mid, "local-vega-train", PARAREAH_CLASS.get(cls, "no_license"), weights_sha256=done["weights_sha256"])
        trainer_file = _trainer_source(self.trainer)
        rcard = runtime_card.mint_runtime_card("trainer:" + self.plan["trainer"], cards.sha256_hex(trainer_file), ["train"],
                                               {"max_wall_seconds": self.plan["max_wall_seconds"], "max_disk_bytes": self.plan["max_disk_bytes"], "steps": self.plan["steps"]}, runtime_license)
        scard = server_card.mint_server_card("local-trainer", "local", ["train"], on_device=True)
        prov = cards.seal({"kind": "MODEL_PROVENANCE", "model_card_hash": mcard["card_hash"], "data_card_hash": self.data["card_hash"], "plan_hash": self.plan["card_hash"],
                           "license_class": cls, "tainted": tainted, "public_ok": not tainted, "base_model": self.plan["base_model"],
                           "note": "provenance of the weights, not a statement that the model is good or safe"})
        freeze_card = cards.seal({"kind": "TRAIN_FREEZE", "plan_hash": self.plan["card_hash"], "env_card_hash": self.env["card_hash"], "data_card_hash": self.data["card_hash"],
                                  "pre_hash": self.pre["card_hash"], "trainer": self.plan["trainer"], "deterministic": bool(getattr(self.trainer, "deterministic", False)),
                                  "checkpoints": [{"step": r["step"], "sha256": r["sha256"]} for r in cps], "weights_sha256": done["weights_sha256"], "final_loss": done["final_loss"],
                                  "metrics_sha256": done["metrics_sha256"], "model_card_hash": mcard["card_hash"], "provenance_hash": prov["card_hash"],
                                  "chain_length_before_freeze": len(rows), "chain_head_before_freeze": rows[-1]["hash"]})
        bundle = {"env.json": self.env, "data.json": self.data, "plan.json": self.plan, "pre.json": self.pre, "runtime.json": rcard, "server.json": scard, "model.json": mcard,
                  "provenance.json": prov, "freeze.json": freeze_card}
        manifest = {}
        for fname, c in bundle.items():
            text = cards.canon(c)
            sp.atomic_write(os.path.join(self.run_dir, "cards", fname), text)
            manifest[fname] = cards.sha256_hex(text.encode("utf-8"))
        mtext = cards.canon({"cards": manifest, "freeze_hash": freeze_card["card_hash"]})
        sp.atomic_write(os.path.join(self.run_dir, "cards", "MANIFEST.json"), mtext)
        self._emit("freezing", freeze_hash=freeze_card["card_hash"])
        self.chain.append("TRAIN_FREEZE", {"freeze_hash": freeze_card["card_hash"], "manifest_sha256": cards.sha256_hex(mtext.encode("utf-8")), "weights_sha256": done["weights_sha256"]})
        return freeze_card


def _trainer_source(trainer):
    import inspect
    try:
        return open(inspect.getsourcefile(type(trainer)), "rb").read()
    except (TypeError, OSError):
        return type(trainer).__qualname__.encode("utf-8")


# ---------------------------------------------------------------------------------------------- the independent verifier
def verify_freeze(run_dir, expected_head=None, min_length=None, data_root=None, replay=False, trainer_factory=None, clock=time.time) -> dict:
    """Recompute every claim of a frozen run from the files. Never raises on a damaged run: the report says what failed. ok is True only if every check passed."""
    checks, problems = {}, []

    def check(name, cond, why=""):
        checks[name] = bool(cond)
        if not cond:
            problems.append(name + (": " + why if why else ""))
        return bool(cond)

    run_dir = os.path.abspath(run_dir)
    chain = RecordChain(os.path.join(run_dir, "RUN.jsonl"))
    ok, probs = chain.verify(expected_head=expected_head, min_length=min_length)
    check("chain_valid", ok, "; ".join(probs or [])[:300])
    if not ok:
        return {"ok": False, "checks": checks, "problems": problems}
    rows = chain.read()
    frz = [r for r in rows if r.get("kind") == "TRAIN_FREEZE"]
    if not check("frozen", len(frz) == 1, "no freeze record" if not frz else "more than one freeze record"):
        return {"ok": False, "checks": checks, "problems": problems}
    try:
        def load(name):
            return json.loads(sp.read_capped(sp.contained(run_dir, "cards", name), 1 << 30).decode("utf-8"))
        manifest = load("MANIFEST.json")
        mtext = cards.canon(manifest)
        check("manifest_matches_chain", cards.sha256_hex(mtext.encode("utf-8")) == frz[0]["manifest_sha256"])
        loaded = {}
        for fname, want in manifest["cards"].items():
            c = load(fname)
            loaded[fname] = c
            check("card_file:" + fname, cards.sha256_hex(cards.canon(c).encode("utf-8")) == want and cards.intact(c), "altered after the freeze")
        fz = loaded["freeze.json"]
        check("freeze_hash_matches_chain", fz["card_hash"] == frz[0]["freeze_hash"] == manifest["freeze_hash"])
        for key, fname in (("env_card_hash", "env.json"), ("data_card_hash", "data.json"), ("plan_hash", "plan.json"), ("pre_hash", "pre.json"), ("model_card_hash", "model.json"),
                           ("provenance_hash", "provenance.json")):
            check("freeze_binds:" + fname, fz[key] == loaded[fname]["card_hash"])
        cps = [r for r in rows if r.get("kind") == "TRAIN_CHECKPOINT"]
        check("checkpoints_match_freeze", [{"step": r["step"], "sha256": r["sha256"]} for r in cps] == fz["checkpoints"])
        check("checkpoint_steps_increase", all(a["step"] < b["step"] for a, b in zip(cps, cps[1:])))
        for r in cps:
            try:
                blob = sp.read_capped(sp.contained(run_dir, "checkpoints", "ckpt_%08d.bin" % r["step"]), 1 << 36)
                good = cards.sha256_hex(blob) == r["sha256"]
            except (OSError, sp.PathRefused):
                good = False
            check("checkpoint:%d" % r["step"], good, "missing or altered")
        weights = sp.read_capped(sp.contained(run_dir, "weights.bin"), 1 << 36)
        wsha = cards.sha256_hex(weights)
        check("weights_match_freeze", wsha == fz["weights_sha256"])
        check("model_card_matches_weights", model_card.check_model(loaded["model.json"], wsha))
        check("provenance_taint_follows_data", loaded["provenance.json"]["tainted"] or not loaded["data.json"]["tainted"])
        if data_root is not None:
            try:
                data_card.load_verified(loaded["data.json"], data_root)
                check("data_still_matches_card", True)
            except data_card.DataRefused as exc:
                check("data_still_matches_card", False, str(exc)[:200])
        if replay:
            if not fz["deterministic"]:
                checks["replay"] = "not replayable: the trainer is not deterministic"
            elif data_root is None:
                check("replay", False, "replay needs data_root")
            else:
                plan = loaded["plan.json"]
                tr = (trainer_factory or trainers.make)(plan["trainer"])
                files = data_card.load_verified(loaded["data.json"], data_root)
                tr.init({**plan["hyper"], "seed": plan["seed"]}, files)
                same, step = True, 0
                while step < plan["steps"]:
                    n = min(plan["checkpoint_every"] - step % plan["checkpoint_every"], plan["steps"] - step)
                    tr.step(n)
                    step += n
                    rec = next((r for r in cps if r["step"] == step), None)
                    same = same and rec is not None and cards.sha256_hex(tr.state_bytes()) == rec["sha256"]
                check("replay_reproduces_checkpoints", same)
                check("replay_reproduces_weights", cards.sha256_hex(tr.weights_bytes()) == fz["weights_sha256"])
    except (OSError, ValueError, KeyError, TypeError, sp.PathRefused, trainers.TrainerRefused, data_card.DataRefused) as exc:
        check("freeze_readable", False, "%s: %s" % (type(exc).__name__, str(exc)[:200]))
    return {"ok": all(v is True for v in checks.values() if isinstance(v, bool)) and not problems, "checks": checks, "problems": problems}
