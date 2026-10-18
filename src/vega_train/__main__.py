"""python -m vega_train - one command line for the whole governed training path. This is what the Vega OS training tab calls; every command prints JSON lines so a UI can read it.

  prepare SPEC.json CARDS_DIR        capture the environment, build the data card, mint the PRE and the plan; writes env.json data.json pre.json plan.json
  start RUN_DIR CARDS_DIR --ledger LEDGER --commitment FILE --data-root DIR     start a run (needs a human commitment for exactly this plan)
  resume RUN_DIR CARDS_DIR --ledger LEDGER --commitment FILE --data-root DIR    resume a halted run (needs a NEW commitment for this checkpoint)
  freeze RUN_DIR CARDS_DIR --data-root DIR      freeze a finished run and write the card bundle
  verify RUN_DIR [--head HASH] [--min-length N] [--data-root DIR] [--replay]    check a frozen run from its files
  status RUN_DIR CARDS_DIR           where the run is, from its record
  action RUN_DIR CARDS_DIR [--resume]    print the exact action name a human must sign for this plan (or for resuming)

SPEC.json: {"name", "trainer", "hyper", "seed", "steps", "checkpoint_every", "max_wall_seconds", "max_disk_bytes", "data": {"root", "files", "license_class", "sources"}, "matrix": "path or absent",
            "pre_lease_seconds" (at most 900: the human-authority lease cap; the PRE is checked when a run starts or resumes)}
The human signs in their own terminal (the jacket's human_cli) and passes the resulting commitment file; this program never signs and never mints one."""
import argparse
import json
import os
import sys

from vega_custody.ledger.backend_local import LocalBackend
from vega_custody.ledger.ledger import Ledger
from vega_harness import use_gate as ug
from vega_harness.records import make_ledger_check
from vega_pareah.cards import pre as pre_mod
from vega_train import cards, data_card, env_card, run


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}, sort_keys=True, allow_nan=False), flush=True)


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(cards.canon(obj))


def prepare(spec_path, cards_dir):
    spec = _load(spec_path)
    os.makedirs(cards_dir, exist_ok=True)
    matrix = ug.load_matrix(spec.get("matrix")) if spec.get("matrix") else {}
    d = spec["data"]
    dcard = data_card.build(spec["name"] + "-data", d["root"], d["files"], d.get("license_class", data_card.OWNED), d.get("sources", []), matrix)
    ecard = env_card.capture()
    pcard = pre_mod.mint_pre("vega-train", ["train"], ["model/" + spec["name"], "dataset/" + dcard["name"]], int(spec.get("pre_lease_seconds", 900)))
    plan = run.make_plan(spec["name"], ecard, dcard, pcard, spec["trainer"], spec.get("hyper"), spec.get("seed", 0), spec["steps"], spec["checkpoint_every"],
                         spec.get("max_wall_seconds", 3600), spec.get("max_disk_bytes", 1 << 30), spec.get("base_model"))
    for name, c in (("env", ecard), ("data", dcard), ("pre", pcard), ("plan", plan)):
        _write(os.path.join(cards_dir, name + ".json"), c)
    emit("prepared", plan_hash=plan["card_hash"], env_hash=ecard["card_hash"], data_hash=dcard["card_hash"], pre_hash=pcard["card_hash"], action=run.start_action(plan),
         license_class=dcard["license_class"], tainted=dcard["tainted"], gpus=len(ecard["gpus"]))


def _open_run(a):
    env, data, pre, plan = (_load(os.path.join(a.cards_dir, n + ".json")) for n in ("env", "data", "pre", "plan"))
    check, burn = None, None
    if getattr(a, "ledger", None):
        check = make_ledger_check(Ledger([LocalBackend(a.ledger)]))
    return run.TrainingRun(a.run_dir, plan, env, data, pre, a.data_root, commitment_check=check, on_event=lambda e: emit(e.pop("event"), **e)), plan


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vega_train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("spec")
    p.add_argument("cards_dir")
    for name in ("start", "resume"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        p.add_argument("cards_dir")
        p.add_argument("--ledger", required=True)
        p.add_argument("--commitment", required=True)
        p.add_argument("--data-root", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("run_dir")
    p.add_argument("cards_dir")
    p.add_argument("--data-root", required=True)
    p = sub.add_parser("verify")
    p.add_argument("run_dir")
    p.add_argument("--head")
    p.add_argument("--min-length", type=int)
    p.add_argument("--data-root")
    p.add_argument("--replay", action="store_true")
    for name in ("status", "action"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
        p.add_argument("cards_dir")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--data-root", default=".")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            prepare(a.spec, a.cards_dir)
        elif a.cmd in ("start", "resume"):
            tr, _ = _open_run(a)
            out = getattr(tr, a.cmd)(_load(a.commitment))
            emit("result", **out)
            return 0 if out["status"] == "done" else 3
        elif a.cmd == "freeze":
            a.ledger = None
            tr, _ = _open_run(a)
            fz = tr.freeze()
            emit("frozen", freeze_hash=fz["card_hash"], weights_sha256=fz["weights_sha256"], cards=os.path.join(a.run_dir, "cards"))
        elif a.cmd == "verify":
            rep = run.verify_freeze(a.run_dir, expected_head=a.head, min_length=a.min_length, data_root=a.data_root, replay=a.replay)
            emit("verified", **rep)
            return 0 if rep["ok"] else 4
        elif a.cmd in ("status", "action"):
            a.ledger = None
            tr, plan = _open_run(a)
            if a.cmd == "action":
                cps = tr.checkpoints()
                emit("action", action=run.resume_action(plan, cps[-1]["sha256"]) if a.resume and cps else run.start_action(plan))
            else:
                cps = tr.checkpoints()
                emit("status", state=tr.state(), checkpoints=len(cps), last_step=cps[-1]["step"] if cps else None, plan_hash=plan["card_hash"])
    except Exception as exc:  # reviewed: the command line reports every failure as one refusal event and a non-zero exit, never a traceback a UI cannot read
        emit("refused", reason="%s: %s" % (type(exc).__name__, exc))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
