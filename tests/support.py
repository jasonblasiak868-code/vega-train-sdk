"""Shared fixtures for the training tests: a tiny linearly-separable data set, a human key with a ledger, a PRE that permits training, and a ready-to-run rig."""
import pathlib
import tempfile
import time
import types

from vega_custody.dag.dag import Dag
from vega_custody.ledger.backend_local import LocalBackend
from vega_custody.ledger.ledger import Ledger
from vega_harness.records import make_ledger_check
from vega_jacket import humanauth
from vega_jacket.commit.commit import Commit
from vega_jacket.humanauth import human_signer
from vega_pareah.cards import pre as pre_mod
from vega_train import data_card, env_card, run

PW, DEV, SURF = "correct horse battery", "dev-train", "surf-train"


def csv_rows(n=60):
    rows = []
    for i in range(n):
        x1, x2 = (i % 10) / 10.0, ((i * 7) % 10) / 10.0
        rows.append("%.3f,%.3f,%d" % (x1, x2, 1 if x1 + x2 > 0.9 else 0))
    return "\n".join(rows) + "\n"


def rig(steps=40, every=10, hyper=None, wall=3600, disk=1 << 26, license_class="OWNED", sources=(), matrix=None, trainer="numpy_mlp", capture_env=False):
    t = pathlib.Path(tempfile.mkdtemp())
    data_dir = t / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_bytes(csv_rows().encode("utf-8"))
    dcard = data_card.build("tiny", str(data_dir), ["train.csv"], license_class, sources, matrix if matrix is not None else {})
    ecard = env_card.capture(query_gpu=False) if capture_env else env_card.capture(query_gpu=False)
    pre = pre_mod.mint_pre("trainer", ["train"], ["model/", "dataset/"], 600)
    plan = run.make_plan("tinyrun", ecard, dcard, pre, trainer, hyper or {"hidden": 4, "lr": 0.8}, seed=7, steps=steps, checkpoint_every=every, max_wall_seconds=wall, max_disk_bytes=disk)
    blob, pub = human_signer.generate(PW)
    dag = Dag([t / "c.jsonl"], [t / "l.jsonl"])
    led = Ledger([LocalBackend(t / "led.jsonl")])
    commit = Commit(humanauth.Verifier(pub, DEV, SURF), led, dag)

    def mint(action, params=None, lease=300):
        params = params if params is not None else {"for": action}
        signed = human_signer.sign(blob, PW, DEV, SURF, action, params, lease)
        return commit.commit(action, params, signed, route="human")["commitment"]
    check = make_ledger_check(led)

    def new_run(run_name="r1", **kw):
        return run.TrainingRun(str(t / run_name), plan, ecard, dcard, pre, str(data_dir), commitment_check=check, **kw)
    return types.SimpleNamespace(t=t, data_dir=data_dir, dcard=dcard, ecard=ecard, pre=pre, plan=plan, mint=mint, check=check, new_run=new_run, led=led, dag=dag,
                                 start_commit=lambda: mint(run.start_action(plan)))
