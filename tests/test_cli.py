"""The command line end to end (what the Vega OS training tab runs): prepare -> human commitment -> start -> freeze -> verify, as separate processes reading JSON lines."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from support import DEV, PW, SURF, csv_rows
from vega_custody.dag.dag import Dag
from vega_custody.ledger.backend_local import LocalBackend
from vega_custody.ledger.ledger import Ledger
from vega_jacket import humanauth
from vega_jacket.commit.commit import Commit
from vega_jacket.humanauth import human_signer


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "vega_train", *args], capture_output=True, text=True, timeout=300, env=os.environ.copy())
    events = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    return r.returncode, events, r.stderr


class T(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp())
        (self.t / "data").mkdir()
        (self.t / "data" / "train.csv").write_bytes(csv_rows().encode())
        self.spec = {"name": "clirun", "trainer": "numpy_mlp", "hyper": {"hidden": 4, "lr": 0.8}, "seed": 3, "steps": 40, "checkpoint_every": 10,
                     "data": {"root": str(self.t / "data"), "files": ["train.csv"], "license_class": "OWNED"}}
        (self.t / "spec.json").write_text(json.dumps(self.spec), encoding="utf-8")
        blob, pub = human_signer.generate(PW)
        self.ledger_path = self.t / "led.jsonl"
        dag = Dag([self.t / "c.jsonl"], [self.t / "l.jsonl"])
        self.commit = Commit(humanauth.Verifier(pub, DEV, SURF), Ledger([LocalBackend(self.ledger_path)]), dag)
        self.blob = blob

    def human_commitment(self, action, name):
        params = {"for": action}
        signed = human_signer.sign(self.blob, PW, DEV, SURF, action, params, 300)
        c = self.commit.commit(action, params, signed, route="human")["commitment"]
        p = self.t / name
        p.write_text(json.dumps(c), encoding="utf-8")
        return str(p)

    def test_the_whole_path_through_the_command_line(self):
        cards_dir, run_dir = str(self.t / "cards"), str(self.t / "runs" / "r1")
        code, ev, err = cli("prepare", str(self.t / "spec.json"), cards_dir)
        self.assertEqual((code, ev[-1]["event"], ev[-1]["tainted"]), (0, "prepared", False), err)
        action = ev[-1]["action"]
        self.assertTrue(action.startswith("train_run:"))
        os.makedirs(os.path.dirname(run_dir), exist_ok=True)
        # without a commitment the run is refused
        bad = self.human_commitment("conductor_step:something", "wrong.json")
        code, ev, _ = cli("start", run_dir, cards_dir, "--ledger", str(self.ledger_path), "--commitment", bad, "--data-root", str(self.t / "data"))
        self.assertEqual((code, ev[-1]["event"]), (2, "refused"))
        good = self.human_commitment(action, "good.json")
        code, ev, err = cli("start", run_dir, cards_dir, "--ledger", str(self.ledger_path), "--commitment", good, "--data-root", str(self.t / "data"))
        self.assertEqual(code, 0, err)
        self.assertEqual([e["event"] for e in ev], ["checkpoint"] * 4 + ["done", "result"])
        self.assertEqual(ev[0]["step"], 10)
        # the same commitment cannot start a second run
        code, ev, _ = cli("start", str(self.t / "runs" / "r2"), cards_dir, "--ledger", str(self.ledger_path), "--commitment", good, "--data-root", str(self.t / "data"))
        self.assertEqual((code, ev[-1]["event"]), (2, "refused"))
        code, ev, _ = cli("status", run_dir, cards_dir)
        self.assertEqual((ev[-1]["state"], ev[-1]["checkpoints"], ev[-1]["last_step"]), ("done", 4, 40))
        code, ev, err = cli("freeze", run_dir, cards_dir, "--data-root", str(self.t / "data"))
        self.assertEqual((code, ev[-1]["event"]), (0, "frozen"), err)
        code, ev, err = cli("verify", run_dir, "--data-root", str(self.t / "data"), "--replay")
        self.assertEqual((code, ev[-1]["event"], ev[-1]["ok"], ev[-1]["problems"]), (0, "verified", True, []), err)
        flipped = pathlib.Path(run_dir, "weights.bin")
        raw = bytearray(flipped.read_bytes())
        raw[-1] ^= 1
        flipped.write_bytes(bytes(raw))
        code, ev, _ = cli("verify", run_dir)
        self.assertEqual((code, ev[-1]["ok"]), (4, False))

    def test_bad_input_is_a_refusal_event_not_a_traceback(self):
        for args in (("prepare", str(self.t / "nope.json"), str(self.t / "c")), ("verify", str(self.t / "nowhere")), ("status", str(self.t / "x"), str(self.t / "y"))):
            code, ev, err = cli(*args)
            self.assertIn(code, (2, 4), err[-300:])
            self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
