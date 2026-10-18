"""Functional tests: the whole governed training path on a tiny model, end to end, with no download."""
import json
import os
import unittest

from support import rig
from vega_train import cards, data_card, env_card, run, trainers


class Cards(unittest.TestCase):
    def test_env_card_has_the_expected_fields_and_no_environment_variables(self):
        c = env_card.capture(query_gpu=False, environ={"CUDA_VISIBLE_DEVICES": "0", "AWS_SECRET_ACCESS_KEY": "hunter2", "PYTHONHASHSEED": "0"})
        self.assertEqual(c["kind"], "ENV")
        self.assertEqual(c["settings"], {"CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0"})
        self.assertNotIn("hunter2", json.dumps(c))
        self.assertTrue(cards.intact(c, "ENV"))
        self.assertIn("numpy", c["packages"])

    def test_two_captures_of_one_setup_have_one_fingerprint_and_drift_names_the_difference(self):
        a, b = env_card.capture(query_gpu=False), env_card.capture(query_gpu=False)
        self.assertEqual(env_card.fingerprint(a), env_card.fingerprint(b))
        self.assertEqual(env_card.drift(a, b), [])
        other = dict(b, cpu_count=(b["cpu_count"] or 1) + 3)
        self.assertTrue(any(x.startswith("cpu_count") for x in env_card.drift(a, other)))

    def test_gpu_lines_are_parsed_strictly(self):
        good = "NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 4096, 610.91"
        self.assertEqual(env_card.parse_gpus(good)[0]["memory_mib"], 4096)
        self.assertEqual(env_card.parse_gpus("error: bad\n" + good + "\n; rm -rf /, 1, 2"), [env_card.parse_gpus(good)[0]])

    def test_data_card_hashes_files_and_holds_no_content(self):
        r = rig()
        c = r.dcard
        self.assertEqual((c["kind"], len(c["files"]), c["license_class"], c["tainted"], c["public_ok"]), ("DATA", 1, "OWNED", False, True))
        self.assertNotIn("0.900", json.dumps(c))
        self.assertEqual(c["files"][0]["lines"], 60)

    def test_split_is_deterministic_and_independent_of_order(self):
        ids = ["a%d" % i for i in range(200)]
        t1, v1 = data_card.split_ids(ids, 200, 5)
        t2, v2 = data_card.split_ids(list(reversed(ids)), 200, 5)
        self.assertEqual((sorted(t1), sorted(v1)), (sorted(t2), sorted(v2)))
        self.assertTrue(20 < len(v1) < 70)


class GovernedRun(unittest.TestCase):
    def test_a_run_needs_a_human_commitment_and_then_completes_and_checkpoints(self):
        r = rig()
        tr = r.new_run()
        with self.assertRaises(run.TrainRefused):
            tr.start(None)
        out = tr.start(r.start_commit())
        self.assertEqual((out["status"], out["final_step"]), ("done", 40))
        cps = tr.checkpoints()
        self.assertEqual([c["step"] for c in cps], [10, 20, 30, 40])
        self.assertTrue(out["final_loss"] < 0.5)
        self.assertEqual(tr.state(), "done")

    def test_the_model_actually_learned_something(self):
        r = rig(steps=200, every=50)
        tr = r.new_run()
        out = tr.start(r.start_commit())
        self.assertLess(out["final_loss"], 0.35)

    def test_freeze_writes_all_cards_and_the_verifier_passes_with_a_replay(self):
        r = rig()
        tr = r.new_run()
        tr.start(r.start_commit())
        fz = tr.freeze()
        self.assertEqual(fz["kind"], "TRAIN_FREEZE")
        for name in ("env", "data", "plan", "pre", "runtime", "server", "model", "provenance", "freeze", "MANIFEST"):
            self.assertTrue(os.path.exists(os.path.join(tr.run_dir, "cards", name + ".json")), name)
        rep = run.verify_freeze(tr.run_dir, expected_head=None, data_root=str(r.data_dir), replay=True)
        self.assertEqual(rep["problems"], [])
        self.assertTrue(rep["ok"])
        self.assertTrue(rep["checks"]["replay_reproduces_weights"])

    def test_a_frozen_run_refuses_everything(self):
        r = rig()
        tr = r.new_run()
        tr.start(r.start_commit())
        tr.freeze()
        for fn in (lambda: tr.start(r.start_commit()), lambda: tr.resume(r.mint("x")), lambda: tr.freeze()):
            with self.assertRaises(run.TrainRefused):
                fn()

    def test_a_halted_run_resumes_from_its_last_checkpoint_only_with_a_new_commitment_and_reaches_the_same_weights(self):
        r = rig()
        tr = r.new_run("halting")
        calls = {"n": 0}
        real_step = tr.trainer.step

        def failing(n):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated crash")
            return real_step(n)
        tr.trainer.step = failing
        out = tr.start(r.start_commit())
        self.assertEqual((out["status"], out["last_checkpoint_step"]), ("halted", 20))
        tr2 = r.new_run("halting")                                              # a fresh process opens the same folder
        self.assertEqual(tr2.state(), "halted")
        with self.assertRaises(run.TrainRefused):
            tr2.resume(None)
        last = tr2.checkpoints()[-1]
        done = tr2.resume(r.mint(run.resume_action(r.plan, last["sha256"])))
        self.assertEqual((done["status"], done["final_step"]), ("done", 40))
        straight = r.new_run("straight")
        ref = straight.start(r.start_commit())
        self.assertEqual(done["weights_sha256"], ref["weights_sha256"])         # interrupted and resumed == run straight through

    def test_the_cli_free_helpers_work_with_the_torch_backend_when_torch_is_installed(self):
        try:
            import torch  # noqa: F401
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("torch or safetensors not installed")
        r = rig(trainer="torch_mlp", hyper={"hidden": 4, "lr": 0.8}, steps=20, every=10)
        tr = r.new_run("torch")
        out = tr.start(r.start_commit())
        self.assertEqual(out["status"], "done")
        blob = open(os.path.join(tr.run_dir, "weights.bin"), "rb").read()
        self.assertNotIn(b"pickle", blob[:64].lower())                          # safetensors, not a pickle
        tr.freeze()
        self.assertTrue(run.verify_freeze(tr.run_dir, data_root=str(r.data_dir), replay=True)["ok"])


class Codec(unittest.TestCase):
    def test_the_checkpoint_format_round_trips_and_is_canonical(self):
        import numpy as np
        a = {"b": np.arange(6, dtype=np.float64).reshape(2, 3), "a": np.ones(2)}
        blob = trainers.pack_arrays(a, {"k": 1})
        self.assertEqual(blob, trainers.pack_arrays({"a": np.ones(2), "b": np.arange(6, dtype=np.float64).reshape(2, 3)}, {"k": 1}))
        back, meta = trainers.unpack_arrays(blob)
        self.assertEqual((meta, back["b"].tolist()), ({"k": 1}, a["b"].tolist()))


if __name__ == "__main__":
    unittest.main()
