"""RED TEAM PASS 1 (training): the run itself. Gates, spent signatures, tampered or swapped checkpoints, a trainer that lies, budgets, concurrency, and a poisoned data folder."""
import math
import os
import pathlib
import pickle
import threading
import time
import unittest

from support import rig
from vega_pareah.cards import pre as pre_mod
from vega_train import cards, run, trainers


class Gates(unittest.TestCase):
    def test_no_commitment_a_loose_checker_or_a_crashing_checker_never_starts_a_run(self):
        r = rig()
        for commitment in (None, {}, "text", {"kind": "WCOMMIT"}):
            with self.assertRaises(run.TrainRefused):
                r.new_run("a").start(commitment)
        for chk in (lambda c, a: "yes", lambda c, a: 1, lambda c, a: 1 / 0, None):
            tr = run.TrainingRun(str(r.t / "b"), r.plan, r.ecard, r.dcard, r.pre, str(r.data_dir), commitment_check=chk)
            with self.assertRaises(run.TrainRefused):
                tr.start(r.start_commit())
        self.assertEqual(r.new_run("a").state(), "new")

    def test_a_commitment_for_another_action_or_another_plan_does_not_start_this_run(self):
        r = rig()
        for action in ("train_run:" + "0" * 64, "conductor_step:train", "run_job", "release_public", run.resume_action(r.plan, "ab" * 32)):
            with self.assertRaises(run.TrainRefused, msg=action):
                r.new_run("a").start(r.mint(action))
        self.assertEqual(r.new_run("a").state(), "new")

    def test_a_spent_commitment_cannot_start_any_run_under_the_same_folder(self):
        r = rig()
        c = r.start_commit()
        self.assertEqual(r.new_run("first").start(c)["status"], "done")
        with self.assertRaises(run.TrainRefused) as cx:
            r.new_run("second").start(c)                                              # a different run folder, the same signature
        self.assertIn("already spent", str(cx.exception))
        self.assertEqual(r.new_run("second").state(), "new")

    def test_the_pre_must_permit_training_this_model_on_this_dataset(self):
        r = rig()
        for i, pre_card in enumerate((pre_mod.mint_pre("t", ["read_file"], ["model/", "dataset/"], 600), pre_mod.mint_pre("t", ["train"], ["model/other"], 600),
                                      pre_mod.mint_pre("t", ["train"], ["model/", "dataset/other"], 600))):
            plan = run.make_plan("tinyrun", r.ecard, r.dcard, pre_card, "numpy_mlp", {"hidden": 2}, 1, 10, 5)
            tr = run.TrainingRun(str(r.t / ("p%d" % i)), plan, r.ecard, r.dcard, pre_card, str(r.data_dir), commitment_check=r.check)
            with self.assertRaises(run.TrainRefused):
                tr.start(r.mint(run.start_action(plan)))
            self.assertEqual(tr.state(), "new")

    def test_an_expired_pre_stops_the_run(self):
        r = rig()
        old = pre_mod.mint_pre("t", ["train"], ["model/", "dataset/"], 60, now=time.time() - 3600)
        plan = run.make_plan("tinyrun", r.ecard, r.dcard, old, "numpy_mlp", {}, 1, 10, 5)
        tr = run.TrainingRun(str(r.t / "e"), plan, r.ecard, r.dcard, old, str(r.data_dir), commitment_check=r.check)
        with self.assertRaises(run.TrainRefused):
            tr.start(r.mint(run.start_action(plan)))

    def test_cards_that_do_not_match_the_plan_or_were_edited_are_refused_at_construction(self):
        r = rig()
        other_env = dict(r.ecard, cpu_count=999)
        with self.assertRaises(run.TrainRefused):
            run.TrainingRun(str(r.t / "x"), r.plan, other_env, r.dcard, r.pre, str(r.data_dir))                      # edited, hash no longer matches
        with self.assertRaises(run.TrainRefused):
            run.TrainingRun(str(r.t / "x"), r.plan, r.ecard, cards.seal({**r.dcard, "name": "other"}), r.pre, str(r.data_dir))   # sealed but not what the plan names
        edited_plan = dict(r.plan, steps=10 ** 6)
        with self.assertRaises(run.TrainRefused):
            run.TrainingRun(str(r.t / "x"), edited_plan, r.ecard, r.dcard, r.pre, str(r.data_dir))

    def test_plan_values_that_are_not_sane_are_refused(self):
        r = rig()
        base = dict(name="n", env=r.ecard, data=r.dcard, pre_card=r.pre, trainer="numpy_mlp")
        for kw in ({"steps": True}, {"steps": 0}, {"steps": -1}, {"steps": 10 ** 9}, {"steps": 5.5}, {"checkpoint_every": 0}, {"steps": 10, "checkpoint_every": 11},
                   {"max_wall_seconds": 0}, {"max_wall_seconds": 10 ** 9}, {"max_disk_bytes": -1}, {"seed": -1}, {"seed": True}, {"seed": 2 ** 70}, {"name": "../x"}, {"name": ""},
                   {"trainer": 5}, {"hyper": [1]}, {"base_model": "gpt"}, {"base_model": {"no": "id"}}):
            args = {**base, "steps": 10, "checkpoint_every": 5, **kw}
            with self.assertRaises(run.TrainRefused if "name" not in kw else Exception, msg=repr(kw)):
                run.make_plan(**args)
        with self.assertRaises(run.TrainRefused):
            run.make_plan("n", {"kind": "ENV"}, r.dcard, r.pre, "numpy_mlp")


class HonestTrainer(unittest.TestCase):
    def go(self, patch, **kw):
        r = rig(**kw)
        tr = r.new_run("t")
        patch(tr)
        return r, tr, tr.start(r.start_commit())

    def test_nan_and_infinite_loss_halt_the_run(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            def patch(tr, bad=bad):
                real = tr.trainer.step
                tr.trainer.step = lambda n: {**real(n), "loss": bad}
            r, tr, out = self.go(patch)
            self.assertEqual(out["status"], "halted")
            self.assertIn("not finite", out["reason"])
            self.assertEqual(tr.state(), "halted")
            self.assertEqual(tr.checkpoints(), [])

    def test_a_trainer_that_misreports_its_step_halts_the_run(self):
        def patch(tr):
            real = tr.trainer.step
            tr.trainer.step = lambda n: {**real(n), "step": 999}
        self.assertIn("expected", self.go(patch)[2]["reason"])

    def test_a_trainer_that_raises_halts_with_the_last_good_checkpoint(self):
        calls = {"n": 0}

        def patch(tr):
            real = tr.trainer.step

            def step(n):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise RuntimeError("secret internal detail")
                return real(n)
            tr.trainer.step = step
        r, tr, out = self.go(patch)
        self.assertEqual((out["status"], out["last_checkpoint_step"]), ("halted", 20))
        self.assertNotIn("secret internal detail", out["reason"])

    def test_the_wall_clock_budget_halts_the_run_after_the_current_chunk(self):
        clock = {"t": 1000.0}
        r = rig(wall=5)

        def tick(*a):
            clock["t"] += 3
            return clock["t"]
        tr = r.new_run("w", clock=tick)
        out = tr.start(r.start_commit())
        self.assertEqual(out["status"], "halted")
        self.assertIn("wall-clock", out["reason"])

    def test_the_disk_budget_halts_the_run_and_keeps_what_was_written(self):
        r = rig(disk=200)
        tr = r.new_run("d")
        out = tr.start(r.start_commit())
        self.assertEqual(out["status"], "halted")
        self.assertIn("disk budget", out["reason"])
        self.assertTrue(tr.checkpoints())

    def test_a_checkpoint_that_cannot_be_written_halts_and_leaves_the_earlier_ones_intact(self):
        r = rig()
        tr = r.new_run("c")
        real = run._atomic_write_bytes
        n = {"i": 0}

        def failing(path, data):
            n["i"] += 1
            if n["i"] == 3:
                raise OSError("disk full")
            return real(path, data)
        run._atomic_write_bytes = failing
        try:
            out = tr.start(r.start_commit())
        finally:
            run._atomic_write_bytes = real
        self.assertEqual((out["status"], out["last_checkpoint_step"]), ("halted", 20))
        self.assertEqual([c["step"] for c in tr.checkpoints()], [10, 20])
        self.assertEqual([p for p in os.listdir(os.path.join(tr.run_dir, "checkpoints")) if p.endswith(".tmp")], [n for n in os.listdir(os.path.join(tr.run_dir, "checkpoints")) if n.endswith(".tmp")])

    def test_an_interrupted_process_leaves_a_running_run_that_can_resume_with_a_new_commitment(self):
        r = rig()
        tr = r.new_run("k")
        real = tr.trainer.step
        n = {"i": 0}

        def interrupt(k):
            n["i"] += 1
            if n["i"] == 3:
                raise KeyboardInterrupt()
            return real(k)
        tr.trainer.step = interrupt
        with self.assertRaises(KeyboardInterrupt):
            tr.start(r.start_commit())
        tr2 = r.new_run("k")
        self.assertEqual(tr2.state(), "running")
        last = tr2.checkpoints()[-1]
        self.assertEqual(tr2.resume(r.mint(run.resume_action(r.plan, last["sha256"])))["status"], "done")


class Checkpoints(unittest.TestCase):
    def halted(self):
        r = rig()
        tr = r.new_run("h")
        real = tr.trainer.step
        n = {"i": 0}

        def stop(k):
            n["i"] += 1
            if n["i"] == 3:
                raise RuntimeError("stop")
            return real(k)
        tr.trainer.step = stop
        tr.start(r.start_commit())
        return r, r.new_run("h")

    def resume_with(self, r, tr):
        last = tr.checkpoints()[-1]
        return tr.resume(r.mint(run.resume_action(r.plan, last["sha256"])))

    def test_a_tampered_checkpoint_is_refused_before_anything_is_trusted_and_the_commitment_is_not_spent(self):
        r, tr = self.halted()
        p = pathlib.Path(tr.run_dir) / "checkpoints" / "ckpt_00000020.bin"
        raw = bytearray(p.read_bytes())
        raw[-1] ^= 1
        p.write_bytes(bytes(raw))
        last = tr.checkpoints()[-1]
        c = r.mint(run.resume_action(r.plan, last["sha256"]))
        with self.assertRaises(run.TrainRefused):
            tr.resume(c)
        p.write_bytes(bytes(raw[:-1]) + bytes([raw[-1] ^ 1]))                            # put it back
        self.assertEqual(tr.resume(c)["status"], "done")                                   # the same commitment still works: it was never spent on the refusal

    def test_a_missing_or_swapped_checkpoint_is_refused(self):
        r, tr = self.halted()
        d = pathlib.Path(tr.run_dir) / "checkpoints"
        keep = (d / "ckpt_00000020.bin").read_bytes()
        (d / "ckpt_00000020.bin").write_bytes((d / "ckpt_00000010.bin").read_bytes())        # an older state under the newer name
        with self.assertRaises(run.TrainRefused):
            self.resume_with(r, tr)
        (d / "ckpt_00000020.bin").unlink()
        with self.assertRaises(run.TrainRefused):
            self.resume_with(r, tr)
        (d / "ckpt_00000020.bin").write_bytes(keep)
        self.assertEqual(self.resume_with(r, tr)["status"], "done")

    def test_resuming_needs_a_commitment_for_exactly_this_checkpoint(self):
        r, tr = self.halted()
        for bad in (None, r.mint("conductor_step:x"), r.mint(run.resume_action(r.plan, "ab" * 32)), r.start_commit()):
            with self.assertRaises(run.TrainRefused):
                tr.resume(bad)
        self.assertEqual(tr.state(), "halted")

    def test_resume_only_works_on_a_halted_or_interrupted_run(self):
        r = rig()
        tr = r.new_run("n")
        with self.assertRaises(run.TrainRefused):
            tr.resume(r.mint("x"))
        tr.start(r.start_commit())
        with self.assertRaises(run.TrainRefused):
            tr.resume(r.mint("y"))
        with self.assertRaises(run.TrainRefused):
            tr.start(r.start_commit())

    def test_a_checkpoint_in_a_pickle_never_executes_and_is_never_loaded(self):
        marker = pathlib.Path(rig().t) / "owned.txt"

        class Evil:
            def __reduce__(self):
                return (open, (str(marker), "w"))
        blob = pickle.dumps(Evil())
        with self.assertRaises(trainers.TrainerRefused):
            trainers.unpack_arrays(blob)
        self.assertFalse(marker.exists())
        r, tr = self.halted()
        p = pathlib.Path(tr.run_dir) / "checkpoints" / "ckpt_00000020.bin"
        p.write_bytes(blob)
        with self.assertRaises(run.TrainRefused):
            self.resume_with(r, tr)
        self.assertFalse(marker.exists())

    def test_the_checkpoint_format_refuses_every_malformed_shape(self):
        good = trainers.pack_arrays({"a": __import__("numpy").ones(3)}, {})
        bad = [b"", b"VTA1", b"VTA1\xff\xff\xff\xff" + b"x" * 10, good[:-1], good + b"x", good[:8] + b"{notjson" + good[8 + 8:], b"VTA2" + good[4:]]
        import json
        import struct
        header = json.dumps({"meta": {}, "arrays": [{"name": "o", "dtype": "O", "shape": [1]}]}).encode()
        bad.append(b"VTA1" + struct.pack("<I", len(header)) + header + b"x" * 8)
        header2 = json.dumps({"meta": {}, "arrays": [{"name": "a", "dtype": "<f8", "shape": [10 ** 12]}]}).encode()
        bad.append(b"VTA1" + struct.pack("<I", len(header2)) + header2)
        for b in bad:
            with self.assertRaises(trainers.TrainerRefused, msg=repr(b[:20])):
                trainers.unpack_arrays(b)


class Concurrency(unittest.TestCase):
    def test_two_processes_cannot_train_the_same_run_at_once(self):
        r = rig(steps=2000, every=500, hyper={"hidden": 64, "lr": 0.1})
        tr1, tr2 = r.new_run("same"), r.new_run("same")
        results, started = {}, threading.Event()
        real = tr1.trainer.step

        def slow(n):
            started.set()
            time.sleep(0.4)
            return real(n)
        tr1.trainer.step = slow
        th = threading.Thread(target=lambda: results.update(first=tr1.start(r.start_commit())))
        th.start()
        started.wait(10)
        with self.assertRaises(run.TrainRefused) as cx:
            tr2.start(r.start_commit())
        self.assertIn("already running", str(cx.exception))
        th.join()
        self.assertEqual(results["first"]["status"], "done")

    def test_a_second_run_folder_with_a_different_plan_cannot_share_a_folder(self):
        r = rig()
        r.new_run("one")
        other = run.make_plan("tinyrun", r.ecard, r.dcard, r.pre, "numpy_mlp", {"hidden": 2}, seed=99, steps=10, checkpoint_every=5)
        with self.assertRaises(run.TrainRefused):
            run.TrainingRun(str(r.t / "one"), other, r.ecard, r.dcard, r.pre, str(r.data_dir))


class PoisonedData(unittest.TestCase):
    def test_data_that_changed_after_the_card_stops_the_run_without_spending_the_commitment(self):
        r = rig()
        (r.data_dir / "train.csv").write_bytes(b"0.1,0.1,1\n" * 60)
        c = r.start_commit()
        with self.assertRaises(run.TrainRefused) as cx:
            r.new_run("p").start(c)
        self.assertIn("data card", str(cx.exception))
        (r.data_dir / "train.csv").write_bytes(__import__("support").csv_rows().encode())          # restore
        self.assertEqual(r.new_run("p").start(c)["status"], "done")                           # the commitment was not burned by the refusal

    def test_rows_that_are_not_numbers_or_labels_are_refused_by_the_trainer_not_by_a_crash(self):
        r = rig()
        for text in (b"a,b,c\n", b"1,2\n3,4,5\n", b"0.1,0.2,7\n", b"0.1,nan,1\n", b"\n\n", b"0.1,0.2,1\n" + b"x" * 5):
            (r.data_dir / "train.csv").write_bytes(text)
            from vega_train import data_card
            dc = data_card.build("tiny", str(r.data_dir), ["train.csv"], "OWNED", (), {})
            plan = run.make_plan("tinyrun", r.ecard, dc, r.pre, "numpy_mlp", {}, 1, 10, 5)
            tr = run.TrainingRun(str(r.t / ("bad%d" % abs(hash(text)))), plan, r.ecard, dc, r.pre, str(r.data_dir), commitment_check=r.check)
            with self.assertRaises(run.TrainRefused, msg=repr(text)):
                tr.start(r.mint(run.start_action(plan)))

    def test_hyper_parameters_out_of_range_are_refused(self):
        r = rig()
        for hyper in ({"hidden": -1}, {"hidden": 10 ** 9}, {"lr": 0}, {"lr": 10 ** 6}, {"batch": -2}, {"l2": 100}, {"hidden": "x"}, {"lr": "fast"}):
            plan = run.make_plan("tinyrun", r.ecard, r.dcard, r.pre, "numpy_mlp", hyper, 1, 10, 5)
            tr = run.TrainingRun(str(r.t / ("h%d" % abs(hash(str(hyper))))), plan, r.ecard, r.dcard, r.pre, str(r.data_dir), commitment_check=r.check)
            with self.assertRaises(run.TrainRefused, msg=repr(hyper)):
                tr.start(r.mint(run.start_action(plan)))


if __name__ == "__main__":
    unittest.main()
