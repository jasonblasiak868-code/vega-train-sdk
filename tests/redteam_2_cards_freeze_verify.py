"""RED TEAM PASS 2 (training): the cards and the freeze. Hostile tool output, secrets in settings or data, poisoned data folders, license and taint propagation, and every way to edit a
frozen run so it still looks frozen."""
import json
import os
import pathlib
import shutil
import tempfile
import unittest

from support import csv_rows, rig
from vega_train import cards, data_card, env_card, run

SECRET = "sk-" + "a1B2c3D4" * 4


def frozen(**kw):
    r = rig(**kw)
    tr = r.new_run("f")
    tr.start(r.start_commit())
    fz = tr.freeze()
    head = tr.rows()[-1]["hash"]
    return r, tr, fz, head


def edit_json(path, fn):
    p = pathlib.Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    fn(obj)
    p.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")), encoding="utf-8")


class EnvCard(unittest.TestCase):
    def test_hostile_tool_output_yields_only_well_formed_gpu_lines(self):
        out = "\n".join(["NVIDIA RTX 4090, 24564, 551.23", "; DROP TABLE, 5, 5", "A" * 5000, "Tesla T4, abc, 1.0", "GPU\x00x, 1, 2", "Quadro P1000, 4096, 470.1", "x, 1, 2, 3", "$(rm -rf /), 4, 5"])
        gpus = env_card.parse_gpus(out)
        self.assertEqual([g["name"] for g in gpus], ["NVIDIA RTX 4090", "Quadro P1000"])
        one = env_card.capture(run=lambda *a, **k: {"status": "ok", "returncode": 0, "out": b"Quadro P1000, 4096, 470.1", "err": b""})
        self.assertEqual(one["gpu_source"], "nvidia-smi (self-reported label, not verified)")
        self.assertEqual(len(env_card.parse_gpus("NVIDIA RTX 4090, 24564, 551.23\n" * 100)), 16)

    def test_a_missing_failing_or_flooding_nvidia_smi_is_recorded_not_fatal(self):
        def missing(*a, **k):
            raise FileNotFoundError()

        def failing(*a, **k):
            return {"status": "ok", "returncode": 9, "out": b"", "err": b"boom"}

        def timeout(*a, **k):
            return {"status": "timeout", "returncode": None, "out": b"", "err": b""}

        def junk(*a, **k):
            return {"status": "ok", "returncode": 0, "out": b"\xff\xfe" * 100, "err": b""}
        for fn in (missing, failing, timeout, junk):
            c = env_card.capture(run=fn)
            self.assertEqual(c["gpus"], [])
            self.assertTrue(c["gpu_note"])
            self.assertTrue(cards.intact(c, "ENV"))

    def test_a_secret_or_odd_value_in_an_allowed_setting_is_not_recorded(self):
        c = env_card.capture(query_gpu=False, environ={"CUDA_VISIBLE_DEVICES": SECRET[:40], "OMP_NUM_THREADS": "4;rm -rf /", "PYTHONHASHSEED": "0", "MKL_NUM_THREADS": "x" * 100})
        self.assertEqual(c["settings"], {"PYTHONHASHSEED": "0"})
        self.assertNotIn("a1B2c3D4", json.dumps(c))

    def test_no_machine_paths_or_user_names_are_recorded(self):
        text = json.dumps(env_card.capture(query_gpu=False))
        self.assertNotIn(os.path.expanduser("~").replace("\\", "\\\\"), text)
        self.assertNotIn(os.path.expanduser("~"), text)

    def test_drift_against_an_altered_card_is_reported_not_trusted(self):
        c = env_card.capture(query_gpu=False)
        forged = dict(c, python="9.9.9")
        self.assertEqual(env_card.drift(forged, c), ["the recorded environment card is altered or not an ENV card"])


class DataCard(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.root = os.path.join(self.d, "data")
        os.makedirs(self.root)
        pathlib.Path(self.root, "a.csv").write_bytes(csv_rows(10).encode())

    def build(self, files=("a.csv",), **kw):
        return data_card.build("d", self.root, list(files), matrix=kw.pop("matrix", {}), **kw)

    def test_a_file_with_a_credential_is_refused_naming_the_file_but_never_the_value(self):
        pathlib.Path(self.root, "leak.csv").write_text("0.1,0.2,1\nnote," + SECRET + ",0\n", encoding="utf-8")
        with self.assertRaises(data_card.DataRefused) as cx:
            self.build(["a.csv", "leak.csv"])
        self.assertIn("leak.csv", str(cx.exception))
        self.assertNotIn("a1B2c3D4", str(cx.exception))

    def test_traversal_absolute_duplicate_and_odd_entries_are_refused(self):
        outside = os.path.join(self.d, "outside.csv")
        pathlib.Path(outside).write_text("1,2,1\n")
        for files in (["../outside.csv"], [outside], ["a.csv", "a.csv"], [""], [5], [None], "a.csv", [], ["nope.csv"], ["/a.csv"], ["sub/../../outside.csv"]):
            with self.assertRaises(data_card.DataRefused, msg=repr(files)):
                self.build(files)

    def test_a_junction_that_points_outside_is_refused(self):
        if os.name != "nt":
            self.skipTest("junctions are a Windows feature")
        import subprocess
        outside = os.path.join(self.d, "outside_dir")
        os.makedirs(outside)
        pathlib.Path(outside, "x.csv").write_text("1,2,1\n")
        r = subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(self.root, "jn"), outside], capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("could not create a junction")
        with self.assertRaises(data_card.DataRefused):
            self.build(["jn/x.csv"])

    def test_license_classes_resolve_to_the_strictest_and_taint_follows(self):
        m = {"gpl-repo": {"repo": "gpl-repo", "license_class": "STRONG_COPYLEFT"}, "mit-repo": {"repo": "mit-repo", "license_class": "PERMISSIVE"}}
        own = self.build()
        self.assertEqual((own["license_class"], own["tainted"], own["public_ok"]), ("OWNED", False, True))
        perm = self.build(license_class="PERMISSIVE")
        self.assertEqual((perm["license_class"], perm["tainted"]), ("PERMISSIVE", False))
        mixed = self.build(license_class="OWNED", sources=["gpl-repo", "mit-repo"], matrix=m)
        self.assertEqual((mixed["license_class"], mixed["tainted"], mixed["public_ok"]), ("STRONG_COPYLEFT", True, False))
        lib = self.build(license_class="OWNED", sources=["mit-repo"], matrix=m)
        self.assertEqual((lib["tainted"], lib["public_ok"]), (True, False))                 # any library source taints, even a permissive one
        unknown = self.build(license_class="OWNED", sources=["not-in-matrix"], matrix=m)
        self.assertEqual(unknown["license_class"], "NO_LICENSE_FILE")
        odd = self.build(license_class="MIT-ish")
        self.assertEqual((odd["license_class"], odd["tainted"]), ("NO_LICENSE_FILE", True))

    def test_the_card_holds_hashes_and_counts_never_content_or_machine_paths(self):
        text = json.dumps(self.build())
        self.assertNotIn(self.d.replace("\\", "\\\\"), text)
        self.assertNotIn("0.100", text)

    def test_load_verified_catches_edits_swaps_and_a_forged_card(self):
        c = self.build()
        self.assertEqual(set(data_card.load_verified(c, self.root)), {"a.csv"})
        pathlib.Path(self.root, "a.csv").write_bytes(csv_rows(10).encode() + b"1,1,1\n")
        with self.assertRaises(data_card.DataRefused):
            data_card.load_verified(c, self.root)
        forged = dict(c, total_bytes=1)
        with self.assertRaises(data_card.DataRefused):
            data_card.load_verified(forged, self.root)
        with self.assertRaises(data_card.DataRefused):
            data_card.load_verified({"kind": "ENV"}, self.root)

    def test_a_card_whose_set_hash_was_resealed_after_a_file_edit_still_fails_against_the_files(self):
        c = self.build()
        evil = cards.seal({**c, "files": [dict(c["files"][0], sha256="00" * 32)]})
        with self.assertRaises(data_card.DataRefused):
            data_card.load_verified(evil, self.root)

    def test_split_rule_rejects_bad_fractions(self):
        for bad in (-1, 1001, True, 1.5, "10", None):
            with self.assertRaises(data_card.DataRefused, msg=repr(bad)):
                data_card.split_ids(["a"], bad, 0)
        self.assertEqual(data_card.split_ids(["a", "b"], 0, 0), (["a", "b"], []))
        self.assertEqual(data_card.split_ids(["a", "b"], 1000, 0), ([], ["a", "b"]))


class Freeze(unittest.TestCase):
    def test_a_clean_run_verifies_with_replay_and_against_the_head_kept_at_freeze_time(self):
        r, tr, fz, head = frozen()
        rep = run.verify_freeze(tr.run_dir, expected_head=head, min_length=len(tr.rows()), data_root=str(r.data_dir), replay=True)
        self.assertEqual(rep["problems"], [])
        self.assertTrue(rep["ok"])

    def test_no_secret_and_no_machine_path_appears_in_any_card_or_record_of_a_run(self):
        r, tr, fz, head = frozen()
        blob = "".join(p.read_text(encoding="utf-8", errors="replace") for p in pathlib.Path(tr.run_dir).rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
        self.assertNotIn(str(r.t), blob)
        self.assertNotIn(str(r.t).replace("\\", "\\\\"), blob)
        self.assertNotIn(os.path.expanduser("~"), blob)
        self.assertNotIn("sk-", blob)

    def tamper_cases(self):
        def card(name, fn):
            return lambda tr, r: edit_json(os.path.join(tr.run_dir, "cards", name), fn)

        return {
            "edit data card": card("data.json", lambda o: o.update(license_class="PERMISSIVE")),
            "edit plan card": card("plan.json", lambda o: o.update(steps=5)),
            "edit env card": card("env.json", lambda o: o.update(cpu_count=1)),
            "edit freeze card": card("freeze.json", lambda o: o.update(final_loss="0.0")),
            "edit model card": card("model.json", lambda o: o.update(model_id="other")),
            "edit the manifest": card("MANIFEST.json", lambda o: o.update(freeze_hash="00" * 32)),
            "flip a weights byte": lambda tr, r: _flip(os.path.join(tr.run_dir, "weights.bin")),
            "flip a checkpoint byte": lambda tr, r: _flip(os.path.join(tr.run_dir, "checkpoints", "ckpt_00000020.bin")),
            "delete a checkpoint": lambda tr, r: os.remove(os.path.join(tr.run_dir, "checkpoints", "ckpt_00000020.bin")),
            "delete a card file": lambda tr, r: os.remove(os.path.join(tr.run_dir, "cards", "env.json")),
            "delete the weights": lambda tr, r: os.remove(os.path.join(tr.run_dir, "weights.bin")),
            "change the training data": lambda tr, r: pathlib.Path(r.data_dir, "train.csv").write_bytes(b"0.1,0.1,1\n" * 60),
        }

    def test_dropping_the_taint_mark_from_a_tainted_model_is_reported(self):
        m = {"gpl": {"repo": "gpl", "license_class": "STRONG_COPYLEFT"}}
        for how in ("edit", "edit and reseal"):
            r, tr, fz, head = frozen(sources=["gpl"], matrix=m)

            def drop(o, how=how):
                o["tainted"], o["public_ok"] = False, True
                if how == "edit and reseal":
                    o["card_hash"] = cards.card_hash(o)
            edit_json(os.path.join(tr.run_dir, "cards", "provenance.json"), drop)
            rep = run.verify_freeze(tr.run_dir, expected_head=head, data_root=str(r.data_dir))
            self.assertFalse(rep["ok"], how)                                         # the manifest and the chain still hold the old hash

    def test_a_card_the_freeze_card_does_not_bind_is_still_tied_to_the_record_by_the_manifest(self):
        """runtime.json and server.json are not bound by the freeze card. An attacker who edits one and rewrites the manifest consistently is caught only because the manifest's hash is in the chain."""
        for name in ("runtime.json", "server.json"):
            r, tr, fz, head = frozen()
            cdir = os.path.join(tr.run_dir, "cards")
            edit_json(os.path.join(cdir, name), lambda o: o.update(capabilities=["train", "network"]))
            manifest = json.loads(pathlib.Path(cdir, "MANIFEST.json").read_text(encoding="utf-8"))
            manifest["cards"][name] = cards.sha256_hex(pathlib.Path(cdir, name).read_bytes())
            pathlib.Path(cdir, "MANIFEST.json").write_text(cards.canon(manifest), encoding="utf-8")
            rep = run.verify_freeze(tr.run_dir, expected_head=head)
            self.assertFalse(rep["checks"]["manifest_matches_chain"], name)
            self.assertFalse(rep["ok"], name)

    def test_every_way_of_editing_a_frozen_run_is_reported(self):
        for name in self.tamper_cases():
            r, tr, fz, head = frozen()
            self.tamper_cases()[name](tr, r)
            rep = run.verify_freeze(tr.run_dir, expected_head=head, data_root=str(r.data_dir))
            self.assertFalse(rep["ok"], name)
            self.assertTrue(rep["problems"], name)

    def test_an_edited_record_breaks_the_chain_and_a_cut_tail_is_caught_by_the_remembered_head(self):
        r, tr, fz, head = frozen()
        chain = pathlib.Path(tr.run_dir, "RUN.jsonl")
        keep = chain.read_text(encoding="utf-8")
        rows = keep.splitlines()
        chain.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")                      # cut off the freeze record itself
        self.assertFalse(run.verify_freeze(tr.run_dir)["ok"])                                 # no freeze record left
        chain.write_text(keep.replace('"step": 10', '"step": 11', 1), encoding="utf-8")
        self.assertFalse(run.verify_freeze(tr.run_dir)["checks"]["chain_valid"])

    def test_a_whole_run_rebuilt_by_an_attacker_does_not_match_the_head_you_kept(self):
        good, tr, fz, head = frozen()
        evil, etr, efz, ehead = frozen(steps=30, every=10)                                    # a different, internally consistent frozen run
        self.assertTrue(run.verify_freeze(etr.run_dir, data_root=str(evil.data_dir))["ok"])    # it verifies on its own...
        self.assertFalse(run.verify_freeze(etr.run_dir, expected_head=head)["ok"])             # ...but it is not the run whose head you remembered

    def test_verify_never_raises_on_garbage_and_missing_things(self):
        d = tempfile.mkdtemp()
        self.assertFalse(run.verify_freeze(d)["ok"])
        self.assertFalse(run.verify_freeze(os.path.join(d, "nope"))["ok"])
        r = rig()
        tr = r.new_run("notdone")
        self.assertFalse(run.verify_freeze(tr.run_dir)["ok"])
        r2, tr2, fz, head = frozen()
        for name in ("MANIFEST.json", "freeze.json"):
            pathlib.Path(tr2.run_dir, "cards", name).write_bytes(b"{broken")
            self.assertFalse(run.verify_freeze(tr2.run_dir)["ok"])

    def test_replay_needs_the_data_and_notices_a_trainer_that_is_not_deterministic(self):
        r, tr, fz, head = frozen()
        self.assertFalse(run.verify_freeze(tr.run_dir, replay=True)["ok"])                     # replay without data_root cannot pass
        rep = run.verify_freeze(tr.run_dir, replay=True, data_root=str(r.data_dir))
        self.assertTrue(rep["ok"])

        class NonDeterministic:
            name, deterministic = "numpy_mlp", False

        r2 = rig()
        tr2 = r2.new_run("nd", trainer=type("T", (run.trainers.NumpyMLP,), {"deterministic": False})())
        tr2.start(r2.start_commit())
        tr2.freeze()
        rep2 = run.verify_freeze(tr2.run_dir, replay=True, data_root=str(r2.data_dir))
        self.assertEqual(rep2["checks"]["replay"], "not replayable: the trainer is not deterministic")

    def test_a_replay_with_other_data_does_not_pass(self):
        r, tr, fz, head = frozen()
        other = tempfile.mkdtemp()
        pathlib.Path(other, "train.csv").write_bytes(b"0.5,0.5,1\n" * 60)
        rep = run.verify_freeze(tr.run_dir, replay=True, data_root=other)
        self.assertFalse(rep["ok"])


def _flip(path):
    raw = bytearray(pathlib.Path(path).read_bytes())
    raw[-1] ^= 1
    pathlib.Path(path).write_bytes(bytes(raw))


class Provenance(unittest.TestCase):
    def test_owned_data_gives_a_clean_permissive_model_and_library_data_taints_it(self):
        r, tr, fz, head = frozen()
        prov = json.loads(pathlib.Path(tr.run_dir, "cards", "provenance.json").read_text(encoding="utf-8"))
        model = json.loads(pathlib.Path(tr.run_dir, "cards", "model.json").read_text(encoding="utf-8"))
        self.assertEqual((prov["tainted"], prov["public_ok"], prov["license_class"], model["license_class"]), (False, True, "OWNED", "permissive"))
        m = {"gpl": {"repo": "gpl", "license_class": "STRONG_COPYLEFT"}}
        r2 = rig(sources=["gpl"], matrix=m)
        tr2 = r2.new_run("g")
        tr2.start(r2.start_commit())
        tr2.freeze()
        prov2 = json.loads(pathlib.Path(tr2.run_dir, "cards", "provenance.json").read_text(encoding="utf-8"))
        model2 = json.loads(pathlib.Path(tr2.run_dir, "cards", "model.json").read_text(encoding="utf-8"))
        self.assertEqual((prov2["tainted"], prov2["public_ok"], prov2["license_class"], model2["license_class"]), (True, False, "STRONG_COPYLEFT", "strong_copyleft"))

    def test_a_restrictive_base_model_makes_the_result_at_least_that_strict(self):
        r = rig()
        plan = run.make_plan("tinyrun", r.ecard, r.dcard, r.pre, "numpy_mlp", {"hidden": 2}, 1, 10, 5, base_model={"id": "some-base", "license_class": "NON_COMMERCIAL_OR_RESTRICTED"})
        tr = run.TrainingRun(str(r.t / "b"), plan, r.ecard, r.dcard, r.pre, str(r.data_dir), commitment_check=r.check)
        tr.start(r.mint(run.start_action(plan)))
        tr.freeze()
        prov = json.loads(pathlib.Path(tr.run_dir, "cards", "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual((prov["license_class"], prov["tainted"], prov["public_ok"]), ("NON_COMMERCIAL_OR_RESTRICTED", True, False))

    def test_the_model_card_matches_only_the_weights_it_was_made_for(self):
        from vega_pareah.cards import model_card
        r, tr, fz, head = frozen()
        model = json.loads(pathlib.Path(tr.run_dir, "cards", "model.json").read_text(encoding="utf-8"))
        wsha = cards.sha256_hex(pathlib.Path(tr.run_dir, "weights.bin").read_bytes())
        self.assertTrue(model_card.check_model(model, wsha))
        self.assertFalse(model_card.check_model(model, "00" * 32))

    def test_a_freeze_refuses_unfinished_runs_changed_weights_and_changed_data(self):
        r = rig()
        tr = r.new_run("u")
        with self.assertRaises(run.TrainRefused):
            tr.freeze()
        tr.start(r.start_commit())
        _flip(os.path.join(tr.run_dir, "weights.bin"))
        with self.assertRaises(run.TrainRefused):
            tr.freeze()
        r2 = rig()
        tr2 = r2.new_run("v")
        tr2.start(r2.start_commit())
        pathlib.Path(r2.data_dir, "train.csv").write_bytes(b"0.9,0.9,1\n" * 60)
        with self.assertRaises(Exception):
            tr2.freeze()

    def test_a_spent_commitment_is_on_the_record(self):
        r = rig()
        tr = r.new_run("s")
        c = r.start_commit()
        tr.start(c)
        spent = [x for x in tr.burn.read() if x.get("kind") == "COMMITMENT_USED"]
        self.assertEqual([x["purpose"] for x in spent], [run.start_action(r.plan)])


if __name__ == "__main__":
    unittest.main()
