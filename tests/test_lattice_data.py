"""Lattice-governed training data: selection, hold-out hygiene, licence rules, ablation plans and the links written back to the lattice."""
import copy
import hashlib
import os
import pathlib
import tempfile
import unittest

from vega_dkg import cards as dkgcards
from vega_train import cards, data_card, lattice_data as L


def card(vid, repo, path, text, kind="doc_node", cls="OWNED", taint=False):
    c = {"kind": kind, "vega_id": vid, "repo": repo, "path": path, "sha256": hashlib.sha256(text.encode()).hexdigest(),
         "code_nodes_in_file": 1, "inherited": {"license_class": cls, "taint_if_used": taint}}
    c["dna_hash"] = dkgcards.h(c)
    return c


def world(n=40, repos=("r1", "r2", "r3", "r4")):
    root = pathlib.Path(tempfile.mkdtemp())
    out = []
    for i in range(n):
        repo = repos[i % len(repos)]
        rel = "%s/f%d.txt" % (repo, i)
        (root / repo).mkdir(exist_ok=True)
        text = "content %d\n" % i
        (root / rel).write_text(text)
        out.append(card("V%03d" % i, repo, rel, text))
    return root, out


class Selection(unittest.TestCase):
    def test_a_plan_is_made_and_sealed_and_is_deterministic(self):
        root, cs = world()
        scores = {c["vega_id"]: i / 100 for i, c in enumerate(cs)}
        p1 = L.select(cs, scores, seed=3)
        p2 = L.select(list(reversed(cs)), scores, seed=3)
        self.assertEqual(p1["card_hash"], p2["card_hash"])                         # input order does not matter
        self.assertTrue(L.plan_intact(p1))
        self.assertEqual(p1["counts"]["train"] + p1["counts"]["held_out"] + p1["counts"]["excluded"], 40)
        self.assertNotEqual(p1["card_hash"], L.select(cs, scores, seed=4)["card_hash"])

    def test_training_and_held_out_never_overlap_and_the_rule_is_rederivable(self):
        root, cs = world(60)
        p = L.select(cs, holdout_permille=200, seed=9)
        train, held = {r["vega_id"] for r in p["train"]}, {r["vega_id"] for r in p["held_out"]}
        self.assertFalse(train & held)
        ids = [c["vega_id"] for c in cs]
        self.assertEqual(held, set(data_card.split_ids(ids, 200, 9)[1]))
        self.assertTrue(held)

    def test_the_same_content_under_two_names_is_trained_once_and_cannot_leak_into_the_test_set(self):
        root, cs = world(10)
        dup = card("V999", "r9", "r9/copy.txt", "content 0\n")                      # identical bytes to V000
        p = L.select(cs + [dup], holdout_permille=0)
        self.assertEqual(sum(1 for e in p["excluded"] if e["why"].startswith("same_content_as")), 1)
        ids = [r["vega_id"] for r in p["train"]]
        self.assertIn("V000", ids)
        self.assertNotIn("V999", ids)

    def test_altered_cards_and_wrong_kinds_are_excluded_not_repaired(self):
        root, cs = world(10)
        bad = copy.deepcopy(cs[0])
        bad["path"] = "r1/other.txt"                                                # edited after its dna_hash was made
        code = card("C1", "r1", "r1/x.py", "x", kind="code_node")
        p = L.select([bad, code] + cs[1:], holdout_permille=0)
        why = {e["vega_id"]: e["why"] for e in p["excluded"]}
        self.assertEqual(why["V000"], "card_altered")
        self.assertEqual(why["C1"], "kind_not_selected")
        self.assertEqual(len(L.select([bad, code] + cs[1:], kinds=("doc_node", "code_node"), holdout_permille=0)["train"]), 10)

    def test_cards_without_a_usable_id_or_path_or_with_a_repeated_id_are_excluded(self):
        root, cs = world(6)
        no_id = card(dkgcards.NS, "r1", "r1/a.txt", "zz1")
        none_id = card(None, "r1", "r1/b.txt", "zz2")
        no_path = card("P1", "r1", "", "zz3")
        twin = card("V000", "r1", "r1/twin.txt", "different bytes")                   # same id as a real node, other content
        p = L.select(cs + [no_id, none_id, no_path, twin], holdout_permille=0)
        whys = sorted(e["why"] for e in p["excluded"])
        self.assertEqual(whys, ["duplicate_id", "no_path", "no_vega_id", "no_vega_id"])
        self.assertEqual(len(p["train"]), 6)

    def test_tainted_and_non_permissive_nodes_are_left_out_unless_the_caller_allows_them(self):
        root, cs = world(6)
        t1 = card("T1", "rt", "rt/a.txt", "a", cls="STRONG_COPYLEFT", taint=True)
        t2 = card("T2", "rt", "rt/b.txt", "b", cls="PERMISSIVE", taint=True)
        t3 = card("T3", "rt", "rt/c.txt", "c", cls="UNKNOWN", taint=False)
        p = L.select(cs + [t1, t2, t3], holdout_permille=0)
        self.assertEqual({e["vega_id"] for e in p["excluded"]}, {"T1", "T2", "T3"})
        self.assertFalse(p["tainted"])
        q = L.select(cs + [t1, t2, t3], holdout_permille=0, allow_tainted=True)
        self.assertTrue(q["tainted"])
        self.assertEqual(q["counts"]["train"], 9)
        silent = card("S1", "rs", "rs/a.txt", "s", cls="OWNED")
        del silent["inherited"]["taint_if_used"]                                     # a card that says nothing about taint is not assumed clean
        silent["dna_hash"] = dkgcards.h({k: v for k, v in silent.items() if k != "dna_hash"})
        self.assertEqual(L.select(cs + [silent], holdout_permille=0)["excluded"][0]["why"], "tainted_or_not_permissive")
        missing = card("M1", "rm", "rm/a.txt", "m")
        del missing["inherited"]
        missing["dna_hash"] = dkgcards.h({k: v for k, v in missing.items() if k != "dna_hash"})
        self.assertEqual(L.select(cs + [missing], holdout_permille=0)["excluded"][0]["why"], "tainted_or_not_permissive")   # no licence info = treated as tainted

    def test_unscored_nodes_are_kept_placed_last_and_said_so(self):
        root, cs = world(8)
        scores = {c["vega_id"]: float(i) for i, c in enumerate(cs[:5])}
        p = L.select(cs, scores, holdout_permille=0)
        bands = [r["band"] for r in p["train"]]
        self.assertEqual(bands[-3:], ["unscored"] * 3)
        self.assertTrue(all(r["score"] is None for r in p["train"][-3:]))
        self.assertEqual(p["counts"]["unscored"], 3)

    def test_strategies_order_by_the_callers_scores(self):
        root, cs = world(9)
        scores = {c["vega_id"]: float(i) for i, c in enumerate(cs)}
        down = [r["score"] for r in L.select(cs, scores, "foundation_first", holdout_permille=0)["train"]]
        up = [r["score"] for r in L.select(cs, scores, "frontier_first", holdout_permille=0)["train"]]
        self.assertEqual(down, sorted(down, reverse=True))
        self.assertEqual(up, sorted(up))
        p = L.select(cs, scores, "foundation_first", bands=3, holdout_permille=0)
        self.assertEqual([r["band"] for r in p["train"]], [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_a_cap_per_repo_stops_one_source_dominating(self):
        root, cs = world(40, repos=("big", "small"))
        p = L.select(cs, holdout_permille=0, max_per_repo=5)
        self.assertEqual({r: sum(1 for x in p["train"] if x["repo"] == r) for r in ("big", "small")}, {"big": 5, "small": 5})
        self.assertEqual(sum(1 for e in p["excluded"] if e["why"] == "repo_cap"), 30)

    def test_bad_arguments_and_an_empty_result_are_refused(self):
        root, cs = world(5)
        for kw in ({"strategy": "random"}, {"bands": 0}, {"bands": True}, {"bands": 99}, {"max_per_repo": 0}, {"max_per_repo": True}):
            with self.assertRaises(L.LatticeRefused):
                L.select(cs, **kw)
        for bad in ("text", b"x", {"a": 1}, 5, None):
            with self.assertRaises(L.LatticeRefused):
                L.select(bad)
        for badscore in (float("nan"), float("inf"), True, "1", None):
            with self.assertRaises(L.LatticeRefused):
                L.select(cs, {"V000": badscore})
        with self.assertRaises(L.LatticeRefused):
            L.select(cs, scores=[1, 2])
        with self.assertRaises(L.LatticeRefused):
            L.select([], holdout_permille=0)
        with self.assertRaises(L.LatticeRefused):
            L.select([card("X", "r", "r/a", "a", cls="UNKNOWN", taint=True)])


class DataCardAndLinks(unittest.TestCase):
    def setUp(self):
        self.root, self.cs = world(20)
        self.plan = L.select(self.cs, {c["vega_id"]: i for i, c in enumerate(self.cs)}, holdout_permille=150, seed=1)

    def test_the_data_card_names_the_nodes_and_the_plan_and_is_a_normal_data_card(self):
        d = L.to_data_card(self.plan, "lattice-set", str(self.root))
        self.assertTrue(cards.intact(d, "DATA"))
        self.assertEqual(d["lattice_plan_sha256"], self.plan["card_hash"])
        self.assertEqual(d["lattice_nodes"], [r["vega_id"] for r in self.plan["train"]])
        self.assertEqual(len(d["files"]), len(self.plan["train"]))
        self.assertTrue(set(d["held_out_nodes"]).isdisjoint(d["lattice_nodes"]))
        got = data_card.load_verified(d, str(self.root))
        self.assertEqual(len(got), len(self.plan["train"]))
        self.assertFalse(d["tainted"])

    def test_an_altered_plan_or_a_changed_file_is_refused(self):
        bad = copy.deepcopy(self.plan)
        bad["train"][0]["path"] = "r1/other.txt"
        with self.assertRaises(L.LatticeRefused):
            L.to_data_card(bad, "x", str(self.root))
        d = L.to_data_card(self.plan, "lattice-set", str(self.root))
        victim = self.plan["train"][0]["path"]
        (self.root / victim).write_text("changed after the card was made")
        with self.assertRaises(data_card.DataRefused):
            data_card.load_verified(d, str(self.root))

    def test_a_tainted_plan_makes_a_tainted_card(self):
        t = card("T1", "rt", "rt/a.txt", "a", cls="PERMISSIVE", taint=True)
        (self.root / "rt").mkdir()
        (self.root / "rt/a.txt").write_text("a")
        p = L.select(self.cs + [t], holdout_permille=0, allow_tainted=True)
        d = L.to_data_card(p, "tainted-set", str(self.root))
        self.assertTrue(d["tainted"])
        self.assertFalse(d["public_ok"])

    def test_ablation_plans_omit_one_repo_each_and_keep_the_test_set_fixed(self):
        abl = L.ablation_plans(self.plan)
        self.assertEqual({a["omitted_repo"] for a in abl}, {"r1", "r2", "r3", "r4"})
        for a in abl:
            self.assertTrue(L.plan_intact(a))
            self.assertEqual(a["held_out"], self.plan["held_out"])
            self.assertTrue(all(r["repo"] != a["omitted_repo"] for r in a["train"]))
            self.assertNotEqual(a["selection_sha256"], self.plan["selection_sha256"])
        bad = copy.deepcopy(self.plan)
        bad["seed"] = 99
        with self.assertRaises(L.LatticeRefused):
            L.ablation_plans(bad)

    def test_links_back_to_the_lattice_only_for_a_card_built_from_this_plan(self):
        d = L.to_data_card(self.plan, "lattice-set", str(self.root))
        links = L.training_links(self.plan, d)
        self.assertEqual(sum(1 for l in links if l["link"] == "TRAINED_ON"), len(self.plan["train"]))
        self.assertEqual(sum(1 for l in links if l["link"] == "HELD_OUT"), len(self.plan["held_out"]))
        self.assertTrue(all(l["data_card"] == d["card_hash"] for l in links))
        other = L.select(self.cs, {c["vega_id"]: i for i, c in enumerate(self.cs)}, holdout_permille=150, seed=1, bands=5)    # same nodes, different plan
        self.assertEqual([r["vega_id"] for r in other["train"]], [r["vega_id"] for r in self.plan["train"]])
        with self.assertRaises(L.LatticeRefused):
            L.training_links(other, d)
        forged = cards.seal(dict({k: v for k, v in d.items() if k != "card_hash"}, lattice_nodes=["V000"]))
        with self.assertRaises(L.LatticeRefused):
            L.training_links(self.plan, forged)
        with self.assertRaises(L.LatticeRefused):
            L.training_links(self.plan, dict(d, name="edited"))                       # card hash no longer matches


if __name__ == "__main__":
    unittest.main()
