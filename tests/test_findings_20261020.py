"""Regression tests (2026-10-20): a PRE revoked in a PreDeck did not stop a training run: the run checked the PRE card it held, never a deck."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support as sup  # noqa: E402
from vega_pareah.cards import pre as pre_mod  # noqa: E402
from vega_train import run  # noqa: E402


class RevokedPreStopsTraining(unittest.TestCase):
    def test_a_revoked_pre_cannot_start_a_run(self):
        r = sup.rig()
        deck = pre_mod.PreDeck()
        deck.add(r.pre)
        deck.revoke(r.pre["card_hash"])
        tr = r.new_run(pre_deck=deck)
        with self.assertRaises(run.TrainRefused) as cm:
            tr.start(r.start_commit())
        self.assertIn("revoked", str(cm.exception))

    def test_a_pre_revoked_during_the_run_halts_it_at_the_next_step(self):
        r = sup.rig(steps=40, every=10)
        deck = pre_mod.PreDeck()
        deck.add(r.pre)
        seen = []

        def listener(ev):
            seen.append(ev["event"])
            if ev["event"] == "checkpoint" and ev["step"] == 10:
                deck.revoke(r.pre["card_hash"])
        tr = r.new_run(pre_deck=deck, on_event=listener)
        out = tr.start(r.start_commit())
        self.assertEqual(out["status"], "halted")
        self.assertIn("revoked", out["reason"])
        self.assertEqual(out["last_checkpoint_step"], 10)

    def test_with_no_deck_or_an_unrevoked_pre_the_run_completes(self):
        r = sup.rig(steps=20, every=10)
        self.assertEqual(r.new_run(run_name="a").start(r.start_commit())["status"], "done")
        deck = pre_mod.PreDeck()
        deck.add(r.pre)
        r2 = sup.rig(steps=20, every=10)
        deck2 = pre_mod.PreDeck()
        deck2.add(r2.pre)
        self.assertEqual(r2.new_run(pre_deck=deck2).start(r2.start_commit())["status"], "done")


if __name__ == "__main__":
    unittest.main()
