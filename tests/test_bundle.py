"""Sealing a training run's cards into one superproof and opening them again."""
import copy
import pathlib
import secrets
import tempfile
import unittest

from vega_custody.dag.dag import Dag
from vega_custody.ledger.backend_local import LocalBackend
from vega_custody.ledger.ledger import Ledger
from vega_superproof.seal import builtin, hashsig, signature
from vega_train import bundle, cards


def world():
    t = pathlib.Path(tempfile.mkdtemp())
    return Dag([t / "c.jsonl"], [t / "l.jsonl"]), Ledger([LocalBackend(t / "led.jsonl")])


def make_cards():
    return [cards.seal({"kind": "ENV", "python": "3.12"}), cards.seal({"kind": "DATA", "name": "d", "files": []}), cards.seal({"kind": "MODEL", "name": "m"})]


class Bundle(unittest.TestCase):
    def setUp(self):
        self.dag, self.led = world()
        self.k1, self.k2 = secrets.token_bytes(32), secrets.token_bytes(32)
        self.sk, self.pub = signature.new_ed25519()

    def test_seal_and_open_round_trip_with_every_card_checked(self):
        cs = make_cards()
        s = bundle.seal_bundle(cs, self.k1, self.k2, self.sk, self.pub, self.dag, self.led)
        out = bundle.open_bundle(s, self.k1, self.k2)
        self.assertEqual(out, {c["kind"]: c for c in cs})
        self.assertNotIn("python", str(s["envelope"]))                              # the cards are not readable from the sealed bundle

    def test_an_altered_card_or_a_repeated_kind_seals_nothing(self):
        cs = make_cards()
        bad = copy.deepcopy(cs)
        bad[0]["python"] = "2.7"
        before = len(self.dag.content.entries())
        for lst in (bad, cs + [cs[0]], [], ["text"], [{"kind": "ENV"}], "x", {"a": 1}, cs * 30):
            with self.assertRaises(bundle.BundleRefused):
                bundle.seal_bundle(lst, self.k1, self.k2, self.sk, self.pub, self.dag, self.led)
        self.assertEqual(len(self.dag.content.entries()), before)

    def test_wrong_keys_fail_and_a_bundle_made_of_something_else_is_refused(self):
        s = bundle.seal_bundle(make_cards(), self.k1, self.k2, self.sk, self.pub, self.dag, self.led)
        with self.assertRaises(Exception):
            bundle.open_bundle(s, secrets.token_bytes(32), self.k2)
        from vega_custody.capsule import capsule as cap
        from vega_superproof.seal import superproof
        for text in ("not json", "{}", '[{"kind":"ENV"}]'):
            other = superproof.seal(cap.make(text, {"source": "x"}), self.k1, self.k2, self.sk, self.pub, self.dag, self.led)
            with self.assertRaises(bundle.BundleRefused):
                bundle.open_bundle(other, self.k1, self.k2)

    def test_the_quantum_proof_tier_works_for_a_bundle(self):
        builtin.enable_quantum_proof(replace=True)
        hs = hashsig.Signer(3)
        s = bundle.seal_bundle(make_cards(), self.k1, self.k2, hs, hs.public, self.dag, self.led, preset="quantum_proof")
        from vega_superproof.seal import superproof
        self.assertTrue(superproof.verify_seal(s, hs.public, self.led, accept_tiers=(2, 3), require_signature=hashsig.ALG))
        self.assertEqual(set(bundle.open_bundle(s, self.k1, self.k2)), {"ENV", "DATA", "MODEL"})


if __name__ == "__main__":
    unittest.main()
