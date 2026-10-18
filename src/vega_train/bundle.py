"""bundle.py - seal a training run's cards (environment, data, PRE, model, freeze, lattice plan) into ONE superproof, and open it again with every card re-checked.

The cards are the evidence of what was trained and under what terms; sealing them with the superproof layer gives them the layered encryption, the lineage root and the
minted top root, so the bundle can be stored or handed over without the cards being readable or silently editable. Any preset the superproof offers can be used
(for the quantum-proof tier enable it first, see vega_superproof.seal.builtin).
  seal_bundle(card_list, key1, key2, signing_key, signing_public, dag, ledger, preset)   every card must be intact (hash matches) or nothing is sealed
  open_bundle(sealed, key1, key2) -> {kind: card}                                          every card is checked intact again; two cards of one kind are refused
HONEST LIMIT: the bundle proves the cards are unchanged since sealing. It does not prove the cards are TRUE (that comes from the run's own checks and commitment)."""
import json

from vega_custody.capsule import capsule as cap
from vega_superproof.seal import superproof
from vega_train import cards

MAX_CARDS = 64


class BundleRefused(ValueError):
    pass


def _check(card_list):
    if isinstance(card_list, (str, bytes, dict)) or not hasattr(card_list, "__len__") or not 1 <= len(card_list) <= MAX_CARDS:
        raise BundleRefused("a bundle is 1 to %d cards" % MAX_CARDS)
    kinds = []
    for c in card_list:
        if not cards.intact(c):
            raise BundleRefused("a card is altered or is not a card: nothing was sealed")
        kinds.append(c["kind"])
    if len(set(kinds)) != len(kinds):
        raise BundleRefused("two cards of the same kind: a bundle holds one card per kind")


def seal_bundle(card_list, key1, key2, signing_key, signing_public, dag, ledger, preset: str = "standard") -> dict:
    card_list = list(card_list)
    _check(card_list)
    body = json.dumps(sorted(card_list, key=lambda c: c["kind"]), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    c = cap.make(body, {"source": "train-bundle"})
    return superproof.seal(c, key1, key2, signing_key, signing_public, dag, ledger, preset)


def open_bundle(sealed: dict, key1, key2) -> dict:
    c = superproof.open_seal(sealed, key1, key2)
    try:
        listing = json.loads(cap.data_of(c))
    except (ValueError, TypeError) as exc:
        raise BundleRefused("the sealed capsule is not a card bundle") from exc
    if not isinstance(listing, list):
        raise BundleRefused("the sealed capsule is not a card bundle")
    _check(listing)
    return {x["kind"]: x for x in listing}
