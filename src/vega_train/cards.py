"""cards.py - the card shape every training card shares: a JSON object, a kind, and a card_hash over everything else.

The hash is INTEGRITY, not authenticity: it shows a card was not edited after it was made. Who made it, and that the human approved the run, comes from the commitment
the run was started under (see run.py), never from the hash alone."""
import hashlib

from vega_harness.records import RecordRefused, canon


class CardRefused(ValueError):
    pass


def card_hash(card: dict) -> str:
    if not isinstance(card, dict):
        raise CardRefused("a card is an object")
    try:
        body = canon({k: v for k, v in card.items() if k != "card_hash"})
    except RecordRefused as exc:
        raise CardRefused(f"card is not plain JSON data: {exc}") from exc
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def seal(card: dict) -> dict:
    card = {k: v for k, v in card.items() if k != "card_hash"}
    card["card_hash"] = card_hash(card)
    return card


def intact(card, kind: str = None) -> bool:
    """True only for a card whose hash matches its content (and, if kind is given, whose kind matches). Never raises."""
    try:
        return isinstance(card, dict) and (kind is None or card.get("kind") == kind) and isinstance(card.get("card_hash"), str) and card_hash(card) == card["card_hash"]
    except CardRefused:
        return False


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
