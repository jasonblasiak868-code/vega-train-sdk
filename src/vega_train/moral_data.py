"""moral_data.py - training examples built from legal precedents and statutes the owner SUPPLIES, so a model is shown, again and again, that a prohibited act has a cost, who bears it, and what the right move was instead.

INPUT is a list of RECORDS you provide (this module reads no law and scrapes nothing; the citations are yours and should be checked by an attorney):
  {id, situation, act, legal: True/False, rule (a rule id in the conduct code), citation, jurisdiction, consequence (what happened or what the law provides), severity (tier 1 to 5), alternative}
OUTPUT is deterministic text from fixed templates, no model involved:
  REFUSE examples      for an illegal act: the situation, the model's check against the rule (with the citation and tier), the consequence, who it falls on (others and the group, not only the actor), and the right alternative
  PREFERENCE PAIRS     chosen = the refusal with its reasons, rejected = the act stated at the same high level
  ALLOW examples       for a legal act: the check is done, the rule is named as not broken, and the work proceeds (this guards against a model that refuses everything)
  REFLECTION text      the meta-audit as the model should do it before acting: what am I about to do, which rule applies, what tier, what does the group lose, what is the right way
The 'how NOT to' side is the SENSITIVE side: rejected texts describe prohibited acts. So: acts are limited to 300 characters of HIGH-LEVEL description, records marked as holding operational detail are refused, credential-shaped
text is refused, and the written dataset is flagged private: it must never be released or published, and its card says so.
HONEST LIMITS: templates teach a pattern; they do not teach judgment, and a model can learn to recite the reasons without acting on them: measure behaviour with behavior_eval.py on held-out probes. A record about a crime is not
an instruction to commit it, but the rejected side is still harmful-looking text: keep it private. Nothing here is legal advice."""
import hashlib
import json
import os

from vega_custody.dag.dag import contains_credential
from vega_train import cards, conduct as C, data_card

MAX_ACT = 300


class MoralDataRefused(ValueError):
    pass


def _check_record(rec, conduct_card):
    if not isinstance(rec, dict) or not isinstance(rec.get("id"), str) or not C.ID.fullmatch(rec["id"]):
        raise MoralDataRefused("each record has a plain id")
    for k, lo, hi in (("situation", 10, 800), ("act", 5, MAX_ACT), ("consequence", 5, 800), ("citation", 3, 300), ("jurisdiction", 1, 40)):
        v = rec.get(k)
        if not isinstance(v, str) or not lo <= len(v.strip()) <= hi:
            raise MoralDataRefused("record %s: %s is %d to %d characters (acts are a high-level description, not instructions)" % (rec["id"], k, lo, hi))
        if contains_credential(v):
            raise MoralDataRefused("record %s: %s holds a credential-shaped value" % (rec["id"], k))
    if rec.get("operational_detail") is True:
        raise MoralDataRefused("record %s holds operational detail: it is not used" % rec["id"])
    if not isinstance(rec.get("legal"), bool):
        raise MoralDataRefused("record %s: legal is true or false" % rec["id"])
    if rec["rule"] not in {r["id"] for r in conduct_card["rules"]}:
        raise MoralDataRefused("record %s names a rule the conduct code does not have" % rec["id"])
    if not rec["legal"]:
        if isinstance(rec.get("severity"), bool) or rec.get("severity") not in (1, 2, 3, 4, 5):
            raise MoralDataRefused("record %s: an illegal act has a severity tier 1 to 5" % rec["id"])
        if not isinstance(rec.get("alternative"), str) or not 5 <= len(rec["alternative"]) <= 500:
            raise MoralDataRefused("record %s: an illegal act comes with the right alternative" % rec["id"])


def _reflection(rec, rule):
    return ("Before acting I check what I am about to do: %s. The rule that applies is %s (%s, %s), tier %d. If I break it the cost is: %s. That cost falls on other people and on the group I work with, "
            "and one serious violation can undo everything the group has done. So: " % (rec["act"].rstrip("."), rule["id"], rule["source"]["citation"], rule["source"]["jurisdiction"], rule["severity"], rec["consequence"].rstrip(".")))


def build_examples(records: list, conduct_card: dict, goal_request: str = "", holdout_permille: int = 100, seed: int = 0) -> dict:
    """-> {'train': [examples], 'probes': [held-out probes], 'counts': {...}}. Examples are deterministic; the held-out records never appear in the training examples."""
    if not cards.intact(conduct_card, "CONDUCT"):
        raise MoralDataRefused("the conduct card is altered")
    if not isinstance(records, list) or not 1 <= len(records) <= 100_000:
        raise MoralDataRefused("1 to 100000 records")
    seen, clean = set(), []
    for rec in records:
        _check_record(rec, conduct_card)
        key = hashlib.sha256((rec["situation"] + "|" + rec["act"]).encode("utf-8")).hexdigest()
        if rec["id"] in seen or key in seen:
            continue
        seen.update((rec["id"], key))
        clean.append(rec)
    train_ids, held_ids = data_card.split_ids([r["id"] for r in clean], holdout_permille, seed)
    held = set(held_ids)
    rules = {r["id"]: r for r in conduct_card["rules"]}
    train, probes = [], []
    for rec in clean:
        rule = rules[rec["rule"]]
        prompt = (rec["situation"] + ("\nGoal: " + goal_request if goal_request else "")).strip()
        if rec["id"] in held:
            probes.append({"id": rec["id"], "prompt": prompt, "expected": "ALLOW" if rec["legal"] else rule["required"] if rule["required"] != "CHECK_THEN_ALLOW" else "ALLOW", "severity": 0 if rec["legal"] else rec["severity"], "rule": rule["id"]})
            continue
        if rec["legal"]:
            resp = _reflection(rec, rule) + "this is allowed: it does not break the rule. I proceed, and I keep the record of the check."
            train.append({"id": rec["id"] + "-allow", "kind": "allow", "prompt": prompt, "response": resp, "rule": rule["id"], "private_negative": False})
        else:
            alt = rec["alternative"].rstrip(".")
            refusal = _reflection(rec, rule) + "I will not do it. Instead: %s." % alt
            train.append({"id": rec["id"] + "-refuse", "kind": "refuse", "prompt": prompt, "response": refusal, "rule": rule["id"], "private_negative": False})
            train.append({"id": rec["id"] + "-pair", "kind": "preference", "prompt": prompt, "chosen": refusal, "rejected": "I will go ahead: %s." % rec["act"].rstrip("."),
                          "rule": rule["id"], "private_negative": True})
    return {"train": train, "probes": probes, "counts": {"records": len(clean), "train_examples": len(train), "probes": len(probes), "duplicates_dropped": len(records) - len(clean)}}


def write_dataset(examples: list, path) -> dict:
    """Write the examples as JSON lines, one file, flagged private; returns {'path', 'sha256', 'examples', 'private_negative'}. Build the DATA card over this file with data_card.build and add the private flags with card_flags()."""
    if not isinstance(examples, list) or not examples:
        raise MoralDataRefused("no examples to write")
    lines = [json.dumps(e, sort_keys=True, ensure_ascii=False) for e in examples]
    body = ("\n".join(lines) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(body)
    return {"path": os.fspath(path), "sha256": hashlib.sha256(body).hexdigest(), "examples": len(examples), "private_negative": sum(1 for e in examples if e.get("private_negative"))}


def private_card(data: dict, dataset: dict) -> dict:
    """Mark a DATA card as holding prohibited-act descriptions: never public, never releasable, whatever its licence class says."""
    if not cards.intact(data, "DATA"):
        raise MoralDataRefused("the data card is altered")
    body = {k: v for k, v in data.items() if k != "card_hash"}
    body.update({"contains_negative_examples": dataset["private_negative"] > 0, "private": True, "tainted": True, "public_ok": False, "moral_dataset_sha256": dataset["sha256"]})
    return cards.seal(body)
