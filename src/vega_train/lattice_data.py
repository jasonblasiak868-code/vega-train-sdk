"""lattice_data.py - lattice-governed training data: pick what a model learns from the DKG lattice, record exactly which lattice nodes it learned from, and keep the
test set clean.

The DKG cards (vega-dkg-lattice-sdk) are the nodes: each has a vega_id, a repo, a path, a content hash and the licence it inherits. This module turns a set of cards into a
SELECTION PLAN and then into a normal DATA card, so everything downstream (training run, freeze, verification) is unchanged:

  select(cards, scores=...)      which nodes to train on, which to hold out, which were left out and WHY. Nothing is read from disk here.
  to_data_card(plan, ...)        the DATA card for those files, tagged with the plan's hash and the lattice node ids it came from
  ablation_plans(plan)           leave-one-repo-out plans with the SAME held-out set, to measure what each source taught the model
  training_links(plan, card)     the TRAINED_ON / HELD_OUT records to write back into the lattice, only if the card really came from this plan

RULES (each one is tested):
  - a card whose dna_hash does not match its content was altered: excluded, never repaired
  - the same content (same sha256) appears once, wherever it came from, so a held-out file cannot also be a training file under another name
  - the held-out set is chosen by a fixed hash rule on the node id (re-derivable, independent of order and of the scores)
  - a node that inherits a tainted or non-permissive licence is excluded unless the caller allows tainted data; if allowed, the plan says tainted
  - ordering uses SCORES the CALLER supplies (for example the galaxy SDK's convergence); a node with no score is 'unscored': it is kept, placed last, and said so
    (unbacked is not false), never given an invented number
  - a per-repo cap keeps one source from dominating
HONEST LIMITS: this chooses and records; it does not train. A curriculum order does not make a model better by itself: measure it (the ablation plans exist for that).
Scores are only as good as whoever computed them. Duplicate detection is by exact content hash: a lightly edited copy is a different file."""
import hashlib
import math

from vega_dkg import cards as dkgcards
from vega_train import cards, data_card

STRATEGIES = ("foundation_first", "frontier_first", "flat")
OK_CLASSES = ("PERMISSIVE", "OWNED")


class LatticeRefused(ValueError):
    pass


def card_intact(card) -> bool:
    """True only if the card's dna_hash matches its content. Never raises."""
    try:
        if not isinstance(card, dict) or not isinstance(card.get("dna_hash"), str):
            return False
        body = {k: v for k, v in card.items() if k != "dna_hash"}
        return dkgcards.h(body) == card["dna_hash"]
    except (TypeError, ValueError):
        return False


def _score(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise LatticeRefused("scores are finite numbers; a node with no score is left out of the dict, never given NaN or a flag")
    return float(v)


def _rank_key(vid):
    return hashlib.sha256(("rank|" + vid).encode("utf-8")).hexdigest()


def select(node_cards, scores=None, strategy="foundation_first", bands=3, max_per_repo=None, allow_tainted=False, holdout_permille=100, seed=0, kinds=("doc_node",)) -> dict:
    if strategy not in STRATEGIES:
        raise LatticeRefused("strategy is one of %s" % (STRATEGIES,))
    if isinstance(bands, bool) or not isinstance(bands, int) or not 1 <= bands <= 20:
        raise LatticeRefused("bands is 1 to 20")
    if max_per_repo is not None and (isinstance(max_per_repo, bool) or not isinstance(max_per_repo, int) or max_per_repo < 1):
        raise LatticeRefused("max_per_repo is a positive whole number")
    if isinstance(node_cards, (str, bytes, dict)) or not hasattr(node_cards, "__iter__"):
        raise LatticeRefused("node_cards is a list of DKG cards")
    scores = {} if scores is None else scores
    if not isinstance(scores, dict):
        raise LatticeRefused("scores is a dict of vega_id -> number")
    sc = {k: _score(v) for k, v in scores.items()}

    excluded, seen_hash, seen_id, pool = [], {}, set(), []
    for c in sorted((x for x in node_cards if isinstance(x, dict)), key=lambda x: str(x.get("vega_id"))):
        vid = c.get("vega_id")
        why = None
        if not card_intact(c):
            why = "card_altered"
        elif c.get("kind") not in kinds:
            why = "kind_not_selected"
        elif not isinstance(vid, str) or vid == dkgcards.NS:
            why = "no_vega_id"
        elif not isinstance(c.get("path"), str) or not c["path"]:
            why = "no_path"
        elif vid in seen_id:
            why = "duplicate_id"
        else:
            inh = c.get("inherited") if isinstance(c.get("inherited"), dict) else {}
            tainted = bool(inh.get("taint_if_used", True)) or inh.get("license_class") not in OK_CLASSES
            if tainted and not allow_tainted:
                why = "tainted_or_not_permissive"
            else:
                h = c.get("sha256")
                if isinstance(h, str) and h in seen_hash:
                    why = "same_content_as:" + seen_hash[h]
                else:
                    if isinstance(h, str):
                        seen_hash[h] = vid
                    seen_id.add(vid)
                    pool.append(dict(c, _tainted=tainted))
        if why:
            excluded.append({"vega_id": vid if isinstance(vid, str) else None, "why": why})

    ids = [c["vega_id"] for c in pool]
    train_ids, held_ids = data_card.split_ids(ids, holdout_permille, seed)
    held = set(held_ids)
    train_pool = [c for c in pool if c["vega_id"] not in held]
    if max_per_repo is not None:
        per, kept = {}, []
        for c in sorted(train_pool, key=lambda x: _rank_key(x["vega_id"])):
            n = per.get(c.get("repo"), 0)
            if n < max_per_repo:
                per[c.get("repo")] = n + 1
                kept.append(c)
            else:
                excluded.append({"vega_id": c["vega_id"], "why": "repo_cap"})
        train_pool = kept
    scored = [c for c in train_pool if c["vega_id"] in sc]
    unscored = [c for c in train_pool if c["vega_id"] not in sc]
    if strategy == "foundation_first":
        scored.sort(key=lambda c: (-sc[c["vega_id"]], c["vega_id"]))
    elif strategy == "frontier_first":
        scored.sort(key=lambda c: (sc[c["vega_id"]], c["vega_id"]))
    else:
        scored.sort(key=lambda c: _rank_key(c["vega_id"]))
    unscored.sort(key=lambda c: c["vega_id"])
    per_band = max(1, -(-len(scored) // bands)) if scored else 1
    train = []
    for i, c in enumerate(scored):
        train.append({"vega_id": c["vega_id"], "repo": c.get("repo"), "path": c["path"], "sha256": c.get("sha256"), "dna_hash": c["dna_hash"], "band": i // per_band, "score": sc[c["vega_id"]]})
    for c in unscored:
        train.append({"vega_id": c["vega_id"], "repo": c.get("repo"), "path": c["path"], "sha256": c.get("sha256"), "dna_hash": c["dna_hash"], "band": "unscored", "score": None})
    held_rows = [{"vega_id": c["vega_id"], "repo": c.get("repo"), "path": c["path"], "sha256": c.get("sha256"), "dna_hash": c["dna_hash"]} for c in sorted(pool, key=lambda x: x["vega_id"]) if c["vega_id"] in held]
    tainted = any(c["_tainted"] for c in train_pool)
    plan = {"kind": "LATTICE_PLAN", "strategy": strategy, "bands": bands, "seed": seed, "holdout_permille": holdout_permille, "max_per_repo": max_per_repo,
            "allow_tainted": bool(allow_tainted), "tainted": tainted, "train": train, "held_out": held_rows,
            "excluded": sorted(excluded, key=lambda e: (str(e["vega_id"]), e["why"])), "omitted_repo": None,
            "counts": {"train": len(train), "held_out": len(held_rows), "excluded": len(excluded), "unscored": len(unscored)}}
    if not train:
        raise LatticeRefused("nothing is left to train on after the rules were applied (see the exclusion reasons by selecting with a looser rule)")
    plan["selection_sha256"] = cards.sha256_hex("\n".join("%s:%s" % (r["vega_id"], r["dna_hash"]) for r in train).encode("utf-8"))
    return cards.seal(plan)


def plan_intact(plan) -> bool:
    return cards.intact(plan, "LATTICE_PLAN")


def to_data_card(plan: dict, name: str, root: str, license_class: str = data_card.OWNED, sources=(), matrix=None) -> dict:
    """The DATA card for this plan's training files. `sources` are the library repos the files came from (their licence is applied by the use gate)."""
    if not plan_intact(plan):
        raise LatticeRefused("the plan is altered or not a lattice plan")
    files = sorted({r["path"] for r in plan["train"]})
    card = data_card.build(name, root, files, license_class=license_class, sources=sources, matrix=matrix)
    body = {k: v for k, v in card.items() if k != "card_hash"}
    body["lattice_plan_sha256"] = plan["card_hash"]
    body["lattice_nodes"] = [r["vega_id"] for r in plan["train"]]
    body["held_out_nodes"] = [r["vega_id"] for r in plan["held_out"]]
    body["tainted"] = bool(card["tainted"] or plan["tainted"])
    body["public_ok"] = not body["tainted"]
    return cards.seal(body)


def ablation_plans(plan: dict):
    """One plan per repo, each with that repo's training nodes removed and the SAME held-out set, so a difference in results is that repo's contribution."""
    if not plan_intact(plan):
        raise LatticeRefused("the plan is altered or not a lattice plan")
    out = []
    for repo in sorted({r["repo"] for r in plan["train"] if r["repo"] is not None}):
        keep = [r for r in plan["train"] if r["repo"] != repo]
        if not keep:
            continue
        p = {k: v for k, v in plan.items() if k != "card_hash"}
        p["train"], p["omitted_repo"] = keep, repo
        p["counts"] = dict(plan["counts"], train=len(keep))
        p["selection_sha256"] = cards.sha256_hex("\n".join("%s:%s" % (r["vega_id"], r["dna_hash"]) for r in keep).encode("utf-8"))
        out.append(cards.seal(p))
    return out


def training_links(plan: dict, data: dict) -> list:
    """TRAINED_ON and HELD_OUT records for the lattice, only for a DATA card that was really built from this plan."""
    if not plan_intact(plan) or not cards.intact(data, "DATA") or data.get("lattice_plan_sha256") != plan["card_hash"]:
        raise LatticeRefused("this DATA card was not built from this plan")
    if data.get("lattice_nodes") != [r["vega_id"] for r in plan["train"]]:
        raise LatticeRefused("the card's node list does not match the plan")
    links = [{"link": "TRAINED_ON", "vega_id": r["vega_id"], "data_card": data["card_hash"], "tainted": data["tainted"]} for r in plan["train"]]
    links += [{"link": "HELD_OUT", "vega_id": r["vega_id"], "data_card": data["card_hash"], "tainted": False} for r in plan["held_out"]]
    return links
