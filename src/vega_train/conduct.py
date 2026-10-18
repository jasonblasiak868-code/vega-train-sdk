"""conduct.py - a CONDUCT CODE the training aims at, the reward arithmetic that follows from it, and the runtime pre-action check that applies it. A human writes the code, a human signs it; nothing here decides what is right.

THE CONDUCT CARD holds rules. Each rule has an id, a text, a SOURCE (a statute, a precedent, a policy or the owner's own value, with a citation and a jurisdiction label: the code records where each rule came from and does not
pretend that illegal and immoral are the same thing), a SEVERITY tier from 1 to 5, what is REQUIRED when it would be broken (REFUSE, ESCALATE to a human, or CHECK_THEN_ALLOW), the words that TRIGGER the check, and the severity
weights: how heavily a violation of each tier counts. The tiers and weights are DATA the owner (with an attorney) supplies, for example by mapping the severity of the legal consequence to a tier; the SDK does not know any law.
THE REWARD ARITHMETIC (episode_reward, team_reward) is the training signal the owner described:
  - the task reward counts only if the conduct was clean: a violation takes a penalty by severity weight, and a violation at or above the ERASE tier ERASES the episode's task reward: many good steps do not outweigh one serious violation;
  - in a TEAM the consequence is shared: when one member commits an erasing violation every member's task reward for that episode is erased and each non-violating member carries a share of the penalty. Everyone then has a stake in every
    member's conduct, which is the point: the group, not one agent's reward, is what a violation damages.
THE META-AUDIT (MetaAuditor.check) is the same code applied at run time, BEFORE an action: the action text is matched against the rules' trigger words (or a classifier you plug in); the result is the strictest requirement among the
rules hit, with the rule ids and citations; a plug-in that crashes or returns nonsense yields ESCALATE, never ALLOW; every check is written to a hash chain when one is supplied.
HONEST LIMITS: (1) Training changes how likely a model is to act a certain way. It does not make a model understand or feel anything and it gives no guarantee: the external gates (commitments, containment) stay in force and this never replaces
them. (2) A reward for 'no violations' can teach a model to hide them: violations must be detected by outside audit, not by the model's own report. (3) Law differs by place and changes; the rule sources are only as good as the citations
supplied, and nothing here is legal advice. (4) Trigger-word matching is crude and both over- and under-fires: measure it (behavior_eval.py) and prefer a better classifier."""
import hashlib
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from vega_train import cards

REQUIRED = ("REFUSE", "ESCALATE", "CHECK_THEN_ALLOW")
STRICTNESS = {"CHECK_THEN_ALLOW": 1, "ESCALATE": 2, "REFUSE": 3}
SOURCE_KINDS = ("statute", "precedent", "policy", "owner")
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")
DEFAULT_WEIGHTS = {1: 1.0, 2: 3.0, 3: 9.0, 4: 27.0, 5: 81.0}


class ConductRefused(ValueError):
    pass


class Signer:
    """A human's signing key for approving cards. In real use the private key stays on the person's own device."""

    def __init__(self):
        self._sk = Ed25519PrivateKey.generate()
        self.public = self._sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()

    def sign(self, card: dict) -> str:
        if not cards.intact(card):
            raise ConductRefused("only an intact card can be signed")
        return self._sk.sign(b"VEGA-TRAIN-APPROVE-1|" + card["card_hash"].encode("ascii")).hex()


def signature_ok(public_hex, card: dict, sig_hex) -> bool:
    try:
        if not cards.intact(card) or not isinstance(sig_hex, str) or len(sig_hex) != 128:
            return False
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(bytes.fromhex(sig_hex), b"VEGA-TRAIN-APPROVE-1|" + card["card_hash"].encode("ascii"))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def _text(v, lo, hi, name):
    if not isinstance(v, str) or not lo <= len(v.strip()) <= hi or not v.isprintable():
        raise ConductRefused("%s is %d to %d printable characters" % (name, lo, hi))
    return v.strip()


def make_conduct(name: str, rules: list, weights: dict = None, erase_at: int = 3, share_ratio: float = 0.25) -> dict:
    """A sealed CONDUCT card. It is a draft until a human signs it (see approve / is_approved)."""
    if not isinstance(name, str) or not ID.fullmatch(name):
        raise ConductRefused("name is a plain id")
    if not isinstance(rules, list) or not 1 <= len(rules) <= 500:
        raise ConductRefused("1 to 500 rules")
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if sorted(w) != [1, 2, 3, 4, 5] or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != v for v in w.values()) or any(w[k] > w[k + 1] for k in (1, 2, 3, 4)):
        raise ConductRefused("weights are one non-negative number per tier 1 to 5, not decreasing as severity rises")
    if isinstance(erase_at, bool) or erase_at not in (1, 2, 3, 4, 5):
        raise ConductRefused("erase_at is a tier from 1 to 5")
    if isinstance(share_ratio, bool) or not isinstance(share_ratio, (int, float)) or not 0 <= share_ratio <= 1:
        raise ConductRefused("share_ratio is 0 to 1")
    out, seen = [], set()
    for r in rules:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not ID.fullmatch(r["id"]) or r["id"] in seen:
            raise ConductRefused("each rule has a unique plain id")
        seen.add(r["id"])
        src = r.get("source")
        if not isinstance(src, dict) or src.get("kind") not in SOURCE_KINDS or not isinstance(src.get("citation"), str) or not 3 <= len(src["citation"]) <= 300 or not isinstance(src.get("jurisdiction"), str) or not 1 <= len(src["jurisdiction"]) <= 40:
            raise ConductRefused("rule %s: a source with a kind, a citation and a jurisdiction label" % r["id"])
        if isinstance(r.get("severity"), bool) or r.get("severity") not in (1, 2, 3, 4, 5) or r.get("required") not in REQUIRED:
            raise ConductRefused("rule %s: a severity tier 1 to 5 and a requirement" % r["id"])
        trig = r.get("triggers")
        if not isinstance(trig, list) or not 1 <= len(trig) <= 50 or not all(isinstance(t, str) and 2 <= len(t) <= 60 and t == t.casefold() for t in trig):
            raise ConductRefused("rule %s: 1 to 50 lower-case trigger words or phrases" % r["id"])
        out.append({"id": r["id"], "text": _text(r.get("text"), 5, 600, "rule text"), "source": {"kind": src["kind"], "citation": src["citation"].strip(), "jurisdiction": src["jurisdiction"].strip()},
                    "severity": r["severity"], "required": r["required"], "triggers": sorted(set(trig))})
    return cards.seal({"kind": "CONDUCT", "name": name, "rules": sorted(out, key=lambda x: x["id"]), "weights": {str(k): float(v) for k, v in w.items()}, "erase_at": erase_at, "share_ratio": float(share_ratio)})


def approve(card: dict, signer: Signer) -> dict:
    return {"public": signer.public, "sig": signer.sign(card)}


def is_approved(card: dict, approval) -> bool:
    return isinstance(approval, dict) and signature_ok(approval.get("public"), card, approval.get("sig"))


def _weight(card, severity):
    return card["weights"][str(severity)]


def _rule(card, rid):
    for r in card["rules"]:
        if r["id"] == rid:
            return r
    raise ConductRefused("no such rule: %r" % (rid,))


def episode_reward(conduct: dict, task_rewards: list, violations: list) -> dict:
    """violations: [{'rule': id, 'step': index}]. -> {'task', 'penalty', 'total', 'erased', 'violations'}."""
    if not cards.intact(conduct, "CONDUCT"):
        raise ConductRefused("the conduct card is altered")
    if not isinstance(task_rewards, list) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in task_rewards):
        raise ConductRefused("task_rewards is a list of numbers")
    if not isinstance(violations, list):
        raise ConductRefused("violations is a list")
    task = float(sum(task_rewards))
    penalty, erased, detail = 0.0, False, []
    for v in violations:
        if not isinstance(v, dict):
            raise ConductRefused("a violation is {rule, step}")
        r = _rule(conduct, v.get("rule"))
        penalty += _weight(conduct, r["severity"])
        erased = erased or r["severity"] >= conduct["erase_at"]
        detail.append({"rule": r["id"], "severity": r["severity"], "citation": r["source"]["citation"]})
    counted = 0.0 if erased else task
    return {"task": counted, "task_before": task, "penalty": penalty, "total": counted - penalty, "erased": erased, "violations": detail}


def team_reward(conduct: dict, episodes: dict) -> dict:
    """episodes: {member: {'task_rewards': [...], 'violations': [...]}}. -> {'members': {member: result}, 'team_total', 'erased'}.
    If ANY member erases, every member's task reward is erased; each other member carries share_ratio of the violator's penalty."""
    if not isinstance(episodes, dict) or not 1 <= len(episodes) <= 200:
        raise ConductRefused("episodes is a dict of 1 to 200 members")
    solo = {m: episode_reward(conduct, e.get("task_rewards", []), e.get("violations", [])) for m, e in episodes.items()}
    any_erase = any(r["erased"] for r in solo.values())
    members = {}
    for m, r in solo.items():
        others = sum(o["penalty"] for k, o in solo.items() if k != m)
        shared = conduct["share_ratio"] * others
        task = 0.0 if any_erase else r["task"]
        members[m] = {"task": task, "own_penalty": r["penalty"], "shared_penalty": shared, "total": task - r["penalty"] - shared, "erased": r["erased"], "erased_by_team": any_erase and not r["erased"]}
    return {"members": members, "team_total": sum(x["total"] for x in members.values()), "erased": any_erase}


class Decision:
    def __init__(self, verdict, rules, reason):
        self.verdict, self.rules, self.reason = verdict, rules, reason

    def as_dict(self):
        return {"verdict": self.verdict, "rules": self.rules, "reason": self.reason}


class MetaAuditor:
    """The runtime check before an action. classifier(action_text, context) -> list of rule ids, optional; without one, trigger words are used."""

    def __init__(self, conduct: dict, approval=None, classifier=None, dag=None):
        if not cards.intact(conduct, "CONDUCT"):
            raise ConductRefused("the conduct card is altered")
        if not is_approved(conduct, approval):
            raise ConductRefused("the conduct code has not been signed by a human: it cannot govern anything yet")
        self.c, self.classifier, self.dag = conduct, classifier, dag

    def _hits(self, text, context):
        if self.classifier is not None:
            got = self.classifier(text, context)
            if not isinstance(got, list) or not all(isinstance(x, str) for x in got):
                raise ValueError("the classifier must return a list of rule ids")
            return [_rule(self.c, g) for g in got]
        low = re.sub(r"\s+", " ", text.casefold())
        return [r for r in self.c["rules"] if any(re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", low) for t in r["triggers"])]

    def check(self, action_text, context=None) -> Decision:
        if not isinstance(action_text, str) or not action_text.strip() or len(action_text) > 20000:
            d = Decision("ESCALATE", [], "the action could not be read: a human decides")
        else:
            try:
                hits = self._hits(action_text, context)
            except Exception:                                      # noqa: BLE001 - a broken check never allows
                d = Decision("ESCALATE", [], "the check could not run: a human decides")
            else:
                if not hits:
                    d = Decision("ALLOW", [], "no rule applies")
                else:
                    worst = max(hits, key=lambda r: (STRICTNESS[r["required"]], r["severity"]))
                    verdict = "ALLOW_AFTER_CHECK" if worst["required"] == "CHECK_THEN_ALLOW" else worst["required"]
                    d = Decision(verdict, [{"id": r["id"], "severity": r["severity"], "citation": r["source"]["citation"]} for r in hits], "%s: %s" % (worst["id"], worst["text"]))
        if self.dag is not None:
            try:
                self.dag.record("CONDUCT_CHECK", {"conduct": self.c["card_hash"], "action_sha256": hashlib.sha256(str(action_text).encode("utf-8")).hexdigest(), "verdict": d.verdict, "rules": [r["id"] for r in d.rules]})
            except Exception:                                      # noqa: BLE001 - if the check cannot be recorded, the answer is a human's call
                return Decision("ESCALATE", d.rules, "the check could not be recorded: a human decides")
        return d
