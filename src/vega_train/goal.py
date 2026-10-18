"""goal.py - the GOAL REQUEST BAR: how a person tells the training system who is asking and what result they want, in a form the rest of the system can check.

A person types what they want a model trained to do. That text is KEPT VERBATIM on a GOAL card together with:
  who is asking     a reference to the person and the identity assurance level they reached when they asked (from the identity SDK; a request below the minimum level is refused: training is a significant act)
  the result        success tests: concrete probes with the verdict expected for each (REFUSE, ESCALATE or ALLOW) so 'done' can be MEASURED, not claimed (behavior_eval.py)
  the constraints   plain statements the training must respect
  the conduct code  the hash of the signed CONDUCT card the training aims at (conduct.py): the moral compass the person chose, with its sources
  who drafted it    'human' or the name of a drafting model: a model may DRAFT the structure from the person's words (draft_from_request), it never approves it
A goal is READY only when (1) the requester has signed it, (2) the conduct code it names is signed by a human and its hash matches, and (3) the assurance level is high enough. Only a ready goal may be the aim of a training plan.
This module does not decide whether a goal is good or lawful: it makes the request explicit, attributable and testable. The training itself still needs the human-committed run (run.py)."""
import hashlib

from vega_train import cards, conduct as C

MIN_ASSURANCE = 2
VERDICTS = ("REFUSE", "ESCALATE", "ALLOW")


class GoalRefused(ValueError):
    pass


def _s(v, lo, hi, name):
    if not isinstance(v, str) or not lo <= len(v.strip()) <= hi:
        raise GoalRefused("%s is %d to %d characters" % (name, lo, hi))
    return v


def make_goal(request_text: str, requester_ref: str, assurance_level: int, conduct_card: dict, success_tests: list, constraints=(), drafted_by: str = "human", min_assurance: int = MIN_ASSURANCE) -> dict:
    _s(request_text, 10, 4000, "the request")
    if not isinstance(requester_ref, str) or not 3 <= len(requester_ref) <= 80:
        raise GoalRefused("requester_ref is a short reference to the person")
    if isinstance(assurance_level, bool) or assurance_level not in (0, 1, 2, 3, 4):
        raise GoalRefused("assurance_level is the identity level 0 to 4")
    if assurance_level < min_assurance:
        raise GoalRefused("the requester's identity level (%d) is below the minimum (%d) for asking for training" % (assurance_level, min_assurance))
    if not cards.intact(conduct_card, "CONDUCT"):
        raise GoalRefused("the conduct card is altered or is not one")
    if not isinstance(success_tests, list) or not 1 <= len(success_tests) <= 500:
        raise GoalRefused("1 to 500 success tests")
    tests, ids = [], set()
    rule_ids = {r["id"] for r in conduct_card["rules"]}
    for t in success_tests:
        if not isinstance(t, dict) or not isinstance(t.get("id"), str) or not C.ID.fullmatch(t["id"]) or t["id"] in ids:
            raise GoalRefused("each test has a unique plain id")
        ids.add(t["id"])
        _s(t.get("prompt"), 3, 2000, "a test prompt")
        if t.get("expect") not in VERDICTS:
            raise GoalRefused("test %s: expect is one of %s" % (t["id"], VERDICTS))
        if t.get("rule") is not None and t["rule"] not in rule_ids:
            raise GoalRefused("test %s names a rule the conduct code does not have" % t["id"])
        tests.append({"id": t["id"], "prompt": t["prompt"], "expect": t["expect"], "rule": t.get("rule")})
    if isinstance(constraints, (str, bytes)) or not isinstance(constraints, (list, tuple)) or len(constraints) > 50 or not all(isinstance(c, str) and 3 <= len(c) <= 500 for c in constraints):
        raise GoalRefused("constraints are up to 50 short statements")
    if not isinstance(drafted_by, str) or not 1 <= len(drafted_by) <= 60:
        raise GoalRefused("drafted_by is 'human' or the drafting model's name")
    return cards.seal({"kind": "GOAL", "request": request_text, "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(), "requester": requester_ref, "assurance": assurance_level,
                       "conduct": conduct_card["card_hash"], "success_tests": tests, "constraints": list(constraints), "drafted_by": drafted_by})


def draft_from_request(request_text: str, requester_ref: str, assurance_level: int, conduct_card: dict, drafter, drafter_name: str) -> dict:
    """A drafting model proposes the structure (success tests and constraints) from the person's own words. The result is a draft card marked with the model's name: the person must still read it and sign it."""
    if not callable(drafter):
        raise GoalRefused("no drafter was supplied: a human writes the structure")
    try:
        proposal = drafter(request_text, conduct_card)
        tests, constraints = proposal["success_tests"], proposal.get("constraints", [])
    except GoalRefused:
        raise
    except Exception as exc:                                                          # noqa: BLE001
        raise GoalRefused("the drafter failed or returned nonsense: %s" % type(exc).__name__) from exc
    return make_goal(request_text, requester_ref, assurance_level, conduct_card, tests, constraints, drafted_by="model:" + str(drafter_name)[:50])


def is_ready(goal: dict, goal_approval, conduct_card: dict, conduct_approval, requester_public: str) -> list:
    """-> list of reasons it is NOT ready (empty means ready)."""
    problems = []
    if not cards.intact(goal, "GOAL"):
        return ["the goal card is altered"]
    if goal["conduct"] != conduct_card.get("card_hash") or not cards.intact(conduct_card, "CONDUCT"):
        problems.append("the conduct card is not the one the goal names")
    if not C.is_approved(conduct_card, conduct_approval):
        problems.append("the conduct code is not signed by a human")
    if not C.is_approved(goal, goal_approval) or (isinstance(goal_approval, dict) and goal_approval.get("public") != requester_public):
        problems.append("the requester has not signed this goal")
    if goal["assurance"] < MIN_ASSURANCE:
        problems.append("the identity level is below the minimum")
    return problems
