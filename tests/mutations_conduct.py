"""Mutation check for the conduct stack: each rule removed in a COPY must make tests/test_conduct.py fail."""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
CO, GO, MO, BE = "conduct.py", "goal.py", "moral_data.py", "behavior_eval.py"
MUTATIONS = [
    ("signature needs an intact card", CO, "if not cards.intact(card) or not isinstance(sig_hex, str) or len(sig_hex) != 128:", "if not isinstance(sig_hex, str) or len(sig_hex) != 128:"),
    ("weights rise with severity", CO, " or any(w[k] > w[k + 1] for k in (1, 2, 3, 4))", ""),
    ("a serious violation erases", CO, 'erased = erased or r["severity"] >= conduct["erase_at"]', "erased = False"),
    ("erased means zero task reward", CO, "counted = 0.0 if erased else task", "counted = task"),
    ("the team's reward is erased too", CO, 'task = 0.0 if any_erase else r["task"]', 'task = r["task"]'),
    ("the others share the cost", CO, 'shared = conduct["share_ratio"] * others', "shared = 0.0"),
    ("the strictest rule wins", CO, 'worst = max(hits, key=lambda r: (STRICTNESS[r["required"]], r["severity"]))', "worst = hits[0]"),
    ("triggers match whole words", CO, 're.search(r"(?<!\\w)" + re.escape(t) + r"(?!\\w)", low)', "(t in low)"),
    ("an unsigned code governs nothing", CO, "if not is_approved(conduct, approval):\n            raise ConductRefused", "if False:\n            raise ConductRefused"),
    ("a broken check never allows", CO, 'd = Decision("ESCALATE", [], "the check could not run: a human decides")', 'd = Decision("ALLOW", [], "x")'),
    ("an unreadable action is escalated", CO, 'd = Decision("ESCALATE", [], "the action could not be read: a human decides")', 'd = Decision("ALLOW", [], "x")'),
    ("a check that cannot be recorded is escalated", CO, 'return Decision("ESCALATE", d.rules, "the check could not be recorded: a human decides")', "return d"),
    ("the classifier's answer is checked", CO, "if not isinstance(got, list) or not all(isinstance(x, str) for x in got):", "if False:"),
    ("minimum identity level", GO, "if assurance_level < min_assurance:", "if False:"),
    ("the conduct card must be intact", GO, 'if not cards.intact(conduct_card, "CONDUCT"):\n        raise GoalRefused("the conduct card is altered or is not one")', 'if False:\n        raise GoalRefused("the conduct card is altered or is not one")'),
    ("test ids are unique", GO, ' or t["id"] in ids:', ":"),
    ("a test's rule must exist", GO, 'if t.get("rule") is not None and t["rule"] not in rule_ids:', "if False:"),
    ("a model draft is labelled", GO, 'drafted_by="model:" + str(drafter_name)[:50]', 'drafted_by="human"'),
    ("ready: the conduct card is the named one", GO, 'if goal["conduct"] != conduct_card.get("card_hash") or not cards.intact(conduct_card, "CONDUCT"):', "if False:"),
    ("ready: the conduct code is signed", GO, "if not C.is_approved(conduct_card, conduct_approval):", "if False:"),
    ("ready: the requester signed", GO, ' or (isinstance(goal_approval, dict) and goal_approval.get("public") != requester_public)', ""),
    ("ready: identity level", GO, 'if goal["assurance"] < MIN_ASSURANCE:', "if False:"),
    ("no operational detail", MO, 'if rec.get("operational_detail") is True:', "if False:"),
    ("no credentials in records", MO, "if contains_credential(v):", "if False:"),
    ("a record names a real rule", MO, 'if rec["rule"] not in {r["id"] for r in conduct_card["rules"]}:', "if False:"),
    ("held-out records stay out", MO, 'train_ids, held_ids = data_card.split_ids([r["id"] for r in clean], holdout_permille, seed)', 'train_ids, held_ids = [r["id"] for r in clean], []'),
    ("duplicates dropped", MO, 'if rec["id"] in seen or key in seen:', "if False:"),
    ("the rejected side is private", MO, '"rule": rule["id"], "private_negative": True})', '"rule": rule["id"], "private_negative": False})'),
    ("the data card is never public", MO, '"private": True, "tainted": True, "public_ok": False', '"private": False, "tainted": False, "public_ok": True'),
    ("probe verdicts follow the rule", MO, 'else rule["required"] if rule["required"] != "CHECK_THEN_ALLOW" else "ALLOW"', 'else "REFUSE"'),
    ("acts stay high-level", MO, "MAX_ACT = 300", "MAX_ACT = 30000"),
    ("probes must be held out", BE, 'if prompt_hash(p["prompt"]) in train_prompt_sha256:', "if False:"),
    ("a crash is the worst case", BE, 'got, errors = "ERROR", errors + 1', 'got, errors = "REFUSE", errors + 1'),
    ("the bound is the Wilson upper bound", BE, "centre = p + z * z / (2 * n)", "centre = p"),
    ("too few probes fail", BE, 'if report["guard_probes"] < min_guard_probes:', "if False:"),
    ("an error is an over-refusal on allowed probes", BE, 'if got in ("REFUSE", "ERROR"):', 'if got in ("REFUSE",):'),
    ("probe ids are unique", BE, ' or p.get("id") in seen:', ":"),
]
if __name__ != "__main__":
    MUTATIONS = []
survivors = []
for name, fname, old, new in MUTATIONS:
    tmp = pathlib.Path(tempfile.mkdtemp())
    shutil.copytree(REPO / "src" / "vega_train", tmp / "vega_train")
    f = tmp / "vega_train" / fname
    text = f.read_text(encoding="utf-8")
    if old not in text:
        print("CANNOT APPLY", name)
        survivors.append(name + " (could not apply)")
        continue
    f.write_text(text.replace(old, new, 1), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "test_conduct"], capture_output=True, text=True, env=env, cwd=str(REPO / "tests"), timeout=600)
    print(("killed   " if r.returncode != 0 else "SURVIVED ") + name, flush=True)
    if r.returncode == 0:
        survivors.append(name)
    shutil.rmtree(tmp, ignore_errors=True)
print("survivors:", survivors or "none")
