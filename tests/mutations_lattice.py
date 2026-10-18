"""Mutation check for lattice_data.py: each rule removed in a COPY must make tests/test_lattice_data.py fail."""
import os, pathlib, shutil, subprocess, sys, tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
F = "lattice_data.py"
MUTATIONS = [
    ("altered cards are caught", F, 'return dkgcards.h(body) == card["dna_hash"]', "return True"),
    ("same content once", F, "if isinstance(h, str) and h in seen_hash:", "if False:"),
    ("tainted excluded by default", F, "if tainted and not allow_tainted:", "if False:"),
    ("licence class must be permissive or owned", F, ' or inh.get("license_class") not in OK_CLASSES', ""),
    ("missing licence info counts as tainted", F, 'bool(inh.get("taint_if_used", True))', "bool(inh.get('taint_if_used', False))"),
    ("unscored are labelled", F, '"band": "unscored"', '"band": 0'),
    ("held-out by the fixed hash rule", F, "train_ids, held_ids = data_card.split_ids(ids, holdout_permille, seed)", "train_ids, held_ids = ids, []"),
    ("per-repo cap", F, "if n < max_per_repo:", "if True:"),
    ("scores must be finite", F, " or not math.isfinite(v):", ":"),
    ("foundation first order", F, 'scored.sort(key=lambda c: (-sc[c["vega_id"]], c["vega_id"]))', 'scored.sort(key=lambda c: (sc[c["vega_id"]], c["vega_id"]))'),
    ("data card needs an intact plan", F, 'if not plan_intact(plan):\n        raise LatticeRefused("the plan is altered or not a lattice plan")\n    files =', 'if False:\n        raise LatticeRefused("the plan is altered or not a lattice plan")\n    files ='),
    ("taint carries to the data card", F, 'body["tainted"] = bool(card["tainted"] or plan["tainted"])', 'body["tainted"] = bool(card["tainted"])'),
    ("ablation removes a repo", F, 'p["train"], p["omitted_repo"] = keep, repo', 'p["train"], p["omitted_repo"] = plan["train"], repo'),
    ("links need the same plan", F, 'data.get("lattice_plan_sha256") != plan["card_hash"]', "False"),
    ("links need the same node list", F, 'if data.get("lattice_nodes") != [r["vega_id"] for r in plan["train"]]:', "if False:"),
    ("empty plan refused", F, "if not train:", "if False:"),
    ("repeated ids excluded", F, "elif vid in seen_id:", "elif False:"),
    ("no id means no node", F, "elif not isinstance(vid, str) or vid == dkgcards.NS:", "elif False:"),
    ("no path means no node", F, 'elif not isinstance(c.get("path"), str) or not c["path"]:', "elif False:"),
    ("kind filter", F, 'elif c.get("kind") not in kinds:', "elif False:"),
]
if __name__ != "__main__":
    MUTATIONS = []                       # importing this file (test discovery) must not start the mutation run
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
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", "test_lattice_data"], capture_output=True, text=True, env=env, cwd=str(REPO / "tests"), timeout=600)
    print(("killed   " if r.returncode != 0 else "SURVIVED ") + name, flush=True)
    if r.returncode == 0:
        survivors.append(name)
    shutil.rmtree(tmp, ignore_errors=True)
print("survivors:", survivors or "none")
