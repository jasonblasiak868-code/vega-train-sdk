"""data_card.py - the DATA card: exactly which bytes a training run learned from, where they came from, and under what terms.

It records, per file: relative path, size, sha256, line count. It records NO file content. Provenance: a license class for the whole set and the library repos any of it came from;
the strictest class wins (the same rule as the use gate). tainted=True and public_ok=False whenever any source is a library repo or the class is not PERMISSIVE or OWNED, so a model
trained on it cannot be released as clean (the model card inherits this).

Safety (data folders are untrusted input): every file must resolve inside the root (symlinks and junctions included), have a plain relative name, and be under the size caps; a file
holding a credential-shaped value is REFUSED (training on a secret would bake it into weights that get shared) and the refusal names the file, never the value.

load_verified(card, root) reads each file ONCE, checks those bytes against the card, and returns them: the bytes trained on are the bytes hashed.
split_ids(ids, val_permille, seed) is a deterministic train/validation rule by hash, so the split can be re-derived and does not depend on file order."""
import hashlib
import os

from vega_custody import safepaths as sp
from vega_custody.dag.dag import contains_credential
from vega_harness import use_gate as ug
from vega_train import cards

OWNED = "OWNED"
CLASS_ORDER = list(ug.STRICT) + [OWNED]              # strictest first; OWNED (the user's own data) is the least strict
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_FILES = 100_000
SCAN_BYTES = 8 * 1024 * 1024                          # credential scan covers the first 8 MiB of each file; larger files are listed with scan_complete False


class DataRefused(ValueError):
    pass


def _class_of(name):
    if name == OWNED:
        return OWNED
    return ug.klass_of(name)


def strictest(classes):
    classes = [_class_of(c) for c in classes]
    return min(classes, key=CLASS_ORDER.index) if classes else ug.DEFAULT_CLASS


def build(name, root, files, license_class=OWNED, sources=(), matrix=None) -> dict:
    sp.safe_name(name)
    if isinstance(files, (str, bytes)) or not isinstance(files, (list, tuple)) or not files or len(files) > MAX_FILES:
        raise DataRefused(f"files is a non-empty list of at most {MAX_FILES} relative paths")
    if isinstance(sources, (str, bytes)) or any(not isinstance(s, str) for s in sources):
        raise DataRefused("sources is a list of library repo names")
    m = matrix if matrix is not None else ug.load_matrix()
    classes = [license_class] + [ug.klass(s, m) for s in sources]
    cls = strictest(classes)
    entries, total, seen = [], 0, set()
    for rel in sorted(files):
        if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or os.path.isabs(rel) or rel in seen:
            raise DataRefused(f"bad or repeated file entry {rel!r}")
        seen.add(rel)
        try:
            path = sp.contained(root, rel)
            size = os.path.getsize(path)
            if not os.path.isfile(path):
                raise DataRefused(f"{rel} is not a file")
            data = sp.read_capped(path, MAX_FILE_BYTES)
        except (OSError, sp.PathRefused) as exc:
            raise DataRefused(f"{rel}: {exc}") from exc
        total += size
        if total > MAX_TOTAL_BYTES:
            raise DataRefused(f"the data set is over the {MAX_TOTAL_BYTES} byte limit")
        head = data[:SCAN_BYTES].decode("utf-8", "replace")
        if contains_credential(head):
            raise DataRefused(f"{rel} holds a credential-shaped value: refusing to train on it (the value is not shown)")
        entries.append({"path": rel.replace("\\", "/"), "bytes": size, "sha256": hashlib.sha256(data).hexdigest(), "lines": data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1),
                        "scan_complete": len(data) <= SCAN_BYTES})
    tainted = bool(sources) or cls not in ("PERMISSIVE", OWNED)
    card = {"kind": "DATA", "name": name, "files": entries, "total_bytes": total, "license_class": cls, "declared_class": license_class, "library_sources": sorted(sources),
            "tainted": tainted, "public_ok": not tainted, "set_sha256": cards.sha256_hex("\n".join(f"{e['path']}:{e['sha256']}" for e in entries).encode("utf-8"))}
    return cards.seal(card)


def load_verified(card: dict, root: str) -> dict:
    """{relative path: bytes}, every file read once and checked against the card. Any mismatch, missing file or altered card raises DataRefused."""
    if not cards.intact(card, "DATA"):
        raise DataRefused("the data card is altered or not a DATA card")
    out = {}
    for e in card["files"]:
        try:
            data = sp.read_capped(sp.contained(root, e["path"]), MAX_FILE_BYTES)
        except (OSError, sp.PathRefused) as exc:
            raise DataRefused(f"{e['path']}: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != e["sha256"] or len(data) != e["bytes"]:
            raise DataRefused(f"{e['path']} changed since the card was made")
        out[e["path"]] = data
    return out


def split_ids(ids, val_permille=100, seed=0):
    """(train_ids, val_ids) by a fixed hash rule, independent of order. val_permille is 0..1000."""
    if isinstance(val_permille, bool) or not isinstance(val_permille, int) or not 0 <= val_permille <= 1000:
        raise DataRefused("val_permille is a whole number from 0 to 1000")
    train, val = [], []
    for i in ids:
        bucket = int(hashlib.sha256(f"{seed}|{i}".encode("utf-8")).hexdigest()[:8], 16) % 1000
        (val if bucket < val_permille else train).append(i)
    return train, val
