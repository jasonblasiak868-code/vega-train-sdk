"""env_card.py - the ENVIRONMENT card: what machine and software a training run happened on, so a result can be compared with, and reproduced under, the same conditions.

It records: Python version and build, operating system, CPU count, the versions of the training-relevant packages (read from package metadata, nothing is imported), the GPU
name/memory/driver as nvidia-smi reports them (a bounded call; no GPU or no nvidia-smi is recorded as such, never invented), whether the torch build is CPU-only, and the value of a
SHORT ALLOW-LIST of non-secret determinism settings. It records NO other environment variable: environments hold tokens and keys, and a card is meant to be shared.

fingerprint(card) hashes the stable parts (not the capture time), so two captures on the same setup compare equal. drift(card, observed) lists exactly what differs.
HONEST LIMIT: a card describes the environment; it does not make a run reproducible. GPU kernels and thread scheduling can differ between identical environments."""
import importlib.metadata
import os
import platform
import re
import sys
import time

from vega_custody import proc
from vega_custody.dag.dag import contains_credential
from vega_train import cards

PACKAGES = ("numpy", "torch", "transformers", "safetensors", "peft", "accelerate", "datasets", "tokenizers", "bitsandbytes", "scipy")
ENV_ALLOWLIST = ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
_SAFE_VALUE = re.compile(r"[A-Za-z0-9_,.:;=-]{0,64}")
_GPU_LINE = re.compile(r"^(?P<name>[A-Za-z0-9 ._()+/-]{1,80}), (?P<mem>\d{1,7}), (?P<driver>[0-9][0-9.]{0,15})$")
NVIDIA_SMI = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
STABLE_EXCLUDED = ("captured_utc", "card_hash", "fingerprint")


def _packages(names):
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def parse_gpus(text: str):
    """Only lines of exactly the expected shape count; anything else (an error banner, a hostile string) is ignored, not trusted."""
    gpus = []
    for line in text.splitlines()[:16]:
        m = _GPU_LINE.match(line.strip())
        if m:
            gpus.append({"name": m["name"].strip(), "memory_mib": int(m["mem"]), "driver": m["driver"]})
    return gpus


def _query_gpus(run=proc.run_bounded):
    try:
        r = run(NVIDIA_SMI, timeout=10, cap=16 * 1024)
    except (OSError, ValueError):
        return [], "nvidia-smi not available"
    if r["status"] != "ok" or r["returncode"] != 0:
        return [], "nvidia-smi did not answer (%s)" % r["status"]
    gpus = parse_gpus(r["out"].decode("utf-8", "replace"))
    return gpus, ("" if gpus else "nvidia-smi answered but no GPU line was understood")


def capture(packages=PACKAGES, query_gpu=True, run=proc.run_bounded, environ=None) -> dict:
    environ = os.environ if environ is None else environ
    gpus, note = _query_gpus(run) if query_gpu else ([], "GPU query skipped")
    pk = _packages(packages)
    card = {"kind": "ENV", "python": platform.python_version(), "implementation": platform.python_implementation(), "executable_name": os.path.basename(sys.executable),
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "packages": pk, "torch_build": ("cpu" if "+cpu" in pk.get("torch", "") else ("accelerated" if "torch" in pk else "absent")),
            "gpus": gpus, "gpu_note": note, "gpu_source": "nvidia-smi (self-reported label, not verified)",
            "settings": {k: environ[k] for k in ENV_ALLOWLIST if k in environ and _SAFE_VALUE.fullmatch(str(environ[k])) and not contains_credential(str(environ[k]))},
            "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    card["fingerprint"] = fingerprint(card)
    return cards.seal(card)


def fingerprint(card: dict) -> str:
    return cards.sha256_hex(cards.canon({k: v for k, v in card.items() if k not in STABLE_EXCLUDED}).encode("utf-8"))


def drift(card: dict, observed: dict) -> list:
    """What differs between a recorded environment and a fresh capture (an empty list = same setup). Raises nothing: an altered card is reported as such."""
    if not cards.intact(card, "ENV"):
        return ["the recorded environment card is altered or not an ENV card"]
    out = []
    for k in sorted((set(card) | set(observed)) - set(STABLE_EXCLUDED)):
        if card.get(k) != observed.get(k):
            out.append(f"{k}: recorded {card.get(k)!r} now {observed.get(k)!r}")
    return out
