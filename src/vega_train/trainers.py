"""trainers.py - the training back ends a governed run can drive, and the rules every back end must follow.

A trainer is a small object: init(config, files) / step(n) / state_bytes() / load_state(bytes) / weights_bytes(). The RUN (run.py) owns governance; the trainer owns only numbers.

Rules that hold for every back end here:
  * NO PICKLE. State is serialised by hand (numpy) or with safetensors (torch). A checkpoint file is never loaded with a format that can run code on load.
  * Serialisation is CANONICAL: the same state always gives the same bytes (no timestamps, no zip metadata), so a checkpoint's hash means something and a replay can match it.
  * `deterministic` is stated honestly. NumpyMLP is deterministic on one machine (float64, seeded generator, no threads). TorchTrainer is deterministic for a full-batch CPU run on one
    machine; it is not claimed to reproduce across machines or on a GPU.
  * Loss must stay finite: step() returns the loss and the run halts on NaN or infinity.

NumpyMLP trains a one-hidden-layer network (or logistic regression when hidden=0) on a CSV whose last column is a 0/1 label. It exists so the whole governed path can be exercised and
replayed with no download; it is a real trainer, but a small one. For a language model you plug in your own Trainer (see TorchTrainer for the shape)."""
import csv
import io
import json
import math
import struct

import numpy as np


class TrainerRefused(ValueError):
    pass


def parse_csv(files: dict, names=None):
    """Rows of floats from the named (or all) CSV files, last column is the label. Returns (X, y). Non-numeric or ragged rows raise TrainerRefused."""
    rows = []
    for name in sorted(names or files):
        text = files[name].decode("utf-8", "replace")
        for i, rec in enumerate(csv.reader(io.StringIO(text, newline="")), 1):
            if not rec:
                continue
            try:
                rows.append([float(x) for x in rec])
            except ValueError as exc:
                raise TrainerRefused(f"{name} line {i}: not numeric") from exc
    if not rows:
        raise TrainerRefused("no training rows")
    width = len(rows[0])
    if width < 2 or any(len(r) != width for r in rows):
        raise TrainerRefused("rows must all have the same width, at least two columns (features then a 0/1 label)")
    a = np.array(rows, dtype=np.float64)
    if not np.isfinite(a).all():
        raise TrainerRefused("training data contains NaN or infinity")
    y = a[:, -1]
    if not np.isin(y, (0.0, 1.0)).all():
        raise TrainerRefused("the last column must be a 0/1 label")
    return a[:, :-1], y


def pack_arrays(arrays: dict, meta: dict = None) -> bytes:
    """Canonical bytes: a JSON header (names sorted, dtype, shape) then the raw little-endian buffers in that order. No timestamps, no pickle."""
    names = sorted(arrays)
    header = {"meta": meta or {}, "arrays": [{"name": n, "dtype": str(arrays[n].dtype), "shape": list(arrays[n].shape)} for n in names]}
    h = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return b"VTA1" + struct.pack("<I", len(h)) + h + b"".join(np.ascontiguousarray(arrays[n]).astype(arrays[n].dtype.newbyteorder("<"), copy=False).tobytes() for n in names)


def unpack_arrays(blob: bytes):
    if not blob.startswith(b"VTA1") or len(blob) < 8:
        raise TrainerRefused("not a vega checkpoint")
    (n,) = struct.unpack("<I", blob[4:8])
    if n > len(blob) - 8:
        raise TrainerRefused("checkpoint header is longer than the file")
    try:
        header = json.loads(blob[8:8 + n].decode("utf-8"))
        pos, out = 8 + n, {}
        for spec in header["arrays"]:
            dt = np.dtype(spec["dtype"])
            if dt.kind not in "fiu" or dt.hasobject:
                raise TrainerRefused("checkpoint holds a data type that is not plain numbers")
            count = int(np.prod(spec["shape"], dtype=np.int64)) if spec["shape"] else 1
            size = count * dt.itemsize
            if size < 0 or pos + size > len(blob):
                raise TrainerRefused("checkpoint is shorter than its header says")
            out[spec["name"]] = np.frombuffer(blob[pos:pos + size], dtype=dt.newbyteorder("<")).reshape(spec["shape"]).astype(dt)
            pos += size
        if pos != len(blob):
            raise TrainerRefused("checkpoint has bytes after its last array")
        return out, header.get("meta", {})
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, TrainerRefused):
            raise
        raise TrainerRefused(f"checkpoint header is damaged: {type(exc).__name__}") from exc


class NumpyMLP:
    name = "numpy_mlp"
    deterministic = True

    def init(self, config: dict, files: dict):
        self.cfg = {"hidden": int(config.get("hidden", 8)), "lr": float(config.get("lr", 0.5)), "seed": int(config.get("seed", 0)), "batch": int(config.get("batch", 0)),
                    "l2": float(config.get("l2", 0.0))}
        if not (0 <= self.cfg["hidden"] <= 4096) or not (0 < self.cfg["lr"] <= 100) or self.cfg["batch"] < 0 or not (0 <= self.cfg["l2"] <= 10):
            raise TrainerRefused("hyper-parameters out of range")
        self.X, self.y = parse_csv(files, config.get("files"))
        self.rng = np.random.Generator(np.random.PCG64(self.cfg["seed"]))
        d, h = self.X.shape[1], self.cfg["hidden"]
        if h:
            self.p = {"w1": self.rng.normal(0, 1 / math.sqrt(d), (d, h)), "b1": np.zeros(h), "w2": self.rng.normal(0, 1 / math.sqrt(h), (h, 1)), "b2": np.zeros(1)}
        else:
            self.p = {"w1": np.zeros((d, 1)), "b1": np.zeros(1)}
        self.steps = 0

    def _forward(self, X):
        if self.cfg["hidden"]:
            a = np.tanh(X @ self.p["w1"] + self.p["b1"])
            z = a @ self.p["w2"] + self.p["b2"]
            return a, z.ravel()
        return None, (X @ self.p["w1"] + self.p["b1"]).ravel()

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def loss(self):
        _, z = self._forward(self.X)
        p = np.clip(self._sig(z), 1e-12, 1 - 1e-12)
        return float(-np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p)))

    def step(self, n: int) -> dict:
        for _ in range(int(n)):
            if self.cfg["batch"] and self.cfg["batch"] < len(self.X):
                idx = self.rng.choice(len(self.X), self.cfg["batch"], replace=False)
                X, y = self.X[idx], self.y[idx]
            else:
                X, y = self.X, self.y
            a, z = self._forward(X)
            g = (self._sig(z) - y) / len(X)
            if self.cfg["hidden"]:
                gw2, gb2 = a.T @ g[:, None], np.array([g.sum()])
                ga = (g[:, None] @ self.p["w2"].T) * (1 - a ** 2)
                grads = {"w1": X.T @ ga, "b1": ga.sum(0), "w2": gw2, "b2": gb2}
            else:
                grads = {"w1": X.T @ g[:, None], "b1": np.array([g.sum()])}
            for k, gr in grads.items():
                self.p[k] = self.p[k] - self.cfg["lr"] * (gr + (self.cfg["l2"] * self.p[k] if k.startswith("w") else 0.0))
            self.steps += 1
        return {"step": self.steps, "loss": self.loss()}

    def weights_bytes(self) -> bytes:
        return pack_arrays(self.p, {"trainer": self.name})

    def state_bytes(self) -> bytes:
        rng_state = json.dumps(self.rng.bit_generator.state, sort_keys=True)
        return pack_arrays(self.p, {"trainer": self.name, "steps": self.steps, "rng": rng_state, "cfg": self.cfg})

    def load_state(self, blob: bytes):
        arrays, meta = unpack_arrays(blob)
        if meta.get("trainer") != self.name or set(arrays) != set(self.p) or any(arrays[k].shape != self.p[k].shape for k in self.p):
            raise TrainerRefused("checkpoint does not fit this model")
        if meta.get("cfg") != self.cfg:
            raise TrainerRefused("checkpoint was made with different hyper-parameters")
        self.p = {k: arrays[k].astype(np.float64) for k in arrays}
        self.steps = int(meta["steps"])
        st = json.loads(meta["rng"])
        self.rng.bit_generator.state = st


def torch_trainer_class():
    """The torch back end, built on first use so importing this module never imports torch. Raises TrainerRefused if torch or safetensors is missing."""
    try:
        import torch
        from safetensors.torch import load as st_load, save as st_save
    except ImportError as exc:
        raise TrainerRefused("the torch back end needs torch and safetensors installed") from exc

    class TorchTrainer:
        """Full-batch SGD on a small MLP, CPU, deterministic algorithms on. State and weights are safetensors: no pickle anywhere. A template for a bigger trainer."""
        name = "torch_mlp"
        deterministic = True

        def init(self, config: dict, files: dict):
            self.cfg = {"hidden": int(config.get("hidden", 8)), "lr": float(config.get("lr", 0.5)), "seed": int(config.get("seed", 0))}
            X, y = parse_csv(files, config.get("files"))
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
            torch.manual_seed(self.cfg["seed"])
            self.X, self.y = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
            d, h = self.X.shape[1], self.cfg["hidden"]
            self.model = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, 1)).double()
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg["lr"])
            self.steps = 0

        def step(self, n: int) -> dict:
            for _ in range(int(n)):
                self.opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.model(self.X).squeeze(1), self.y)
                loss.backward()
                self.opt.step()
                self.steps += 1
            with torch.no_grad():
                return {"step": self.steps, "loss": float(torch.nn.functional.binary_cross_entropy_with_logits(self.model(self.X).squeeze(1), self.y))}

        def weights_bytes(self) -> bytes:
            return st_save({k: v.detach().contiguous() for k, v in self.model.state_dict().items()}, metadata={"meta": json.dumps({"trainer": self.name}, sort_keys=True)})

        def state_bytes(self) -> bytes:
            # ONE metadata key holding sorted-key JSON: safetensors writes metadata keys in dict-hash order, which changes between processes, so several keys would give different bytes for the same state
            return st_save({k: v.detach().contiguous() for k, v in self.model.state_dict().items()}, metadata={"meta": json.dumps({"trainer": self.name, "steps": self.steps, "cfg": self.cfg}, sort_keys=True)})

        def load_state(self, blob: bytes):
            tensors = st_load(blob)
            if set(tensors) != set(self.model.state_dict()):
                raise TrainerRefused("checkpoint does not fit this model")
            self.model.load_state_dict(tensors)
            self.steps = self._steps_in(blob)

        @staticmethod
        def _steps_in(blob):
            n = struct.unpack("<Q", blob[:8])[0]
            return int(json.loads(json.loads(blob[8:8 + n].decode("utf-8")).get("__metadata__", {}).get("meta", "{}")).get("steps", 0))
    return TorchTrainer


REGISTRY = {"numpy_mlp": NumpyMLP}


def make(name: str):
    if name == "torch_mlp":
        return torch_trainer_class()()
    if name not in REGISTRY:
        raise TrainerRefused(f"unknown trainer {name!r}; known: {sorted(REGISTRY) + ['torch_mlp']}")
    return REGISTRY[name]()
