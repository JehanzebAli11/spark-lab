"""Classify why a job failed from the tail of its log. Standard library only.

Two classifiers share one interface:

* `RuleClassifier`: hand-written regexes, the baseline. High precision on the error lines
  it knows, silent on anything else.
* `LinearClassifier`: TF-IDF + multinomial logistic regression trained offline on the
  chaos-job log dataset (`labml/logs.py`) and exported to JSON, so inference here needs
  nothing outside the standard library.

`classify()` returns a `Diagnosis`. When the model's top probability is below the
threshold the label becomes `unknown`, which the healer escalates instead of guessing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CLASSES = ("oom", "disk_full", "cuda_error", "network", "bad_data", "flaky", "code_bug")
UNKNOWN = "unknown"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "log_classifier.json"
TAIL_LINES = 60

_ATTEMPT = re.compile(r"^=== .* attempt \d+ :: ", re.M)
_HEX = re.compile(r"0x[0-9a-f]+")
_NUM = re.compile(r"\d+")
_TOKEN = re.compile(r"[a-z_]+|[0-9]+")
_PATH = re.compile(r"(/[\w.@+-]+){2,}")


@dataclass
class Diagnosis:
    label: str
    confidence: float
    source: str  # "rules", "model", "model+rules"
    evidence: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        d["scores"] = {k: round(v, 4) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])[:3]}
        return d


def last_attempt(log_text: str) -> str:
    """What the most recent attempt wrote, without the queue's header line (it holds the cmd)."""
    starts = [m.start() for m in _ATTEMPT.finditer(log_text)]
    if not starts:
        return log_text
    body = log_text[starts[-1]:]
    return body.split("\n", 1)[1] if "\n" in body else ""


def tail(log_text: str, n: int = TAIL_LINES) -> str:
    return "\n".join(last_attempt(log_text).splitlines()[-n:])


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit|Failure|Warning)?: |"
                       r"^[A-Za-z_][\w.]*(Error|Exception)$|^Killed$|^=== exit ")


def _words(text: str) -> list[str]:
    text = _PATH.sub(" path ", text)
    text = _CAMEL.sub(" ", text).replace("_", " ").lower()
    text = _HEX.sub(" hex ", text)
    text = _NUM.sub("0", text)
    return _TOKEN.findall(text)


def error_region(text: str) -> str:
    """The lines that say what went wrong: exception lines and the last few log lines."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    exc = [ln for ln in lines if _EXC_LINE.search(ln.strip())]
    return "\n".join(exc[-3:] + lines[-4:])


# ---- rules ---------------------------------------------------------------------------

RULES: list[tuple[str, re.Pattern[str]]] = [
    ("oom", re.compile(r"out of memory|OutOfMemoryError|MemoryError|Cannot allocate memory|"
                       r"oom-kill|Killed process .* total-vm|CUBLAS_STATUS_ALLOC_FAILED", re.I)),
    ("disk_full", re.compile(r"No space left on device|ENOSPC|Disk quota exceeded|errno 28\b", re.I)),
    ("cuda_error", re.compile(r"CUDA error|cudaError|CUDA driver|NCCL error|Xid \d+|ECC error|"
                              r"cuDNN error|CUBLAS_STATUS_(?!ALLOC)|GPU has fallen off", re.I)),
    ("network", re.compile(r"Connection(Refused|Reset)Error|Connection refused|Connection reset|"
                           r"Name or service not known|Temporary failure in name resolution|"
                           r"Read timed out|ConnectTimeout|Max retries exceeded|503 Service|502 Bad Gateway|"
                           r"429 Too Many Requests|Network is unreachable|URLError|RemoteDisconnected", re.I)),
    ("bad_data", re.compile(r"JSONDecodeError|UnicodeDecodeError|ParserError|CSV|corrupt|"
                            r"truncated|Unexpected end of|missing column|KeyError: '(text|label|input|"
                            r"output|prompt|image)'|BadZipFile|invalid load key|empty dataset", re.I)),
    ("flaky", re.compile(r"Resource temporarily unavailable|Text file busy|Stale file handle|"
                         r"Interrupted system call|database is locked|Address already in use|"
                         r"Device or resource busy", re.I)),
    ("code_bug", re.compile(r"(TypeError|AttributeError|NameError|IndexError|AssertionError|"
                            r"ImportError|ModuleNotFoundError|SyntaxError|ZeroDivisionError|"
                            r"NotImplementedError|UnboundLocalError|RecursionError):", re.I)),
]


def tokens(text: str) -> list[str]:
    """Features of a log tail: uni/bigrams of the whole tail plus `e:`-prefixed words of the
    error region. CamelCase and snake_case are split, so `OutOfMemoryError` and
    `_ArrayMemoryError` share `memory error`. Regex rule hits in the error region are
    added as `r:<label>` features, so the model learns when to trust a rule. Shared by
    training and inference."""
    words = _words(text)
    err = _words(error_region(text))
    region = error_region(text)
    hits = [f"r:{label}" for label, rx in RULES if rx.search(region)]
    return (words + [f"{a} {b}" for a, b in zip(words, words[1:])]
            + [f"e:{w}" for w in err] + [f"e:{a} {b}" for a, b in zip(err, err[1:])] + hits * 3)


class RuleClassifier:
    name = "rules"

    def predict(self, text: str) -> Diagnosis:
        """The rule that matches the latest line wins; error lines come last in a traceback."""
        best: tuple[int, int, str, str] | None = None
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for rank, (label, rx) in enumerate(RULES):
                if rx.search(line):
                    key = (i, -rank, label, line.strip())
                    if best is None or key[:2] > best[:2]:
                        best = key
        if best is None:
            return Diagnosis(UNKNOWN, 0.0, "rules")
        return Diagnosis(best[2], 1.0, "rules", evidence=best[3][:200], scores={best[2]: 1.0})


# ---- linear model --------------------------------------------------------------------


class LinearClassifier:
    """TF-IDF (sublinear tf, l2 norm) + multinomial logistic regression, loaded from JSON."""

    name = "model"

    def __init__(self, spec: dict):
        self.classes: list[str] = spec["classes"]
        self.vocab: dict[str, int] = spec["vocab"]
        self.idf: list[float] = spec["idf"]
        self.coef: list[list[float]] = spec["coef"]  # [n_classes][n_features]
        self.intercept: list[float] = spec["intercept"]
        self.threshold: float = float(spec.get("threshold", 0.5))
        self.mode: str = spec.get("mode", "model")  # "model" or "hybrid" (rules on the error region win)
        self.meta: dict = spec.get("meta", {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> LinearClassifier:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def probabilities(self, text: str) -> dict[str, float]:
        counts: dict[int, int] = {}
        for t in tokens(text):
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        x = {j: (1.0 + math.log(c)) * self.idf[j] for j, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in x.values())) or 1.0
        logits = [b + sum(w[j] * v / norm for j, v in x.items()) for w, b in zip(self.coef, self.intercept)]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        s = sum(exps)
        return {c: e / s for c, e in zip(self.classes, exps)}

    def predict(self, text: str) -> Diagnosis:
        p = self.probabilities(text)
        label = max(p, key=p.get)
        return Diagnosis(label, p[label], "model", scores=p)


_SIGNAL_EXIT = re.compile(r"^=== exit -\d+")


def has_evidence(text: str) -> bool:
    """An exception line, a signal exit or a rule hit. Without one, any label is a guess."""
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("=== "):
            if _SIGNAL_EXIT.match(ln):
                return True
            continue
        if _EXC_LINE.search(ln) or any(rx.search(ln) for _, rx in RULES):
            return True
    return False


def _evidence(text: str, label: str) -> str:
    rx = dict(RULES).get(label)
    if rx:
        for line in reversed(text.splitlines()):
            if rx.search(line):
                return line.strip()[:200]
    for line in reversed(text.splitlines()):
        if line.strip() and not line.startswith("==="):
            return line.strip()[:200]
    return ""


class Triage:
    """The classifier the healer uses: the model when its weights exist, else the rules.

    Below `threshold` the label becomes `unknown` and the healer escalates.
    """

    def __init__(self, model: LinearClassifier | None = None, threshold: float | None = None):
        self.model = model
        self.rules = RuleClassifier()
        self.threshold = threshold if threshold is not None else (model.threshold if model else 0.5)

    @classmethod
    def default(cls, path: str | Path = DEFAULT_MODEL) -> Triage:
        try:
            return cls(LinearClassifier.load(path))
        except (OSError, ValueError, KeyError):
            return cls(None)

    def classify(self, log_text: str) -> Diagnosis:
        text = tail(log_text)
        if not has_evidence(text):
            return Diagnosis(UNKNOWN, 0.0, "no-evidence", evidence=_evidence(text, UNKNOWN))
        if self.model is None:
            d = self.rules.predict(text)
        else:
            d = self.model.predict(text)
            d.evidence = _evidence(text, d.label)
            if self.model.mode == "hybrid":
                r = self.rules.predict(error_region(text))
                if r.label != UNKNOWN:
                    r.source = "rules" if r.label != d.label else "model+rules"
                    r.scores = d.scores
                    return r
        if d.label != UNKNOWN and d.confidence < self.threshold:
            d.scores = {**d.scores}
            d.evidence = f"low confidence {d.confidence:.2f} for {d.label}; " + d.evidence
            d.label = UNKNOWN
        return d
