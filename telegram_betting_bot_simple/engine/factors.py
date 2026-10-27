"""Typed factors: every projection is a baseline plus sized, sourced factors that sum to the number shown.

This is the additive contract SHAP explanations use ("the sum of the base value and all the individual
feature SHAP values equals the final output value"): ``baseline + sum(factor deltas) == value``, with each
delta in the stat's own units. The app's models are mostly multiplicative (a rate x minutes x context
multipliers), so ``allocate`` puts additive terms (a minutes change, an injury's points) in first as their
own deltas, then splits the multiplicative uplift across the multipliers by each one's share of the
log-uplift:

    c_i = (P - B') * ln m_i / sum_j ln m_j,   with B' = base + sum(additive) and P = B' * prod(m_j)

That split is exact, independent of order and in the stat's units. Worked example (report, "Every projection
becomes a waterfall"): 25.1 points in 34.0 minutes, projected for 35 minutes, opponent x1.02, pace x1.025,
usage x1.02, back-to-back x0.985 gives "27.1 pts = 25.1 season + 0.7 minutes + 0.7 pace + 0.5 opponent +
0.5 usage - 0.4 back-to-back".

Factors the literature finds weak (defence vs position, referees, wind, small home/away splits) carry the
confidence "small, noisy" so a tiny effect is not dressed up as a finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

NORMAL = "normal"
NOISY = "small, noisy"
CONFIDENCES = (NORMAL, NOISY)
DIRECTIONS = ("up", "down", "none")

# Below this a share-of-change split falls back to the local slope (the change is ~0 but parts may not be)
_TINY = 1e-12


def _direction(delta: float) -> str:
    if delta > _TINY:
        return "up"
    if delta < -_TINY:
        return "down"
    return "none"


@dataclass
class Factor:
    """One sized reason: ``delta`` (in the projection's units) moves the number from its baseline."""

    key: str
    label: str
    delta: float
    unit: str = ""
    direction: str = ""  # "up" | "down" | "none": always set from the sign of ``delta``
    evidence: str = ""  # a short fact behind the number, e.g. "team projected 118.2 vs 114.0 from player lines"
    source: str = ""  # where the input came from, e.g. "ESPN standings" or "Odds API consensus"
    as_of: Optional[str] = None  # ISO time of the input, when known
    confidence: str = NORMAL  # "normal" | "small, noisy"

    def __post_init__(self) -> None:
        self.delta = float(self.delta)
        self.direction = _direction(self.delta)
        if self.confidence not in CONFIDENCES:
            raise ValueError(f"Unknown factor confidence {self.confidence!r}")

    def to_dict(self, delta: Optional[float] = None) -> Dict[str, Any]:
        """A JSON-ready dict (``delta`` overrides the stored one, e.g. a display-rounded value)."""
        value = self.delta if delta is None else delta
        return {"key": self.key, "label": self.label, "delta": value, "unit": self.unit,
                "direction": self.direction, "evidence": self.evidence, "source": self.source,
                "as_of": self.as_of, "confidence": self.confidence}


@dataclass
class Projection:
    """A number with its explanation: ``baseline`` (labelled) plus every factor's delta equals ``value``.

    ``key`` and ``label`` (optional) name what is projected, e.g. "margin" / "Home margin (home minus away)".
    """

    baseline_label: str
    baseline: float
    factors: List[Factor] = field(default_factory=list)
    value: float = 0.0
    unit: str = ""
    key: str = ""
    label: str = ""

    def residual(self) -> float:
        """How far baseline + deltas misses the value (0 when the factors explain it exactly)."""
        return self.baseline + math.fsum(f.delta for f in self.factors) - self.value

    def to_dict(self, digits: Optional[int] = None) -> Dict[str, Any]:
        """A JSON-ready dict. With ``digits`` the baseline and value are rounded and the deltas are rounded
        by largest remainder, so the numbers sent still add up exactly to the value shown."""
        deltas = [f.delta for f in self.factors]
        baseline, value = self.baseline, self.value
        if digits is not None:
            baseline, value = round(baseline, digits), round(value, digits)
            deltas = round_to_sum(deltas, value - baseline, digits)
        return {"key": self.key, "label": self.label, "baseline_label": self.baseline_label,
                "baseline": baseline, "value": value, "unit": self.unit,
                "factors": [f.to_dict(d) for f, d in zip(self.factors, deltas)]}


def check_sum(projection: Projection, tol: float = 0.01) -> bool:
    """True when baseline + the factors' deltas equals the projection's value to within ``tol``."""
    return all(math.isfinite(x) for x in (projection.baseline, projection.value, *(f.delta for f in projection.factors))) \
        and abs(projection.residual()) <= tol


def round_to_sum(values: Sequence[float], total: float, digits: int) -> List[float]:
    """Round ``values`` to ``digits`` decimals so the rounded values add up to ``total`` (itself rounded):
    largest-remainder rounding, the way seat apportionment keeps totals exact."""
    if not all(math.isfinite(v) for v in (*values, total)):
        return [round(v, digits) for v in values]  # nothing sensible to apportion
    scale = 10 ** digits
    target = round(total * scale)
    scaled = [v * scale for v in values]
    floors = [math.floor(x) for x in scaled]
    short = target - sum(floors)
    order = sorted(range(len(values)), key=lambda i: scaled[i] - floors[i], reverse=short >= 0)
    ints = list(floors)
    if order:
        step = 1 if short >= 0 else -1
        for n in range(abs(short)):
            ints[order[n % len(order)]] += step
    return [round(i / scale, digits) for i in ints]


def split_by_share(change: float, weights: Mapping[str, float], slope: float) -> Dict[str, float]:
    """Split ``change`` across parts in proportion to their (signed) ``weights``: c_i = change * w_i / sum(w).

    When the weights cancel out (their sum ~0, so the change is ~0 too) the ratio change/sum(w) is replaced
    by its limit, ``slope`` (the derivative of the output with respect to the summed weights), so opposing
    parts still show their own sizes and still sum to the change.
    """
    total = math.fsum(weights.values())
    ratio = change / total if abs(total) > _TINY else slope
    return {key: ratio * w for key, w in weights.items()}


Spec = Union[Tuple[Any, ...], List[Any]]


def _spec(key: str, spec: Spec) -> Tuple[str, float, str, str, str, Optional[str]]:
    """(label, number, evidence, source, confidence, as_of) from a (label, number, ...) tuple."""
    if len(spec) < 2:
        raise ValueError(f"Factor {key!r} needs at least (label, value)")
    padded = list(spec) + [""] * max(0, 5 - len(spec))
    label, number, evidence, source, confidence = padded[:5]
    as_of = padded[5] if len(padded) > 5 else None
    return str(label), float(number), str(evidence or ""), str(source or ""), confidence or NORMAL, as_of


def allocate(base: float, multipliers: Optional[Mapping[str, Spec]] = None,
             additive: Optional[Mapping[str, Spec]] = None, unit: str = "") -> List[Factor]:
    """Factors for P = (base + sum(additive deltas)) * prod(multipliers).

    ``multipliers``: {key: (label, m, evidence, source[, confidence[, as_of]])}, m > 0.
    ``additive``: {key: (label, delta, evidence, source[, confidence[, as_of]])}.
    Additive terms come first, each as its own delta; the multiplicative uplift P - B' is then split by log
    share, c_i = (P - B') * ln m_i / sum_j ln m_j. A multiplier of exactly 1 gets a zero delta. The deltas
    always sum to P - base (see ``project`` for the value itself).
    """
    factors: List[Factor] = []
    shifted = float(base)
    for key, spec in (additive or {}).items():
        label, delta, evidence, source, confidence, as_of = _spec(key, spec)
        factors.append(Factor(key, label, delta, unit, "", evidence, source, as_of, confidence))
        shifted += delta
    specs = {key: _spec(key, spec) for key, spec in (multipliers or {}).items()}
    for key, (label, m, *_rest) in specs.items():
        if not (m > 0 and math.isfinite(m)):
            raise ValueError(f"Multiplier {key!r} ({label}) must be a positive number, got {m}")
    logs = {key: math.log(spec[1]) for key, spec in specs.items()}
    s = math.fsum(logs.values())
    # P - B' = B' (e^S - 1); expm1 keeps it exact for small uplifts, and its limit B' covers S ~ 0
    uplift = shifted * math.expm1(s)
    parts = split_by_share(uplift, logs, slope=shifted)
    for key, (label, _m, evidence, source, confidence, as_of) in specs.items():
        factors.append(Factor(key, label, parts[key], unit, "", evidence, source, as_of, confidence))
    return factors


def project(baseline_label: str, base: float, multipliers: Optional[Mapping[str, Spec]] = None,
            additive: Optional[Mapping[str, Spec]] = None, unit: str = "", key: str = "",
            label: str = "", value: Optional[float] = None) -> Projection:
    """A Projection from ``allocate``: value = (base + additive) * prod(m), unless ``value`` is given (the
    caller's own computation of the same product, kept so displays match the model to the last digit)."""
    factors = allocate(base, multipliers, additive, unit)
    if value is None:
        shifted = base + math.fsum(_spec(k, s)[1] for k, s in (additive or {}).items())
        value = shifted * math.prod(_spec(k, s)[1] for k, s in (multipliers or {}).items())
    return Projection(baseline_label, float(base), factors, float(value), unit, key, label)


def factor_lines(factors: Iterable[Factor], digits: int = 1) -> List[str]:
    """Plain one-line renderings, e.g. "Pace: +0.7 pts" (for text surfaces with no waterfall)."""
    lines = []
    for f in factors:
        unit = f" {f.unit}" if f.unit else ""
        lines.append(f"{f.label}: {f.delta:+.{digits}f}{unit}" + (" (small, noisy)" if f.confidence == NOISY else ""))
    return lines


def fold(projection: Projection, n: int = 6, digits: Optional[int] = None) -> Dict[str, Any]:
    """``projection.to_dict(digits)`` with the factors sorted by size (largest first), the top ``n`` kept and the
    rest folded into one "Other (k factors)" entry (key "folded", with ``folded`` = k), so the numbers still add up
    exactly to the value shown (a single leftover factor is kept as itself). For text surfaces and compact tool
    results; the app's waterfall folds the same way."""
    shown = projection.to_dict(digits)
    factors = sorted(shown["factors"], key=lambda f: -abs(f["delta"]))
    keep, rest = factors[:max(int(n), 0)], factors[max(int(n), 0):]
    if len(rest) == 1:  # one factor shows as itself rather than as "Other (1 factor)"
        keep, rest = factors, []
    if rest:
        total = math.fsum(f["delta"] for f in rest)
        if digits is not None:
            total = round(total, digits)
        keep.append({"key": "folded", "label": f"Other ({len(rest)} factors)",
                     "delta": total, "unit": projection.unit, "direction": _direction(total), "evidence": "",
                     "source": "", "as_of": None, "confidence": NORMAL, "folded": len(rest)})
    shown["factors"] = keep
    return shown
