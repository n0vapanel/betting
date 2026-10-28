"""The parameter registry: every knob the learning loop may move, its bounds and per-cycle step, and the live values.

The allow-list is method.md section 4.1's table (plus the context priors and SDs and the new count models). Every
knob's default is read from the engine module that owns it, at call time, so the defaults are today's constants
exactly; with no stored version (or in safe mode) ``current`` serves them and every caller's output is unchanged.

Knobs are named ``<area>.<name>``, the names the AI's proposals use:

- ``consensus.weight.<kind>`` (sharp, us_books, prediction_market, model_rating, team_model): the blend's kind
  weights, each in [0, 0.8], market kinds together at least 0.70 of the total, at most 0.05 a cycle per kind;
- ``consensus.sharpness``: the pool's extremising exponent ``a`` (1 today), in [0.8, 1.25], 0.05 a cycle, and
  structural: it moves off 1 only through the superiority gate;
- ``consensus.near_start_factor`` / ``consensus.near_start_seconds`` and ``consensus.low_confidence_shrink``;
- ``soccer.rho`` (Dixon-Coles) and ``soccer.btts_nudge`` (soccer only);
- ``context.adjust_weight``, ``context.home_prior``, ``context.margin_sd``, ``context.total_sd`` and (soccer)
  ``context.draw_prior`` (sports with context parameters);
- ``nba.market_weight``, ``nba.espn_weight``, ``nba.market_total_weight`` (NBA only; market at least 0.75);
- ``counts.<corners|cards>.<intercept|home|supremacy|total|size>``: the new negative binomial count models (soccer
  only, structural; their values are shown to nobody until they beat the climatology baseline through the gate).

Sports are the context sport keys (``nfl``, ``ncaaf``, ``nba``, ``wnba``, ``ncaab``, ``nhl``, ``mlb``,
``soccer``) plus ``other`` (tennis, MMA, the CFL, ...: only the blend knobs apply). ``sport_key`` maps a league key.

``current(league)`` returns that sport's live ``Params``: cached in memory and re-read from the store (``brain_params``
champions, ``brain_state`` epoch and safe mode) when the epoch moves, checked at most every ``CHECK_SECONDS``. A
store that can't be read gives the defaults: the registry never breaks a page.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..engine import consensus, context, nba_model, soccer

log = logging.getLogger(__name__)

SPORTS: Tuple[str, ...] = ("nfl", "ncaaf", "nba", "wnba", "ncaab", "nhl", "mlb", "soccer", "other")
KINDS: Tuple[str, ...] = consensus.KINDS
MARKET_KINDS: Tuple[str, ...] = consensus.MARKET_KINDS
MARKET_FLOOR = 0.70  # market kinds' least share of the combined weight
NBA_MARKET_FLOOR = 0.75
CHECK_SECONDS = 15.0  # how often ``current`` looks at the store's epoch
COUNT_TARGETS: Tuple[str, ...] = ("corners", "cards")
COUNT_TERMS: Tuple[str, ...] = ("intercept", "home", "supremacy", "total", "size")
COUNT_TOTAL_CENTRE = 2.7  # goals: the count models' total term is (expected goals - 2.7)

# Starting values of the count models (new, shown to nobody until they beat climatology): per team,
# log mean = intercept +- home / 2 + supremacy x (own minus opponent's expected goals) + total x (expected goals -
# 2.7), negative binomial with ``size`` (variance = mean + mean^2 / size). Corners: about 5.1 a team, 5.6 home and
# 4.7 away in the top leagues, favourites win more; yellow cards about 1.9 a team, underdogs and away sides more.
COUNT_DEFAULTS: Mapping[str, Mapping[str, float]] = MappingProxyType({
    "corners": MappingProxyType({"intercept": 1.629, "home": 0.18, "supremacy": 0.10, "total": 0.05, "size": 15.0}),
    "cards": MappingProxyType({"intercept": 0.642, "home": -0.12, "supremacy": -0.08, "total": -0.03, "size": 12.0}),
})
_COUNT_BOUNDS = {
    "corners": {"intercept": (1.0, 2.3), "home": (-0.5, 0.6), "supremacy": (-0.6, 0.6), "total": (-0.5, 0.5),
                "size": (2.0, 500.0)},
    "cards": {"intercept": (-0.2, 1.5), "home": (-0.6, 0.5), "supremacy": (-0.6, 0.6), "total": (-0.5, 0.5),
              "size": (2.0, 500.0)},
}


@dataclasses.dataclass(frozen=True)
class Knob:
    """One allow-listed parameter.

    ``mode`` sets how ``low``/``high``/``step`` read: "abs" (absolute bounds and step), "rel" (multiples of the
    sport's default, e.g. SDs in [0.8, 1.25] x default, 5% a cycle) or "around" (the default plus ``low``/``high``,
    an absolute step). ``rel_step`` makes the step a share of the current value (count dispersion). ``structural``
    knobs change only through the superiority gate (method.md 4.3).
    """

    name: str
    label: str
    group: str  # blend, soccer, context, nba, counts
    low: float
    high: float
    step: float
    mode: str = "abs"
    sports: Optional[Tuple[str, ...]] = None  # None: every sport
    structural: bool = False
    rel_step: bool = False
    unit: str = ""

    def applies(self, sport: str) -> bool:
        return self.sports is None or sport in self.sports

    def default(self, sport: str) -> float:
        return float(default_value(self.name, sport))

    def bounds(self, sport: str) -> Tuple[float, float]:
        if self.mode == "rel":
            base = self.default(sport)
            return base * self.low, base * self.high
        if self.mode == "around":
            base = self.default(sport)
            return base + self.low, base + self.high
        return self.low, self.high

    def max_step(self, sport: str, value: Optional[float] = None) -> float:
        if self.rel_step:
            return abs(value if value is not None else self.default(sport)) * self.step
        if self.mode == "rel":
            return self.default(sport) * self.step
        return self.step


def _knobs() -> Dict[str, Knob]:
    out: List[Knob] = []
    labels = {"sharp": "sharp books", "us_books": "US books", "prediction_market": "prediction markets",
              "model_rating": "ratings", "team_model": "our team model"}
    for kind in KINDS:
        out.append(Knob(f"consensus.weight.{kind}", f"Blend weight of {labels[kind]}", "blend", 0.0, 0.8, 0.05))
    out += [
        Knob("consensus.sharpness", "Blend sharpness (how far the pooled chance leans from 50%)", "blend", 0.8, 1.25,
             0.05, structural=True),
        Knob("consensus.near_start_factor", "Prediction-market weight near the start", "blend", 0.25, 1.0, 0.1),
        Knob("consensus.near_start_seconds", "How close to the start counts as near", "blend", 300.0, 3600.0, 300.0,
             unit="s"),
        Knob("consensus.low_confidence_shrink", "Pull of ratings-only chances toward the base rate", "blend", 0.1,
             0.5, 0.05),
        Knob("soccer.rho", "Goals model low-score correction (Dixon-Coles rho)", "soccer", -0.25, 0.05, 0.02,
             sports=("soccer",)),
        Knob("soccer.btts_nudge", "Both-teams-to-score calibration nudge", "soccer", -0.03, 0.03, 0.005,
             sports=("soccer",)),
        Knob("context.adjust_weight", "Share of a late weather or roof change applied", "context", 0.0, 1.0, 0.1,
             sports=SPORTS[:-1]),
        Knob("context.home_prior", "Home-win prior", "context", -0.08, 0.08, 0.01, mode="around",
             sports=SPORTS[:-1]),
        Knob("context.draw_prior", "Draw prior", "context", -0.08, 0.08, 0.01, mode="around", sports=("soccer",)),
        Knob("context.margin_sd", "Spread of final margins", "context", 0.8, 1.25, 0.05, mode="rel",
             sports=SPORTS[:-1]),
        Knob("context.total_sd", "Spread of final totals", "context", 0.8, 1.25, 0.05, mode="rel",
             sports=SPORTS[:-1]),
        Knob("nba.market_weight", "NBA model: market share of the margin", "nba", NBA_MARKET_FLOOR, 1.0, 0.03,
             sports=("nba",)),
        Knob("nba.espn_weight", "NBA model: ESPN share of the margin", "nba", 0.0, 0.25, 0.03, sports=("nba",)),
        Knob("nba.market_total_weight", "NBA model: market share of the total", "nba", NBA_MARKET_FLOOR, 1.0, 0.03,
             sports=("nba",)),
    ]
    for target in COUNT_TARGETS:
        for term in COUNT_TERMS:
            low, high = _COUNT_BOUNDS[target][term]
            out.append(Knob(f"counts.{target}.{term}", f"{target.title()} model: {term}", "counts", low, high,
                            0.25 if term == "size" else 0.05, sports=("soccer",), structural=True,
                            rel_step=term == "size"))
    return {knob.name: knob for knob in out}


KNOBS: Mapping[str, Knob] = MappingProxyType(_knobs())
WEIGHT_KNOBS: Tuple[str, ...] = tuple(f"consensus.weight.{kind}" for kind in KINDS)


def default_value(name: str, sport: str) -> float:
    """Today's constant for ``name`` in ``sport``, read from the engine module that owns it."""
    if name.startswith("consensus.weight."):
        return consensus.DEFAULT_WEIGHTS[name.rsplit(".", 1)[1]]
    if name == "consensus.sharpness":
        return 1.0
    if name == "consensus.near_start_factor":
        return consensus.NEAR_START_FACTOR
    if name == "consensus.near_start_seconds":
        return consensus.NEAR_START_SECONDS
    if name == "consensus.low_confidence_shrink":
        return consensus.LOW_CONFIDENCE_SHRINK
    if name == "soccer.rho":
        return soccer.DC_RHO
    if name == "soccer.btts_nudge":
        return soccer.BTTS_NUDGE
    if name == "context.adjust_weight":
        return context.ADJUST_WEIGHT
    if name.startswith("context."):
        value = getattr(context.SPORTS[sport], name.split(".", 1)[1])
        if value is None:
            raise KeyError(f"{name} has no default for {sport}")
        return value
    if name.startswith("nba."):
        return getattr(nba_model.Params(), name.split(".", 1)[1])
    if name.startswith("counts."):
        _, target, term = name.split(".")
        return COUNT_DEFAULTS[target][term]
    raise KeyError(f"Unknown knob {name!r}")


# --- sports and the allow-list ------------------------------------------------------------------------------------

def sport_key(league: Any) -> str:
    """The registry's sport for a league or sport key ("basketball_nba" -> "nba", "soccer_epl" -> "soccer"; tennis,
    MMA and other leagues with no context parameters -> "other")."""
    text = str(league or "").strip().lower()
    if text in SPORTS:
        return text
    return context.sport_of(text) or "other"


def knobs_for(sport: str) -> List[Knob]:
    sport = sport_key(sport)
    return [knob for knob in KNOBS.values() if knob.applies(sport)]


def knob(name: str) -> Knob:
    try:
        return KNOBS[name]
    except KeyError:
        raise ValueError(f"{name!r} is not an allow-listed knob") from None


def allowed(name: str, sport: Any) -> bool:
    """Whether ``name`` is on the allow-list for ``sport`` (a sport or league key)."""
    return name in KNOBS and KNOBS[name].applies(sport_key(sport))


def defaults(sport: Any) -> Dict[str, float]:
    """Every knob of ``sport`` at today's constant."""
    sport = sport_key(sport)
    return {k.name: default_value(k.name, sport) for k in knobs_for(sport)}


def describe(sport: Any, values: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """The allow-list of ``sport`` for a fact sheet or the page: name, label, value, default, bounds, step."""
    sport = sport_key(sport)
    values = values if values is not None else current(sport).values
    out = []
    for k in knobs_for(sport):
        low, high = k.bounds(sport)
        value = values.get(k.name, k.default(sport))
        out.append({"knob": k.name, "label": k.label, "group": k.group, "value": value, "default": k.default(sport),
                    "low": low, "high": high, "step": k.max_step(sport, value), "structural": k.structural,
                    "unit": k.unit})
    return out


def check(sport: Any, values: Mapping[str, Any]) -> Dict[str, float]:
    """A complete, valid value set for ``sport``: ``values`` over the defaults; ValueError for an unknown or
    inapplicable knob, a value out of bounds, or a broken joint rule (market kinds under 0.70 of the weights, NBA
    market plus ESPN past 1)."""
    sport = sport_key(sport)
    out = defaults(sport)
    for name, value in values.items():
        if name not in out:
            raise ValueError(f"{name!r} is not a knob of {sport}")
        number = _finite(value, name)
        low, high = KNOBS[name].bounds(sport)
        if not low - 1e-12 <= number <= high + 1e-12:
            raise ValueError(f"{name} must be between {low:g} and {high:g} for {sport}, got {value!r}")
        out[name] = number
    weights = [out[name] for name in WEIGHT_KNOBS]
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("The blend weights can't all be 0")
    market = math.fsum(out[f"consensus.weight.{kind}"] for kind in MARKET_KINDS)
    if market / total < MARKET_FLOOR - 1e-9:
        raise ValueError(f"Market kinds need at least {MARKET_FLOOR:.0%} of the blend weight, got "
                         f"{market / total:.1%}")
    if sport == "nba" and out["nba.market_weight"] + out["nba.espn_weight"] > 1 + 1e-12:
        raise ValueError("NBA market and ESPN weights can't add up past 1")
    return out


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def changed(sport: Any, old: Mapping[str, float], new: Mapping[str, float], tol: float = 1e-12) -> List[str]:
    """The knobs whose value differs between two value sets."""
    sport = sport_key(sport)
    base = defaults(sport)
    return [name for name in base if abs(float(new.get(name, base[name])) - float(old.get(name, base[name]))) > tol]


def clip_step(sport: Any, old: Mapping[str, float], new: Mapping[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """``new`` moved no further from ``old`` than one cycle's step and kept inside the bounds and joint rules: the
    blend weights are clipped per kind, renormalised to sum to 1 and clipped again (a few rounds) and the market
    share floor enforced. Returns (values, the knobs whose move was cut)."""
    sport = sport_key(sport)
    base = check(sport, old)
    out = dict(base)
    cut: List[str] = []
    for name, value in new.items():
        if name not in base:
            raise ValueError(f"{name!r} is not a knob of {sport}")
        k = KNOBS[name]
        low, high = k.bounds(sport)
        step = k.max_step(sport, base[name])
        wanted = _finite(value, name)
        lo, hi = max(low, base[name] - step), min(high, base[name] + step)
        out[name] = min(max(wanted, lo), hi)
        if abs(out[name] - wanted) > 1e-12:
            cut.append(name)
    if any(abs(out[n] - base[n]) > 1e-15 for n in WEIGHT_KNOBS):
        out.update(_project_weights(sport, {n: base[n] for n in WEIGHT_KNOBS}, {n: out[n] for n in WEIGHT_KNOBS}))
    if sport == "nba" and out["nba.market_weight"] + out["nba.espn_weight"] > 1:
        out["nba.espn_weight"] = max(1 - out["nba.market_weight"], 0.0)
        cut.append("nba.espn_weight")
    check(sport, out)
    return out, sorted(set(cut))


def _project_weights(sport: str, old: Mapping[str, float], new: Mapping[str, float]) -> Dict[str, float]:
    """Weights summing to 1, each within its step of ``old`` (itself rescaled to sum 1) and inside [0, 0.8], with the
    market kinds at least MARKET_FLOOR: alternate clipping and renormalising until both hold."""
    total_old = math.fsum(old.values())
    ref = {n: v / total_old for n, v in old.items()}
    w = dict(new)
    for _ in range(50):
        total = math.fsum(w.values())
        w = {n: v / total for n, v in w.items()}
        market = math.fsum(w[f"consensus.weight.{kind}"] for kind in MARKET_KINDS)
        if market < MARKET_FLOOR:
            scale_m = MARKET_FLOOR / market
            scale_r = (1 - MARKET_FLOOR) / max(1 - market, 1e-12)
            w = {n: v * (scale_m if n.rsplit(".", 1)[1] in MARKET_KINDS else scale_r) for n, v in w.items()}
        clipped = {}
        for n, v in w.items():
            step = KNOBS[n].max_step(sport)
            low, high = KNOBS[n].bounds(sport)
            clipped[n] = min(max(v, max(low, ref[n] - step)), min(high, ref[n] + step))
        if all(abs(clipped[n] - w[n]) < 1e-12 for n in w) and abs(math.fsum(w.values()) - 1) < 1e-12:
            return w
        w = clipped
    total = math.fsum(w.values())
    return {n: v / total for n, v in w.items()}


def is_nudge(sport: Any, old: Mapping[str, float], new: Mapping[str, float]) -> bool:
    """A change every knob of which is within one cycle's step (and touches no structural knob): the nudge gate
    applies. Anything else (sharpness, a count model, a bigger move) needs the superiority gate."""
    sport = sport_key(sport)
    names = changed(sport, old, new)
    if not names:
        return True
    base = defaults(sport)
    for name in names:
        k = KNOBS[name]
        if k.structural:
            return False
        before = float(old.get(name, base[name]))
        if abs(float(new[name]) - before) > k.max_step(sport, before) + 1e-9:
            return False
    return True


# --- the live values ----------------------------------------------------------------------------------------------

class Params:
    """One sport's live knob values and where they came from. ``version`` 0 is the defaults (no stored champion, or
    safe mode). The ``*_kwargs``/``*_params`` helpers give exactly what each engine call takes, and nothing at all
    for a knob at its default, so a default call is the unchanged call."""

    __slots__ = ("sport", "version", "values", "source", "effective_from", "safe_mode", "epoch")

    def __init__(self, sport: str, values: Optional[Mapping[str, Any]] = None, version: int = 0,
                 source: str = "default", effective_from: Optional[float] = None, safe_mode: bool = False,
                 epoch: int = 0) -> None:
        self.sport = sport_key(sport)
        self.values: Mapping[str, float] = MappingProxyType(check(self.sport, values or {}))
        self.version = int(version)
        self.source = source
        self.effective_from = effective_from
        self.safe_mode = bool(safe_mode)
        self.epoch = int(epoch)

    def __repr__(self) -> str:
        return f"Params({self.sport!r}, version={self.version}, changed={self.changed()})"

    def get(self, name: str) -> float:
        return self.values[name]

    def changed(self) -> List[str]:
        """Knobs not at today's constant."""
        return changed(self.sport, defaults(self.sport), self.values)

    @property
    def is_default(self) -> bool:
        return not self.changed()

    def _differs(self, name: str) -> bool:
        return name in self.values and self.values[name] != default_value(name, self.sport)

    @property
    def weights(self) -> Dict[str, float]:
        return {kind: self.values[f"consensus.weight.{kind}"] for kind in KINDS}

    def blend_kwargs(self) -> Dict[str, Any]:
        """Keywords for ``engine.consensus.blend``: ``weights``, ``sharpness``, ``near_start_factor``,
        ``near_start_seconds``, ``low_confidence_shrink``, each only when not at its default ({} at the defaults)."""
        out: Dict[str, Any] = {}
        if any(self._differs(name) for name in WEIGHT_KNOBS):
            out["weights"] = self.weights
        for name, key in (("consensus.sharpness", "sharpness"), ("consensus.near_start_factor", "near_start_factor"),
                          ("consensus.near_start_seconds", "near_start_seconds"),
                          ("consensus.low_confidence_shrink", "low_confidence_shrink")):
            if self._differs(name):
                out[key] = self.values[name]
        return out

    def soccer_kwargs(self) -> Dict[str, float]:
        """Keywords for ``engine.soccer.derive``: ``rho`` and ``nudge``, each only when not at its default."""
        out: Dict[str, float] = {}
        if self._differs("soccer.rho"):
            out["rho"] = self.values["soccer.rho"]
        if self._differs("soccer.btts_nudge"):
            out["nudge"] = self.values["soccer.btts_nudge"]
        return out

    @property
    def rho(self) -> float:
        return self.values.get("soccer.rho", soccer.DC_RHO)

    @property
    def btts_nudge(self) -> float:
        return self.values.get("soccer.btts_nudge", soccer.BTTS_NUDGE)

    @property
    def adjust_weight(self) -> float:
        """``weight`` for ``engine.context.adjustment``/``weather_adjustment`` (ADJUST_WEIGHT itself by default)."""
        if self._differs("context.adjust_weight"):
            return self.values["context.adjust_weight"]
        return context.ADJUST_WEIGHT

    def context_overrides(self) -> Dict[str, float]:
        """{field: value} for ``engine.context.sport_params`` (margin_sd, total_sd, home_prior, draw_prior), only
        the ones not at their default."""
        out = {}
        for field in context.TUNABLE:
            name = f"context.{field}"
            if self._differs(name):
                out[field] = self.values[name]
        return out

    def context_params(self) -> Optional[context.SportParams]:
        """The sport's ``engine.context.SportParams`` with the tuned priors and SDs (the SPORTS entry itself by
        default); None for "other"."""
        if self.sport not in context.SPORTS:
            return None
        return context.sport_params(self.sport, self.context_overrides())

    def nba_overrides(self) -> Dict[str, float]:
        out = {}
        for field in nba_model.TUNABLE:
            if self._differs(f"nba.{field}"):
                out[field] = self.values[f"nba.{field}"]
        return out

    def nba_params(self, base: Optional[nba_model.Params] = None) -> nba_model.Params:
        """``engine.nba_model.Params`` with the tuned weights (``base``, or ``Params()``, itself by default)."""
        return nba_model.params_with(self.nba_overrides(), base)

    def counts(self, target: str) -> Dict[str, float]:
        """The count model of ``target`` ("corners", "cards"): {intercept, home, supremacy, total, size}."""
        if target not in COUNT_TARGETS:
            raise ValueError(f"Unknown count target {target!r}")
        return {term: self.values.get(f"counts.{target}.{term}", COUNT_DEFAULTS[target][term])
                for term in COUNT_TERMS}

    def to_dict(self) -> Dict[str, Any]:
        return {"sport": self.sport, "version": self.version, "source": self.source,
                "effective_from": self.effective_from, "safe_mode": self.safe_mode, "values": dict(self.values),
                "changed": self.changed()}


_lock = threading.RLock()
_state: Dict[str, Any] = {"store": None, "key": None, "epoch": None, "checked": 0.0, "params": {}, "safe": False}
_defaults: Dict[Tuple[str, bool, int], "Params"] = {}  # the defaults' Params, built once per sport, safe mode, epoch


def _default_params(sport: str, safe: bool = False, epoch: int = 0) -> "Params":
    key = (sport, bool(safe), int(epoch))
    found = _defaults.get(key)
    if found is None:
        if len(_defaults) > 200:
            _defaults.clear()
        found = _defaults[key] = Params(sport, safe_mode=safe, epoch=epoch)
    return found


def bind(store: Any) -> None:
    """Point ``current`` at a store (a data/store.Store, a learn.brain_store.BrainStore or a SQLite path); None
    unbinds (the defaults). Resets the cache."""
    with _lock:
        _state["store"] = store
        invalidate()


def invalidate() -> None:
    """Forget the cached values: the next ``current`` re-reads the store (call after a promotion in-process)."""
    with _lock:
        _state.update({"key": None, "epoch": None, "checked": 0.0, "params": {}, "safe": False})


def _store_key(store: Any) -> Any:
    return getattr(store, "path", None) or id(store)


def _load(store: Any) -> Tuple[int, bool, Dict[str, Params]]:
    from .brain_store import BrainStore, champions, epoch, safe_mode

    with BrainStore(store).connect() as db:
        number = epoch(db)
        safe = safe_mode(db)
        rows = champions(db) if not safe else {}
    out: Dict[str, Params] = {}
    for sport, row in rows.items():
        if sport not in SPORTS:
            continue
        try:
            stored = {k: v for k, v in (row.get("params") or {}).items() if k in KNOBS and KNOBS[k].applies(sport)}
            out[sport] = Params(sport, stored, version=row["version"], source=row.get("source") or "refit",
                                effective_from=row.get("effective_from"), epoch=number)
        except ValueError as error:  # a stored version the allow-list no longer accepts: serve the defaults
            log.warning("brain_params version %s for %s ignored: %s", row.get("version"), sport, error)
    return number, bool(safe), out


def current(league: Any = None, store: Any = None, *, now: Optional[float] = None) -> Params:
    """The live Params of a league's sport (``league`` None: "other"). ``store`` overrides the bound one. Defaults
    (version 0) with no store, no champion, in safe mode, or when the store can't be read."""
    sport = sport_key(league)
    with _lock:
        target = store if store is not None else _state["store"]
        if target is None:
            return _default_params(sport)
        clock = time.monotonic() if now is None else now
        key = _store_key(target)
        if key != _state["key"] or clock - _state["checked"] >= CHECK_SECONDS or clock < _state["checked"]:
            try:
                from .brain_store import BrainStore, epoch

                with BrainStore(target).connect() as db:
                    number = epoch(db)
                if key != _state["key"] or number != _state["epoch"]:
                    number, safe, loaded = _load(target)
                    _state.update({"key": key, "epoch": number, "params": loaded, "safe": safe})
                _state["checked"] = clock
            except Exception as error:  # noqa: BLE001 - the registry must never break a page
                log.warning("Brain registry unreadable, serving the defaults: %s", error)
                _state.update({"key": key, "epoch": None, "params": {}, "safe": False, "checked": clock})
        found = _state["params"].get(sport)
        if found is not None:
            return found
        return _default_params(sport, _state["safe"], _state["epoch"] or 0)


def snapshot(store: Any = None) -> Dict[str, Params]:
    """Every sport's live Params."""
    return {sport: current(sport, store) for sport in SPORTS}


def version_of(league: Any, store: Any = None) -> int:
    """The live version number of a league's sport (0: the defaults): what a logged row records."""
    return current(league, store).version


def values_of(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """{knob: value} from [{"knob", "value"}] (``describe`` rows)."""
    return {row["knob"]: float(row["value"]) for row in rows}
