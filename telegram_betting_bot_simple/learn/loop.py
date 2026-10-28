"""The learner: a thread in the dashboard process that runs the learning loop on a schedule (method.md section 7).

``Learner(store, ...).tick()`` is called about once a minute (``learn_forever``). It takes the store lease
``"learner"`` (so only one process runs it; 15 minutes, renewed every tick, before every step and every 5 minutes
while a long step runs, so a restarted dashboard takes over soon after), collects any AI batch that has finished,
then runs whatever is due, in server time:

- **nightly at 02:30**: (1) grade what is due (the grader writes each finished game's result, box stats and last
  lines into the history tables as it goes); (2) the backfill's budget of old games (learn/history.py; it only
  works between 01:00 and 06:00); (3) the week-to-date scores and (4) the tripwire, the rollback watch and the
  shadow scores (learn/tune.py ``run_night``; never a promotion at night), with each sport's headline; (5) the AI's
  note on the worst misses (at most 2 calls; a template note when no AI is linked);
- **weekly, Monday 03:30**: (1-3) candidates, the gate, promotions and rollbacks (``tune.run_week``); (4) the AI's
  proposals, each frozen as a challenger judged only on games after it; (5) the "what changed and why" texts
  (template, then the AI's when they pass the checks); (6) a three-line summary to the owner's Telegram chat with a
  link to the page;
- **monthly, the 1st at 04:30**: the review memo (1 call).

Runs are idempotent and resumable: each run is a ``brain_state`` mark ``run:<kind>:<period>`` with its steps; a step
done is never done again for that period, a step that failed is tried again on the next tick (three times at most),
and a restart carries on from the next step. Only the current period runs (no backlog after a long outage).

Pausing (the owner's switch, ``brain_state`` "paused", or LEARN_ENABLED=0) freezes the current values: the scores,
the history capture and the tripwire go on, but nothing is promoted or rolled back, no old games are fetched and the
AI is not asked. The clock is injectable (``clock``, ``tz``) so the whole schedule runs in tests on a fake clock.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import math
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import brain_store as bs
from . import evaluate as ev
from . import registry, stats, tune
from .analyst import Answer, BrainAnalyst, finite, known_evidence

log = logging.getLogger(__name__)

LEASE = "learner"
LEASE_SECONDS = 900.0  # renewed every tick, before every step and every LEASE_RENEW while a step runs; a restarted
LEASE_RENEW = 300.0  # process (a new holder) takes over at most 15 minutes after the old one stopped
TICK_SECONDS = 60.0
POLL_SECONDS = 600.0  # an AI batch still running is looked at again after this
JOB_EXPIRY = 26 * 3600.0  # a batch with no answer after this is given up (templates stay)
MAX_TRIES = 3  # a step that keeps failing is marked failed after this many ticks
NIGHT_AT = datetime.time(2, 30)
WEEK_AT = (0, datetime.time(3, 30))  # Monday
MONTH_AT = (1, datetime.time(4, 30))  # the 1st
BRAIN_PATH = "/#/bets?tab=brain"  # the Brain progress page (My bets, AI record), after PUBLIC_URL
NOTES_KEPT = 14
CHANGELOG_KEPT = 300
MISSES = 20
SPORT_LABELS = {"nfl": "NFL", "ncaaf": "College football", "nba": "NBA", "wnba": "WNBA", "ncaab": "College basketball",
                "nhl": "NHL", "mlb": "MLB", "soccer": "Soccer", "other": "Other sports"}
STEPS = {"night": ("grade", "backfill", "scores", "note"),
         "week": ("gate", "proposals", "changelog", "telegram"),
         "month": ("memo",)}


# --- the schedule ---------------------------------------------------------------------------------------------------

def _local(now: float, tz: Optional[datetime.tzinfo]) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(now, tz) if tz is not None else datetime.datetime.fromtimestamp(now)


def _at(day: datetime.date, at: datetime.time, tz: Optional[datetime.tzinfo]) -> datetime.datetime:
    return datetime.datetime.combine(day, at, tzinfo=tz) if tz is not None else datetime.datetime.combine(day, at)


def period(kind: str, now: float, tz: Optional[datetime.tzinfo] = None) -> Tuple[str, float]:
    """(the period's key, when its run was due) for the latest scheduled time of ``kind`` at or before ``now``:
    night "2026-10-28" (02:30), week "2026-W44" (its Monday 03:30), month "2026-10" (the 1st, 04:30)."""
    local = _local(now, tz)
    if kind == "night":
        due = _at(local.date(), NIGHT_AT, tz)
        if local < due:
            due = _at(local.date() - datetime.timedelta(days=1), NIGHT_AT, tz)
        return due.date().isoformat(), due.timestamp()
    if kind == "week":
        monday = local.date() - datetime.timedelta(days=local.weekday())
        due = _at(monday, WEEK_AT[1], tz)
        if local < due:
            due = _at(monday - datetime.timedelta(days=7), WEEK_AT[1], tz)
        year, week, _ = due.date().isocalendar()
        return f"{year}-W{week:02d}", due.timestamp()
    if kind == "month":
        first = local.date().replace(day=1)
        due = _at(first, MONTH_AT[1], tz)
        if local < due:
            first = (first - datetime.timedelta(days=1)).replace(day=1)
            due = _at(first, MONTH_AT[1], tz)
        return due.strftime("%Y-%m"), due.timestamp()
    raise ValueError(f"Unknown run {kind!r}")


# --- small helpers --------------------------------------------------------------------------------------------------

def _connect(store: Any) -> sqlite3.Connection:
    db = store._connect() if hasattr(store, "_connect") else sqlite3.connect(str(store), timeout=10,
                                                                            isolation_level=None)
    bs.init_brain_schema(db)
    return db


def iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sport_label(sport: str) -> str:
    return SPORT_LABELS.get(sport, sport.upper())


FAMILY_LABELS = {"weights": "Blend weights", "sharpness": "Blend sharpness",
                 "near_start": "Prediction-market weight near the start", "shrink": "Pull of ratings-only chances",
                 "rho": "Goals model low-score correction", "btts": "Both-teams-to-score nudge",
                 "adjust_weight": "Share of a late weather or roof change applied", "home_prior": "Home and draw priors",
                 "total_sd": "Spread of final totals", "margin_sd": "Spread of final margins",
                 "nba": "NBA model's market and ESPN weights", "counts.corners": "Corners model",
                 "counts.cards": "Cards model", "show.corners": "Showing the corners model",
                 "show.cards": "Showing the cards model", "challenger": "A challenger in shadow",
                 "health": "Data health", "-": "Every sport"}


def knob_label(knob: str) -> str:
    """A knob's plain name (several, comma separated, each named; a refit family by its plain name)."""
    names = [k for k in str(knob or "").split(",") if k]
    labels = [registry.KNOBS[k].label if k in registry.KNOBS else FAMILY_LABELS.get(k, k) for k in names]
    return ", ".join(dict.fromkeys(labels)) or "-"


def _value(v: Any) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return f"{float(v):g}" if abs(float(v)) >= 1 else f"{float(v):.3g}"
    return str(v)


def _values(values: Any) -> str:
    if isinstance(values, Mapping):
        return ", ".join(_value(v) for v in values.values()) or "-"
    return _value(values) if values is not None else "-"


def _moves(old: Any, new: Any, arrow: bool = True) -> str:
    """Each knob of a change named with its own values ("Blend weight of US books 0.2 → 0.218; ..."), in the
    registry's order, so no value is ever read against the wrong name."""
    if not isinstance(new, Mapping):
        return f"{_values(old)} → {_values(new)}" if arrow else _values(new)
    old = old if isinstance(old, Mapping) else {}
    order = list(registry.KNOBS)
    names = sorted(new, key=lambda k: order.index(k) if k in order else len(order))
    parts = []
    for k in names:
        label = registry.KNOBS[k].label if k in registry.KNOBS else k
        parts.append(f"{label} {_value(old.get(k))} → {_value(new[k])}" if arrow and k in old else
                     f"{label} {_value(new[k])}")
    return "; ".join(parts) or "-"


def _signed(x: Optional[float]) -> str:
    return "-" if x is None else stats.signed(x)


def telegram_notifier(token: Optional[str], chat_id: Optional[str]) -> Optional[Callable[[str], bool]]:
    """The owner's Telegram chat (TELEGRAM_CHANNEL_ID) through the bot (bot/notify.py), or None without a token or
    chat."""
    if not token or not chat_id:
        return None
    from ..bot.notify import owner_notifier
    return owner_notifier(token, chat_id)


# --- fact sheets (built from the evaluator's own numbers; the AI only reads these) ---------------------------------

def _p(p: Any) -> Dict[str, float]:
    return {str(k): round(float(v), 3) for k, v in (p or {}).items() if isinstance(v, (int, float))}


def worst_misses(db: sqlite3.Connection, now: float, days: float = 7.0, limit: int = MISSES) -> Tuple[int, List[Dict[str, Any]]]:
    """(graded games of the last ``days``, their ``limit`` worst by the brain's log loss as fact rows)."""
    rows = ev.consensus_records(db, since=now - days * 86400, until=now)
    scored = [(r, s) for r in rows if (s := ev.game_scores(r)) is not None]
    scored.sort(key=lambda rs: -rs[1]["brain_ll"])
    out = []
    for r, s in scored[:limit]:
        out.append({"id": f"miss:{r['game_key']}", "league": r["league"], "sport": r["sport"], "day": r["day"],
                    "outcome": r["outcome"], "brain_ll": finite(s["brain_ll"]), "market_ll": finite(s["market_ll"]),
                    "blend": _p((r.get("blend") or {}).get("p")),
                    "minutes_before_start": int(round((r["start"] - r["logged_at"]) / 60)),
                    "sources": [{"name": src.get("name"), "kind": src.get("kind"), "status": src.get("status"),
                                 "p": _p(src.get("p"))} for src in r.get("sources") or [] if isinstance(src, Mapping)]})
    return len(scored), out


def _calibration(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in bs.scores(db, target="result"):
        latest[row["sport"]] = row
    return [{"id": f"calibration:{sport}", "sport": sport, "week": row["week"], "n": row["n"],
             **{k: finite(row.get(k)) for k in ("slope", "slope_lo", "slope_hi", "intercept", "intercept_lo",
                                               "intercept_hi", "gap", "gap_lo", "gap_hi")}}
            for sport, row in sorted(latest.items())]


def _trial_fact(t: Mapping[str, Any]) -> Dict[str, Any]:
    return {"id": f"trial:{t['id']}", "week": t["week"], "sport": t["sport"], "knob": t["knob"],
            "setting": knob_label(t["knob"]), "source": t["source"], "old": t.get("old"), "new": t.get("new"),
            "eval_n": t.get("eval_n"), "dll": finite(t.get("dll")), "lo": finite(t.get("lo")),
            "hi": finite(t.get("hi")), "p": finite(t.get("p")), "decision": t["decision"], "reason": t.get("reason")}


def night_facts(db: sqlite3.Connection, now: float) -> Dict[str, Any]:
    """The nightly error analysis' input: the 20 worst misses of the last 7 days, calibration per sport, drift flags
    and the last 7 days' decisions."""
    games, misses = worst_misses(db, now)
    drift = {}
    for sport in registry.SPORTS:
        flag = bs.get_state(db, f"freeze:{sport}")
        if isinstance(flag, Mapping) and flag.get("frozen"):
            drift[sport] = "calibration drifted: promotions frozen"
    changes = [_trial_fact(t) for t in bs.trials(db, since=now - 7 * 86400, limit=30)]
    return {"kind": "night", "as_of": iso(now), "games_7_days": games, "misses": misses,
            "calibration": _calibration(db), "drift": drift, "changes": changes}


def _champion_values(db: sqlite3.Connection, sport: str) -> Dict[str, float]:
    row = bs.champion(db, sport)
    if row is None or bs.safe_mode(db):
        return registry.defaults(sport)
    stored = {k: v for k, v in (row.get("params") or {}).items() if k in registry.KNOBS
              and registry.KNOBS[k].applies(sport)}
    try:
        return registry.check(sport, stored)
    except ValueError:
        return registry.defaults(sport)


def week_facts(db: sqlite3.Connection, now: float, notes: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """The weekly proposals' input: the week's notes and hypotheses, the worst misses, the registry (allowed knobs,
    bounds, steps and current values) of the sports with graded games, the latest scores and the last 8 weeks'
    ledger (so nothing tried is proposed again)."""
    _, misses = worst_misses(db, now)
    sports = sorted({row["sport"] for row in bs.scores(db)} | {m["sport"] for m in misses})
    registry_rows = {sport: [{k: v for k, v in row.items() if k in ("knob", "label", "value", "low", "high", "step",
                                                                   "structural")}
                             for row in registry.describe(sport, _champion_values(db, sport))]
                     for sport in sports if sport in registry.SPORTS}
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in bs.scores(db):
        latest[(row["sport"], row["target"])] = row
    scores = [{"id": f"score:{sport}:{target}", "sport": sport, "target": target, "week": row["week"], "n": row["n"],
               **{k: finite(row.get(k)) for k in ("brain_ll", "market_ll", "baseline_ll", "gap", "gap_lo", "gap_hi")}}
              for (sport, target), row in sorted(latest.items())]
    ledger = [_trial_fact(t) for t in bs.trials(db, since=now - 8 * 7 * 86400, limit=60)]
    return {"kind": "week", "as_of": iso(now), "week": tune.decision_week(now),
            "notes": [{"text": n.get("text"), "hypotheses": n.get("hypotheses") or [], "at": iso(n.get("at"))}
                      for n in notes][-7:],
            "misses": misses, "calibration": _calibration(db), "registry": registry_rows, "scores": scores,
            "ledger": ledger}


def week_changes(db: sqlite3.Connection, now: float, limit: int = 20) -> List[Dict[str, Any]]:
    """The decisions to explain this week: the decided week's trials and anything rolled back, undone or switched
    to safe mode in the last 7 days, the ones that changed something first."""
    week = tune.decision_week(now)
    found = {t["id"]: t for t in bs.trials(db, week=week, limit=200)}
    for t in bs.trials(db, since=now - 7 * 86400, limit=200):
        if t["decision"] in ("rolled_back", "undone", "safe_mode", "cleared", "promoted", "shadow"):
            found[t["id"]] = t
    order = {"promoted": 0, "rolled_back": 1, "undone": 2, "safe_mode": 3, "shadow": 4, "cleared": 5, "rejected": 6,
             "held": 7}
    ranked = sorted(found.values(), key=lambda t: (order.get(t["decision"], 9), -t["id"]))
    return [_trial_fact(t) for t in ranked[:limit]]


def month_facts(db: sqlite3.Connection, now: float, month: str, ai: Mapping[str, Any]) -> Dict[str, Any]:
    """The monthly memo's input: the month's trials by decision, the AI's proposals' record, its spend against the
    cap, and each sport's latest scores."""
    since = now - 31 * 86400
    trials = bs.trials(db, since=since, limit=None)
    by_decision: Dict[str, int] = {}
    for t in trials:
        by_decision[t["decision"]] = by_decision.get(t["decision"], 0) + 1
    latest: Dict[str, Dict[str, Any]] = {}
    for row in bs.scores(db, target="result"):
        latest[row["sport"]] = row
    return {"kind": "month", "month": month, "as_of": iso(now), "candidates_tried": len(trials),
            "decisions": by_decision, "ai_proposals": ai_record(db), "ai_spend": dict(ai),
            "scores": [{"sport": sport, "week": row["week"], "n": row["n"], "gap": finite(row.get("gap")),
                        "gap_lo": finite(row.get("gap_lo")), "gap_hi": finite(row.get("gap_hi")),
                        "slope": finite(row.get("slope"))} for sport, row in sorted(latest.items())]}


def ai_record(db: sqlite3.Connection, since: Optional[float] = None) -> Dict[str, int]:
    """The AI's proposals: made (frozen or refused), passed (promoted), failed (rejected after the forward test),
    pending (challengers still in shadow) and refused (failed the code-side checks)."""
    made = passed = failed = refused = 0
    for t in bs.trials(db, since=since, limit=None):
        if not str(t.get("source") or "").startswith("ai:"):
            continue
        if t["decision"] == "shadow":
            made += 1
        elif t["decision"] == "rejected" and str(t.get("reason") or "").startswith("Proposal refused"):
            made += 1
            refused += 1
        elif t["decision"] == "promoted":
            passed += 1
        elif t["decision"] == "rejected":
            failed += 1
    pending = sum(1 for v in bs.versions(db, status="challenger") if str(v.get("source") or "").startswith("ai:"))
    return {"made": made, "passed": passed, "failed": failed, "pending": pending, "refused": refused}


# --- template texts (code writes them from the same facts; the AI's replace them only when they pass the checks) ---

def template_note(facts: Mapping[str, Any]) -> str:
    misses = facts.get("misses") or []
    games = facts.get("games_7_days") or 0
    if not misses:
        return "No graded games in the last 7 days, so there is nothing to review yet."
    worst = misses[0]
    chance = (worst.get("blend") or {}).get(worst.get("outcome"))
    said = f"the brain gave that result {chance * 100:.1f}%" if isinstance(chance, (int, float)) else "the brain missed"
    market = (f"; the market's log loss was {worst['market_ll']:.4f}" if worst.get("market_ll") is not None else "")
    drift = facts.get("drift") or {}
    tail = (f" Calibration drifted in {', '.join(sport_label(s) for s in sorted(drift))}, so promotions there wait for "
            "the weekly review." if drift else
            " Most misses are the games' own uncertainty; only the weekly gate decides any change.")
    return (f"{games} graded games in the last 7 days. The biggest miss ({sport_label(worst['sport'])}, "
            f"{worst['day']}): {said} (log loss {worst['brain_ll']:.4f}{market}).{tail}")


def template_change(change: Mapping[str, Any]) -> str:
    """One "what changed and why" line from a ledger row (its reason is already plain words from the gate)."""
    knob, sport = knob_label(change.get("knob")), sport_label(str(change.get("sport")))
    decision = change.get("decision")
    reason = str(change.get("reason") or "").strip()
    numbers = ""
    if change.get("dll") is not None and change.get("eval_n"):
        numbers = (f" Tested on {change['eval_n']} games: change in log loss {_signed(change['dll'])} "
                   f"(95% range {_signed(change.get('lo'))} to {_signed(change.get('hi'))}).")
    if decision == "promoted":
        return f"{sport}: {_moves(change.get('old'), change.get('new'))}.{numbers} {reason}".strip()
    if decision == "shadow":
        return f"{sport}: {knob} is now tested in shadow on games after today; nothing live changes yet. {reason}".strip()
    if decision == "rolled_back":
        return (f"{sport}: back to the version before it ({_moves(change.get('old'), change.get('new'), arrow=False)})."
                f" {reason}").strip()
    if decision == "undone":
        return f"{sport}: {knob} was undone by the owner."
    if decision == "safe_mode":
        return f"Safe mode: the default settings serve every sport. {reason}".strip()
    if decision == "cleared":
        return "Safe mode was cleared by the owner."
    return f"{sport}: {knob}: no change.{numbers} {reason}".strip()


def template_memo(facts: Mapping[str, Any]) -> str:
    decisions = facts.get("decisions") or {}
    ai = facts.get("ai_proposals") or {}
    spend = facts.get("ai_spend") or {}
    lines = [f"Month {facts.get('month')}: {facts.get('candidates_tried', 0)} candidates tried, "
             f"{decisions.get('promoted', 0)} promoted, {decisions.get('rolled_back', 0)} rolled back, "
             f"{decisions.get('held', 0)} held."]
    if ai.get("made") or ai.get("pending"):
        lines.append(f"AI proposals: {ai.get('made', 0)} made, {ai.get('passed', 0)} passed, {ai.get('failed', 0)} "
                     f"failed, {ai.get('pending', 0)} still being judged.")
    if spend.get("provider"):
        lines.append(f"AI cost this month: {spend.get('spent_cents', 0):.0f} of {spend.get('cap_cents', 0)} cents.")
    for row in facts.get("scores") or []:
        if row.get("gap") is not None:
            lines.append(f"{sport_label(row['sport'])}, week {row['week']}: log loss {_signed(row['gap'])} against "
                         f"the market over {row['n']} games.")
    return "\n".join(lines)


HEADLINE_MIN_GAMES = 100  # fewer graded games than this: the headline says so instead of "ahead" or "behind"


def sport_headline(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """evaluate.headline over a sport's graded rows (its last 200 with a market price), with its word held back
    ("too few games to say yet") below HEADLINE_MIN_GAMES games or without a 95% range: a handful of games says
    nothing about the brain against the market."""
    h = dict(ev.headline(rows))
    out = {k: (finite(v, 6) if isinstance(v, float) else v) for k, v in h.items()}
    if out.get("n") and (out["n"] < HEADLINE_MIN_GAMES or out.get("lo") is None or out.get("hi") is None):
        out["word"] = "too few games to say yet"
    return out


def headline_text(sport: str, h: Mapping[str, Any]) -> str:
    """"NBA, last 200 games: about level with the sharp market (log loss +0.0004, 95% range -0.0011 to +0.0019)"."""
    if not h or not h.get("n"):
        return f"{sport_label(sport)}: no graded games with a market price yet."
    games = f"{h['n']} game" + ("s" if h["n"] != 1 else "")
    if h.get("gap") is None or h.get("lo") is None or h.get("hi") is None:
        return f"{sport_label(sport)}, last {games}: {h.get('word') or 'too few games to say yet'}."
    return (f"{sport_label(sport)}, last {games}: {h['word']} (log loss {_signed(h['gap'])}, 95% range "
            f"{_signed(h.get('lo'))} to {_signed(h.get('hi'))}).")


def gate_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    """A weekly run's report (tune.run_week) as the page, the summary and the run mark keep it."""
    return {"week": report.get("week"), "skipped": report.get("skipped"),
            "trials": len(report.get("trials") or []),
            "promoted": [{k: v for k, v in p.items() if k in ("sport", "version", "knobs", "source")}
                         for p in report.get("promoted") or []],
            "rolled_back": [{k: v for k, v in r.items() if k in ("sport", "version", "now_live", "reason")}
                            for r in report.get("rolled_back") or []],
            "shadow": len(report.get("shadow") or []), "held": len(report.get("held") or []),
            "rejected": len(report.get("rejected") or [])}


def weekly_summary(week: str, report: Optional[Mapping[str, Any]], headlines: Mapping[str, Mapping[str, Any]],
                   url: Optional[str]) -> str:
    """The owner's three lines: what the gate did (from its counts; None: the weekly check didn't run, and no
    counts are given), the busiest sport's headline, the link."""
    report = report if report is not None else {}
    tested = int(report.get("trials") or 0)
    promoted = len(report.get("promoted") or [])
    rolled = len(report.get("rolled_back") or [])
    cands = f"{tested} candidate" + ("" if tested == 1 else "s")
    if not report:
        first = f"Brain, week {week}: the weekly check couldn't run this time, so there is no decision to report."
    elif report.get("skipped") in ("paused", "safe mode"):
        why = "learning is paused" if report["skipped"] == "paused" else "the brain is in safe mode"
        first = f"Brain, week {week}: {why}, so nothing changed ({cands} scored)."
    elif promoted or rolled:
        first = (f"Brain, week {week}: {cands} tested, {promoted} promoted, {rolled} rolled back. "
                 "Every change can be undone on the page.")
    else:
        first = f"Brain, week {week}: {cands} tested, no change this week (most weeks end this way)."
    busiest = max(headlines.items(), key=lambda kv: kv[1].get("n") or 0, default=None)
    second = headline_text(busiest[0], busiest[1]) if busiest and busiest[1].get("n") else \
        "No graded games with a market price yet."
    third = f"Details: {url}" if url else "Details: the Brain progress page (My bets, AI record)."
    return "\n".join((first, second, third))


# --- the learner ----------------------------------------------------------------------------------------------------

class Learner:
    """The learning loop's scheduler (see the module's docstring).

    ``store`` is the data/store.Store; ``service`` the DashboardService (its grader); ``cache`` the shared TTLCache
    (the backfill's fetches); ``config`` the app's settings (LEARN_*, the AI keys, PUBLIC_URL, the bot token and
    TELEGRAM_CHANNEL_ID). Tests hand in ``clock``/``tz`` (a fake clock), ``analyst`` (a BrainAnalyst over fake
    clients), ``notify`` (a Telegram stand-in), ``backfill`` (``backfill(now) -> report``) and ``lease``."""

    def __init__(self, store: Any, *, service: Any = None, cache: Any = None, config: Any = None,
                 analyst: Optional[BrainAnalyst] = None, notify: Optional[Callable[[str], bool]] = None,
                 backfill: Optional[Callable[[float], Mapping[str, Any]]] = None,
                 clock: Optional[Callable[[], float]] = None, tz: Optional[datetime.tzinfo] = None,
                 lease: Optional[Callable[[], bool]] = None, public_url: Optional[str] = None,
                 enabled: Optional[bool] = None, use_notify: bool = True) -> None:
        self.store = store
        self.service = service
        self.cache = cache
        self.config = config
        self.clock = clock or time.time
        self.tz = tz
        self.enabled = bool(getattr(config, "learn_enabled", True)) if enabled is None else bool(enabled)
        self.per_night = int(getattr(config, "learn_backfill_per_night", 1500) or 0)
        self.analyst = analyst or BrainAnalyst(
            store, mode=str(getattr(config, "learn_ai", "auto") or "auto"), keys=self._keys,
            monthly_cents=int(getattr(config, "learn_ai_monthly_cents", 500)), clock=self.clock, tz=tz)
        if notify is None and use_notify and config is not None:
            notify = telegram_notifier(getattr(config, "telegram_token", None),
                                       getattr(config, "telegram_channel_id", None))
        self.notify = notify
        self._backfill = backfill
        self.lease = lease or (lambda: bool(store.try_lease(LEASE, LEASE_SECONDS)))
        url = public_url if public_url is not None else getattr(config, "public_url", None)
        self.page_url = (str(url).rstrip("/") + BRAIN_PATH) if url else None
        self._lock = threading.Lock()
        self.renew_every = LEASE_RENEW

    def _keys(self) -> Dict[str, Any]:
        c = self.config
        return {"anthropic_api_key": getattr(c, "anthropic_api_key", None),
                "deepseek_api_key": getattr(c, "deepseek_api_key", None),
                "anthropic_model": getattr(c, "anthropic_model", None),
                "deepseek_model": getattr(c, "deepseek_model", None)}

    # -- state --

    def _state(self, key: str, default: Any = None) -> Any:
        with closing(_connect(self.store)) as db:
            return bs.get_state(db, key, default)

    def _set(self, key: str, value: Any, now: float) -> None:
        with closing(_connect(self.store)) as db:
            bs.set_state(db, key, value, now)

    def paused(self) -> Optional[Dict[str, Any]]:
        """{"paused": True, "by", "at"} while learning is paused (LEARN_ENABLED=0 or the owner's switch), else
        None."""
        if not self.enabled:
            return {"paused": True, "by": "LEARN_ENABLED", "at": None}
        state = self._state("paused")
        if isinstance(state, Mapping) and state.get("paused"):
            return {"paused": True, "by": state.get("by") or "owner", "at": state.get("at")}
        return None

    def set_paused(self, paused: bool, *, by: str = "owner", now: Optional[float] = None) -> Dict[str, Any]:
        """The owner's Pause switch. Returns the learning state ({"paused", "by", "at", "locked"}: locked when
        LEARN_ENABLED=0 holds it paused whatever the switch says)."""
        now = self.clock() if now is None else now
        before = self.paused()
        self._set("paused", {"paused": bool(paused), "by": by, "at": now}, now)
        after = self.paused()
        if bool(before) != bool(after):
            self._tell("Learning paused: the brain keeps its current settings until it is switched back on."
                       if after else "Learning switched back on.")
        return {"paused": bool(after), "by": (after or {}).get("by"), "at": now, "locked": not self.enabled}

    def undo(self, version: int, *, by: str = "owner", now: Optional[float] = None) -> Dict[str, Any]:
        """The owner's Undo of one change (tune.undo: back to its parent, or a challenger withdrawn)."""
        now = self.clock() if now is None else now
        # Not under the run lock: a night's backfill can take most of an hour, and every write here is its own
        # SQLite transaction (a weekly promotion and an Undo can't interleave inside one)
        result = tune.undo(self.store, int(version), now=now, by=by)
        if result.get("undone") is not None:
            registry.invalidate()
            if result.get("withdrawn"):
                self._tell(f"Withdrawn by the owner: brain version {result['undone']} was an experiment in shadow; "
                           "nothing live changed.")
            else:
                self._tell(f"Undone by the owner: brain version {result['undone']}; version {result['now_live']} is "
                           "live again." if result.get("now_live") else
                           f"Undone by the owner: brain version {result['undone']}; the default settings are live "
                           "again.")
        return result

    def clear_safe_mode(self, *, by: str = "owner", now: Optional[float] = None) -> Dict[str, Any]:
        """The owner clears safe mode (set by the tripwire): the live versions serve again. Returns {"cleared"}."""
        now = self.clock() if now is None else now
        cleared = tune.clear_safe_mode(self.store, now=now, by=by)
        if cleared:
            registry.invalidate()
            self._tell("Safe mode cleared by the owner: the brain's live settings serve again.")
        return {"cleared": bool(cleared)}

    def _tell(self, text: str) -> bool:
        if self.notify is None:
            return False
        try:
            return bool(self.notify(text if not self.page_url else f"{text}\n{self.page_url}"))
        except Exception:  # noqa: BLE001
            log.exception("Couldn't send the brain's message")
            return False

    # -- the schedule --

    def due(self, now: Optional[float] = None) -> List[Tuple[str, str]]:
        """The runs due at ``now``: (kind, period key), nightly first."""
        now = self.clock() if now is None else now
        out = []
        for kind in ("night", "week", "month"):
            key, at = period(kind, now, self.tz)
            mark = self._state(f"run:{kind}:{key}")
            if now >= at and not (isinstance(mark, Mapping) and mark.get("status") == "done"):
                out.append((kind, key))
        return out

    def tick(self, now: Optional[float] = None) -> Dict[str, Any]:
        """One scheduler step: take the lease, collect finished AI batches, run what is due. Returns what it did."""
        now = self.clock() if now is None else now
        if not self.lease():
            return {"at": now, "skipped": "another process holds the learner lease"}
        with self._lock:
            report: Dict[str, Any] = {"at": now, "ai": [], "ran": []}
            try:
                report["ai"] = self.poll_ai(now)
            except Exception:  # noqa: BLE001 - a broken batch never holds up the schedule
                log.exception("Couldn't collect the learner's AI batches")
            for kind, key in self.due(now):
                report["ran"].append(self.run(kind, key, now))
            return report

    def run(self, kind: str, key: str, now: float) -> Dict[str, Any]:
        """Run (or carry on with) one period's run; each step is recorded as it finishes."""
        mark_key = f"run:{kind}:{key}"
        mark = dict(self._state(mark_key) or {})
        if mark.get("status") == "done":
            return {"kind": kind, "key": key, "skipped": "already done"}
        mark.setdefault("started_at", now)
        mark["status"] = "running"
        steps: Dict[str, Any] = dict(mark.get("steps") or {})
        for step in STEPS[kind]:
            entry = dict(steps.get(step) or {})
            if entry.get("done"):
                continue
            if not self.lease():
                mark["steps"] = steps
                self._set(mark_key, mark, now)
                return {"kind": kind, "key": key, "stopped": "lease lost", "steps": steps}
            try:
                with self._keep_lease():
                    result = getattr(self, f"_{kind}_{step}")(now, key)
                entry = {"done": True, "at": now, "result": result}
            except Exception as exc:  # noqa: BLE001 - a failing step is retried next tick, then given up
                log.exception("Learner %s %s step %s failed", kind, key, step)
                tries = int(entry.get("tries") or 0) + 1
                entry = {"done": tries >= MAX_TRIES, "tries": tries, "error": f"{type(exc).__name__}: {exc}"[:300],
                         "at": now}
                steps[step] = entry
                mark["steps"] = steps
                self._set(mark_key, mark, now)
                if not entry["done"]:
                    return {"kind": kind, "key": key, "stopped": f"{step} failed", "steps": steps}
                continue
            steps[step] = entry
            mark["steps"] = steps
            self._set(mark_key, mark, now)
        mark.update({"status": "done", "finished_at": now, "steps": steps})
        self._set(mark_key, mark, now)
        self._set(f"last_{kind}", {"key": key, "at": now}, now)
        return {"kind": kind, "key": key, "done": True, "steps": steps}

    @contextlib.contextmanager
    def _keep_lease(self) -> Iterator[None]:
        """Renew the lease every LEASE_RENEW seconds while a long step (the backfill) runs."""
        stop = threading.Event()

        def renew() -> None:
            while not stop.wait(self.renew_every):
                try:
                    self.lease()
                except Exception:  # noqa: BLE001
                    log.exception("Couldn't renew the learner's lease")

        keeper = threading.Thread(target=renew, name="learner-lease", daemon=True)
        keeper.start()
        try:
            yield
        finally:
            stop.set()
            keeper.join(timeout=5)

    # -- nightly steps --

    def _night_grade(self, now: float, key: str) -> Dict[str, Any]:
        if self.service is None or not hasattr(self.service, "grade_due"):
            return {"skipped": "no grader here"}
        return {"graded": int(self.service.grade_due() or 0)}

    def _night_backfill(self, now: float, key: str) -> Dict[str, Any]:
        if self.paused():
            return {"skipped": "paused"}
        if self._backfill is not None:
            report = dict(self._backfill(now) or {})
        else:
            from .history import Backfill
            report = Backfill(self.store, self.cache, self.clock, tz=self.tz, per_night=self.per_night).run_night()
        return {k: report.get(k) for k in ("ran", "stopped", "used", "requests", "games", "budget")}

    def _night_scores(self, now: float, key: str) -> Dict[str, Any]:
        paused = bool(self.paused())
        mark = self._state(f"run:night:{key}") or {}
        since = float(mark.get("started_at") or now) if isinstance(mark, Mapping) else now
        try:
            report = tune.run_night(self.store, now=now, paused=paused)
            headlines = self.write_headlines(now)
        finally:
            # From the ledger, not the report: a rollback or safe mode written before a later part of the run failed
            # is still told (once) when the step stops, and again never
            self._tell_changes(since)
        trip = report.get("tripwire") or {}
        return {"scores": report.get("scores"), "rolled_back": len(report.get("rolled_back") or []),
                "tripwire": {k: trip.get(k) for k in ("fired", "share", "sharp_games", "safe_mode", "reason")},
                "headlines": len(headlines)}

    def _tell_changes(self, since: float) -> int:
        """Tell the owner of each automatic rollback and safe-mode switch in the ledger since ``since`` not told
        yet (brain_state ``notified:<trial id>``). Returns the messages sent."""
        with closing(_connect(self.store)) as db:
            found = [t for t in bs.trials(db, since=since, limit=None)
                     if t["decision"] in ("rolled_back", "safe_mode") and t.get("gate") in ("rollback", "tripwire")
                     and not bs.get_state(db, f"notified:{t['id']}")]
        sent = 0
        for t in sorted(found, key=lambda t: t["id"]):
            if t["decision"] == "rolled_back":
                text = (f"Brain: {sport_label(t['sport'])} rolled back automatically "
                        f"({str(t.get('reason') or '').rstrip('.')}).")
            else:
                why = (t.get("details") or {}).get("reason") or t.get("reason")
                text = f"Brain: safe mode is on, so the default settings serve every sport ({str(why).rstrip('.')})."
            if self._tell(text):
                sent += 1
                self._set(f"notified:{t['id']}", True, time.time())
        return sent

    def write_headlines(self, now: float) -> Dict[str, Dict[str, Any]]:
        """Each sport's headline over its last 200 graded games with a market price (evaluate.headline), kept in
        brain_state "headlines" for the page, the bot and the AI tool."""
        with closing(_connect(self.store)) as db:
            rows = ev.consensus_records(db, until=now)
            out = {}
            for sport in registry.SPORTS:
                sport_rows = [r for r in rows if r["sport"] == sport]
                if not sport_rows:
                    continue
                out[sport] = sport_headline(sport_rows)
                out[sport]["text"] = headline_text(sport, out[sport])
            bs.set_state(db, "headlines", {"at": now, "sports": out}, now)
        return out

    def _night_note(self, now: float, key: str) -> Dict[str, Any]:
        with closing(_connect(self.store)) as db:
            facts = night_facts(db, now)
        note = {"text": template_note(facts), "hypotheses": [], "source": "template", "model": None, "at": now,
                "run": f"night:{key}"}
        self._keep_note(note, now)
        if self.paused():
            return {"note": "template", "ai": "paused"}
        return {"note": "template", "ai": self._ask("note", facts, now, f"night-{key}")}

    def _keep_note(self, note: Mapping[str, Any], now: float) -> None:
        self._set("note:latest", dict(note), now)
        notes = list(self._state("notes") or [])
        notes = [n for n in notes if n.get("run") != note.get("run")] + [dict(note)]
        self._set("notes", notes[-NOTES_KEPT:], now)

    # -- weekly steps --

    def _week_gate(self, now: float, key: str) -> Dict[str, Any]:
        report = tune.run_week(self.store, now=now, paused=bool(self.paused()))
        summary = gate_summary(report)
        if report.get("skipped") != "already ran":
            self._set(f"week_report:{summary['week']}", summary, now)
            self._set("week_report:latest", summary, now)
        return summary

    def _this_weeks_report(self, key: str, now: float) -> Tuple[str, Optional[Dict[str, Any]]]:
        """(the decided week, this run's gate summary): the gate step's own result, else the week's stored report
        or its stored decisions; None when the gate decided nothing this week (never last week's report)."""
        mark = self._state(f"run:week:{key}") or {}
        started = float(mark.get("started_at") or now) if isinstance(mark, Mapping) else now
        week = tune.decision_week(started)
        gate = ((mark.get("steps") or {}).get("gate") or {}) if isinstance(mark, Mapping) else {}
        result = gate.get("result") if gate.get("done") and not gate.get("error") else None
        if isinstance(result, Mapping) and result.get("skipped") != "already ran":
            return str(result.get("week") or week), dict(result)
        stored = self._state(f"week_report:{week}")
        if isinstance(stored, Mapping):
            return week, dict(stored)
        decided = self._state(f"week_decided:{week}")
        if isinstance(decided, Mapping) and isinstance(decided.get("report"), Mapping):
            return week, gate_summary(decided["report"])
        return week, None

    def _week_proposals(self, now: float, key: str) -> Dict[str, Any]:
        if self.paused():
            return {"skipped": "paused"}
        notes = list(self._state("notes") or [])
        with closing(_connect(self.store)) as db:
            facts = week_facts(db, now, [n for n in notes if (n.get("at") or 0) >= now - 7 * 86400])
        return {"ai": self._ask("proposals", facts, now, f"week-{key}")}

    def _week_changelog(self, now: float, key: str) -> Dict[str, Any]:
        with closing(_connect(self.store)) as db:
            changes = week_changes(db, now)
        log_ = dict(self._state("changelog") or {})
        for change in changes:
            ident = change["id"].split(":", 1)[1]
            entry = log_.get(ident)
            if not entry or entry.get("source") == "template":
                log_[ident] = {"text": template_change(change), "source": "template", "model": None, "at": now}
        self._set("changelog", _trim_log(log_), now)
        if not changes:
            return {"changes": 0}
        if self.paused():
            return {"changes": len(changes), "ai": "paused"}
        facts = {"kind": "changelog", "week": tune.decision_week(now), "changes": changes}
        return {"changes": len(changes), "ai": self._ask("changelog", facts, now, f"changelog-{key}")}

    def _week_telegram(self, now: float, key: str) -> Dict[str, Any]:
        week, report = self._this_weeks_report(key, now)
        headlines = (self._state("headlines") or {}).get("sports") or {}
        if report is not None and not report.get("trials") and not any((h or {}).get("n") for h in headlines.values()):
            return {"sent": False, "why": "nothing to report yet"}
        text = weekly_summary(week, report, headlines, self.page_url)
        if self.notify is None:
            return {"sent": False, "why": "no owner chat (TELEGRAM_CHANNEL_ID) or bot token", "text": text}
        sent = False
        try:
            sent = bool(self.notify(text))
        except Exception:  # noqa: BLE001
            log.exception("Couldn't send the weekly summary")
        return {"sent": sent, "text": text}

    # -- monthly step --

    def _month_memo(self, now: float, key: str) -> Dict[str, Any]:
        status = self.ai_status(now)
        with closing(_connect(self.store)) as db:
            facts = month_facts(db, now, key, status)
        self._set("memo:latest", {"text": template_memo(facts), "ideas": [], "source": "template", "model": None,
                                  "at": now, "month": key}, now)
        if self.paused():
            return {"memo": "template", "ai": "paused"}
        return {"memo": "template", "ai": self._ask("memo", facts, now, f"month-{key}")}

    # -- the AI --

    def _jobs(self) -> List[Dict[str, Any]]:
        return [dict(j) for j in self._state("ai_jobs") or [] if isinstance(j, Mapping)]

    def reserved(self) -> float:
        """Dollars promised to AI batches still out (counted against the monthly cap)."""
        return sum(float((j.get("batch") or {}).get("estimate") or 0) for j in self._jobs())

    def ai_status(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if now is None else now
        status = self.analyst.status(now, self.reserved())
        status["pending"] = len(self._jobs())
        return status

    def _ask(self, kind: str, facts: Mapping[str, Any], now: float, job: str) -> Dict[str, Any]:
        """Ask the AI (a batch with Claude when it can); apply the answer, or queue the batch."""
        answer = self.analyst.ask(kind, facts, now=now, reserved=self.reserved(), batch=True, job=job)
        if answer.status == "pending" and answer.batch:
            jobs = [j for j in self._jobs() if j.get("job") != job]
            jobs.append({"job": job, "kind": kind, "batch": {**answer.batch, "kind": kind}, "submitted_at": now,
                         "polled_at": now})
            self._set(f"ai_facts:{job}", dict(facts), now)
            self._set("ai_jobs", jobs, now)
        else:
            self._apply(answer, facts, now, job)
        return {"status": answer.status, "provider": answer.provider, "reason": answer.reason,
                "cost": round(answer.cost, 6), "calls": answer.calls}

    def poll_ai(self, now: float) -> List[Dict[str, Any]]:
        """Collect finished batches (each looked at every POLL_SECONDS); give up on ones past JOB_EXPIRY."""
        done: List[Dict[str, Any]] = []
        jobs = self._jobs()
        if not jobs:
            return done
        kept: List[Dict[str, Any]] = []
        paused = bool(self.paused())
        for i, job in enumerate(jobs):
            if now - float(job.get("polled_at") or 0) < POLL_SECONDS and now - float(job.get("submitted_at") or 0) \
                    < JOB_EXPIRY:
                kept.append(job)
                continue
            facts = self._state(f"ai_facts:{job['job']}") or {}
            answer = None
            if now - float(job.get("submitted_at") or 0) < JOB_EXPIRY:
                others = sum(float((j.get("batch") or {}).get("estimate") or 0) for j in kept + jobs[i + 1:])
                try:
                    # Paused: the answer is read but never asked again (no new call while learning is paused)
                    answer = self.analyst.finish(job["batch"], facts, now=now, kind=job["kind"], reserved=others,
                                                 retry=not paused)
                except Exception as exc:  # noqa: BLE001 - an answer that can't be read is given up, not re-billed
                    log.exception("Couldn't read the learner's AI batch %s", job.get("job"))
                    answer = Answer(job["kind"], "failed", provider=(job.get("batch") or {}).get("provider"),
                                    reason=f"the batch's answer couldn't be read ({type(exc).__name__})")
            if answer is None and now - float(job.get("submitted_at") or 0) < JOB_EXPIRY:
                kept.append({**job, "polled_at": now})
                continue
            if answer is None:
                answer = Answer(job["kind"], "failed", provider=(job.get("batch") or {}).get("provider"),
                                reason="the batch gave no answer within a day")
            # The job leaves the queue before its answer is used: it is collected (and its cost logged) once only,
            # whatever happens next
            self._set("ai_jobs", kept + jobs[i + 1:], now)
            status = answer.status
            try:
                self._apply(answer, facts, now, job["job"])
            except Exception:  # noqa: BLE001 - one answer that can't be used never holds up the others
                log.exception("Couldn't use the learner's AI answer %s", job.get("job"))
                status = "failed"
            self._set(f"ai_facts:{job['job']}", None, now)
            done.append({"job": job["job"], "kind": job["kind"], "status": status})
        self._set("ai_jobs", kept, now)
        return done

    def _apply(self, answer: Answer, facts: Mapping[str, Any], now: float, job: str) -> None:
        """Use a checked answer: the note, the proposals (frozen as challengers), the change texts, the memo."""
        rejected = len(answer.rejected) + (1 if answer.status == "failed" and "facts" in answer.reason else 0)
        if rejected:
            self._set("ai_rejected", int(self._state("ai_rejected", 0) or 0) + rejected, now)
        self._set("ai_last", {"kind": answer.kind, "status": answer.status, "reason": answer.reason,
                              "provider": answer.provider, "model": answer.model, "at": now,
                              "cost": round(answer.cost, 6), "job": job, "calls": answer.calls,
                              "trimmed": answer.trimmed, "rejected": len(answer.rejected)}, now)
        if answer.status != "done" or not answer.data:
            return
        source = f"ai:{answer.provider}"
        data = answer.data
        if answer.kind == "note" and data.get("note"):
            run = f"night:{job[len('night-'):]}" if job.startswith("night-") else job
            self._keep_note({"text": data["note"], "hypotheses": data.get("hypotheses") or [], "source": source,
                             "model": answer.model, "at": now, "run": run}, now)
        elif answer.kind == "proposals":
            proposals = data.get("proposals") or []
            outcomes: List[Dict[str, Any]] = []
            if proposals and self.paused():  # an answer collected while paused: kept, nothing frozen or tuned
                outcomes = [{"rejected": "not frozen: learning is paused"} for _ in proposals]
            elif proposals:
                with closing(_connect(self.store)) as db:
                    corpus = ev.load_corpus(db, until=now)
                    outcomes = tune.freeze_proposals(db, proposals, corpus, provider=str(answer.provider), now=now,
                                                     known_evidence=known_evidence(facts))
            self._set("proposals:latest", {"at": now, "provider": answer.provider, "model": answer.model,
                                           "none_reason": data.get("none_reason"),
                                           "proposals": [{**p, "outcome": o} for p, o in zip(proposals, outcomes)]},
                      now)
        elif answer.kind == "changelog":
            log_ = dict(self._state("changelog") or {})
            known = {c["id"].split(":", 1)[1] for c in facts.get("changes") or [] if isinstance(c, Mapping)}
            for item in data.get("texts") or []:
                ident = str(item.get("id") or "").split(":", 1)[-1]
                if ident in known and item.get("text"):
                    log_[ident] = {"text": item["text"], "source": source, "model": answer.model, "at": now}
            self._set("changelog", _trim_log(log_), now)
        elif answer.kind == "memo" and data.get("memo"):
            memo = dict(self._state("memo:latest") or {})
            memo.update({"text": data["memo"], "ideas": data.get("data_ideas") or [], "source": source,
                         "model": answer.model, "at": now})
            self._set("memo:latest", memo, now)


def _trim_log(log_: Mapping[str, Any]) -> Dict[str, Any]:
    ordered = sorted(log_.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)
    return dict(ordered[-CHANGELOG_KEPT:])


def learn_forever(learner: Learner, every: float = TICK_SECONDS, rounds: Optional[int] = None,
                  sleep: Callable[[float], None] = time.sleep) -> None:
    """The dashboard's "learner" thread: a tick every minute (``rounds`` stops it early, for tests)."""
    done = 0
    while rounds is None or done < rounds:
        try:
            learner.tick()
        except Exception:  # noqa: BLE001 - the loop never dies
            log.exception("The learner's run failed")
        done += 1
        if rounds is None or done < rounds:
            sleep(every)


__all__ = ["Learner", "learn_forever", "period", "telegram_notifier", "night_facts", "week_facts", "week_changes",
           "month_facts", "ai_record", "template_note", "template_change", "template_memo", "weekly_summary",
           "gate_summary",
           "headline_text", "sport_label", "knob_label", "iso", "LEASE", "BRAIN_PATH", "STEPS", "SPORT_LABELS"]
