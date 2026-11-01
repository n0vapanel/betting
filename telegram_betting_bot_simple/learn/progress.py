"""The Brain progress page's numbers (GET /api/brain/progress), the bot's /brain lines and the AI tool's facts, all
read from the store (nothing is fetched or computed from the network; method.md section 8). Every number comes from
the evaluator and the gate; every status word ("promoted", "rolled back", "held") is the gate's decision as stored.

``progress(store, learner=None, owner=False, now=None)`` returns::

    {"as_of", "served_at", "owner",        # as_of: when the numbers were scored (the nightly run); owner: whether
                                           # the viewer may Undo and pause
     "learning": {"enabled", "paused", "by", "at", "locked"},
     "safe_mode": None | {"since", "reason"},
     "sports": [{"sport", "label", "version": {"number", "since", "source", "changed", "labels"},
                 "headline": {"n", "gap", "lo", "hi", "word", "text", "since", "until"} | None,
                 "series": [{"week", "n", "gap", "lo", "hi", "base_gap", "n_market"}],
                 "markers": [{"week", "at", "kind", "version", "knob", "label"}],
                 "calibration": {"n", "slope", "slope_lo", "slope_hi", "intercept", "intercept_lo", "intercept_hi",
                                 "words", "curve": {"x", "y", "lo", "hi"} | None} | None,
                 "targets": [{"target", "label", "week", "n", "brain_ll", "market_ll", "baseline_ll", "gap",
                              "gap_lo", "gap_hi", "vs", "shown", "status", "pit"}],
                 "adjustments": {"week", "games", "adjusted", "dll", "lo", "hi", "brier", "slope", "causes"} | None}],
     "changes": [{"id", "version", "sport", "sport_label", "knob", "label", "old", "new", "at", "week", "decision",
                  "gate", "source", "provider", "evidence": {"n", "train_n", "dll", "lo", "hi", "p", "holm_p",
                  "tried"}, "reason", "text", "text_source", "text_model", "rolled_back", "undo"}],
     "held": {"week", "tested", "passed", "frozen", "words", "items": [{"id", "sport", "label", "decision",
              "reason", "eval_n", "needed"}]},
     "experiments": [{"version", "sport", "label", "knobs", "source", "n", "needed", "progress", "dll", "lo",
                      "hi", "since"}],
     "ai": {"mode", "provider", "model", "reason", "note", "proposals", "latest_proposals", "cost", "rejected_texts",
            "pending", "last", "memo"},
     "health": {"graded", "voided", "backlog", "backfill", "history", "runs", "trials"},
     "capture": {"enabled", "stored", "as_of", "last_stored",
                 "coverage": {"days", "within_hours", "target", "sports": [{"sport", "label", "finals", "covered",
                              "missed", "pending", "share", "unread"}], "total": {...},
                              "unread": [{"league", "day", "tries", "final"}]},
                 "size": {"raw_bytes", "disk_total", "disk_free", "disk_full", "floor_bytes", "under_floor",
                          "full_until", "full_minutes", "paused"} | None,
                 "requests": {"day", "capture", "prices", "per_day", "backfill", "backfill_night", "per_night",
                              "night", "night_open"},
                 "http": {"since", "requests", "errors", "share", "counted", "days": {"today", "yesterday"}},
                 "backfill_http": {"since", "requests", "errors", "share"} | None,
                 "backfill": [{"league", "seasons": [{"season", "done", "games_seen", "games_stored", "updated_at"}],
                               "done", "begun", "planned", "games_stored", "updated_at"}],
                 "errors": [{"at", "league", "game_key", "what", "error"}],
                 "shapes": [{"source", "endpoint", "label", "ok", "missing", "json", "at", "league"}],
                 "alarms": [{"kind": "http" | "backfill" | "boards" | "shape" | "disk", "text"}]}}

``brain_lines(store)`` gives the bot's /brain text (a headline per sport and the last change); ``tool_facts(store)``
the Ask AI tool's answer to "is the brain getting better?".

The "capture" block (v4 P1) reports the raw capture's health from raw.db (capture/raw_store.py). raw.db is read with
plain SQL on a read-only connection, and one is never created. The tables and columns are the store's stable
contract (plan section 3.3, plus ``capture_budget`` and ``backfill_progress``). See ``capture_health``.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import shutil
import sqlite3
import threading
import time
import urllib.parse
import zlib
from contextlib import closing
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..data.espn_client import ESPN_DAY_TZ, espn_path
from . import brain_store as bs
from . import evaluate as ev
from . import registry, stats, tune
from .analyst import finite
from .loop import (SPORT_LABELS, Learner, _connect, headline_text, iso, knob_label, period, sport_headline,
                   sport_label, template_change)

CHANGE_DECISIONS = ("promoted", "rolled_back", "undone", "safe_mode", "cleared", "shadow")
TARGET_LABELS = {"result": ("Who wins", "the sharp market"), "total": ("Totals (over/under)", "the books' own chance"),
                 "goals": ("Exact score", "climatology"), "btts": ("Both teams to score", "the base rate"),
                 "corners": ("Corners", "climatology"), "cards": ("Cards", "climatology")}
CHANGES_SHOWN = 40
SERIES_WEEKS = 52
_cache_lock = threading.Lock()
_headline_cache: Dict[Any, Tuple[Any, Dict[str, Any]]] = {}


def calibration_words(row: Mapping[str, Any]) -> str:
    """The calibration slope in words, from its 95% range (code, not the AI)."""
    slope, lo, hi = row.get("slope"), row.get("slope_lo"), row.get("slope_hi")
    if slope is None or lo is None or hi is None:
        return "Not enough graded games yet to judge calibration."
    shown = f"slope {slope:.2f}, 95% range {lo:.2f} to {hi:.2f}"
    if lo <= 1 <= hi:
        return f"Calibrated within the noise: when it says 60%, it happens about 60% of the time ({shown})."
    if hi < 1:
        return f"A little overconfident: its chances lean too far from 50% ({shown})."
    return f"A little timid: its chances could lean further from 50% ({shown})."


def _headlines(db: sqlite3.Connection, store: Any) -> Dict[str, Dict[str, Any]]:
    """The nightly run's headlines; before the first night, worked out from the graded games (kept until a game is
    graded)."""
    stored = bs.get_state(db, "headlines")
    if isinstance(stored, Mapping) and stored.get("sports"):
        return dict(stored["sports"])
    mark = None
    try:
        mark = store.consensus_graded_mark() if hasattr(store, "consensus_graded_mark") else None
    except Exception:  # noqa: BLE001
        mark = None
    key = getattr(store, "path", None) or id(store)
    with _cache_lock:
        cached = _headline_cache.get(key)
        if cached is not None and mark is not None and cached[0] == mark:
            return cached[1]
    rows = ev.consensus_records(db)
    out: Dict[str, Dict[str, Any]] = {}
    for sport in registry.SPORTS:
        sport_rows = [r for r in rows if r["sport"] == sport]
        if sport_rows:
            out[sport] = sport_headline(sport_rows)
            out[sport]["text"] = headline_text(sport, out[sport])
    with _cache_lock:
        _headline_cache[key] = (mark, out)
    return out


def _version(db: sqlite3.Connection, sport: str) -> Dict[str, Any]:
    row = None if bs.safe_mode(db) else bs.champion(db, sport)
    if row is None:
        return {"number": 0, "since": None, "source": "default", "changed": [], "labels": []}
    values = {k: v for k, v in (row.get("params") or {}).items() if k in registry.KNOBS
              and registry.KNOBS[k].applies(sport)}
    try:
        changed = registry.changed(sport, registry.defaults(sport), registry.check(sport, values))
    except ValueError:
        changed = []
    return {"number": row["version"], "since": iso(row.get("effective_from")), "source": row.get("source"),
            "changed": changed, "labels": [knob_label(k) for k in changed]}


def _targets(db: sqlite3.Connection, store: Any, sport: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in bs.scores(db, sport=sport):
        latest[row["target"]] = row
    targets, adjustments = [], None
    for target, row in latest.items():
        label, vs = TARGET_LABELS.get(target, (target, "a baseline"))
        details = row.get("details") or {}
        entry = {"target": target, "label": label, "week": row["week"], "n": row["n"],
                 **{k: finite(row.get(k), 6) for k in ("brain_ll", "market_ll", "baseline_ll", "gap", "gap_lo",
                                                       "gap_hi")},
                 "vs": vs, "shown": True, "status": None, "pit": details.get("pit")}
        if target in registry.COUNT_TARGETS:
            shown = tune.counts_shown(db, target)
            entry["shown"] = shown
            entry["status"] = "model only, low confidence" if shown else \
                "record only: not shown until it beats climatology through the gate"
        targets.append(entry)
        if target == "total" and details.get("adjusted"):
            adjustments = {"week": row["week"], "games": row["n"], "adjusted": details.get("adjusted"),
                           "dll": finite(row.get("gap"), 6), "lo": finite(row.get("gap_lo"), 6),
                           "hi": finite(row.get("gap_hi"), 6), "brier": details.get("brier_gap"),
                           "slope": details.get("slope"), "causes": details.get("causes")}
    order = list(TARGET_LABELS)
    targets.sort(key=lambda t: order.index(t["target"]) if t["target"] in order else 99)
    return targets, adjustments


def _sport_block(db: sqlite3.Connection, store: Any, sport: str, headline: Optional[Mapping[str, Any]],
                 trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = bs.scores(db, sport=sport, target="result")[-SERIES_WEEKS:]
    series = [{"week": r["week"], "n": r["n"], "gap": finite(r.get("gap"), 6), "lo": finite(r.get("gap_lo"), 6),
               "hi": finite(r.get("gap_hi"), 6), "n_market": r.get("n_market"),
               "base_gap": finite((r["brain_ll"] - r["baseline_ll"]) if r.get("brain_ll") is not None
                                  and r.get("baseline_ll") is not None else None, 6)} for r in results]
    markers = [{"week": stats.week_of(t["created_at"]), "at": iso(t["created_at"]), "kind": t["decision"],
                "version": t.get("version"), "knob": t["knob"], "label": knob_label(t["knob"])}
               for t in reversed(trials) if t["sport"] == sport and t["decision"] in ("promoted", "rolled_back",
                                                                                     "undone")]
    calibration = None
    if results:
        last = results[-1]
        reliability = (last.get("details") or {}).get("reliability")
        calibration = {"n": (last.get("details") or {}).get("calibration_n"),
                       **{k: finite(last.get(k)) for k in ("slope", "slope_lo", "slope_hi", "intercept",
                                                          "intercept_lo", "intercept_hi")},
                       "words": calibration_words(last),
                       "curve": {k: [finite(v) for v in reliability.get(k) or []] for k in ("x", "y", "lo", "hi")}
                       if reliability else None}
    targets, adjustments = _targets(db, store, sport)
    shown = None
    if headline:
        shown = dict(headline)
        for key in ("since", "until"):
            if isinstance(shown.get(key), (int, float)):
                shown[key] = iso(shown[key])
    return {"sport": sport, "label": sport_label(sport), "version": _version(db, sport),
            "headline": shown, "series": series, "markers": markers,
            "calibration": calibration, "targets": targets, "adjustments": adjustments}


def _changes(db: sqlite3.Connection, trials: List[Dict[str, Any]], owner: bool) -> List[Dict[str, Any]]:
    changelog = bs.get_state(db, "changelog") or {}
    tried: Dict[Tuple[str, str], int] = {}
    for t in gate_rows(trials):  # candidates the weekly gate tested (not rollbacks, undos, safe mode or AI freezes)
        tried[(t["week"], t["sport"])] = tried.get((t["week"], t["sport"]), 0) + 1
    later: Dict[int, Dict[str, Any]] = {}
    for t in trials:  # newest first: a version's rollback or undo (a rollback can take back a run's several changes)
        if t["decision"] in ("rolled_back", "undone") and t.get("version") is not None:
            for version in [t["version"], *((t.get("details") or {}).get("versions") or [])]:
                later.setdefault(int(version), t)
    statuses = {v["version"]: v["status"] for v in bs.versions(db)}
    out = []
    for t in trials:
        if t["decision"] not in CHANGE_DECISIONS:
            continue
        fact = {**t, "dll": finite(t.get("dll")), "lo": finite(t.get("lo")), "hi": finite(t.get("hi"))}
        entry = changelog.get(str(t["id"])) or {}
        text = entry.get("text") or template_change(fact)
        version = t.get("version")
        back = later.get(int(version)) if version is not None and t["decision"] in ("promoted", "shadow") else None
        status = statuses.get(version)
        source = str(t.get("source") or "")
        out.append({
            "id": t["id"], "version": version, "sport": t["sport"], "sport_label": sport_label(t["sport"]),
            "knob": t["knob"], "label": knob_label(t["knob"]), "old": t.get("old"), "new": t.get("new"),
            "at": iso(t["created_at"]), "week": t["week"], "decision": t["decision"], "gate": t.get("gate"),
            "source": source, "provider": source.split(":", 1)[1] if source.startswith("ai:") else None,
            "evidence": {"n": t.get("eval_n"), "train_n": t.get("train_n"), "dll": finite(t.get("dll")),
                         "lo": finite(t.get("lo")), "hi": finite(t.get("hi")), "p": finite(t.get("p")),
                         "holm_p": finite(t.get("holm_p")), "tried": tried.get((t["week"], t["sport"]), 0)},
            "reason": t.get("reason"), "text": text, "text_source": entry.get("source") or "template",
            "text_model": entry.get("model"),
            "rolled_back": ({"at": iso(back["created_at"]), "reason": back.get("reason"), "decision": back["decision"]}
                            if back else None),
            "undo": bool(owner and version is not None and status in ("champion", "challenger")
                         and t["decision"] in ("promoted", "shadow"))})
        if len(out) >= CHANGES_SHOWN:
            break
    return out


GATE_DECISIONS = ("promoted", "shadow", "held", "rejected")  # what the weekly gate decides (not undo or rollback)


def is_gate_row(t: Mapping[str, Any]) -> bool:
    """A candidate the weekly gate judged (tune._record writes its family): not a rollback, an Undo, safe mode, nor
    an AI proposal frozen (or refused by the code checks) when it was made."""
    return bool((t.get("details") or {}).get("family"))


def gate_rows(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The gate's rows, one per candidate (a challenger's ruling once per version and week, the newest)."""
    out, seen = [], set()
    for t in trials:  # newest first
        if not is_gate_row(t):
            continue
        key = (t["week"], t["sport"], t["version"]) if t.get("version") is not None else ("id", t["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def is_freeze_row(t: Mapping[str, Any]) -> bool:
    """An AI proposal frozen as a challenger: queued to be judged on later games, not a gate decision."""
    return (t["decision"] == "shadow" and str(t.get("source") or "").startswith("ai:") and not is_gate_row(t))


def _held(db: sqlite3.Connection, now: float) -> Dict[str, Any]:
    week = tune.decision_week(now)
    rows = gate_rows(bs.trials(db, week=week, limit=500))
    if not rows:  # no weekly run for that week yet: the latest week the gate decided on
        latest = next((t for t in bs.trials(db, limit=500) if is_gate_row(t)), None)
        if latest:
            week = latest["week"]
            rows = gate_rows(bs.trials(db, week=week, limit=500))
    frozen = sum(1 for t in bs.trials(db, week=week, limit=500) if is_freeze_row(t)) if rows else 0
    # Passed: promoted, or a structural test passed into shadow with its evidence (never an AI freeze, which has none)
    passed = sum(1 for t in rows if t["decision"] == "promoted"
                 or (t["decision"] == "shadow" and (t.get("eval_n") is not None or t.get("p") is not None)))
    items = []
    for t in rows:
        if t["decision"] not in ("held", "rejected"):
            continue
        n = t.get("eval_n")
        needed = max(tune.MIN_EVAL - int(n), 0) if isinstance(n, (int, float)) and n < tune.MIN_EVAL else None
        items.append({"id": t["id"], "sport": t["sport"], "sport_label": sport_label(t["sport"]), "knob": t["knob"],
                      "label": knob_label(t["knob"]), "decision": t["decision"], "reason": t.get("reason"),
                      "eval_n": n, "needed": needed})
    tested = len(rows)
    many = "candidate" if tested == 1 else "candidates"
    # A week run while learning was paused (or in safe mode) changed nothing whatever the gate found (tune.run_week
    # keeps that in the week's report, as the owner's Telegram summary says it)
    decided = bs.get_state(db, f"week_decided:{week}") if rows else None
    skipped = ((decided.get("report") or {}).get("skipped") if isinstance(decided, Mapping) else None)
    if not tested:
        words = "No weekly test has run yet."
    elif skipped in ("paused", "safe mode"):
        why = "learning was paused" if skipped == "paused" else "the brain was in safe mode"
        words = f"This week: {why}, so nothing changed ({tested} {many} scored)."
    elif passed:
        words = f"This week: {tested} {many} tested, {passed} passed the gate."
    else:
        words = (f"This week: {tested} {many} tested, none passed. Most weeks end this way: the brain stays as it "
                 "is.")
    if frozen:
        words += (f" {frozen} AI proposal{'s' if frozen != 1 else ''} queued to be judged on later games.")
    return {"week": week if rows else None, "tested": tested, "passed": passed, "frozen": frozen, "words": words,
            "items": items}


def _experiments(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    out = []
    for v in bs.versions(db, status="challenger"):
        shadow = bs.get_state(db, f"shadow:{v['version']}") or {}
        knobs = [k for k in str(v.get("knob") or "").split(",") if k] or list(shadow.get("knobs") or [])
        n = int(shadow.get("n") or 0)
        needed = int(shadow.get("needed") or tune.MIN_EVAL)
        out.append({"version": v["version"], "sport": v["sport"], "sport_label": sport_label(v["sport"]),
                    "label": knob_label(",".join(knobs)), "knobs": knobs, "source": v.get("source"), "n": n,
                    "needed": needed, "progress": round(min(1.0, n / needed), 3) if needed else None,
                    "dll": finite(shadow.get("dll")), "lo": finite(shadow.get("lo")), "hi": finite(shadow.get("hi")),
                    "since": iso(v.get("effective_from") or v.get("created_at"))})
    return out


def _health(db: sqlite3.Connection, store: Any, now: float) -> Dict[str, Any]:
    graded: Dict[str, int] = {}
    voided: Dict[str, Dict[str, Any]] = {}
    backlog = 0
    try:
        columns = {r[1] for r in db.execute("PRAGMA table_info(consensus_log)")}
        snap = " AND kind = 'last'" if "kind" in columns else ""
        for league, n in db.execute(f"SELECT league, COUNT(*) FROM consensus_log WHERE status = 'graded'{snap} "
                                    "GROUP BY league"):
            sport = registry.sport_key(league)
            graded[sport] = graded.get(sport, 0) + int(n)
        for league, result in db.execute(f"SELECT league, result FROM consensus_log WHERE status = 'void'{snap}"):
            sport = registry.sport_key(league)
            entry = voided.setdefault(sport, {"n": 0, "reasons": {}})
            entry["n"] += 1
            try:
                reason = (json.loads(result) or {}).get("reason") or (json.loads(result) or {}).get("status") or \
                    "called off"
            except (TypeError, ValueError):
                reason = "called off"
            entry["reasons"][str(reason)] = entry["reasons"].get(str(reason), 0) + 1
        cutoff = datetime.datetime.fromtimestamp(now - 2 * 86400, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        backlog = int(db.execute(f"SELECT COUNT(*) FROM consensus_log WHERE status = 'pending' AND start < ?{snap}",
                                 (cutoff,)).fetchone()[0])
    except sqlite3.OperationalError:
        pass
    try:
        from .history import progress as backfill_progress
        from .history_store import counts as history_counts
        backfill = backfill_progress(db)
        history = history_counts(db)
    except Exception:  # noqa: BLE001 - a store without the history tables
        backfill, history = {}, {}
    runs = {}
    for kind in ("night", "week", "month"):
        last = bs.get_state(db, f"last_{kind}") or {}
        key, at = period(kind, now)
        runs[kind] = {"last": iso(last.get("at")), "period": last.get("key"), "current": key}
    month_since = now - 31 * 86400
    month_trials = bs.trials(db, since=month_since, limit=None)
    return {"graded": graded, "voided": voided, "backlog": backlog, "backfill": backfill, "history": history,
            "runs": runs, "trials": {"month": len(gate_rows(month_trials)),
                                     "promoted": sum(1 for t in month_trials if t["decision"] == "promoted"),
                                     "rolled_back": sum(1 for t in month_trials if t["decision"] == "rolled_back")}}


def _ai(db: sqlite3.Connection, learner: Optional[Learner], now: float) -> Dict[str, Any]:
    from .loop import ai_record
    if learner is not None:
        status = learner.ai_status(now)
    else:
        last = bs.get_state(db, "ai_last") or {}
        status = {"mode": None, "provider": last.get("provider"), "model": last.get("model"), "reason": None,
                  "spent_cents": None, "reserved_cents": None, "cap_cents": None, "pending": None}
    note = bs.get_state(db, "note:latest")
    memo = bs.get_state(db, "memo:latest")
    latest = bs.get_state(db, "proposals:latest")
    return {"mode": status.get("mode"), "provider": status.get("provider"), "model": status.get("model"),
            "reason": status.get("reason"),
            "note": ({"text": note.get("text"), "hypotheses": note.get("hypotheses") or [],
                      "source": note.get("source"), "model": note.get("model"), "at": iso(note.get("at")),
                      "run": note.get("run")} if isinstance(note, Mapping) else None),
            "proposals": ai_record(db), "latest_proposals": latest if isinstance(latest, Mapping) else None,
            "cost": {"spent_cents": status.get("spent_cents"), "reserved_cents": status.get("reserved_cents"),
                     "cap_cents": status.get("cap_cents")},
            "rejected_texts": int(bs.get_state(db, "ai_rejected", 0) or 0), "pending": status.get("pending"),
            "last": bs.get_state(db, "ai_last"),
            "memo": ({"text": memo.get("text"), "ideas": memo.get("ideas") or [], "source": memo.get("source"),
                      "model": memo.get("model"), "at": iso(memo.get("at")), "month": memo.get("month")}
                     if isinstance(memo, Mapping) else None)}


def learning_state(db: sqlite3.Connection, learner: Optional[Learner]) -> Dict[str, Any]:
    if learner is not None:
        paused = learner.paused()
        enabled = learner.enabled
    else:
        state = bs.get_state(db, "paused")
        paused = {"paused": True, "by": state.get("by"), "at": state.get("at")} \
            if isinstance(state, Mapping) and state.get("paused") else None
        enabled = True
    return {"enabled": enabled, "paused": bool(paused), "by": (paused or {}).get("by"),
            "at": iso((paused or {}).get("at")), "locked": not enabled}


def _scored_at(db: sqlite3.Connection) -> Optional[float]:
    """When the page's numbers were worked out: the nightly run that wrote the headlines (else the last nightly or
    weekly scoring); None before any run (the numbers are then worked out for this request)."""
    for key in ("headlines", "last_night_run", "last_week_run"):
        state = bs.get_state(db, key)
        if isinstance(state, Mapping) and isinstance(state.get("at"), (int, float)):
            return float(state["at"])
    return None


# --- the raw capture's health (v4 P1: capture/ and raw.db) --------------------------------------------------------------

log = logging.getLogger(__name__)

CAPTURE_DAYS = 7  # coverage counts the games that started in the last week ...
CAPTURE_WITHIN = 36 * 3600  # ... a final is covered when a raw copy was stored within this of the game's end
CAPTURE_TARGET = 0.98  # plan section 1.4, D1: at least 98% of finals with a raw bundle within 36 h
# A game's end, at the long end of usual, in hours after its start (overtime, extra innings, extra time and penalties,
# a five-set match, a whole card): a game is a final to count from then
GAME_HOURS = {"basketball": 3.5, "football": 4.5, "hockey": 3.5, "baseball": 5.0, "soccer": 3.0, "mma": 7.0,
              "tennis": 6.0}
OTHER_GAME_HOURS = 5.0
CAPTURE_SPORTS = (("basketball", "Basketball"), ("football", "American football"), ("hockey", "Ice hockey"),
                  ("baseball", "Baseball"), ("soccer", "Soccer"), ("mma", "MMA"), ("tennis", "Tennis"),
                  ("other", "Other"))
DISK_ALARM = 0.70  # plan section 9.4: the alarm at 70% of the disk. Under capture/raw_store.py's floor (disk_room:
# max(CAPTURE_MIN_FREE_GB GiB, DISK_FLOOR_SHARE of the disk) free) the alarm says capture and the backfill are paused
HTTP_ALARM = 0.25  # at least this share of forward capture's requests failing (today and yesterday together, UTC, or
HTTP_ALARM_MIN = 50  # either day alone, so a fresh outage isn't diluted) ... out of at least this many
BACKFILL_HTTP_ALARM = 0.25  # the older-games backfill's own share (today and yesterday, UTC) ...
BACKFILL_HTTP_ALARM_MIN = 200  # ... out of at least this many of its requests
# capture_budget's request counters (by UTC day): forward capture (the slate's jobs, the sweep, the static files and
# prices) and the older-games backfill apart, so a night of backfill can't hide a forward outage
FORWARD_KINDS = ("http_ok", "http_error")
BACKFILL_KINDS = ("backfill_http_ok", "backfill_http_error")
COUNTER_KINDS = ("capture", "prices") + FORWARD_KINDS + BACKFILL_KINDS
RECENT_ROWS = 20000  # the newest raw_payloads rows the shape check, the stored errors and their counts look through
ERRORS_SHOWN = 5
ERRORS_DAYS = 7  # the latest errors come from this many days (capture_jobs' updated_at index for jobs with an error)
ERROR_CHARS = 200
SWEEP_TRIES = 3  # capture/worker.py SWEEP_TRIES (read from there when it loads): a board unread after them is given up
# The newest 2xx answer of each JSON kind must carry these top-level keys (state None: any state). Every payload of the
# audit has them (tests/fixtures/capture); one gone missing, or an answer that isn't a JSON object, is a shape change
SHAPES: Tuple[Tuple[str, str, Optional[str], Tuple[str, ...], str], ...] = (
    ("espn_site", "summary", "final", ("header", "boxscore"), "ESPN game summaries"),
    ("espn_site", "summary", "pre", ("header",), "ESPN pre-game summaries"),
    ("espn_site", "scoreboard", None, ("events", "leagues"), "ESPN scoreboards"),
    ("espn_core", "core_plays", None, ("items", "pageCount"), "ESPN play-by-play pages"),
    ("espn_core", "core_status", None, ("type",), "ESPN bout status"),
    ("espn_core", "core_competitor_stats", None, ("splits",), "ESPN fighter statistics"),
    ("espn_fightcenter", "fightcenter", None, (), "ESPN fight centre"),
    ("mlb_statsapi", "feed_live", None, ("gameData", "liveData"), "MLB StatsAPI live feeds"),
    ("espn_propbets", "propbets", None, ("items", "pageCount"), "ESPN DraftKings props"),
    ("kalshi", "events", None, ("events",), "Kalshi series"),
)
JOB_WORDS = {"todo": "capture job, retrying", "partial": "capture job, partly stored", "done": "capture job",
             "gave_up": "capture job, given up", "void": "capture job, called off"}
# The backfill's league order (the spec's waves: wave 1, then wave 2); any other league after them, by key
BACKFILL_ORDER = ("basketball_nba", "americanfootball_nfl", "soccer_epl", "soccer_efl_champ", "soccer_spain_la_liga",
                  "soccer_germany_bundesliga", "soccer_italy_serie_a", "soccer_france_ligue_one", "soccer_spl",
                  "soccer_netherlands_eredivisie", "soccer_portugal_primeira_liga", "soccer_uefa_champs_league",
                  "icehockey_nhl", "mma_mixed_martial_arts", "baseball_mlb")
SHAPE_CACHE_SIZE = 1024  # one check per (kind, league) kept: about 150 on the 58 paths
_shape_lock = threading.Lock()
_shape_cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}  # (raw.db, row id, sha256) -> its check


def capture_sport(league: Any) -> str:
    """The sport of a capture job's league, for the coverage rows: ESPN's sport for the league's path (basketball,
    football, hockey, baseball, soccer, tennis, mma); "other" when it has none."""
    key = str(league or "")
    path = espn_path(key)
    if path:
        return path[0] if path[0] in GAME_HOURS else "other"
    for prefix, sport in (("tennis_", "tennis"), ("mma_", "mma")):
        if key.startswith(prefix):
            return sport
    return "other"


def _setting(learner: Any, env: Mapping[str, str], name: str, default: Any) -> Any:
    """A capture setting from the learner's config (config.py), else the environment, else ``default``."""
    value = getattr(getattr(learner, "config", None), name, None)
    if value is not None and value != "":
        return value
    text = str(env.get(name.upper()) or "").strip()
    if not text:
        return default
    if isinstance(default, bool):
        return text.lower() not in ("0", "false", "no", "off")
    if isinstance(default, (int, float)):
        try:
            return type(default)(float(text)) if isinstance(default, float) else int(text)
        except ValueError:
            return default
    return text


def raw_db_path(store: Any, learner: Any = None) -> Optional[str]:
    """Where raw.db is: the capture's own rule (capture/raw_store.py resolve_raw_db_path), so this reads the file the
    worker writes: RAW_DB_PATH (the learner's config, else the environment), else beside betting.db. None for a
    store with no file."""
    from ..capture.raw_store import resolve_raw_db_path  # here: the page never fails over the capture package
    path = getattr(store, "path", None)
    if not isinstance(path, str) or not path or path == ":memory:":
        path = None
    config = getattr(learner, "config", None)
    if path is None and not (getattr(config, "raw_db_path", None) or os.environ.get("RAW_DB_PATH")):
        return None
    return resolve_raw_db_path(config, path)


def _connect_ro(path: str) -> sqlite3.Connection:
    """raw.db, read-only; a missing file raises sqlite3.OperationalError and is never created."""
    db = sqlite3.connect("file:" + urllib.parse.quote(os.path.abspath(path)) + "?mode=ro", uri=True, timeout=5)
    db.execute("PRAGMA query_only = ON")
    return db


def _utc_day(ts: float) -> str:
    return datetime.datetime.fromtimestamp(float(ts), datetime.timezone.utc).date().isoformat()


def _pct(share: float) -> str:
    return f"{round(100 * share)}%"


def _bytes_text(n: Optional[float]) -> str:
    """A size as df -h shows it (1024-based): "1.4 GB", "2.0 GB" for 2 GiB."""
    if n is None:
        return "unknown"
    for unit, size in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{int(n)} B"


def _floor_rule() -> Optional[Tuple[Any, float, Any]]:
    """capture/raw_store.py's disk floor, as the worker applies it: (disk_room, DISK_FLOOR_SHARE, min_free_bytes);
    None when the capture package can't load (the page never fails over it)."""
    try:
        from ..capture.raw_store import DISK_FLOOR_SHARE, disk_room, min_free_bytes
    except Exception:  # noqa: BLE001
        return None
    return disk_room, float(DISK_FLOOR_SHARE), min_free_bytes


def backfill_tz_of(config: Any) -> Optional[datetime.tzinfo]:
    """The older-games backfill's time zone, by its own rule (capture/backfill.py backfill_tz: the config's TIMEZONE,
    New York without one; server-local time, None, for a name that isn't a zone), as the worker builds it."""
    from ..capture.backfill import backfill_tz  # here, not at the top: capture.backfill imports learn.history
    return backfill_tz(config)


def backfill_night(now: float, tz: Optional[datetime.tzinfo]) -> Tuple[str, bool]:
    """(night, open): the backfill's budget day at ``now`` by RawBackfill's own rule (capture/backfill.py: ``night``
    is the local date in ``tz``, the day before for the small hours of a window that crosses midnight; ``in_window``
    whether ``now`` is inside the night window, WINDOW_HOURS)."""
    from ..capture.backfill import WINDOW_HOURS
    start, end = int(WINDOW_HOURS[0]), int(WINDOW_HOURS[1])
    moment = datetime.datetime.fromtimestamp(now, tz) if tz is not None else datetime.datetime.fromtimestamp(now)
    inside = start <= moment.hour < end if start <= end else (moment.hour >= start or moment.hour < end)
    if start > end and moment.hour < end:
        moment -= datetime.timedelta(days=1)
    return moment.date().isoformat(), inside


def _espn_day(ts: float) -> datetime.date:
    """ESPN's day for a moment (US Eastern, as the sweep keys its boards)."""
    return datetime.datetime.fromtimestamp(ts, ESPN_DAY_TZ).date()


def _sweep_tries() -> int:
    try:
        from ..capture.worker import SWEEP_TRIES as tries
        return int(tries)
    except Exception:  # noqa: BLE001
        return SWEEP_TRIES


def _cut(text: Any, limit: int = ERROR_CHARS) -> str:
    words = " ".join(str(text or "").split())
    return words if len(words) <= limit else words[:limit - 1].rstrip() + "…"


def _empty_coverage() -> Dict[str, Any]:
    return {"days": CAPTURE_DAYS, "within_hours": CAPTURE_WITHIN // 3600, "target": CAPTURE_TARGET, "sports": [],
            "total": {"finals": 0, "covered": 0, "missed": 0, "pending": 0, "share": None}, "unread": []}


def _covered_share(covered: int, judged: int) -> Optional[float]:
    """covered / judged to 4 decimals, rounded down: 24,999 of 25,000 is 0.9999, never a whole 1."""
    return math.floor(covered * 10000 / judged) / 10000 if judged else None


def _skeleton(now: float, enabled: Optional[bool] = True, settings: Optional[Mapping[str, Any]] = None
              ) -> Dict[str, Any]:
    """The capture block with nothing read yet."""
    settings = settings or {}
    return {"enabled": enabled, "stored": False, "as_of": iso(now), "last_stored": None, "coverage": _empty_coverage(),
            "size": None, "requests": {"day": _utc_day(now), "capture": 0, "prices": 0,
                                       "per_day": settings.get("per_day"), "backfill_night": None, "backfill": 0,
                                       "per_night": settings.get("per_night"), "night": None, "night_open": False},
            "http": None, "backfill_http": None, "backfill": [], "errors": [], "shapes": [], "alarms": []}


def _has_final_kind(kinds_done: Any) -> bool:
    try:
        kinds = json.loads(kinds_done) if isinstance(kinds_done, str) else list(kinds_done or [])
    except (TypeError, ValueError):
        return False
    return any(isinstance(kind, str) and kind.startswith("final:") for kind in kinds)


def _coverage(db: sqlite3.Connection, now: float, unread: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Finals of the last CAPTURE_DAYS days per sport: covered (a raw final stored within CAPTURE_WITHIN of the end),
    missed (stored later, or not by the deadline, or given up), pending (still inside the 36 h). Void jobs (called off)
    aren't finals. A job whose final lives under another key (a tennis day board, a card's fight centre) counts as
    stored when its job says so, at the job's last update.

    The finals are the jobs the slate and the nightly sweep made, so a board the sweep couldn't read (``unread``:
    _unread_boards) leaves its games out unseen: the sport's share and the total's are then unknown (None), each
    row saying how many of its boards are unread. Reads capture_jobs by its start index."""
    since = now - CAPTURE_DAYS * 86400
    rows = db.execute(
        "SELECT j.league, j.start, j.state, j.kinds_done, j.updated_at, "
        "(SELECT MIN(p.fetched_at) FROM raw_payloads p WHERE p.game_key = j.game_key AND p.state = 'final' "
        "AND p.http_status BETWEEN 200 AND 299) "
        "FROM capture_jobs j WHERE j.start >= ? AND j.start < ? AND j.state != 'void' "
        "AND substr(j.game_key, 1, 7) != 'static:'", (since, now)).fetchall()
    counts: Dict[str, Dict[str, int]] = {}
    for league, start, state, kinds_done, updated_at, first in rows:
        sport = capture_sport(league)
        end = float(start) + GAME_HOURS.get(sport, OTHER_GAME_HOURS) * 3600
        if end > now:
            continue  # still being played: not a final yet
        deadline = end + CAPTURE_WITHIN
        stored = float(first) if first is not None else None
        if stored is None and updated_at is not None and _has_final_kind(kinds_done):
            stored = float(updated_at)
        entry = counts.setdefault(sport, {"finals": 0, "covered": 0, "missed": 0, "pending": 0})
        entry["finals"] += 1
        if stored is not None and stored <= deadline:
            entry["covered"] += 1
        elif stored is not None or now > deadline or state == "gave_up":
            entry["missed"] += 1
        else:
            entry["pending"] += 1
    unread = list(unread or [])
    blind: Dict[str, int] = {}
    for board in unread:
        sport = capture_sport(board["league"])
        blind[sport] = blind.get(sport, 0) + 1
    out = _empty_coverage()
    out["unread"] = unread
    total = out["total"]
    for sport, label in CAPTURE_SPORTS:
        entry = counts.get(sport)
        if not entry and not blind.get(sport):
            continue
        entry = entry or {"finals": 0, "covered": 0, "missed": 0, "pending": 0}
        share = None if blind.get(sport) else _covered_share(entry["covered"], entry["covered"] + entry["missed"])
        out["sports"].append({"sport": sport, "label": label, **entry, "share": share, "unread": blind.get(sport, 0)})
        for key in ("finals", "covered", "missed", "pending"):
            total[key] += entry[key]
    total["share"] = None if unread else _covered_share(total["covered"], total["covered"] + total["missed"])
    return out


def _unread_boards(db: sqlite3.Connection, now: float) -> List[Dict[str, Any]]:
    """The boards the nightly sweep couldn't read (capture/worker.py keeps each ESPN day's sweep in capture_state as
    "sweep:<day>": {"leagues", "made", "tries", "failed", "next"}), for the days the coverage window spans:
    [{"league", "day", "tries", "final"}]. ``final``: the sweep gave up on it (SWEEP_TRIES tries, or a later day's
    sweep has begun); otherwise it tries again."""
    first, last = _espn_day(now - CAPTURE_DAYS * 86400), _espn_day(now)
    days = [first + datetime.timedelta(days=n) for n in range((last - first).days + 1)]
    keys = [f"sweep:{day.isoformat()}" for day in days] + ["sweep:last"]
    found = {str(key): value for key, value in db.execute(
        f"SELECT key, value FROM capture_state WHERE key IN ({', '.join('?' for _ in keys)})", keys)}

    def parsed(value: Any) -> Any:
        try:
            return json.loads(value) if isinstance(value, (str, bytes)) else None
        except ValueError:
            return None

    swept = parsed(found.get("sweep:last"))
    yesterday = last - datetime.timedelta(days=1)
    tries_most = _sweep_tries()
    out = []
    for day in days:
        state = parsed(found.get(f"sweep:{day.isoformat()}"))
        if not isinstance(state, dict) or not isinstance(state.get("failed"), list):
            continue
        try:
            tries = int(state.get("tries") or 0)
        except (TypeError, ValueError):
            tries = 0
        final = tries >= tries_most or day < yesterday or swept == day.isoformat()
        for league in sorted({str(item) for item in state["failed"] if isinstance(item, str) and item}):
            out.append({"league": league, "day": day.isoformat(), "tries": tries, "final": bool(final)})
    return out


def _full_pause(path: str, now: Optional[float]) -> Tuple[Optional[float], int]:
    """(until, minutes): when capture's pause after SQLite's "database or disk is full" ends, if one is on in this
    process (capture/raw_store.py full_pause: the worker's RawStore stops for FULL_PAUSE after such an error, though the
    disk may show free blocks: a full inode table, a quota), and how long each pause is."""
    from ..capture.raw_store import FULL_PAUSE, full_pause
    return full_pause(path, now), max(1, int(float(FULL_PAUSE) // 60))


def _disk(path: str, min_free_gb: Any = None, now: Optional[float] = None) -> Dict[str, Any]:
    """raw.db's size on disk (its write-ahead log included), its disk's size and free space, and the worker's own
    rules for pausing capture and the backfill (capture/raw_store.py): under the floor (``under_floor``: not
    disk_room, the free space it needs being ``floor_bytes``: max(min_free_bytes(CAPTURE_MIN_FREE_GB), DISK_FLOOR_SHARE
    of the disk)), or after SQLite said the disk is full (``full_until``: full_pause). ``paused``: either."""
    raw = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            raw += os.path.getsize(path + suffix)
        except OSError:
            pass
    until, minutes = _full_pause(path, now)
    try:
        usage = shutil.disk_usage(os.path.dirname(os.path.abspath(path)))
    except OSError:
        return {"raw_bytes": raw, "disk_total": None, "disk_free": None, "disk_full": None, "floor_bytes": None,
                "under_floor": False, "full_until": iso(until), "full_minutes": minutes, "paused": bool(until)}
    full = round((usage.total - usage.free) / usage.total, 4) if usage.total else None
    floor, short = None, False
    rule = _floor_rule()
    if rule is not None:
        disk_room, share, min_free_bytes = rule
        need = min_free_bytes(min_free_gb)
        floor = int(max(float(need), share * float(usage.total)))
        short = not disk_room(path, need)
    return {"raw_bytes": raw, "disk_total": int(usage.total), "disk_free": int(usage.free), "disk_full": full,
            "floor_bytes": floor, "under_floor": bool(short), "full_until": iso(until), "full_minutes": minutes,
            "paused": bool(short or until)}


def _requests(db: sqlite3.Connection, now: float, settings: Mapping[str, Any]
              ) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], int]]:
    """Today's capture and price requests (UTC) and the backfill's latest night, against their caps; and the counters
    of today and yesterday by (kind, day), for the HTTP shares.

    The backfill counts by its night (capture/backfill.py: the window's start date in its time zone, backfill_tz).
    ``night`` is the night that began last (``night_open``: still running), in that zone (``settings["tz"]``, else
    backfill_tz with no config); ``backfill_night`` the latest night at or before it the backfill spent in, which is
    ``night`` itself when it ran last night, an earlier date when it didn't."""
    today, yesterday = _utc_day(now), _utc_day(now - 86400)
    used = {(str(kind), str(day)): int(n or 0) for kind, day, n in db.execute(
        f"SELECT kind, day, used FROM capture_budget WHERE kind IN ({', '.join('?' for _ in COUNTER_KINDS)}) "
        "AND day >= ?", (*COUNTER_KINDS, yesterday))}
    tz = settings["tz"] if "tz" in settings else backfill_tz_of(None)
    night, night_open = backfill_night(now, tz)
    latest = db.execute("SELECT day, used FROM capture_budget WHERE kind = 'backfill' AND day <= ? "
                        "ORDER BY day DESC LIMIT 1", (night,)).fetchone()
    return ({"day": today, "capture": used.get(("capture", today), 0), "prices": used.get(("prices", today), 0),
             "per_day": settings.get("per_day"), "backfill_night": str(latest[0]) if latest else None,
             "backfill": int(latest[1] or 0) if latest else 0, "per_night": settings.get("per_night"),
             "night": night, "night_open": night_open}, used)


def _share_of(ok: int, bad: int) -> Dict[str, Any]:
    total = ok + bad
    return {"requests": total, "errors": bad, "share": round(bad / total, 4) if total else None}


def _since_yesterday(now: float) -> Optional[str]:
    """Midnight UTC starting yesterday: where the counters' two days begin."""
    return iso(datetime.datetime.combine(datetime.date.fromisoformat(_utc_day(now - 86400)), datetime.time(),
                                         datetime.timezone.utc).timestamp())


def _http(db: sqlite3.Connection, now: float, used: Mapping[Tuple[str, str], int], floor: int) -> Dict[str, Any]:
    """The share of forward capture's requests that failed today and yesterday (UTC), from its request counters
    (http_ok and http_error: the slate's jobs, the sweep, the static files and prices, never the backfill), with each
    day's own share (``days``: a fresh outage shows in today's before the two days' share catches up). Without any
    request counter (a worker that doesn't count yet), the non-2xx answers stored in the last 24 hours (an identical
    answer is one row)."""
    yesterday, today = _utc_day(now - 86400), _utc_day(now)
    if any(kind in FORWARD_KINDS + BACKFILL_KINDS for kind, _day in used):
        days = {name: _share_of(used.get(("http_ok", day), 0), used.get(("http_error", day), 0))
                for name, day in (("yesterday", yesterday), ("today", today))}
        ok = sum(d["requests"] - d["errors"] for d in days.values())
        bad = sum(d["errors"] for d in days.values())
        return {"since": _since_yesterday(now), **_share_of(ok, bad), "counted": "requests", "days": days}
    total = bad = 0
    for klass, n in db.execute("SELECT http_status / 100, COUNT(*) FROM raw_payloads WHERE id > ? AND fetched_at >= ? "
                               "GROUP BY http_status / 100", (floor, now - 86400)):
        total += int(n)
        bad += int(n) if int(klass) != 2 else 0
    return {"since": iso(now - 86400), "requests": total, "errors": bad,
            "share": round(bad / total, 4) if total else None, "counted": "stored answers", "days": None}


def _backfill_http(now: float, used: Mapping[Tuple[str, str], int]) -> Optional[Dict[str, Any]]:
    """The older-games backfill's own failed share today and yesterday (UTC; backfill_http_ok and
    backfill_http_error); None when it made no request in those days."""
    days = (_utc_day(now - 86400), _utc_day(now))
    ok = sum(used.get(("backfill_http_ok", day), 0) for day in days)
    bad = sum(used.get(("backfill_http_error", day), 0) for day in days)
    if not ok + bad:
        return None
    return {"since": _since_yesterday(now), **_share_of(ok, bad)}


def _errors(db: sqlite3.Connection, floor: int, now: float) -> List[Dict[str, Any]]:
    """The newest problems of the last ERRORS_DAYS days, newest first: jobs' last errors (by capture_jobs' index on
    updated_at for jobs with an error) and non-2xx answers stored (among the newest RECENT_ROWS rows)."""
    labels: Dict[Tuple[str, str], str] = {}
    for source, endpoint, _state, _keys, label in SHAPES:
        labels.setdefault((source, endpoint), label)  # a summary's error is a game summary's, whatever its state
    found = []
    for at, key, league, state, error in db.execute(
            "SELECT updated_at, game_key, league, state, last_error FROM capture_jobs WHERE last_error IS NOT NULL "
            "AND updated_at >= ? AND last_error != '' ORDER BY updated_at DESC LIMIT ?",
            (now - ERRORS_DAYS * 86400, ERRORS_SHOWN)):
        found.append((float(at or 0), {"league": league, "game_key": key,
                                       "what": JOB_WORDS.get(str(state), "capture job"), "error": _cut(error)}))
    for at, key, league, source, endpoint, status in db.execute(
            "SELECT fetched_at, game_key, league, source, endpoint, http_status FROM raw_payloads WHERE id > ? AND "
            "http_status NOT BETWEEN 200 AND 299 ORDER BY id DESC LIMIT ?", (floor, ERRORS_SHOWN)):
        what = labels.get((source, endpoint)) or f"{source} {endpoint}"
        found.append((float(at), {"league": league, "game_key": key, "what": what, "error": f"HTTP {int(status)}"}))
    found.sort(key=lambda item: -item[0])
    return [{"at": iso(at), **entry} for at, entry in found[:ERRORS_SHOWN]]


def _decode_body(blob: Any, codec: str) -> bytes:
    if codec == "zlib6":
        return zlib.decompress(blob)
    if codec == "identity":
        return bytes(blob)
    raise ValueError(f"unknown codec {codec!r}")


def _shape_of(db: sqlite3.Connection, path: str, row_id: int, keys: Tuple[str, ...]) -> Dict[str, Any]:
    """One stored answer's check: a JSON object with ``keys`` at its top. Kept per row and body (a stored row never
    changes; the sha256 keeps a raw.db made again, its ids from 1, from reading an old row's check)."""
    league, at, digest = db.execute("SELECT league, fetched_at, sha256 FROM raw_payloads WHERE id = ?",
                                    (int(row_id),)).fetchone()
    cache_key = (os.path.abspath(path), int(row_id), str(digest))
    with _shape_lock:
        hit = _shape_cache.get(cache_key)
    if hit is not None and hit.get("keys") == list(keys):
        return {k: v for k, v in hit.items() if k != "keys"}
    codec, blob = db.execute("SELECT codec, body FROM raw_payloads WHERE id = ?", (int(row_id),)).fetchone()
    try:
        doc = json.loads(_decode_body(blob, str(codec)).decode("utf-8-sig"))
    except (ValueError, zlib.error, UnicodeDecodeError):
        doc = None
    is_object = isinstance(doc, dict)
    missing = [key for key in keys if not is_object or key not in doc]
    result = {"ok": is_object and not missing, "json": is_object, "missing": missing, "at": iso(at), "league": league}
    with _shape_lock:
        _shape_cache[cache_key] = {**result, "keys": list(keys)}
        while len(_shape_cache) > SHAPE_CACHE_SIZE:
            _shape_cache.pop(next(iter(_shape_cache)))
    return result


def _shapes(db: sqlite3.Connection, path: str, floor: int) -> List[Dict[str, Any]]:
    """The newest 2xx answer of each JSON kind in SHAPES for each league (among the newest RECENT_ROWS rows), checked:
    one league's changed answer shows even when another league's is newer."""
    newest: Dict[Tuple[str, str, str, str], int] = {}
    for source, endpoint, state, league, row_id in db.execute(
            "SELECT source, endpoint, state, league, MAX(id) FROM raw_payloads WHERE id > ? AND http_status BETWEEN 200 "
            "AND 299 GROUP BY source, endpoint, state, league", (floor,)):
        newest[(str(source), str(endpoint), str(state), str(league))] = int(row_id)
    out = []
    for source, endpoint, state, keys, label in SHAPES:
        by_league: Dict[str, int] = {}
        for (s, e, st, league), row_id in newest.items():
            if s == source and e == endpoint and (state is None or st == state):
                by_league[league] = max(row_id, by_league.get(league, 0))
        for league in sorted(by_league):
            out.append({"source": source, "endpoint": endpoint, "label": label,
                        **_shape_of(db, path, by_league[league], keys)})
    return out


def _planned_seasons(league: str, seasons: Any, now: float) -> Optional[int]:
    """How many seasons of ``league`` the backfill has in scope now (capture/backfill.py's own rule: the last
    ``seasons`` completed ones, and the one being played if one is); seasons + 1 when that can't be worked out."""
    if not isinstance(seasons, int) or seasons < 0:
        return None
    try:
        from ..capture import backfill as raw_backfill
        return len(raw_backfill.seasons_for(league, raw_backfill.settled_day(now), seasons))
    except Exception:  # noqa: BLE001 - an unknown league, or the capture package can't load
        return seasons + 1


def _backfill_rows(db: sqlite3.Connection, settings: Mapping[str, Any], now: float) -> List[Dict[str, Any]]:
    """The older-games backfill per league (capture/backfill.py's backfill_progress), seasons newest first."""
    by_league: Dict[str, List[Dict[str, Any]]] = {}
    for league, season, done, seen, stored, updated in db.execute(
            "SELECT league, season, done, games_seen, games_stored, updated_at FROM backfill_progress"):
        by_league.setdefault(str(league), []).append({
            "season": int(season), "done": bool(done), "games_seen": int(seen or 0), "games_stored": int(stored or 0),
            "updated_at": None if updated is None else float(updated)})
    seasons = settings.get("seasons")
    order = {league: i for i, league in enumerate(BACKFILL_ORDER)}
    out = []
    for league in sorted(by_league, key=lambda key: (order.get(key, len(order)), key)):
        rows = sorted(by_league[league], key=lambda row: -row["season"])
        times = [row["updated_at"] for row in rows if row["updated_at"] is not None]
        out.append({"league": league, "seasons": [{**row, "updated_at": iso(row["updated_at"])} for row in rows],
                    "done": sum(row["done"] for row in rows), "begun": len(rows),
                    "planned": _planned_seasons(league, seasons, now),
                    "games_seen": sum(row["games_seen"] for row in rows),
                    "games_stored": sum(row["games_stored"] for row in rows),
                    "updated_at": iso(max(times)) if times else None})
    return out


def _tripped(part: Mapping[str, Any], least: int, share: float) -> bool:
    return (part.get("requests") or 0) >= least and (part.get("share") or 0) >= share


def _http_alarm(http: Mapping[str, Any]) -> Optional[str]:
    """Forward capture's HTTP alarm: today and yesterday together, else today alone, else yesterday alone, each at
    HTTP_ALARM over at least HTTP_ALARM_MIN requests."""
    if _tripped(http, HTTP_ALARM_MIN, HTTP_ALARM):
        when = "today and yesterday (UTC)" if http.get("counted") == "requests" else "in the last 24 hours"
        what = "capture requests" if http.get("counted") == "requests" else "answers stored"
        return f"{_pct(http['share'])} of {what} failed {when}: {http['errors']:,} of {http['requests']:,}."
    for name in ("today", "yesterday"):
        day = (http.get("days") or {}).get(name) or {}
        if _tripped(day, HTTP_ALARM_MIN, HTTP_ALARM):
            return f"{_pct(day['share'])} of capture requests failed {name} (UTC): {day['errors']:,} of " \
                   f"{day['requests']:,}."
    return None


def _listed(items: List[str], most: int = 3) -> str:
    return ", ".join(items[:most]) + (f" and {len(items) - most} more" if len(items) > most else "")


def _alarms(block: Mapping[str, Any]) -> List[Dict[str, str]]:
    """The block's alarms, worded here: forward capture's HTTP error share and the backfill's own, the boards the
    sweep couldn't read, payload-shape changes (one alarm per change, its leagues listed), and the disk."""
    alarms = []
    text = _http_alarm(block.get("http") or {})
    if text:
        alarms.append({"kind": "http", "text": text})
    backfill = block.get("backfill_http") or {}
    if _tripped(backfill, BACKFILL_HTTP_ALARM_MIN, BACKFILL_HTTP_ALARM):
        alarms.append({"kind": "backfill", "text": f"{_pct(backfill['share'])} of the older-games backfill's "
                                                   f"requests failed today and yesterday (UTC): "
                                                   f"{backfill['errors']:,} of {backfill['requests']:,}."})
    unread = (block.get("coverage") or {}).get("unread") or []
    if unread:
        boards = _listed([f"{board['league']} ({board['day']})" for board in unread], 4)
        one = len(unread) == 1
        retrying = not all(board.get("final") for board in unread)
        alarms.append({"kind": "boards", "text": f"The sweep couldn't read {len(unread)} ESPN board"
                                                 f"{'' if one else 's'}{' yet' if retrying else ''}: {boards}. "
                                                 f"{'Its' if one else 'Their'} finals aren't counted, so coverage "
                                                 "is unknown."})
    changed: Dict[Tuple[str, bool, Tuple[str, ...]], Dict[str, Any]] = {}
    for shape in block.get("shapes") or []:
        if shape.get("ok"):
            continue
        group = changed.setdefault((str(shape.get("label")), bool(shape.get("json")),
                                    tuple(shape.get("missing") or ())), {"leagues": [], "at": ""})
        if shape.get("league"):
            group["leagues"].append(str(shape["league"]))
        group["at"] = max(group["at"], str(shape.get("at") or ""))
    for (label, is_json, missing), group in changed.items():
        where = ", ".join(part for part in (_listed(sorted(set(group["leagues"]))), group["at"][:10]) if part)
        if not is_json:
            text = f"{label} changed shape: the newest answer isn't a JSON object any more"
        else:
            text = (f"{label} changed shape: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing "
                    "from the newest answer")
        alarms.append({"kind": "shape", "text": f"{text}{f' ({where})' if where else ''}."})
    size = block.get("size") or {}
    if size.get("under_floor"):  # under the floor: at any fullness, capture and the backfill wait for room
        full = f"{_pct(size['disk_full'])} full" if size.get("disk_full") is not None else "nearly full"
        alarms.append({"kind": "disk", "text": f"The disk is {full}: {_bytes_text(size.get('disk_free'))} free. "
                                               "Capture and the backfill are paused until "
                                               f"{_bytes_text(size.get('floor_bytes'))} is free."})
    elif size.get("full_until"):  # SQLite refused a write as "disk is full" though the free blocks look fine
        shows = f", though the disk shows {_bytes_text(size['disk_free'])} free" \
            if size.get("disk_free") is not None else ""
        alarms.append({"kind": "disk", "text": "raw.db can't store more: SQLite said the database or disk is full"
                                               f"{shows} (check its inodes or quota). Capture and the backfill are "
                                               f"paused, trying again every {size.get('full_minutes') or 5} "
                                               "minutes."})
    elif size.get("disk_full") is not None and size["disk_full"] >= DISK_ALARM:
        alarms.append({"kind": "disk", "text": f"The disk is {_pct(size['disk_full'])} full: "
                                               f"{_bytes_text(size.get('disk_free'))} free, raw.db is "
                                               f"{_bytes_text(size.get('raw_bytes'))}."})
    return alarms


def capture_health(path: Optional[str], now: Optional[float] = None, *, enabled: Optional[bool] = True,
                   settings: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The progress page's "capture" block (see the module's docstring) from raw.db at ``path``. It only reads: a
    missing raw.db is "nothing stored yet" and is never created, and a part it can't read is left empty. ``settings``
    may give per_day, per_night, seasons and min_free_gb (CAPTURE_PER_DAY, CAPTURE_BACKFILL_PER_NIGHT,
    CAPTURE_BACKFILL_SEASONS, CAPTURE_MIN_FREE_GB), and tz: the backfill's time zone (capture/backfill.py
    backfill_tz; when absent, its rule with no config)."""
    now = time.time() if now is None else float(now)
    settings = dict(settings or {})
    block = _skeleton(now, enabled, settings)
    if not path or not os.path.isfile(path):
        return block
    block["size"] = _disk(path, settings.get("min_free_gb"), now)
    try:
        with closing(_connect_ro(path)) as db:
            tables = {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if "raw_payloads" in tables:
                block["stored"] = True
                top = db.execute("SELECT MAX(id) FROM raw_payloads").fetchone()[0]
                floor = int(top or 0) - RECENT_ROWS
                last = db.execute("SELECT fetched_at FROM raw_payloads WHERE id = ?", (top,)).fetchone() \
                    if top is not None else None
                block["last_stored"] = iso(last[0]) if last else None
                block["shapes"] = _shapes(db, path, floor)
            else:
                floor = 0
            used: Dict[Tuple[str, str], int] = {}
            if "capture_budget" in tables:
                block["requests"], used = _requests(db, now, settings)
                block["backfill_http"] = _backfill_http(now, used)
            if "raw_payloads" in tables:
                block["http"] = _http(db, now, used, floor)
            if "capture_jobs" in tables and "raw_payloads" in tables:
                unread = _unread_boards(db, now) if "capture_state" in tables else []
                block["coverage"] = _coverage(db, now, unread)
                block["errors"] = _errors(db, floor, now)
            if "backfill_progress" in tables:
                block["backfill"] = _backfill_rows(db, settings, now)
    except (sqlite3.Error, OSError, ValueError, zlib.error) as exc:
        log.warning("Brain progress: raw.db couldn't be read for the capture block (%s)", type(exc).__name__)
        block["unreadable"] = type(exc).__name__
    block["alarms"] = _alarms(block)
    return block


def _capture(store: Any, learner: Optional[Learner], now: float) -> Dict[str, Any]:
    """The capture block for this store and learner (raw.db beside betting.db unless RAW_DB_PATH says otherwise).
    Never raises: the progress page doesn't depend on capture."""
    env = os.environ
    try:
        config = getattr(learner, "config", None)
        settings = {"per_day": _setting(learner, env, "capture_per_day", 3000),
                    "per_night": _setting(learner, env, "capture_backfill_per_night", 6000),
                    "seasons": _setting(learner, env, "capture_backfill_seasons", 3),
                    # None: min_free_bytes reads CAPTURE_MIN_FREE_GB itself, as the worker does
                    "min_free_gb": getattr(config, "capture_min_free_gb", None),
                    # the backfill's nights are dates in its own time zone (TIMEZONE), as the worker builds it
                    "tz": backfill_tz_of(config)}
        return capture_health(raw_db_path(store, learner), now,
                              enabled=bool(_setting(learner, env, "capture_enabled", True)), settings=settings)
    except Exception:  # noqa: BLE001 - the page is about the brain: capture's health must never cost it
        log.exception("Brain progress: the capture block failed")
        return {**_skeleton(now), "unreadable": "error"}


def progress(store: Any, learner: Optional[Learner] = None, *, owner: bool = False,
             now: Optional[float] = None) -> Dict[str, Any]:
    """GET /api/brain/progress (see the module's docstring). Reads the store only."""
    now = time.time() if now is None else now
    with closing(_connect(store)) as db:
        headlines = _headlines(db, store)
        trials = bs.trials(db, limit=500)
        sports = sorted({row["sport"] for row in bs.scores(db)} | set(headlines),
                        key=lambda s: registry.SPORTS.index(s) if s in registry.SPORTS else 99)
        safe = bs.safe_mode(db)
        page = {
            "as_of": iso(_scored_at(db) or now), "served_at": iso(now), "owner": bool(owner),
            "learning": learning_state(db, learner),
            "safe_mode": {"since": iso(safe.get("since")), "reason": safe.get("reason")} if safe else None,
            "sports": [_sport_block(db, store, sport, headlines.get(sport), trials) for sport in sports
                       if sport in registry.SPORTS],
            "changes": _changes(db, trials, owner), "held": _held(db, now), "experiments": _experiments(db),
            "ai": _ai(db, learner, now), "health": _health(db, store, now)}
    page["capture"] = _capture(store, learner, now)  # raw.db, read after betting.db is closed
    return page


def brain_lines(store: Any, now: Optional[float] = None) -> List[str]:
    """The bot's /brain: each sport's headline and today's version, then the last change (from the store)."""
    now = time.time() if now is None else now
    with closing(_connect(store)) as db:
        headlines = _headlines(db, store)
        lines = []
        for sport in registry.SPORTS:
            h = headlines.get(sport)
            if not h:
                continue
            version = _version(db, sport)
            since = f", since {version['since'][:10]}" if version["since"] else ""
            lines.append(f"{h.get('text') or headline_text(sport, h)} Settings: version {version['number']}"
                         f"{' (today’s defaults)' if not version['number'] else ''}{since}.")
        changes = _changes(db, bs.trials(db, limit=200), owner=False)
        safe = bs.safe_mode(db)
    if not lines:
        lines.append("No graded games with a market price yet, so there is nothing to score.")
    if safe:
        lines.append(f"Safe mode is on: the default settings serve every sport ({safe.get('reason')}).")
    if changes:
        last = changes[0]
        lines.append(f"Last change ({(last['at'] or '')[:10]}): {last['text']}")
    else:
        lines.append("No change yet: the brain runs on today's defaults until a change passes the gate.")
    return lines


def tool_facts(store: Any, learner: Optional[Learner] = None, now: Optional[float] = None) -> Dict[str, Any]:
    """The Ask AI brain_progress tool's facts: per sport the headline (n, log-loss gap to the sharp market with its
    95% range, the gate's word), calibration in words, the version live; the last changes; this week's held
    candidates; the AI's provider and spend. All numbers from the store."""
    data = progress(store, learner, now=now)
    sports = [{"sport": s["label"], "headline": s["headline"], "calibration": (s["calibration"] or {}).get("words"),
               "version": s["version"]["number"], "version_since": s["version"]["since"],
               "changed_settings": s["version"]["labels"]} for s in data["sports"]]
    return {"promise": "Improving means: better-calibrated chances, log loss on unseen games moving toward or below "
                       "the sharp market's, and new targets beating simple baselines. Most weeks nothing changes.",
            "sports": sports,
            "changes": [{k: c[k] for k in ("at", "sport_label", "label", "decision", "text")}
                        for c in data["changes"][:5]],
            "held": {k: data["held"][k] for k in ("week", "tested", "passed", "words")},
            "learning": data["learning"], "safe_mode": data["safe_mode"],
            "ai": {"provider": data["ai"]["provider"], "cost": data["ai"]["cost"]}}


def set_paused(store: Any, paused: bool, *, by: str = "owner", now: Optional[float] = None) -> Dict[str, Any]:
    """The Pause switch without a running learner (the bot process, tests): the same brain_state key."""
    now = time.time() if now is None else now
    with closing(_connect(store)) as db:
        bs.set_state(db, "paused", {"paused": bool(paused), "by": by, "at": now}, now)
    return {"paused": bool(paused), "by": by, "at": now, "locked": False}


__all__ = ["progress", "brain_lines", "tool_facts", "set_paused", "calibration_words", "learning_state",
           "SPORT_LABELS"]
