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
     "health": {"graded", "voided", "backlog", "backfill", "history", "runs", "trials"}}

``brain_lines(store)`` gives the bot's /brain text (a headline per sport and the last change); ``tool_facts(store)``
the Ask AI tool's answer to "is the brain getting better?".
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
        return {
            "as_of": iso(_scored_at(db) or now), "served_at": iso(now), "owner": bool(owner),
            "learning": learning_state(db, learner),
            "safe_mode": {"since": iso(safe.get("since")), "reason": safe.get("reason")} if safe else None,
            "sports": [_sport_block(db, store, sport, headlines.get(sport), trials) for sport in sports
                       if sport in registry.SPORTS],
            "changes": _changes(db, trials, owner), "held": _held(db, now), "experiments": _experiments(db),
            "ai": _ai(db, learner, now), "health": _health(db, store, now)}


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
