"""Runs the AI analysts: when each is asked, what it costs, how it is graded,
and how much of its forecast the AI takes on.

The dashboard process asks the analysts about NBA games a few hours before
tip-off (again if the bookmakers' line moves a lot), and anyone can ask one
from a game's Analysts tab. Every call is logged with its estimated cost and
counts toward a daily limit per analyst. After the games, each forecast is
graded next to the AI's and the bookmakers' chances from the same moment,
and the trust fitted from those grades decides how far each analyst moves
the AI's own win chance (nothing, until it has earned it).
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.store import Store
from ..engine import nba_model
from ..engine.calls import winner_points
from . import learning
from .dossier import build_dossier, dossier_text, news_request
from .forecast import AnalystError, Reply

LEAGUES = {"basketball_nba"}
LABELS = {"claude": "Claude", "deepseek": "DeepSeek"}
REASK_MOVE = 0.04       # ask again before tip-off when the bookmakers' chance has moved this much
MIN_GAP = 30 * 60       # never ask the same analyst about the same game more often than this
NEWS_FRESH = 6 * 3600   # Claude's news check is shared with DeepSeek for this long
TRUST_TTL = 600         # seconds a trust fit is reused
_NORMAL = NormalDist()


def _start(card: Dict[str, Any]) -> datetime.datetime:
    return datetime.datetime.fromisoformat(card["start"].replace("Z", "+00:00"))


class AnalystDesk:
    def __init__(self, store: Store, analysts: Sequence[Any], daily_limit: int = 25,
                 lookahead_hours: float = 6.0) -> None:
        self.store = store
        # Claude goes first: with web search on, its news check also goes into DeepSeek's dossier
        self.analysts = {a.key: a for a in sorted(analysts, key=lambda a: a.key != "claude")}
        self.daily_limit = daily_limit
        self.lookahead = lookahead_hours * 3600
        self.lock = threading.Lock()
        self._trust: Tuple[float, Dict[str, float]] = (0.0, {})

    # --- asking ---------------------------------------------------------------------

    def setup(self) -> List[Dict[str, Any]]:
        """The analysts switched on, and their use over the last 24 hours."""
        out = []
        for key, analyst in self.analysts.items():
            calls, cost = self.store.analyst_spend(key, time.time() - 86400)
            out.append({"key": key, "label": analyst.label, "model": analyst.model,
                        "web_search": bool(getattr(analyst, "web_search", False)),
                        "calls_24h": calls, "cost_24h": round(cost, 3), "daily_limit": self.daily_limit})
        return out

    def _skip(self, key: str, card: Dict[str, Any], latest: Optional[Dict[str, Any]], asked: bool) -> Optional[str]:
        """Why an analyst shouldn't be asked about a game right now (None: go ahead)."""
        calls, _ = self.store.analyst_spend(key, time.time() - 86400)
        if calls >= self.daily_limit:
            return f"Daily limit reached: {self.daily_limit} calls in 24 hours (AI_DAILY_LIMIT on the server)"
        last = self.store.last_analyst_call(key, card["id"])
        if last and time.time() - last[0] < MIN_GAP:
            return "Asked less than 30 minutes ago" if last[1] else "It failed less than 30 minutes ago: try later"
        if asked or latest is None:
            return None
        now = (card["prediction"]["sources"].get("market") or {}).get("home_win_prob")
        then = latest["payload"].get("market_home_win")
        if now is None or then is None or abs(now - then) < REASK_MOVE:
            return "Up to date"
        return None

    def _known_news(self, league: str, event_id: str) -> Tuple[str, List[Dict[str, str]]]:
        for row in self.store.analyses(league=league, event_id=event_id):
            if row["payload"].get("news") and time.time() - row["created_at"] < NEWS_FRESH:
                return row["payload"]["news"], row["payload"].get("sources") or []
        return "", []

    def analyze(self, service: Any, league: str, event_id: str, only: Optional[str] = None,
                asked: bool = False) -> Dict[str, Dict[str, str]]:
        """Ask the analysts that are due about one game (or just ``only``, when someone asks)."""
        if league not in LEAGUES:
            raise AnalystError("The analysts cover NBA games for now")
        with self.lock:
            card = next((c for c in service.games(league)["games"] if c["id"] == event_id), None)
            if card is None:
                raise AnalystError("That game isn't on the board any more")
            if card["live"]:
                raise AnalystError("This game has started: the analysts forecast before tip-off")
            if not card.get("prediction"):
                raise AnalystError("The AI has no prediction for this game yet")
            results: Dict[str, Dict[str, str]] = {}
            todo = []
            for key in [only] if only else list(self.analysts):
                if key not in self.analysts:
                    results[key] = {"status": "error", "message": f"{LABELS.get(key, key)} isn't set up on the server"}
                    continue
                reason = self._skip(key, card, self.store.analysis(key, league, event_id), asked)
                if reason:
                    results[key] = {"status": "skipped", "message": reason}
                else:
                    todo.append(key)
            if not todo:
                return results
            card, info, players = service.detail(league, event_id)
            news, sources = self._known_news(league, event_id)
            for key in todo:
                results[key] = self._ask(key, league, card, info, players, news, sources)
                saved = self.store.analysis(key, league, event_id)
                if results[key]["status"] == "done" and saved and saved["payload"].get("news"):
                    news, sources = saved["payload"]["news"], saved["payload"].get("sources") or []
            if any(result["status"] == "done" for result in results.values()):
                service.forget(league)  # the game cards pick up the new forecasts
            return results

    def _ask(self, key: str, league: str, card: Dict[str, Any], info: Dict[str, Any],
             players: Dict[str, Dict[str, Any]], news: str, sources: List[Dict[str, str]]) -> Dict[str, str]:
        analyst = self.analysts[key]
        cost = 0.0
        try:
            if getattr(analyst, "web_search", False):
                today = datetime.datetime.now(datetime.timezone.utc).strftime("%A %B %-d, %Y (UTC)")
                try:
                    news, sources, cost = analyst.check_news(news_request(card, today))
                except AnalystError as exc:
                    cost += exc.cost  # forecast without the news rather than not at all
                    logging.warning("%s news check failed for %s: %s", analyst.label, card["id"], exc)
            note = learning.feedback_note(learning.record(self.store.analyses(analyst=key, status="graded")))
            reply = analyst.analyze(dossier_text(build_dossier(card, info, players, note, news)))
            cost += reply.cost
            self._save(key, league, card, info, reply, news, sources, cost)
            self.store.log_analyst_call(key, reply.model, card["id"], reply.input_tokens, reply.output_tokens,
                                        cost, True)
            return {"status": "done"}
        except AnalystError as exc:
            self.store.log_analyst_call(key, analyst.model, card["id"], 0, 0, cost + exc.cost, False)
            logging.warning("%s couldn't forecast %s: %s", analyst.label, card["id"], exc)
            return {"status": "error", "message": str(exc)}
        except Exception:
            logging.exception("%s failed on %s", analyst.label, card["id"])
            self.store.log_analyst_call(key, analyst.model, card["id"], 0, 0, cost, False)
            return {"status": "error", "message": f"{analyst.label} failed unexpectedly (see the dashboard's log)"}

    def _save(self, key: str, league: str, card: Dict[str, Any], info: Dict[str, Any], reply: Reply, news: str,
              sources: List[Dict[str, str]], cost: float) -> None:
        prediction = card["prediction"]
        before = prediction["sources"].get("before_analysts", {})
        self.store.save_analysis({
            "analyst": key, "league": league, "event_id": card["id"],
            "espn_id": (info.get("match") or {}).get("event_id"),
            "home": card["home"], "away": card["away"], "start": card["start"],
            "payload": {
                **reply.forecast,
                "model": reply.model,
                # The AI's own chance (before any analyst moved it) and the bookmakers', at this moment
                "base_home_win": before.get("home_win_prob", prediction["home_win"]),
                "market_home_win": (prediction["sources"].get("market") or {}).get("home_win_prob"),
                "news": news,
                "sources": sources,
                "cost": round(cost, 5),
            },
        })

    def run_due(self, service: Any) -> int:
        """Ask about NBA games starting soon that the analysts haven't covered, or whose line moved."""
        done = 0
        now = datetime.datetime.now(datetime.timezone.utc)
        for league in LEAGUES:
            for card in service.games(league)["games"]:
                if card["live"] or not card.get("prediction") or (_start(card) - now).total_seconds() > self.lookahead:
                    continue
                try:
                    results = self.analyze(service, league, card["id"])
                except AnalystError:
                    continue
                done += sum(result["status"] == "done" for result in results.values())
        return done

    # --- grading and trust --------------------------------------------------------------

    def grade_due(self, service: Any, started_before: str, give_up: str) -> int:
        """Grade forecasts on finished games next to the AI's and the bookmakers' chances from then."""
        graded = 0
        for row in self.store.analyses(status="pending", started_before=started_before):
            key = (row["analyst"], row["league"], row["event_id"])
            try:
                final = service.final(row)
                if final["state"] == "pending":
                    if row["start"] < give_up:
                        self.store.grade_analysis(*key, "void", {"reason": "no result"})
                    continue
                if final["state"] == "void":
                    self.store.grade_analysis(*key, "void", final)
                    continue
                payload = row["payload"]
                won = final["home"] > final["away"]
                p = payload["home_win_prob"]
                result = {
                    "home": final["home"], "away": final["away"], "home_won": won,
                    "brier": round((p - won) ** 2, 4),
                    "points": winner_points(p if won else 1 - p),
                    "margin_miss": round(payload["home_margin"] - (final["home"] - final["away"]), 1),
                    "total_miss": round(payload["total"] - (final["home"] + final["away"]), 1),
                }
                for name, chance in (("ai_points", payload.get("base_home_win")),
                                     ("market_points", payload.get("market_home_win"))):
                    if chance is not None:
                        result[name] = winner_points(chance if won else 1 - chance)
                self.store.grade_analysis(*key, "graded", result)
                graded += 1
            except Exception:
                logging.exception("Couldn't grade %s's forecast for %s", row["analyst"], row["event_id"])
        if graded:
            self._trust = (0.0, {})
        return graded

    def trust(self) -> Dict[str, float]:
        """Each analyst's earned trust (0-1), refitted from its graded forecasts every few minutes."""
        fitted_at, trust = self._trust
        if time.time() - fitted_at > TRUST_TTL:
            trust = learning.fit_trust(learning.training_rows(self.store.analyses(status="graded")))
            self._trust = (time.time(), trust)
        return trust

    def adjust(self, league: str, event_id: str, prediction: nba_model.GamePrediction,
               params: nba_model.Params) -> None:
        """Move the AI's win chance (and margin with it) by each analyst's forecast, times its trust."""
        trust = {key: weight for key, weight in self.trust().items() if weight > 0}
        if not trust:
            return
        shift, used = 0.0, []
        for key, weight in trust.items():
            row = self.store.analysis(key, league, event_id)
            if row is None or row["status"] != "pending" or row["payload"].get("base_home_win") is None:
                continue
            payload = row["payload"]
            shift += weight * (learning.logit(payload["home_win_prob"]) - learning.logit(payload["base_home_win"]))
            used.append(LABELS.get(key, key))
        if not used or abs(shift) < 1e-6:
            return
        before, margin = prediction.home_win_prob, prediction.margin
        after = learning.sigmoid(learning.logit(before) + shift)
        win_sd = nba_model.spreads_for(prediction.total, params)[1]
        moved = win_sd * (_NORMAL.inv_cdf(min(max(after, 0.01), 0.99)) - _NORMAL.inv_cdf(min(max(before, 0.01), 0.99)))
        prediction.sources["before_analysts"] = {"home_win_prob": before, "margin": margin}
        prediction.home_win_prob = after
        prediction.margin += moved
        prediction.home_score += moved / 2
        prediction.away_score -= moved / 2
        prediction.factors.append(
            f"AI analysts ({', '.join(used)}), weighted by their graded record: home win chance {after - before:+.1%}"
        )

    # --- views ------------------------------------------------------------------------------

    def briefs(self, league: str) -> Dict[str, List[Dict[str, Any]]]:
        """Each upcoming game's latest forecasts, for the game cards."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        rows = sorted(self.store.analyses(league=league, status="pending"), key=lambda r: r["analyst"] != "claude")
        for row in rows:
            out.setdefault(row["event_id"], []).append({
                "analyst": row["analyst"], "label": LABELS.get(row["analyst"], row["analyst"].title()),
                "home_win": row["payload"]["home_win_prob"],
            })
        return out

    def game_view(self, league: str, event_id: str) -> Dict[str, Any]:
        """Every analyst's latest forecast for a game, with its reasons, news and cost."""
        trust = self.trust()
        rows = {row["analyst"]: row for row in self.store.analyses(league=league, event_id=event_id)}
        view = []
        for key in list(self.analysts) + sorted(set(rows) - set(self.analysts)):
            analyst, row = self.analysts.get(key), rows.get(key)
            entry: Dict[str, Any] = {"key": key, "label": LABELS.get(key, key.title()), "enabled": analyst is not None,
                                     "model": getattr(analyst, "model", None), "trust": trust.get(key, 0.0)}
            if row:
                payload = row["payload"]
                entry.update({
                    "forecast": {name: payload.get(name) for name in
                                 ("home_win_prob", "home_margin", "total", "confidence", "summary", "factors", "edge")},
                    "model": payload.get("model"), "news": payload.get("news") or "", "sources": payload.get("sources") or [],
                    "cost": payload.get("cost"), "created_at": row["created_at"], "status": row["status"],
                    "base_home_win": payload.get("base_home_win"), "market_home_win": payload.get("market_home_win"),
                })
            view.append(entry)
        return {"analysts": view, "setup": self.setup()}

    def record(self) -> List[Dict[str, Any]]:
        """Each analyst's graded record, earned trust and spend, for the AI Record page."""
        trust = self.trust()
        graded = self.store.analyses(status="graded")
        keys = list(self.analysts) + sorted({row["analyst"] for row in graded} - set(self.analysts))
        out = []
        for key in keys:
            stats = learning.record([row for row in graded if row["analyst"] == key])
            calls_24h, cost_24h = self.store.analyst_spend(key, time.time() - 86400)
            calls, cost = self.store.analyst_spend(key)
            out.append({**stats, "key": key, "label": LABELS.get(key, key.title()), "enabled": key in self.analysts,
                        "model": getattr(self.analysts.get(key), "model", None), "trust": trust.get(key, 0.0),
                        "min_games": learning.MIN_GAMES, "calls_24h": calls_24h, "cost_24h": round(cost_24h, 3),
                        "calls": calls, "cost": round(cost, 2), "daily_limit": self.daily_limit})
        return out
