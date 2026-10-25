"""People's predictions: the AI's view of them, scoring, grading from results and the AI's track record."""

from __future__ import annotations

import math

import pytest

from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.engine import calls
from telegram_betting_bot_simple.engine.nba_model import GamePrediction


def prediction(margin=4.0, total=226.0) -> GamePrediction:
    from statistics import NormalDist
    return GamePrediction(home_score=(total + margin) / 2, away_score=(total - margin) / 2, margin=margin,
                          total=total, home_win_prob=NormalDist().cdf(margin / 12.5))


def test_winner_points_follow_fivethirtyeights_game() -> None:
    assert calls.winner_points(0.75) == 18.8  # 75% on the winner: 18.75, rounded half up
    assert calls.winner_points(0.25) == -31.3  # 75% on the loser
    assert calls.winner_points(0.5) == 0
    assert calls.winner_points(1.0) == 25 and calls.winner_points(0.0) == -75
    assert calls.winner_points(0.704) == calls.winner_points(0.70)  # probabilities count in whole percents
    assert calls.forecast_points(0.6, calls.PROP_WEIGHT) == 3.6  # a player line at 40% weight


def test_score_points_reward_close_margins_and_totals() -> None:
    assert calls.score_points(6, 220, 113, 107) == 15 + 20 + 10  # exact: both scores within 2, bullseye
    assert calls.score_points(10, 230, 113, 107) == 11 + 10  # margin off by 4, total by 10
    assert calls.score_points(-5, 260, 113, 107) == 4 + 0  # wrong winner by 11, total way off


def test_agreement_bands() -> None:
    assert calls.agreement(0.6) == "AI agrees" and calls.agreement(0.59) == "Coin flip for the AI"
    assert calls.agreement(0.39) == "AI disagrees"
    assert calls.compare_confidence(0.72, 0.70) == "Same as the AI"
    assert calls.compare_confidence(0.90, 0.70) == "Bolder than the AI"
    assert calls.compare_confidence(0.55, 0.70) == "More cautious than the AI"
    assert calls.compare_confidence(0.80, 0.30) == "Against the AI"


def test_stat_probabilities() -> None:
    # A plain Poisson (variance = mean): P(X >= 2) with mean 2 is 1 - 3e^-2
    assert 1 - calls._negative_binomial_cdf(1, 2.0, 1.0) == pytest.approx(1 - 3 * math.exp(-2))
    assert calls._negative_binomial_cdf(60, 6.0, 1.4) == pytest.approx(1.0, abs=1e-9)
    # Lines below the projection are likely, above it unlikely, and it falls as the line rises
    over = [calls.stat_over_prob("pts", 25.0, line) for line in (19.5, 24.5, 29.5, 34.5)]
    assert over == sorted(over, reverse=True) and over[0] > 0.7 and over[-1] < 0.2  # points vary a lot
    assert 0.4 < calls.stat_over_prob("pts", 25.0, 24.5) < 0.6
    assert 0.35 < calls.stat_over_prob("ast", 8.0, 7.5) < 0.6
    assert calls.stat_over_prob("reb", 0.0, 0.5) == 0.0
    # A whole-number line can push: the three outcomes add up to 1
    over_, under, push = calls.stat_probabilities("reb", 6.0, 6)
    assert push > 0.1 and over_ + under + push == pytest.approx(1.0)
    # Bench players' numbers swing more
    assert calls.stat_over_prob("pts", 8.0, 14.5, minutes=14) > calls.stat_over_prob("pts", 8.0, 14.5, minutes=30)


def test_the_ai_judges_each_part_of_a_call() -> None:
    p = prediction(margin=4.0)  # home favoured
    p.sources = {"market": {"home_win_prob": 0.75, "margin": 8.0, "total": 224.0}}
    players = {"7": {"name": "Star Guard", "pts": 27.0, "reb": 5.0, "ast": 6.0}}
    view = calls.judge_call(p, {"winner": "away", "confidence": 0.7, "home_score": 108, "away_score": 112,
                                "props": [{"player_id": "7", "stat": "pts", "line": 29.5, "side": "over"}]},
                            "Home Team", "Away Team", players)
    assert view["winner"]["ai_prob"] == pytest.approx(1 - p.home_win_prob)
    assert view["winner"]["market_prob"] == pytest.approx(0.25)
    assert view["winner"]["label"] == "AI disagrees" and view["winner"]["compare"] == "Against the AI"
    assert "You: 70%, against the AI." in view["verdict"]
    # The AI (62%) is 13 points off the bookmakers (75%): flagged
    assert view["caution"] and "Caution" in view["verdict"]
    # Away by 4 or more when the AI has home by 4: well under half
    assert view["score"]["wins_by_at_least"] < 0.3
    assert view["props"][0]["ai_mean"] == 27.0 and view["props"][0]["ai_prob"] < 0.5
    assert view["props"][0]["confidence"] == 0.6  # the default


def test_grading_scores_the_person_the_ai_and_the_market() -> None:
    p = prediction(margin=4.0, total=226.0)
    p.sources = {"market": {"home_win_prob": 0.6, "margin": 3.0, "total": 225.0}}
    call = {"winner": "home", "confidence": 0.8, "home_score": 115, "away_score": 108,
            "props": [{"player_id": "7", "stat": "pts", "line": 24.5, "side": "over", "confidence": 0.9},
                      {"player_id": "8", "stat": "reb", "line": 8.5, "side": "over"}]}
    ai = calls.judge_call(p, call, "Home", "Away", {"7": {"name": "A", "pts": 26.0}, "8": {"name": "B", "reb": 6.0}})
    box = {"7": {"pts": 30, "dnp": False}, "8": {"dnp": True}}
    # The AI's last number before tip-off (66%) is what it's scored on, not the one saved with the call
    tip_off = {"home_win": 0.66, "margin": 5.0, "total": 224.0, "market_home_win": 0.62}
    graded = calls.grade_call(call, ai, {"home": 116, "away": 110}, box, tip_off)
    winner, score, prop, void = graded["parts"]
    assert winner["hit"] and winner["points"] == calls.winner_points(0.8)
    assert winner["ai_points"] == calls.winner_points(0.66) and winner["market_points"] == calls.winner_points(0.62)
    assert score["margin_off"] == 1 and score["total_off"] == -3
    assert score["points"] == calls.score_points(7, 223, 116, 110) == 14 + 17 + 10  # 115-108 vs 116-110: bullseye
    assert score["ai_points"] == calls.score_points(5.0, 224.0, 116, 110)
    assert prop["hit"] and prop["points"] == calls.forecast_points(0.9, calls.PROP_WEIGHT)
    assert prop["ai_points"] == calls.forecast_points(prop_ai(ai), calls.PROP_WEIGHT)
    assert void["void"] and void["points"] == 0
    assert graded["points"] == pytest.approx(sum(part["points"] for part in graded["parts"]), abs=0.05)
    assert graded["vs_ai"] == pytest.approx(graded["points"] - graded["ai_points"], abs=0.05)
    assert graded["market_points"] == winner["market_points"]


def prop_ai(ai):
    return ai["props"][0]["ai_prob"]


def test_track_record_measures_accuracy_brier_and_record_against_the_close() -> None:
    rows = [
        {"home": "H", "away": "A", "start": "2026-10-20T23:00:00Z",
         "payload": {"home_win": 0.7, "margin": 5.0, "total": 220.0, "market_home_win": 0.65},
         "result": {"home": 110, "away": 100, "closing": {"home_spread": -4.5, "total": 221.5}}},
        {"home": "H2", "away": "A2", "start": "2026-10-21T23:00:00Z",
         "payload": {"home_win": 0.4, "margin": -2.0, "total": 230.0, "market_home_win": 0.45},
         "result": {"home": 120, "away": 112, "closing": {"home_spread": 1.5, "total": 228.5}}},
    ]
    record = calls.track_record(rows)
    assert record["games"] == 2 and record["winner_accuracy"] == 0.5
    assert record["brier"] == pytest.approx(((0.7 - 1) ** 2 + (0.4 - 1) ** 2) / 2)
    assert record["market_brier"] == pytest.approx(((0.65 - 1) ** 2 + (0.45 - 1) ** 2) / 2)
    # Game 1: AI had home by 5 vs -4.5 (home covers) and home won by 10: win. Game 2: AI had away
    # covering +1.5 but home won by 8: loss. Totals: under 221.5 lost (210 is under: won), over 228.5 won
    assert record["ats"] == {"won": 1, "lost": 1, "push": 0}
    assert record["totals"] == {"won": 2, "lost": 0, "push": 0}
    assert record["calibration"][2]["games"] == 1  # the 70% call


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


class FinishedGames:
    """ESPN stand-in: every game is over, 110-100 to the home team."""

    def find_game(self, *args):
        return {"event_id": "401", "home_id": "1", "away_id": "2"}

    def game_result(self, league, event_id):
        return {"state": "final", "home": 110, "away": 100}

    def box_score(self, league, event_id):
        return {}

    def closing_line(self, league, event_id):
        return {"home_spread": -6.5, "total": 215.5}


def test_calls_are_saved_judged_graded_and_ranked(service, store) -> None:
    service.store = store
    saved = service.save_call("web:abc12345", "Sam", "basketball_nba", "g1", {"winner": "away", "confidence": 0.6})
    assert saved["ai"]["winner"]["team"] == "Los Angeles Clippers"
    # Saving again replaces the earlier call
    service.save_call("web:abc12345", "Sam", "basketball_nba", "g1", {"winner": "home", "confidence": 0.9})
    assert [c["call"]["winner"] for c in service.my_calls("web:abc12345")] == ["home"]
    # The AI's own pre-game predictions are logged when the games are shown
    service.games("basketball_nba")
    assert {row["event_id"] for row in store.ai_predictions()} == {"g1", "g2"}

    # Once the games are over (pretend they started a day ago) everything gets graded
    with store._connect() as db:
        db.execute("UPDATE predictions SET start = '2020-01-01T00:00:00Z'")
        db.execute("UPDATE ai_predictions SET start = '2020-01-01T00:00:00Z'")
    service.espn = FinishedGames()
    assert service.grade_due() == 3
    call = service.my_calls("web:abc12345")[0]
    assert call["status"] == "graded" and call["user_points"] == calls.winner_points(0.9)
    board = service.leaderboard()
    assert board[0]["name"] == "Sam" and board[0]["calls"] == 1
    assert board[0]["vs_ai"] == round(board[0]["points"] - board[0]["ai_points"], 1)
    record = service.track_record()
    assert record["games"] == 2 and record["winner_accuracy"] == 1.0
    assert [t["tier"] for t in record["tiers"]] == ["safe", "risky", "crazy"]


def test_the_ais_daily_tickets_are_logged_and_graded(service, store) -> None:
    service.store = store
    built = service.ai_bets("basketball_nba")
    top = built["tiers"]["safe"]["bets"][0]
    logged = {t["tier"]: t for t in store.ai_tickets()}
    assert set(logged) == {key for key, tier in built["tiers"].items() if tier["bets"]}
    assert logged["safe"]["payload"]["legs"][0]["home"] == "New York Knicks"
    with store._connect() as db:
        db.execute("UPDATE ai_tickets SET first_start = '2020-01-01T00:00:00Z', last_start = '2020-01-01T00:00:00Z'")
    service.espn = FinishedGames()  # every game 110-100 to the home team
    service.grade_due()
    ticket = next(t for t in store.ai_tickets() if t["tier"] == "safe")
    assert ticket["status"] == ("won" if top["legs"][0]["side"] == "home" else "lost")
    safe = service.track_record()["tiers"][0]
    assert safe["tickets"] == 1 and safe["predicted"] == pytest.approx(top["probability"])


def test_calls_close_at_tip_off(service, store) -> None:
    from telegram_betting_bot_simple.dashboard.service import CallRejected
    service.store = store
    games = service.odds.league_games("basketball_nba")
    service.odds.league_games = lambda league, refresh=False: [{**g, "commence_time": "2020-01-01T00:00:00Z"} for g in games]
    with pytest.raises(CallRejected):
        service.save_call("web:abc12345", "Sam", "basketball_nba", "g1", {"winner": "home", "confidence": 0.6})
