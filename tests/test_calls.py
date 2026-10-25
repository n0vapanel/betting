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
    assert calls.winner_points(0.75) == pytest.approx(18.75)  # 75% on the winner
    assert calls.winner_points(0.25) == pytest.approx(-31.25)  # 75% on the loser
    assert calls.winner_points(0.5) == 0
    assert calls.winner_points(1.0) == 25 and calls.winner_points(0.0) == -75


def test_closeness_points_fall_off_linearly() -> None:
    assert calls.closeness_points(0.5, 1, 12) == 10
    assert calls.closeness_points(-6.5, 1, 12) == pytest.approx(5.0)
    assert calls.closeness_points(15, 1, 12) == 0


def test_stat_probabilities() -> None:
    # A plain Poisson (variance = mean): P(X >= 2) with mean 2 is 1 - 3e^-2
    assert 1 - calls._negative_binomial_cdf(1, 2.0, 1.0) == pytest.approx(1 - 3 * math.exp(-2))
    assert calls._negative_binomial_cdf(60, 6.0, 1.4) == pytest.approx(1.0, abs=1e-9)
    # Lines below the projection are likely, above it unlikely, and it falls as the line rises
    over = [calls.stat_over_prob("pts", 25.0, line) for line in (19.5, 24.5, 29.5, 34.5)]
    assert over == sorted(over, reverse=True) and over[0] > 0.75 and over[-1] < 0.1
    assert 0.4 < calls.stat_over_prob("pts", 25.0, 24.5) < 0.6
    assert 0.35 < calls.stat_over_prob("ast", 8.0, 7.5) < 0.6
    assert calls.stat_over_prob("reb", 0.0, 0.5) == 0.0


def test_the_ai_judges_each_part_of_a_call() -> None:
    p = prediction(margin=4.0)  # home favoured
    players = {"7": {"name": "Star Guard", "pts": 27.0, "reb": 5.0, "ast": 6.0}}
    view = calls.judge_call(p, {"winner": "away", "confidence": 0.7, "home_score": 108, "away_score": 112,
                                "props": [{"player_id": "7", "stat": "pts", "line": 29.5, "side": "over"}]},
                            "Home Team", "Away Team", players)
    assert view["winner"]["ai_prob"] == pytest.approx(1 - p.home_win_prob)
    assert "disagrees" in view["verdict"]
    # Away by 4 or more when the AI has home by 4: well under half
    assert view["score"]["wins_by_at_least"] < 0.3
    assert view["props"][0]["ai_mean"] == 27.0 and view["props"][0]["ai_prob"] < 0.5


def test_grading_scores_the_person_and_the_ai_on_the_same_parts() -> None:
    p = prediction(margin=4.0, total=226.0)
    call = {"winner": "home", "confidence": 0.8, "home_score": 115, "away_score": 108,
            "props": [{"player_id": "7", "stat": "pts", "line": 24.5, "side": "over"},
                      {"player_id": "8", "stat": "reb", "line": 8.5, "side": "over"}]}
    ai = calls.judge_call(p, call, "Home", "Away", {"7": {"name": "A", "pts": 26.0}, "8": {"name": "B", "reb": 6.0}})
    box = {"7": {"pts": 30, "dnp": False}, "8": {"dnp": True}}
    graded = calls.grade_call(call, ai, {"home": 116, "away": 110}, box)
    winner, score, prop, void = graded["parts"]
    assert winner["hit"] and winner["points"] == pytest.approx(calls.winner_points(0.8))
    assert winner["ai_points"] == pytest.approx(calls.winner_points(p.home_win_prob))
    assert score["margin_off"] == 1 and score["total_off"] == -3
    assert prop["hit"] and prop["points"] == 10 and prop["ai_points"] == 10  # the AI also rated the over likely
    assert void["void"] and void["points"] == 0
    assert graded["points"] == pytest.approx(sum(part["points"] for part in graded["parts"]), abs=0.01)


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
    saved = service.save_call("web:abc12345", "Aidan", "basketball_nba", "g1", {"winner": "away", "confidence": 0.6})
    assert saved["ai"]["winner"]["team"] == "Los Angeles Clippers"
    # Saving again replaces the earlier call
    service.save_call("web:abc12345", "Aidan", "basketball_nba", "g1", {"winner": "home", "confidence": 0.9})
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
    assert call["status"] == "graded" and call["user_points"] == pytest.approx(calls.winner_points(0.9))
    board = service.leaderboard()
    assert board[0]["name"] == "Aidan" and board[0]["calls"] == 1
    record = service.track_record()
    assert record["games"] == 2 and record["winner_accuracy"] == 1.0


def test_calls_close_at_tip_off(service, store) -> None:
    from telegram_betting_bot_simple.dashboard.service import CallRejected
    service.store = store
    games = service.odds.league_games("basketball_nba")
    service.odds.league_games = lambda league, refresh=False: [{**g, "commence_time": "2020-01-01T00:00:00Z"} for g in games]
    with pytest.raises(CallRejected):
        service.save_call("web:abc12345", "Aidan", "basketball_nba", "g1", {"winner": "home", "confidence": 0.6})
