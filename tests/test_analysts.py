"""The AI analysts (Claude, DeepSeek): requests, limits, grading and earned trust (no network)."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from telegram_betting_bot_simple.ai import learning
from telegram_betting_bot_simple.ai.claude import ClaudeAnalyst
from telegram_betting_bot_simple.ai.deepseek import DeepSeekAnalyst
from telegram_betting_bot_simple.ai.desk import AnalystDesk
from telegram_betting_bot_simple.ai.forecast import AnalystError, Reply, parse_forecast
from telegram_betting_bot_simple.bot.handlers import game_card_text
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard.app import create_app
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.services import build_analysts

FORECAST = {"home_win_prob": 0.83, "home_margin": 9.5, "total": 226.0, "confidence": "low",
            "summary": "The Thunder are rested and deep. The Clippers are on a back-to-back.",
            "factors": ["Rest", "Depth"], "edge": "None: I agree with the market"}


# --- the forecast format ---------------------------------------------------------------------

def test_forecasts_are_validated_and_kept_inside_2_to_98_percent() -> None:
    parsed = parse_forecast({**FORECAST, "home_win_prob": 1.0, "factors": ["a", " ", "b", "c", "d", "e", "f"]})
    assert parsed["home_win_prob"] == 0.98 and parsed["factors"] == ["a", "b", "c", "d", "e"]
    for bad in ({**FORECAST, "home_win_prob": 1.5}, {**FORECAST, "confidence": "certain"},
                {k: v for k, v in FORECAST.items() if k != "total"}, ["not", "an", "object"]):
        with pytest.raises(AnalystError):
            parse_forecast(bad)


# --- Claude ------------------------------------------------------------------------------------

def usage(fresh=1000, out=2000, read=3000, searches=0):
    return SimpleNamespace(input_tokens=fresh, output_tokens=out, cache_creation_input_tokens=0,
                           cache_read_input_tokens=read,
                           server_tool_use=SimpleNamespace(web_search_requests=searches) if searches else None)


def message(content, stop="end_turn", used=None, model="claude-opus-5-5"):
    return SimpleNamespace(content=content, stop_reason=stop, usage=used or usage(), model=model)


def text(value, citations=None):
    return SimpleNamespace(type="text", text=value, citations=citations)


class FakeAnthropic:
    def __init__(self, *responses):
        self.requests, self.responses = [], list(responses)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_claude_forecasts_with_structured_output_cached_instructions_and_fallbacks() -> None:
    client = FakeAnthropic(message([SimpleNamespace(type="thinking", thinking=""), text(json.dumps(FORECAST))]))
    reply = ClaudeAnalyst("sk-ant-test", client=client).analyze('{"game": {}}')
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": request["output_config"]["format"]["schema"]}}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in request and "thinking" not in request  # Opus 5.5 always thinks; effort sets how much
    assert reply.forecast["home_win_prob"] == 0.83 and reply.model == "claude-opus-5-5"
    assert reply.cost == pytest.approx((1000 * 4 + 3000 * 0.2 + 2000 * 20) / 1e6)

    # Models without server-side fallbacks don't get the parameter
    other = FakeAnthropic(message([text(json.dumps(FORECAST))], model="claude-haiku-5-5"))
    ClaudeAnalyst("sk-ant-test", model="claude-haiku-5-5", client=other).analyze("{}")
    assert "fallbacks" not in other.requests[0] and "betas" not in other.requests[0]


def test_claude_refusals_cut_offs_and_api_errors_become_plain_messages() -> None:
    refused = FakeAnthropic(message([], stop="refusal"))
    with pytest.raises(AnalystError, match="declined") as caught:
        ClaudeAnalyst("sk-ant-test", client=refused).analyze("{}")
    assert caught.value.cost > 0  # a refused request can still cost money
    with pytest.raises(AnalystError, match="cut off"):
        ClaudeAnalyst("sk-ant-test", client=FakeAnthropic(message([text('{"home')], stop="max_tokens"))).analyze("{}")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    bad_key = anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=request), body=None)
    with pytest.raises(AnalystError, match="ANTHROPIC_API_KEY") as caught:
        ClaudeAnalyst("sk-ant-secret-value", client=FakeAnthropic(bad_key)).analyze("{}")
    assert "secret" not in str(caught.value)


def test_claudes_news_check_searches_the_web_and_resumes_paused_turns() -> None:
    citation = SimpleNamespace(url="https://www.nba.com/news/injury-report", title="NBA injury report")
    client = FakeAnthropic(
        message([SimpleNamespace(type="server_tool_use")], stop="pause_turn", used=usage(searches=2)),
        message([text("Shai Gilgeous-Alexander is probable (NBA injury report, Oct 25).", [citation])],
                used=usage(searches=1)),
    )
    news, sources, cost = ClaudeAnalyst("sk-ant-test", web_search=True, client=client).check_news("Game: ...")
    assert client.requests[0]["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]
    assert client.requests[1]["messages"][-1]["role"] == "assistant"  # the paused turn is sent back to carry on
    assert news.startswith("Shai Gilgeous-Alexander is probable")
    assert sources == [{"url": "https://www.nba.com/news/injury-report", "title": "NBA injury report"}]
    assert cost == pytest.approx(2 * (1000 * 4 + 3000 * 0.2 + 2000 * 20) / 1e6 + 3 * 0.01)


# --- DeepSeek -----------------------------------------------------------------------------------

class FakeSession:
    def __init__(self, *responses):
        self.calls, self.responses = [], list(responses)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        status, body = self.responses.pop(0)
        return SimpleNamespace(status_code=status, json=lambda: body)


def completion(content, finish="stop"):
    return {"model": "deepseek-v4-pro", "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 4000, "prompt_cache_hit_tokens": 3000, "prompt_cache_miss_tokens": 1000,
                      "completion_tokens": 2500}}


def test_deepseek_uses_json_mode_and_asks_again_without_thinking_when_empty() -> None:
    session = FakeSession((200, completion("")), (200, completion(json.dumps(FORECAST))))
    reply = DeepSeekAnalyst("sk-deepseek-test", session=session).analyze('{"game": {}}')
    first, second = session.calls
    assert first["url"] == "https://api.deepseek.com/chat/completions"
    assert first["headers"] == {"Authorization": "Bearer sk-deepseek-test"}
    assert first["json"]["response_format"] == {"type": "json_object"} and "thinking" not in first["json"]
    assert second["json"]["thinking"] == {"type": "disabled"}
    assert reply.forecast["total"] == 226.0
    assert reply.cost == pytest.approx(2 * (3000 * 0.044 + 1000 * 1.32 + 2500 * 3.96) / 1e6)
    for status, words in ((401, "DEEPSEEK_API_KEY"), (402, "out of credit"), (429, "rate-limiting")):
        with pytest.raises(AnalystError, match=words):
            DeepSeekAnalyst("sk-x", session=FakeSession((status, {}))).analyze("{}")


# --- the desk: asking, limits, grading ------------------------------------------------------------

class FakeAnalyst:
    def __init__(self, key, label, p=0.83, web_search=False):
        self.key, self.label, self.model, self.web_search, self.p = key, label, f"{key}-test", web_search, p
        self.dossiers = []

    def check_news(self, request):
        return "Kawhi Leonard is out (Clippers, Oct 25).", [{"url": "https://example.com/x", "title": "Report"}], 0.02

    def analyze(self, dossier):
        self.dossiers.append(json.loads(dossier))
        return Reply({**FORECAST, "home_win_prob": self.p}, self.model, 0.05, 1000, 500)


@pytest.fixture()
def store(tmp_path):
    return Store(str(tmp_path / "betting.db"))


@pytest.fixture()
def desk(service, store):
    claude, deepseek = FakeAnalyst("claude", "Claude", web_search=True), FakeAnalyst("deepseek", "DeepSeek", p=0.88)
    desk = AnalystDesk(store, [deepseek, claude], daily_limit=3)
    service.store, service.analysts = store, desk
    return desk


def test_analysts_forecast_a_game_with_its_dossier_and_their_costs_are_logged(service, store, desk) -> None:
    results = desk.analyze(service, "basketball_nba", "g1")
    assert results == {"claude": {"status": "done"}, "deepseek": {"status": "done"}}
    claude, deepseek = desk.analysts["claude"], desk.analysts["deepseek"]
    dossier = deepseek.dossiers[0]
    assert dossier["game"]["home"] == "Oklahoma City Thunder" and dossier["bookmakers"]["home_win"] > 0.8
    assert dossier["latest_news"].startswith("Kawhi Leonard is out")  # Claude's news check, shared
    assert dossier["your_record"] == "No graded forecasts yet."
    saved = store.analysis("claude", "basketball_nba", "g1")
    assert saved["payload"]["home_win_prob"] == 0.83 and saved["payload"]["cost"] == pytest.approx(0.07)
    assert saved["payload"]["base_home_win"] == pytest.approx(service.games("basketball_nba")["games"][0]["prediction"]["home_win"])
    assert store.analyst_spend("claude") == (1, pytest.approx(0.07))

    # The game card shows each pick; the game's own view has the reasons
    card = next(g for g in service.games("basketball_nba")["games"] if g["id"] == "g1")
    assert [(a["label"], a["home_win"]) for a in card["analysts"]] == [("Claude", 0.83), ("DeepSeek", 0.88)]
    view = service.game("basketball_nba", "g1")["analysts"]
    assert view["analysts"][0]["forecast"]["summary"].startswith("The Thunder") and view["analysts"][0]["trust"] == 0.0
    assert "🧠 <b>AI analysts</b>" in game_card_text({**service.game("basketball_nba", "g1")}, None)

    # Asked again straight away: nothing new is spent
    assert desk.analyze(service, "basketball_nba", "g1")["claude"]["status"] == "skipped"
    assert len(claude.dossiers) == 1


def test_the_daily_limit_and_the_background_run(service, store, desk) -> None:
    desk.lookahead = 10 ** 10  # the fake games are years away
    assert desk.run_due(service) == 4  # two games, two analysts
    assert desk.run_due(service) == 0  # up to date: the bookmakers haven't moved
    with store._connect() as db:
        db.execute("UPDATE analyst_calls SET ts = ts - 3600")  # past the 30-minute gap
    result = desk.analyze(service, "basketball_nba", "g1", only="claude", asked=True)
    assert result["claude"]["status"] == "done"  # someone asked: allowed within the limit (3 a day)
    with store._connect() as db:
        db.execute("UPDATE analyst_calls SET ts = ts - 3600")
    blocked = desk.analyze(service, "basketball_nba", "g2", only="claude", asked=True)["claude"]
    assert blocked["status"] == "skipped" and "Daily limit" in blocked["message"]


def test_forecasts_are_graded_after_the_game(service, store, desk) -> None:
    from tests.test_calls import FinishedGames
    desk.analyze(service, "basketball_nba", "g1")
    with store._connect() as db:
        db.execute("UPDATE analyses SET start = '2020-01-01T00:00:00Z'")
    service.espn = FinishedGames()  # 110-100 to the home team
    service.grade_due()
    graded = store.analysis("deepseek", "basketball_nba", "g1")
    assert graded["status"] == "graded" and graded["result"]["home_won"] is True
    assert graded["result"]["brier"] == pytest.approx((0.88 - 1) ** 2, abs=1e-4)
    assert graded["result"]["margin_miss"] == -0.5 and graded["result"]["total_miss"] == 16.0
    record = {row["key"]: row for row in service.track_record()["analysts"]}
    assert record["deepseek"]["games"] == 1 and record["deepseek"]["accuracy"] == 1.0
    assert record["claude"]["trust"] == 0.0 and record["claude"]["min_games"] == learning.MIN_GAMES
    # Its record goes into its next dossier
    note = learning.feedback_note(learning.record(store.analyses(analyst="deepseek", status="graded")))
    assert note.startswith("1 graded forecast (a small sample")


# --- earned trust ---------------------------------------------------------------------------------

def simulated_rows(n, seed, informative):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        base = rng.uniform(0.25, 0.85)
        hidden = rng.gauss(0, 1)
        won = int(rng.random() < learning.sigmoid(learning.logit(base) + 0.25 * hidden))
        move = 0.2 * hidden + rng.gauss(0, 0.08) if informative else rng.gauss(0, 0.15)
        rows.append((learning.logit(base), won, {"analyst": move}))
    return rows


def test_trust_is_earned_slowly_and_noise_earns_none() -> None:
    assert learning.fit_trust(simulated_rows(learning.MIN_GAMES - 1, 1, True)) == {"analyst": 0.0}
    useful = [learning.fit_trust(simulated_rows(1500, seed, True))["analyst"] for seed in range(5)]
    noise = [learning.fit_trust(simulated_rows(1500, seed, False))["analyst"] for seed in range(5)]
    assert min(useful) > 0.2 and max(noise) < 0.15


def test_earned_trust_moves_the_ai_and_the_dossier_keeps_the_ais_own_numbers(service, store, desk) -> None:
    before = next(g for g in service.games("basketball_nba")["games"] if g["id"] == "g1")["prediction"]
    desk.analyze(service, "basketball_nba", "g1")
    desk.trust = lambda: {"claude": 0.5, "deepseek": 0.0}
    service.forget("basketball_nba")
    after = next(g for g in service.games("basketball_nba")["games"] if g["id"] == "g1")["prediction"]
    move = 0.5 * (learning.logit(0.83) - learning.logit(before["home_win"]))
    assert after["home_win"] == pytest.approx(learning.sigmoid(learning.logit(before["home_win"]) + move))
    assert (after["margin"] - before["margin"]) * (after["home_win"] - before["home_win"]) > 0
    assert any("AI analysts (Claude)" in factor for factor in after["factors"])
    assert after["sources"]["before_analysts"]["home_win_prob"] == pytest.approx(before["home_win"], abs=1e-3)
    # A new forecast is still measured from the AI's own number, not the moved one
    with store._connect() as db:
        db.execute("DELETE FROM analyst_calls")
    desk.analyze(service, "basketball_nba", "g1", only="deepseek", asked=True)
    assert desk.analysts["deepseek"].dossiers[-1]["ai_model"]["home_win"] == pytest.approx(before["home_win"], abs=1e-3)


# --- API, settings ----------------------------------------------------------------------------------

def test_the_ask_endpoint(service, store, desk) -> None:
    client = TestClient(create_app(service, "s3cret", login_delay=0))
    client.post("/api/login", json={"password": "s3cret"})
    asked = client.post("/api/analysts/ask", json={"league": "basketball_nba", "id": "g2", "analyst": "deepseek"})
    assert asked.status_code == 200 and asked.json()["analysts"][1]["forecast"]["home_win_prob"] == 0.88
    again = client.post("/api/analysts/ask", json={"league": "basketball_nba", "id": "g2", "analyst": "deepseek"})
    assert again.status_code == 400 and "30 minutes" in again.json()["detail"]
    assert client.post("/api/analysts/ask", json={"league": "basketball_nba", "id": "g2", "analyst": "gpt"}).status_code == 422
    service.analysts = None
    off = client.post("/api/analysts/ask", json={"league": "basketball_nba", "id": "g2", "analyst": "claude"})
    assert off.status_code == 400 and "ANTHROPIC_API_KEY" in off.json()["detail"]


def test_analysts_are_switched_on_by_their_keys(store) -> None:
    assert build_analysts(Config(TELEGRAM_BOT_TOKEN="123456:test"), store) is None
    desk = build_analysts(Config(TELEGRAM_BOT_TOKEN="123456:test", DEEPSEEK_API_KEY="sk-deepseek-test",
                                 ANTHROPIC_API_KEY="sk-ant-test-key", AI_WEB_SEARCH="1", AI_DAILY_LIMIT="10"), store)
    assert list(desk.analysts) == ["claude", "deepseek"] and desk.daily_limit == 10
    assert desk.analysts["claude"].web_search is True and desk.analysts["claude"].model == "claude-opus-5-5"
    with pytest.raises(ValueError):
        Config(TELEGRAM_BOT_TOKEN="123456:test", AI_EFFORT="ultra")
