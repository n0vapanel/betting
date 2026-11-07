"""StatMuse is a stat lookup beside the ticket. It is not a price."""

from zoneinfo import ZoneInfo

from telegram_betting_bot_simple.ai.assistant import SYSTEM_PROMPT
from telegram_betting_bot_simple.ai.tools import INPUTS, PICK_TOOLS, TOOLS, Toolbox
from telegram_betting_bot_simple.data.apis.statmuse import StatMuseClient


def test_an_nba_question_hits_the_nba_ask_endpoint() -> None:
    seen = {}

    def fetch(url, params=None, headers=None, timeout=10.0, **kwargs):
        seen["url"] = url
        seen["params"] = params
        seen["header"] = (headers or {}).get("x-statmuse-api-key")
        return {
            "type": "stats",
            "answer": {"text": "Shai Gilgeous-Alexander averaged 32.7 points a game."},
            "data": {"results": [{
                "context": [{"type": "player", "longName": "Shai Gilgeous-Alexander"}],
                "stats": {"PointsPerGame": {"value": 32.7, "display": "32.7"}},
            }]},
        }

    out = StatMuseClient("test-key", fetch=fetch).ask("basketball_nba", "  who   led scoring  ")
    assert seen["url"] == "https://api.statmuse.com/sports/nba/v1/ask"
    assert seen["params"] == {"input": "who led scoring"}
    assert seen["header"] == "test-key"
    assert out["supported"] is True and out["domain"] == "nba"
    assert out["answer"] == "Shai Gilgeous-Alexander averaged 32.7 points a game."
    assert out["rows"][0]["who"] == ["Shai Gilgeous-Alexander"]
    assert out["rows"][0]["stats"]["PointsPerGame"] == 32.7
    assert "test-key" not in str(out)


def test_a_fight_and_a_lower_league_are_not_asked() -> None:
    def fetch(*args, **kwargs):
        raise AssertionError("StatMuse was called")

    client = StatMuseClient("test-key", fetch=fetch)
    fight = client.ask("mma_mixed_martial_arts", "who wins")
    mls = client.ask("soccer_usa_mls", "goals leader")
    assert fight["supported"] is False and mls["supported"] is False
    assert fight["error"] == "statmuse_unsupported"


def test_no_key_makes_no_request() -> None:
    def fetch(*args, **kwargs):
        raise AssertionError("StatMuse was called")

    out = StatMuseClient("", fetch=fetch).ask("basketball_nba", "scoring leader")
    assert out["error"] == "statmuse_off"


def test_a_question_is_cut_to_the_api_limit() -> None:
    seen = {}

    def fetch(url, params=None, headers=None, timeout=10.0, **kwargs):
        seen["input"] = params["input"]
        return {"type": "unknown", "answer": {"text": "no"}}

    StatMuseClient("test-key", fetch=fetch).ask("soccer_epl", "x" * 400)
    assert seen["input"] == "x" * 200
    assert StatMuseClient("k", fetch=fetch).ask("soccer_uefa_champs_league", "table")["domain"] == "fc"


def test_the_chat_tool_quotes_stats_and_does_not_price(monkeypatch) -> None:
    def fake_ask(self, league, question):
        assert league == "basketball_nba" and question == "scoring leader"
        return {"supported": True, "domain": "nba", "type": "stats", "answer": "32.7 points a game.", "rows": []}

    monkeypatch.setattr(StatMuseClient, "ask", fake_ask)
    monkeypatch.setenv("STATMUSE_API_KEY", "test-key")
    box = Toolbox(None, "tg:1", ZoneInfo("America/New_York"), paused=True)
    run = box.run("statmuse_ask", {"league": "basketball_nba", "question": "scoring leader"})
    assert run.ok, run.content
    assert run.content["answer"] == "32.7 points a game."
    assert "Not a win probability" in run.content["note"]
    assert "statmuse_ask" not in PICK_TOOLS
    tool = next(item for item in TOOLS if item["name"] == "statmuse_ask")
    assert "win probability" in tool["description"]
    assert INPUTS["statmuse_ask"].model_validate({"league": "basketball_nba", "question": "scoring"}).question == "scoring"
    assert "statmuse_ask looks up a counting stat" in SYSTEM_PROMPT


def test_the_tool_list_grew_by_the_stat_lookup() -> None:
    assert len(TOOLS) == 19
