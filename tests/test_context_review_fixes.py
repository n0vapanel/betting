"""Stage 4 review fixes (python): a retractable roof whose state isn't announced, geocoding by the address's state,
the Builder's Why naming the condition the attribution holds, the AI ticket tools' adjusted note, both unit systems in
the text the server writes for shared entries, the goals model following an adjusted total, Open-Meteo's credit on the
bot's line, an adjusted leg's chance source and its ticket warning, and unit-ordered factor evidence.

The world is test_context_wiring's (the real venue table, Open-Meteo and MLB StatsAPI clients with their HTTP faked,
the real ContextService, slate, DashboardService, Lab and engine): no network."""

from __future__ import annotations

import datetime
import json
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from telegram_betting_bot_simple.ai import tools as tools_mod
from telegram_betting_bot_simple.ai.tools import Toolbox
from telegram_betting_bot_simple.data import context as context_mod
from telegram_betting_bot_simple.data.cache import TTLCache
from telegram_betting_bot_simple.data.context import ContextService, why_line, why_lines
from telegram_betting_bot_simple.data.mlb_stats import parse_weather
from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.data.weather import WeatherClient, region_name
from telegram_betting_bot_simple.engine import context as engine
from telegram_betting_bot_simple.engine.tickets import TicketLeg, price_ticket
from tests import test_context_wiring as wiring
from tests.test_context_wiring import EPL, FIXTURES, MLB, NBA, NFL, NOW, NY, UTC, VENUES, Response, sums

# The wiring tests' fixtures: no network, the default book roles, and the world after its two rounds
no_network, default_roles, world = wiring.no_network, wiring.default_roles, wiring.world

UID = "tg:5151"


def iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hours(start: float, **conditions: float) -> List[Dict[str, Any]]:
    base = {"temp_c": 10.0, "wind_kmh": 5.0, "wind_dir": 270, "gust_kmh": 8.0, "precip_prob": 0, "precip_mm": 0.0}
    return [{"t": datetime.datetime.fromtimestamp(start + h * 3600, UTC).strftime("%Y-%m-%dT%H:%MZ"),
             **{**base, **conditions}} for h in range(-1, 4)]


def lone_service(tmp_path) -> ContextService:
    """A ContextService over the real venue table and a real store, fed forecasts by hand (build())."""
    return ContextService(VENUES, None, None, Store(str(tmp_path / "s.db")), TTLCache(name="fixes"))


def nfl_game(home_id: str, start: float, odds_at: float, total: float = 44.5) -> Dict[str, Any]:
    return {"league": NFL, "id": f"{NFL}:g{home_id}", "odds_id": f"g{home_id}", "status": "pre", "_ts": start,
            "start": iso(start), "odds_as_of": iso(odds_at), "home": {"name": "Home"}, "away": {"name": "Away"},
            "_ids": (home_id, "99"), "_lines": {"spread": -3.5, "total": total, "p_over": 0.5}}


def ready_user(world) -> str:
    world.store.get_or_create_user(UID, now=NOW)
    world.store.confirm_age(UID, now=NOW)
    return UID


# --- 1. a retractable roof with no announced state ------------------------------------------------------------------

def test_a_retractable_nfl_roof_never_announced_moves_and_explains_no_weather(tmp_path) -> None:
    """AT&T Stadium and Lucas Oil (NFL roofs are never announced): a forecast turning windy and freezing after the odds
    gives no weather factor, no weather badge and no adjustment; the panel's line says the state isn't announced.
    Lambeau (open air) in the same forecasts is adjusted."""
    svc = lone_service(tmp_path)
    start = datetime.datetime(2026, 12, 6, 21, 25, tzinfo=UTC).timestamp()
    first, odds_at, later = start - 30 * 3600, start - 24 * 3600, start - 5 * 3600
    seen = {}
    for venue_id, home_id in (("att-stadium", "6"), ("lucas-oil-stadium", "11"), ("lambeau-field", "9")):
        venue, game = VENUES.by_id(venue_id), nfl_game(home_id, start, odds_at)
        svc.build(game, venue, forecast={"hours": hours(start, wind_kmh=14, temp_c=8), "fetched_at": first}, now=first)
        block, _ = svc.build(game, venue, forecast={"hours": hours(start, wind_kmh=32, temp_c=-2), "fetched_at": later},
                             now=later)
        seen[venue_id] = block
    for venue_id in ("att-stadium", "lucas-oil-stadium"):
        block = seen[venue_id]
        assert block["venue"]["roof"] == "retractable" and block["venue"]["roof_state"] is None
        assert block["adjustments"] == [] and block["badges"] == []
        assert block["line"] == {"metric": "Retractable roof: state not announced",
                                 "imperial": "Retractable roof: state not announced"}
        total = block["attribution"]["total"]
        assert [f["key"] for f in total["factors"]] == ["other"] and sums(total)
        assert "wasn't announced" in total["factors"][0]["evidence"]
        assert context_mod.total_adjustment(block) is None
    lambeau = seen["lambeau-field"]
    assert lambeau["badges"][0] == "adj" and lambeau["adjustments"][0]["applied"] == -2.0


def test_mlbs_routine_roof_closed_after_an_unknown_state_moves_nothing(tmp_path) -> None:
    """Chase Field in July: nothing announced when the odds were fetched (so the line holds no weather), then MLB's
    routine "Roof Closed": the roof chip, no adjustment."""
    svc = lone_service(tmp_path)
    start = datetime.datetime(2026, 7, 19, 2, 10, tzinfo=UTC).timestamp()
    first, odds_at, later = start - 20 * 3600, start - 8 * 3600, start - 2 * 3600
    venue = VENUES.by_id("chase-field")
    game = {"league": MLB, "id": f"{MLB}:m1", "odds_id": "m1", "status": "pre", "_ts": start, "start": iso(start),
            "odds_as_of": iso(odds_at), "home": {"name": "Arizona Diamondbacks"}, "away": {"name": "Away"},
            "_ids": ("29", "99"), "_lines": {"total": 9.5, "p_over": 0.5}, "chance": {"home": 0.55, "away": 0.45}}
    forecast = {"hours": hours(start, temp_c=40.5, wind_kmh=12, wind_dir=200), "fetched_at": first}
    before, _ = svc.build(game, venue, forecast=forecast, now=first)
    assert {f["key"] for f in before["attribution"]["total"]["factors"]} == {"park", "other"}
    assert before["badges"] == [] and before["venue"]["roof_state"] is None
    announced = {"weather": parse_weather({"condition": "Roof Closed", "temp": "76", "wind": "0 mph, None"}),
                 "fetched_at": later}
    after, _ = svc.build(game, venue, forecast=forecast, announced=announced, now=later)
    assert after["venue"]["roof_state"] == "closed" and after["badges"] == ["roof"] and after["adjustments"] == []
    assert after["line"]["imperial"] == "Roof closed"


def test_a_roof_announced_open_at_the_odds_then_closed_is_adjusted(world) -> None:
    """The wiring world's Rogers Centre: MLB announced the roof open (sunny, wind out to CF) before the odds were
    fetched, and closed since: newer news, so the weather the line held is taken back."""
    toronto = world.block(MLB, "mlb2")
    [moved] = toronto["adjustments"]
    assert moved["reason"].startswith("The roof was announced closed") and moved["effect_at_odds"] > 0
    assert toronto["badges"][:2] == ["adj", "roof"]


# --- 2. unknown venues placed by the address's state -----------------------------------------------------------------

class Geocoder:
    """Open-Meteo's geocoder from the captured same-name answers (count asked for is recorded)."""

    def __init__(self) -> None:
        self.answers = json.loads((FIXTURES / "geocodes_same_name.json").read_text())["answers"]
        self.calls: List[Dict[str, Any]] = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(dict(params or {}))
        return Response(self.answers.get((params or {}).get("name"), {"results": []}))


def test_region_names_match_whole_states_not_substrings() -> None:
    assert region_name("AR", "USA") == "arkansas" and region_name("MO", "US") == "missouri"
    assert region_name("ON", "Canada") == "ontario" and region_name("Utah", "USA") == "utah"
    assert region_name(None, "England") == "england" and region_name(None, "USA") is None
    assert region_name("ZZ", "USA") is None and region_name("", None) is None


def test_the_geocoder_prefers_the_addresss_state(tmp_path) -> None:
    session = Geocoder()
    weather = WeatherClient(TTLCache(name="geo"), None, session=session)
    plain = weather.geocode("Fayetteville", "USA")
    assert plain["admin1"] == "North Carolina"  # the most populous, as before, without a state
    arkansas = weather.geocode("Fayetteville", "USA", region="AR")
    assert (arkansas["admin1"], arkansas["timezone"]) == ("Arkansas", "America/Chicago")
    assert (round(arkansas["lat"], 2), round(arkansas["lon"], 2)) == (36.06, -94.16)
    assert len(session.calls) == 2 and session.calls[1]["count"] == 30  # its own cache entry, more names asked for
    assert weather.geocode("Fayetteville", "USA", region="AR") == arkansas and len(session.calls) == 2  # kept
    # A state the answers don't hold: the first match, as without one
    assert weather.geocode("Fayetteville", "USA", region="WY")["admin1"] == "North Carolina"


@pytest.mark.parametrize("city, state, admin1, tz", [("Fayetteville", "AR", "Arkansas", "America/Chicago"),
                                                     ("Columbia", "MO", "Missouri", "America/Chicago"),
                                                     ("Manhattan", "KS", "Kansas", "America/Chicago"),
                                                     ("Athens", "OH", "Ohio", "America/New_York")])
def test_an_unknown_college_venue_lands_in_its_state(city, state, admin1, tz) -> None:
    weather = WeatherClient(TTLCache(name="geo"), None, session=Geocoder())
    svc = ContextService(VENUES, weather, None, None, TTLCache(name="ctx"))
    game = {"league": "americanfootball_ncaaf", "id": "x", "odds_id": "o1",
            "venue": {"name": f"{city} Stadium", "city": city, "indoor": False},
            "_place": {"id": f"999{city[:3]}", "state": state, "country": "USA"}, "_ids": ("1", "2")}
    row = svc._venue(game, True)
    assert row["approx"] and row["region"] == admin1 and row["tz"] == tz


# --- 3. the Builder's Why names the condition the attribution holds -----------------------------------------------

def test_the_why_line_names_the_forecast_at_the_odds_time(tmp_path) -> None:
    svc = lone_service(tmp_path)
    start = datetime.datetime(2026, 11, 8, 18, 0, tzinfo=UTC).timestamp()
    first, odds_at, later = start - 30 * 3600, start - 24 * 3600, start - 5 * 3600
    game, venue = nfl_game("9", start, odds_at, total=41.5), VENUES.by_id("lambeau-field")
    block, _ = svc.build(game, venue, forecast={"hours": hours(start, wind_kmh=19.3, temp_c=8), "fetched_at": first},
                         now=first)
    block, _ = svc.build(game, venue, forecast={"hours": hours(start, wind_kmh=19.3, temp_c=8), "fetched_at": odds_at},
                         now=odds_at + 60)
    assert why_line(block, "totals") == "Wind 12 mph at kick-off: the line already has about −2.2 pts for it"
    # The forecast rises to 20 mph: the attribution still holds the 12 mph the odds saw, and the line says so
    block, _ = svc.build(game, venue, forecast={"hours": hours(start, wind_kmh=32.2, temp_c=8), "fetched_at": later},
                         now=later)
    assert [f["delta"] for f in block["attribution"]["total"]["factors"] if f["key"] == "wind"] == [-2.2]
    assert block["weather"]["at_odds"]["changed"] == ["wind"]
    assert why_lines(block, "totals") == {
        "imperial": "Wind 12 mph at kick-off as forecast when the odds were fetched: the line already has about "
                    "−2.2 pts for it",
        "metric": "Wind 19 km/h at kick-off as forecast when the odds were fetched: the line already has about "
                  "−2.2 pts for it"}


# --- 4. the AI ticket tools carry the adjusted note ------------------------------------------------------------------

def test_the_ai_ticket_tools_say_a_leg_was_adjusted(world) -> None:
    uid = ready_user(world)
    box = Toolbox(world.lab, uid, NY, limits={"stake": [Decimal("10")]},
                  now=lambda: datetime.datetime.fromtimestamp(NOW, UTC))
    leg = {"league": NFL, "id": "nfl1", "market": "totals", "side": "under"}
    run = box.run("ticket_price", {"type": "single", "stake": "10", "legs": [leg]})
    assert run.ok, run.content
    [fact] = run.content["ticket"]["legs"]
    assert fact["adjusted"]["applied"] == -2.0 and fact["adjusted"]["reason"].startswith("The forecast changed")
    assert fact["adjusted"]["unit"] == "pts" and fact["ai_chance"] > fact["market_chance"]
    brain = box.run("brain_price_legs", {"legs": [leg]})
    assert brain.ok and brain.content["legs"][0]["adjusted"]["applied"] == -2.0
    plain = box.run("brain_price_legs", {"legs": [{**leg, "id": "nfl2"}]})
    assert plain.ok and "adjusted" not in plain.content["legs"][0]
    assert tools_mod.CHANCE_SOURCES["adjusted"] == "books_adjusted"


# --- 5 and 10. both unit systems in the text the server writes ------------------------------------------------------

def test_shared_legs_carry_both_unit_systems(world) -> None:
    by_key = world.service.builder_candidates(now=NOW)["by_key"]
    over = by_key[f"{NFL}:nfl1:totals:over:41.5"]
    assert set(over.adjusted["reasons"]) == {"metric", "imperial"}
    assert "km/h" in over.adjusted["reasons"]["metric"] and "mph" in over.adjusted["reasons"]["imperial"]
    assert over.why == over.why_units["imperial"]  # the NFL's "auto"
    assert "km/h" in over.why_units["metric"][0] and "mph" not in over.why_units["metric"][0]
    assert over.why_units["metric"][1] == "Freezing at kick-off (-5°C): the line already has about −1.0 pts for it"
    assert by_key[f"{NFL}:nfl2:totals:over:47.5"].why_units is None  # nothing in it depends on units
    # A Builder run's legs carry them too
    client = world.client()
    result = client.post("/api/builder/run", json={"controls": {"preset": "risky", "leagues": [NFL],
                                                                "markets": ["totals"]}}).json()
    legs = [leg for ticket in result.get("tickets") or [] for leg in ticket["legs"] if leg["ref"]["id"] == "nfl1"]
    assert legs and all(set(leg["why_units"]) == {"metric", "imperial"} for leg in legs)
    assert all(set(leg["adjusted"]["reasons"]) == {"metric", "imperial"} for leg in legs)


def test_the_lab_note_follows_the_persons_units(world) -> None:
    client = world.client()
    body = {"type": "single", "stake": "10", "legs": [{"league": NFL, "id": "nfl1", "market": "totals",
                                                        "side": "under"}]}
    imperial = client.post("/api/tickets/analyse", json={**body, "explain": False}).json()
    note = next(n for n in imperial["legs"][0]["notes"] if n.startswith("Adjusted:"))
    assert "mph" in note and "km/h" not in note
    assert client.put("/api/me", json={"units": "metric"}).json()["units"] == "metric"
    metric = client.post("/api/tickets/analyse", json={**body, "explain": False}).json()
    note = next(n for n in metric["legs"][0]["notes"] if n.startswith("Adjusted:"))
    assert "km/h" in note and "mph" not in note
    priced = client.post("/api/tickets/price", json=body).json()
    assert set(priced["legs"][0]["adjusted"]["reasons"]) == {"metric", "imperial"}


# --- 6. the goals model follows an adjusted total --------------------------------------------------------------------

def test_the_goals_model_follows_an_adjusted_soccer_total(world) -> None:
    block = world.block(EPL, "epl1")
    moved = context_mod.total_adjustment(block)
    assert moved is not None and moved["applied"] > 0  # warmer since the odds: more goals
    event = next(g for g in world.odds.saved_games(EPL)[1] if g["id"] == "epl1")
    inputs, derived = world.service.soccer_derived(EPL, event)
    books = inputs["over_2_5_before"]
    assert inputs["over_2_5"] == pytest.approx(context_mod.adjusted_chance(EPL, moved, "over", 2.5, books), abs=1e-12)
    assert inputs["over_2_5"] > books
    from telegram_betting_bot_simple.engine import soccer
    plain = soccer.derive(inputs["home"], inputs["draw"], inputs["away"], books)
    assert sum(derived["rates"].values()) > sum(plain["rates"].values())
    # The game page's goals model says so, with the same chance as the board's adjusted Over 2.5
    card = world.service.game(EPL, "epl1")
    assert card["soccer"]["adjusted"]["applied"] == moved["applied"] and card["soccer"]["inputs"]["over_2_5"] == \
        round(inputs["over_2_5"], 4)
    assert card["markets"]["totals"]["over"]["ai"] == pytest.approx(inputs["over_2_5"], abs=1e-9)
    # The Builder's team totals and BTTS carry the note; draw no bet and double chance (sides) don't
    candidates = [c for c in world.service.builder_candidates(now=NOW)["candidates"] if c.ref["id"] == "epl1"]
    derived_legs = {c.market: c for c in candidates if c.chance_source == "model"}
    assert derived_legs["team_totals"].adjusted["applied"] == moved["applied"]
    assert derived_legs["btts"].adjusted["applied"] == moved["applied"]
    assert all(c.adjusted is None for c in candidates if c.market in ("draw_no_bet", "double_chance"))


# --- 8. an adjusted leg's chance is not the books' -------------------------------------------------------------------

def test_an_adjusted_legs_chance_source_is_not_the_books(world) -> None:
    by_key = world.service.builder_candidates(now=NOW)["by_key"]
    assert by_key[f"{NFL}:nfl1:totals:over:41.5"].chance_source == "adjusted"
    assert by_key[f"{NFL}:nfl1:totals:under:41.5"].chance_source == "adjusted"
    assert by_key[f"{NFL}:nfl2:totals:over:47.5"].chance_source == "books"
    client = world.client()
    result = client.post("/api/builder/run", json={"controls": {"preset": "risky", "leagues": [NFL],
                                                                "markets": ["totals"]}}).json()
    legs = [leg for ticket in result.get("tickets") or [] for leg in ticket["legs"]]
    assert legs and all(leg["chance_source"] == ("adjusted" if leg["ref"]["id"] == "nfl1" else "books")
                        for leg in legs)


# --- 9. an adjusted leg's ticket warning -----------------------------------------------------------------------------

def _leg(ai: float, market: float, adjusted=None) -> TicketLeg:
    return TicketLeg(key="k", league=NFL, event_id="e", market="totals", side="over", point=42.5,
                     price=Decimal("1.91"), label="Over 42.5", game="Bears @ Packers", start="2030-01-01T00:00:00Z",
                     ai_prob=ai, market_prob=market, adjusted=adjusted)


def test_an_adjusted_leg_is_not_the_ai_far_from_the_market() -> None:
    moved = {"applied": -1.9, "reason": "The forecast changed after the odds were fetched.", "before": 0.5}
    warnings = price_ticket([_leg(0.44, 0.5, moved)], "single", 10)["warnings"]
    assert not any("injury news" in w for w in warnings)
    assert ("Over 42.5: chance moved 50% → 44% for a forecast or roof change since the odds (What moves it)."
            in warnings)
    # A model already far from the market before the adjustment still says so
    far = price_ticket([_leg(0.54, 0.5, {**moved, "before": 0.60})], "single", 10)["warnings"]
    assert "The AI is far from the market on Over 42.5: check the injury news." in far
    # Without an adjustment nothing changed
    plain = price_ticket([_leg(0.44, 0.5)], "single", 10)["warnings"]
    assert "The AI is far from the market on Over 42.5: check the injury news." in plain
    assert not any("chance moved" in w for w in plain)


def test_the_slip_and_builder_warn_of_the_adjustment_not_injuries(world) -> None:
    uid = ready_user(world)
    priced = world.lab.price(uid, {"type": "single", "stake": "10",
                                   "legs": [{"league": NFL, "id": "nfl1", "market": "totals", "side": "over"}]})
    assert not any("injury news" in w for w in priced["warnings"])
    assert any(w.startswith("Over 41.5: chance moved 50% → ") for w in priced["warnings"])
    client = world.client()
    result = client.post("/api/builder/run", json={"controls": {"preset": "risky", "leagues": [NFL],
                                                                "markets": ["totals"]}}).json()
    for ticket in result.get("tickets") or []:
        assert not any("injury news" in w for w in ticket["warnings"]), ticket["warnings"]


# --- 11. factor evidence in the person's units -----------------------------------------------------------------------

def test_weather_evidence_comes_in_both_orders() -> None:
    windy = {"wind_kmh": 30.0, "gust_kmh": 43.0, "temp_c": -2.0}
    highmark = VENUES.by_id("highmark-stadium")
    metric = {f.key: f.evidence for f in engine.weather_factors("nfl", highmark, windy, units="metric")}
    imperial = {f.key: f.evidence for f in engine.weather_factors("nfl", highmark, windy, units="imperial")}
    assert metric["wind"].startswith("30 km/h (19 mph) sustained, gusting 43 km/h (27 mph), open-air: 0 at 13 km/h")
    assert imperial["wind"].startswith("19 mph (30 km/h) sustained, gusting 27 mph (43 km/h), open-air: 0 at 8 mph")
    assert metric["cold"].startswith("-2°C (28°F): −1 pt at 0°C (32°F)")
    assert imperial["cold"].startswith("28°F (-2°C): −1 pt at 32°F (0°C)")
    # The sizes never depend on the wording
    assert [f.delta for f in engine.weather_factors("nfl", highmark, windy, units="metric")] == \
        [f.delta for f in engine.weather_factors("nfl", highmark, windy)]
    wrigley = VENUES.by_id("wrigley-field")
    out = engine.weather_factors("mlb", wrigley, {"wind_kmh": 24.0, "wind_dir": (wrigley["azimuth_deg"] + 180) % 360,
                                                  "temp_c": 30.0}, units="metric")
    wind = next(f for f in out if f.key == "wind")
    assert "24 km/h (15 mph) out towards centre field" in wind.evidence
    assert "per mph (1.6 km/h) above 5 mph (8 km/h)" in wind.evidence


def test_the_block_sends_each_factors_evidence_in_both_unit_systems(world) -> None:
    total = world.block(NFL, "nfl1")["attribution"]["total"]
    cold = next(f for f in total["factors"] if f["key"] == "cold")
    assert cold["evidence"] == cold["evidence_units"]["imperial"] and cold["evidence_units"]["metric"].startswith("-5°C")
    other = next(f for f in total["factors"] if f["key"] == "other")
    assert "evidence_units" not in other and sums(total)
    margin = world.block(NBA, "nba1")["attribution"]["margin"]
    altitude = next(f for f in margin["factors"] if f["key"] == "altitude")
    assert "5,1" in altitude["evidence_units"]["imperial"].split(" ft")[0]
    assert altitude["evidence_units"]["metric"] == altitude["evidence"] and sums(margin)
