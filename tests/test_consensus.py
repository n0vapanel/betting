"""The markets brain (engine/consensus.py): no-vig, book consensus and the blend, against hand-computed values."""

from __future__ import annotations

import datetime
import math

import pytest

from telegram_betting_bot_simple.engine.consensus import (
    BASE_RATES,
    DEFAULT_WEIGHTS,
    KINDS,
    SHARP_BOOKS,
    base_rate,
    blend,
    book_consensus,
    book_sources,
    log_pool,
    make_source,
    no_vig,
    prediction_market_source,
    score,
)

AS_OF = "2026-10-27T18:00:00Z"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def src(name, kind, home, away=None, draw=None, status="ok", as_of=AS_OF):
    p = {"home": home, "away": 1 - home if away is None and draw is None else away}
    if draw is not None:
        p["draw"] = draw
    return {"name": name, "kind": kind, "p": p, "as_of": as_of, "status": status, "url": None, "detail": None}


# --- constants -------------------------------------------------------------------------------------------

def test_contract_constants() -> None:
    assert KINDS == ("sharp", "us_books", "prediction_market", "model_rating", "team_model")
    assert DEFAULT_WEIGHTS == {"sharp": .50, "us_books": .20, "prediction_market": .20, "model_rating": .05,
                               "team_model": .05}
    assert math.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0)
    assert SHARP_BOOKS == ("pinnacle", "betfair_ex_eu", "matchbook")


# --- no-vig and the book consensus -------------------------------------------------------------------

def test_no_vig_is_multiplicative() -> None:
    assert no_vig({"home": 1.91, "away": 1.91}) == {"home": 0.5, "away": 0.5}
    # 1/1.5 = .666667 and 1/2.8 = .357143 add up to 1.023810: each is divided by that
    fair = no_vig({"home": 1.5, "away": 2.8})
    assert fair["home"] == pytest.approx(0.651163, abs=1e-6) and fair["away"] == pytest.approx(0.348837, abs=1e-6)
    # Three-way: .476190 + .294118 + .277778 = 1.048086
    fair = no_vig({"home": 2.10, "draw": 3.40, "away": 3.60})
    assert fair == pytest.approx({"home": 0.454343, "draw": 0.280624, "away": 0.265033}, abs=1e-6)
    assert math.isclose(sum(fair.values()), 1.0)


@pytest.mark.parametrize("prices", [{}, {"home": 1.9}, {"home": 1.0, "away": 3.0}, {"home": 0.5, "away": 2.0},
                                    {"home": "x", "away": 2.0}, {"home": None, "away": 2.0},
                                    {"home": float("nan"), "away": 2.0}, None])
def test_no_vig_refuses_prices_it_cannot_read(prices) -> None:
    assert no_vig(prices) == {}


BOOKS = {
    "pinnacle": {"home": 1.70, "away": 2.25},       # no-vig home .569620
    "betfair_ex_eu": {"home": 1.74, "away": 2.30},  # .569307
    "draftkings": {"home": 1.67, "away": 2.20},     # .568475
    "fanduel": {"home": 1.65, "away": 2.25},        # .576923
    "betmgm": {"home": 1.62, "away": 2.30},         # .586735
    "kalshi": {"home": 1.50, "away": 3.00},         # a prediction market listed as a book: not a US book
}


def test_book_consensus_splits_sharp_and_us_books() -> None:
    sharp, us = book_consensus(BOOKS)
    # Sharp: the median of Pinnacle and Betfair (two books: their mean)
    assert sharp["home"] == pytest.approx((0.569620 + 0.569307) / 2, abs=1e-6)
    # US: the median of DraftKings, FanDuel and BetMGM; Kalshi is left out (it has its own source)
    assert us["home"] == pytest.approx(0.576923, abs=1e-6)
    assert sharp["home"] + sharp["away"] == pytest.approx(1.0) and us["home"] + us["away"] == pytest.approx(1.0)
    # Without sharp books there is no sharp figure, and vice versa
    assert book_consensus({k: v for k, v in BOOKS.items() if k not in SHARP_BOOKS})[0] is None
    assert book_consensus({k: BOOKS[k] for k in ("pinnacle",)}) == (
        pytest.approx({"home": 0.569620, "away": 0.430380}, abs=1e-6), None)
    # A custom sharp list
    sharp, us = book_consensus(BOOKS, sharp=("draftkings",))
    assert sharp["home"] == pytest.approx(0.568475, abs=1e-6)
    # ...and every other book is "US": the median of Betfair .569307, Pinnacle .569620, FanDuel .576923 and
    # BetMGM .586735 is the mean of the middle two
    assert us["home"] == pytest.approx((0.569620 + 0.576923) / 2, abs=1e-6)
    assert book_consensus({}) == (None, None)


def test_book_consensus_three_way_and_incomplete_books() -> None:
    books = {
        "pinnacle": {"home": 2.10, "draw": 3.40, "away": 3.60},
        "draftkings": {"home": 2.05, "draw": 3.30, "away": 3.50},
        "fanduel": {"home": 2.00, "draw": 3.40, "away": 3.60},
        "bovada": {"home": 2.10, "away": 3.60},  # no draw price: can't be de-vigged with the others
    }
    sharp, us = book_consensus(books)
    assert sharp == pytest.approx({"home": 0.454343, "draw": 0.280624, "away": 0.265033}, abs=1e-6)
    dk, fd = no_vig(books["draftkings"]), no_vig(books["fanduel"])
    medians = {o: (dk[o] + fd[o]) / 2 for o in ("home", "draw", "away")}
    assert us == pytest.approx({o: v / sum(medians.values()) for o, v in medians.items()})
    assert list(us) == ["home", "away", "draw"]


def test_book_sources_name_the_books() -> None:
    sharp, us = book_sources(BOOKS, AS_OF)
    assert (sharp["name"], sharp["kind"], sharp["status"], sharp["as_of"]) == ("Sharp", "sharp", "ok", AS_OF)
    assert sharp["detail"] == "no-vig: Pinnacle · Betfair"
    assert (us["name"], us["kind"], us["detail"]) == ("US books", "us_books", "median no-vig of 3 books")
    assert us["p"]["home"] == pytest.approx(0.576923, abs=1e-6)
    assert book_sources({"draftkings": BOOKS["draftkings"]})[0]["detail"] == "median no-vig of 1 book"
    assert book_sources({}) == []


# --- sources ------------------------------------------------------------------------------------------

def test_make_source_checks_and_normalises() -> None:
    espn = make_source("ESPN", "model_rating", {"home": 0.62, "away": 0.38, "as_of": AS_OF, "source": "BPI"})
    assert espn == {"name": "ESPN", "kind": "model_rating", "p": {"home": 0.62, "away": 0.38}, "as_of": AS_OF,
                    "status": "ok", "url": None, "detail": None}
    # Prices that don't quite add up are rescaled
    assert make_source("X", "sharp", {"home": 0.51, "away": 0.51})["p"] == {"home": 0.5, "away": 0.5}
    # No usable chance: "missing" (and never used)
    assert make_source("Elo", "model_rating", None)["status"] == "missing"
    assert make_source("Elo", "model_rating", {"home": 0.6})["status"] == "missing"
    assert make_source("Elo", "model_rating", {"home": "?", "away": 0.4})["status"] == "missing"
    stamped = make_source("Model", "team_model", {"home": 0.5, "away": 0.5},
                          datetime.datetime(2026, 10, 27, 18, tzinfo=datetime.timezone.utc).timestamp())
    assert stamped["as_of"] == AS_OF
    with pytest.raises(ValueError):
        make_source("X", "tipster", {"home": 0.5, "away": 0.5})
    with pytest.raises(ValueError):
        make_source("X", "sharp", {"home": 0.5, "away": 0.5}, status="great")


def quote(p, thin=False, liquidity=25000.0):
    return {"p": p, "bid": p - 0.01, "ask": p + 0.01, "last": p, "liquidity": liquidity, "volume": 1e5, "thin": thin}


def test_prediction_market_source_from_a_market_event() -> None:
    event = {"source": "kalshi", "league": "basketball_nba", "home": "BOS", "away": "NYK", "start": None,
             "url": "https://kalshi.com/markets/kxnbagame", "as_of": AS_OF,
             "markets": {"h2h": {"home": quote(0.63), "away": quote(0.38, liquidity=12000.0)}}}
    source = prediction_market_source(event)
    assert (source["name"], source["kind"], source["status"], source["url"]) == (
        "Kalshi", "prediction_market", "ok", "https://kalshi.com/markets/kxnbagame")
    assert source["p"]["home"] == pytest.approx(0.63 / 1.01)
    assert source["detail"] == "mid price; $12,000 liquidity"
    flipped = prediction_market_source(event, flip=True)
    assert flipped["p"]["home"] == pytest.approx(0.38 / 1.01)
    thin = {**event, "source": "polymarket", "markets": {"h2h": {"home": quote(0.5, thin=True), "away": quote(0.5)}}}
    assert prediction_market_source(thin)["status"] == "thin" and prediction_market_source(thin)["name"] == "Polymarket"
    assert prediction_market_source({**event, "markets": {}}) is None
    soccer = {**event, "markets": {"h2h": {"home": quote(0.45), "away": quote(0.28), "draw": quote(0.27)}}}
    assert set(prediction_market_source(soccer)["p"]) == {"home", "away", "draw"}


# --- the blend ---------------------------------------------------------------------------------------

def test_two_way_blend_against_a_hand_computed_value() -> None:
    sources = [src("Sharp", "sharp", 0.60), src("US books", "us_books", 0.58),
               src("Kalshi", "prediction_market", 0.62), src("Polymarket", "prediction_market", 0.64),
               src("ESPN", "model_rating", 0.55), src("Model", "team_model", 0.70)]
    result = blend(sources, "basketball_nba")
    # logit p = .5 logit .60 + .2 logit .58 + .1 logit .62 + .1 logit .64 + .05 logit .55 + .05 logit .70
    #         = .2027326 + .0645547 + .0489548 + .0575364 + .0100335 + .0423649 = .4261769 -> p = .604960
    assert result["p"]["home"] == pytest.approx(0.604960, abs=1e-6)
    assert result["p"]["home"] + result["p"]["away"] == pytest.approx(1.0)
    expected = 0.5 * logit(.60) + 0.2 * logit(.58) + 0.1 * logit(.62) + 0.1 * logit(.64) + 0.05 * logit(.55) \
        + 0.05 * logit(.70)
    assert logit(result["p"]["home"]) == pytest.approx(expected)
    # Kalshi and Polymarket share the prediction-market weight
    assert result["weights"] == pytest.approx({"Sharp": .5, "US books": .2, "Kalshi": .1, "Polymarket": .1,
                                               "ESPN": .05, "Model": .05})
    assert result["used"] == ["Sharp", "US books", "Kalshi", "Polymarket", "ESPN", "Model"]
    assert result["confidence"] == "normal" and result["n_sources"] == 6 and result["base"] is None
    # Model 70% against ESPN 55%: a 15-point spread
    assert result["spread_pp"] == 15.0 and result["disagree"] is True
    assert result["as_of"] == AS_OF and result["near_start"] is False


def test_three_way_blend_is_a_normalised_log_linear_pool() -> None:
    sources = [src("Sharp", "sharp", .45, .27, .28), src("US books", "us_books", .47, .26, .27),
               src("Polymarket", "prediction_market", .44, .27, .29),
               src("Elo", "model_rating", .60, .40)]  # two-way: can't join a three-way pool
    result = blend(sources, "soccer_epl")
    # Weights .5/.2/.2 renormalised over .9; p_k proportional to prod p_sk^w_s
    assert result["p"] == pytest.approx({"home": 0.452209, "away": 0.267806, "draw": 0.279984}, abs=1e-6)
    assert sum(result["p"].values()) == pytest.approx(1.0)
    assert result["weights"] == pytest.approx({"Sharp": .5 / .9, "US books": .2 / .9, "Polymarket": .2 / .9})
    assert "Elo" not in result["used"]
    assert result["spread_pp"] == 3.0 and result["disagree"] is False


def test_log_pool_two_way_matches_the_logit_formula() -> None:
    pooled = log_pool([{"home": .7, "away": .3}, {"home": .4, "away": .6}], [.25, .75])
    assert logit(pooled["home"]) == pytest.approx(.25 * logit(.7) + .75 * logit(.4))


def test_weights_renormalise_over_the_kinds_present() -> None:
    sources = [src("Sharp", "sharp", 0.60), src("Kalshi", "prediction_market", 0.55),
               src("Polymarket", "prediction_market", 0.40, status="thin"),  # thin: not used
               src("US books", "us_books", 0.50, status="stale"),             # stale: not used
               src("ESPN", "model_rating", 0.90, status="missing")]
    result = blend(sources, "basketball_nba")
    # Only sharp (.5) and one prediction market (.2) are left: .5/.7 and .2/.7
    assert result["weights"] == pytest.approx({"Sharp": 0.714286, "Kalshi": 0.285714}, abs=1e-6)
    assert result["p"]["home"] == pytest.approx(0.585878, abs=1e-6)
    assert result["used"] == ["Sharp", "Kalshi"] and result["n_sources"] == 2
    # The ignored sources don't count toward disagreement either: 60 vs 55 is exactly 5 points
    assert result["spread_pp"] == 5.0 and result["disagree"] is True
    # One source alone carries everything
    alone = blend([src("US books", "us_books", 0.58)], "basketball_nba")
    assert alone["p"]["home"] == pytest.approx(0.58) and alone["weights"] == {"US books": 1.0}
    assert alone["spread_pp"] == 0.0 and alone["disagree"] is False


def test_sources_of_one_kind_share_its_weight() -> None:
    one = blend([src("Sharp", "sharp", 0.6), src("Kalshi", "prediction_market", 0.5)], "nba")
    two = blend([src("Sharp", "sharp", 0.6), src("Kalshi", "prediction_market", 0.5),
                 src("Polymarket", "prediction_market", 0.5)], "nba")
    assert one["weights"]["Kalshi"] == pytest.approx(2 / 7)
    assert two["weights"]["Kalshi"] == two["weights"]["Polymarket"] == pytest.approx(1 / 7)
    # Two identical markets weigh exactly what one did
    assert two["p"]["home"] == pytest.approx(one["p"]["home"])
    # Three ratings share .05
    three = blend([src("Sharp", "sharp", 0.6)] + [src(n, "model_rating", 0.5) for n in ("ESPN", "Elo", "Torvik")], "nba")
    assert three["weights"]["ESPN"] == pytest.approx(0.05 / 3 / 0.55)


def test_prediction_markets_get_half_weight_near_the_start() -> None:
    sources = [src("Sharp", "sharp", 0.60), src("Kalshi", "prediction_market", 0.55)]
    start = datetime.datetime(2026, 10, 27, 23, 0, tzinfo=datetime.timezone.utc)
    early = blend(sources, "nba", now=start - datetime.timedelta(minutes=20), start=start)
    assert early["weights"]["Kalshi"] == pytest.approx(0.2 / 0.7) and early["near_start"] is False
    for minutes in (14, 0, -30):  # within 15 minutes, at the start and after it
        late = blend(sources, "nba", now=start - datetime.timedelta(minutes=minutes), start=start)
        # .5 and .1 -> 5/6 and 1/6
        assert late["weights"] == pytest.approx({"Sharp": 5 / 6, "Kalshi": 1 / 6})
        assert late["p"]["home"] == pytest.approx(0.591781, abs=1e-6) and late["near_start"] is True
    # ISO strings and epoch seconds work the same
    iso = blend(sources, "nba", now="2026-10-27T22:50:00Z", start="2026-10-27T23:00:00Z")
    epoch = blend(sources, "nba", now=start.timestamp() - 600, start=start.timestamp())
    assert iso["weights"]["Kalshi"] == epoch["weights"]["Kalshi"] == pytest.approx(1 / 6)
    # No start time: no down-weighting; a prediction market alone keeps all the weight
    assert blend(sources, "nba", now=start, start=None)["near_start"] is False
    solo = blend([src("Kalshi", "prediction_market", 0.55)], "nba", now=start, start=start)
    assert solo["weights"] == {"Kalshi": 1.0}


def test_low_confidence_fallback_shrinks_ratings_toward_the_base_rate() -> None:
    sources = [src("ESPN", "model_rating", 0.60), src("Elo", "model_rating", 0.70),
               src("Sharp", "sharp", 0.90, status="stale"), src("Kalshi", "prediction_market", 0.2, status="thin")]
    result = blend(sources, "basketball_nba")
    # Equal weights: logit p = (logit .6 + logit .7) / 2 -> .651669; then 75% of that + 25% of NBA's 55.4%
    assert result["confidence"] == "low"
    assert result["weights"] == {"ESPN": 0.5, "Elo": 0.5}
    assert result["p"]["home"] == pytest.approx(0.75 * 0.651669 + 0.25 * 0.554, abs=1e-6)
    assert result["p"]["home"] == pytest.approx(0.627251, abs=1e-6)
    assert result["base"] == {"home": 0.554, "away": 0.446}
    assert result["spread_pp"] == 10.0 and result["disagree"] is True
    # Our model counts as a rating when there's no market, with an equal say
    with_model = blend(sources + [src("Model", "team_model", 0.65)], "basketball_nba")
    assert with_model["weights"] == pytest.approx({"ESPN": 1 / 3, "Elo": 1 / 3, "Model": 1 / 3})
    # Soccer: 45/27/28
    soccer = blend([src("Elo", "model_rating", .50, .25, .25)], "soccer_spain_la_liga")
    assert soccer["p"] == pytest.approx({"home": .4875, "away": .2575, "draw": .255})
    # Neutral sites and sports without a base rate shrink toward an even split
    neutral = blend([src("FPI", "model_rating", 0.70)], "americanfootball_nfl", neutral=True)
    assert neutral["p"]["home"] == pytest.approx(0.75 * 0.70 + 0.25 * 0.5)
    tennis = blend([src("Elo", "model_rating", 0.70)], "tennis_atp_paris")
    assert tennis["p"]["home"] == pytest.approx(0.65) and tennis["base"] == {"home": 0.5, "away": 0.5}
    nfl = blend([src("FPI", "model_rating", 0.70)], "americanfootball_nfl")
    assert nfl["p"]["home"] == pytest.approx(0.75 * 0.70 + 0.25 * 0.539)


def test_base_rates() -> None:
    assert base_rate("basketball_nba") == {"home": 0.554, "away": 0.446}
    assert base_rate("nhl") == {"home": 0.522, "away": 0.478}
    assert base_rate("baseball_mlb")["home"] == 0.529
    assert base_rate("soccer_epl", ("home", "away", "draw")) == {"home": 0.45, "away": 0.28, "draw": 0.27}
    assert base_rate("soccer_epl") == {"home": 0.5, "away": 0.5}  # a two-way soccer market: no base rate for it
    assert base_rate(None) == {"home": 0.5, "away": 0.5}
    assert all(sum(rate.values()) == pytest.approx(1.0) for rate in BASE_RATES.values())


def test_disagreement_rule() -> None:
    def pair(a, b):
        return blend([src("Sharp", "sharp", a), src("US books", "us_books", b)], "nba")
    assert pair(0.60, 0.55)["disagree"] is True       # 5 points exactly, despite float rounding
    assert pair(0.60, 0.551)["disagree"] is False      # 4.9 points
    assert pair(0.60, 0.551)["spread_pp"] == 4.9
    assert pair(0.30, 0.40)["spread_pp"] == 10.0
    # Any outcome counts: the draw here
    soccer = blend([src("Sharp", "sharp", .45, .30, .25), src("US books", "us_books", .45, .36, .19)], "soccer_epl")
    assert soccer["spread_pp"] == 6.0 and soccer["disagree"] is True


def test_blend_edge_cases() -> None:
    assert blend([], "nba") is None
    assert blend([src("Sharp", "sharp", 0.6, status="stale")], "nba") is None
    assert blend([{"name": "Tipster", "kind": "tipster", "p": {"home": .9, "away": .1}, "status": "ok"}], "nba") is None
    # Weight overrides merge over the defaults; a kind given no weight isn't used
    sources = [src("Sharp", "sharp", 0.60), src("US books", "us_books", 0.50)]
    custom = blend(sources, "nba", weights={"us_books": 0.0})
    assert custom["used"] == ["Sharp"] and custom["p"]["home"] == pytest.approx(0.60)
    assert blend(sources, "nba", weights={"sharp": 0, "us_books": 0})["weights"] == {"Sharp": .5, "US books": .5}
    # The blend is as of its newest source
    stamped = [src("Sharp", "sharp", 0.6, as_of="2026-10-27T17:00:00Z"),
               src("Kalshi", "prediction_market", 0.6, as_of="2026-10-27T18:05:00+00:00")]
    assert blend(stamped, "nba")["as_of"] == "2026-10-27T18:05:00+00:00"
    assert blend([src("Sharp", "sharp", 0.6, as_of=None)], "nba", now=0)["as_of"] == "1970-01-01T00:00:00Z"
    # Chances of 0 or 1 are kept just inside the interval
    extreme = blend([src("Sharp", "sharp", 1.0, 0.0), src("US books", "us_books", 0.5)], "nba")
    assert 0.99 < extreme["p"]["home"] < 1.0


def test_score_brier_and_log_loss() -> None:
    two = score({"home": 0.7, "away": 0.3}, "home")
    assert two["brier"] == pytest.approx(0.09) and two["log_loss"] == pytest.approx(-math.log(0.7))
    assert score({"home": 0.7, "away": 0.3}, "away")["brier"] == pytest.approx(0.49)
    three = score({"home": 0.5, "draw": 0.3, "away": 0.2}, "draw")
    assert three["brier"] == pytest.approx(0.5 * (0.25 + 0.49 + 0.04))
    assert three["log_loss"] == pytest.approx(-math.log(0.3))
    assert score({"home": 0.6, "away": 0.4}, "draw") is None  # a tie on a two-way market isn't graded
    assert score({"home": 1.0, "away": 0.0}, "away")["log_loss"] == pytest.approx(-math.log(1e-4))
