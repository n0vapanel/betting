"""The learning loop's evaluation: byte-identical defaults, scoring rules, paired tests, calibration, the leakage
checklist, the replays through the real engine and the weekly scores (no network)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sqlite3

import numpy as np
import pytest

import learn_sim as sim
from telegram_betting_bot_simple.engine import consensus, context, nba_model, soccer
from telegram_betting_bot_simple.learn import brain_store as bs
from telegram_betting_bot_simple.learn import evaluate as ev
from telegram_betting_bot_simple.learn import registry, stats

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "learn_eval", "defaults_golden.json")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN) as handle:
        return json.load(handle)


@pytest.fixture(autouse=True)
def unbound_registry():
    registry.bind(None)
    yield
    registry.bind(None)


def _dump(x):
    if dataclasses.is_dataclass(x):
        x = dataclasses.asdict(x)
    return json.dumps(x, sort_keys=True, default=str)


# --- byte-identical defaults ---------------------------------------------------------------------------------------

def _empty_store(tmp_path):
    path = str(tmp_path / "brain.db")
    with sqlite3.connect(path) as db:
        bs.init_brain_schema(db)
    return path


@pytest.mark.parametrize("bound", [False, True])
def test_blend_defaults_byte_identical(golden, tmp_path, bound):
    """With an empty registry (no store, or a store with no versions) every blend is byte-identical to the engine
    before the learning loop (golden outputs captured from b9e3111), through the kwargs the wiring passes."""
    if bound:
        registry.bind(_empty_store(tmp_path))
    for case in golden["blend"]:
        params = registry.current(case["league"])
        assert params.version == 0 and params.blend_kwargs() == {}
        out = consensus.blend(case["sources"], case["league"], now=case["now"], start=case["start"],
                              neutral=case["neutral"], **params.blend_kwargs())
        assert json.dumps(out, sort_keys=True) == case["out"]


def test_blend_explicit_defaults_identical(golden):
    """Passing today's values explicitly (sharpness 1 is a no-op) changes nothing either."""
    for case in golden["blend"]:
        out = consensus.blend(case["sources"], case["league"], now=case["now"], start=case["start"],
                              neutral=case["neutral"], weights=dict(consensus.DEFAULT_WEIGHTS), sharpness=1.0,
                              near_start_factor=consensus.NEAR_START_FACTOR,
                              near_start_seconds=consensus.NEAR_START_SECONDS,
                              low_confidence_shrink=consensus.LOW_CONFIDENCE_SHRINK)
        assert json.dumps(out, sort_keys=True) == case["out"]


def test_soccer_defaults_byte_identical(golden):
    params = registry.current("soccer_epl")
    assert params.soccer_kwargs() == {}
    for case in golden["soccer"]:
        h, d, a, o = case["in"]
        for derived in (soccer.derive(h, d, a, o, **params.soccer_kwargs()),
                        soccer.derive(h, d, a, o, rho=params.rho, nudge=params.btts_nudge)):
            assert json.dumps(derived, sort_keys=True) == case["derive"]
            legs = list(soccer._legs(derived, "Home FC", "Away FC"))
            assert json.dumps(legs, sort_keys=True, default=str) == case["legs"]


def test_context_defaults_byte_identical(golden):
    """The context layer through a registry SportParams (the SPORTS entry itself by default) and its weight."""
    for case in golden["context"]:
        params = registry.current(case["sport"])
        sp = params.context_params()
        assert sp is context.SPORTS[case["sport"]]
        assert params.adjust_weight is context.ADJUST_WEIGHT
        args = case["args"]
        fn = case["fn"]
        if fn == "adjustment":
            out = context.adjustment(sp, "total", args[0], 0.0, odds_at=1_793_000_000, weight=params.adjust_weight)
        elif fn == "adjustment_side_full":
            out = context.adjustment(sp, "side", args[0], 0.0, has_market=False, weight=params.adjust_weight)
        elif fn == "adjusted_over":
            out = context.adjusted_over(sp, *args)
        elif fn == "implied_margin":
            out = context.implied_margin(sp, args[0])
        elif fn == "margin_to_prob":
            out = context.margin_to_prob(sp, args[0])
        elif fn == "home_edge":
            out = context.home_edge(sp)
        elif fn == "attribute_margin":
            out = context.attribute_margin(sp, market_margin=args[0], as_of="2026-10-27T12:00:00Z")
        else:
            out = context.attribute_total(sp, market_total=args[0], as_of="2026-10-27T12:00:00Z")
        assert _dump(out) == case["out"], (case["sport"], fn)


def test_nba_params_default_identity():
    params = registry.current("basketball_nba")
    assert params.nba_overrides() == {}
    assert params.nba_params() == nba_model.Params()
    base = nba_model.Params(home_court=2.5)
    assert params.nba_params(base) is base


def test_tuned_values_reach_the_engine():
    values = registry.defaults("soccer")
    values.update({"soccer.rho": -0.11, "soccer.btts_nudge": 0.01, "context.total_sd": 1.8,
                   "context.adjust_weight": 0.6, "consensus.sharpness": 1.05})
    params = registry.Params("soccer", values, version=7)
    assert params.soccer_kwargs() == {"rho": -0.11, "nudge": 0.01}
    assert params.blend_kwargs() == {"sharpness": 1.05}
    assert params.context_params().total_sd == 1.8 and params.context_params().margin_sd == 1.7
    assert params.adjust_weight == 0.6
    derived = soccer.derive(0.45, 0.27, 0.28, 0.52, **params.soccer_kwargs())
    assert derived["rho"] == -0.11 and derived["btts_nudge"] == 0.01
    legs = list(soccer._legs(derived, "H", "A"))
    assert "plus a 1-point calibration nudge" in legs[0][6]


def test_sharpness_extremises_the_pool():
    sources = [{"name": "Sharp", "kind": "sharp", "p": {"home": 0.62, "away": 0.38}, "status": "ok"},
               {"name": "Books", "kind": "us_books", "p": {"home": 0.6, "away": 0.4}, "status": "ok"}]
    plain = consensus.blend(sources, "basketball_nba", now=0, start=10_000)
    sharper = consensus.blend(sources, "basketball_nba", now=0, start=10_000, sharpness=1.2)
    assert "sharpness" not in plain and sharper["sharpness"] == 1.2
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    assert logit(sharper["p"]["home"]) == pytest.approx(1.2 * logit(plain["p"]["home"]), rel=1e-12)
    three = [{"name": "Sharp", "kind": "sharp", "p": {"home": 0.5, "draw": 0.27, "away": 0.23}, "status": "ok"}]
    a, b = consensus.blend(three, "soccer_epl", now=0, start=1), consensus.blend(three, "soccer_epl", now=0, start=1,
                                                                                   sharpness=0.9)
    ratio = math.log(b["p"]["home"] / b["p"]["away"]) / math.log(a["p"]["home"] / a["p"]["away"])
    assert ratio == pytest.approx(0.9, rel=1e-9)
    with pytest.raises(ValueError):
        consensus.blend(sources, "basketball_nba", sharpness=0)


# --- scoring rules -------------------------------------------------------------------------------------------------

def test_lgamma_and_nb_pmf():
    x = np.array([0.5, 1.0, 3.3, 17.0, 250.5])
    assert np.max(np.abs(stats.lgamma(x) - np.array([math.lgamma(v) for v in x]))) < 1e-10
    table = stats.nb_pmf_table([4.5, 1.2], [8.0, 40.0], 120)
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-9)
    k = np.arange(121)
    assert table[0] @ k == pytest.approx(4.5, rel=1e-8)
    assert table[0] @ (k ** 2) - 4.5 ** 2 == pytest.approx(4.5 + 4.5 ** 2 / 8.0, rel=1e-6)


def test_count_scores_crps_pit_and_variance_ratio():
    rng = np.random.default_rng(5)
    mu = rng.uniform(2, 8, 4000)
    y = rng.poisson(mu)
    table = np.exp(stats.poisson_logpmf(np.arange(40)[None, :], mu[:, None]))
    scored = stats.count_scores(table, y)
    assert scored["variance_ratio"] == pytest.approx(1.0, abs=0.08)
    assert max(abs(v - 1) for v in scored["pit"]) < 0.12  # calibrated: a flat PIT histogram
    # CRPS by brute force on one forecast
    cdf = np.cumsum(table[0])
    brute = sum((cdf[k] - (y[0] <= k)) ** 2 for k in range(40))
    assert scored["per_game_crps"][0] == pytest.approx(brute)
    # Too confident (half the spread): a U-shaped PIT
    narrow = stats.nb_pmf_table(mu, 500.0, 39)
    wide_y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    wide_y = np.minimum(wide_y, 39)
    u = stats.count_scores(narrow, wide_y)["pit"]
    assert u[0] > 1.3 and u[-1] > 1.3 and u[5] < 0.9


def test_rps_and_log_loss():
    assert stats.rps([1.0, 0.0, 0.0], 0) == 0.0
    assert stats.rps([0.5, 0.3, 0.2], 2) == pytest.approx(((0.5) ** 2 + (0.8) ** 2) / 2)
    assert stats.log_loss([0.8], [1])[0] == pytest.approx(-math.log(0.8))
    assert stats.log_loss([1.0], [0])[0] == pytest.approx(-math.log(stats.P_FLOOR))


def test_t_cdf_and_betainc():
    assert stats.t_cdf(2.0, 10) == pytest.approx(0.963306, abs=1e-6)
    # t with 3 df has a closed form
    t = -1.5
    closed = 0.5 + (math.atan(t / math.sqrt(3)) + (t / math.sqrt(3)) / (1 + t * t / 3)) / math.pi
    assert stats.t_cdf(t, 3) == pytest.approx(closed, abs=1e-12)
    assert stats.t_cdf(0.0, 7) == pytest.approx(0.5)
    assert stats.t_cdf(1.96, 1e6) == pytest.approx(0.975, abs=1e-4)


def test_diebold_mariano_hln_matches_the_formula():
    rng = np.random.default_rng(2)
    d = rng.normal(-0.001, 0.03, 600)
    days = [f"d{i // 10:03d}" for i in range(600)]
    out = stats.diebold_mariano(d, days, lags=0)
    sums = np.array([d[i:i + 10].sum() for i in range(0, 600, 10)])
    T = sums.size
    dm = sums.mean() / math.sqrt(sums.var() / T)
    hln = dm * math.sqrt((T + 1 - 2 + 0) / T)
    assert out["days"] == T and out["dm"] == pytest.approx(dm) and out["stat"] == pytest.approx(hln)
    assert out["p"] == pytest.approx(stats.t_cdf(hln, T - 1))
    assert out["p"] + out["p_worse"] == pytest.approx(1.0)


def test_diebold_mariano_size_on_noise():
    """On pure noise the one-sided test rejects at about its design rate (5%)."""
    rejected = 0
    runs = 400
    for seed in range(runs):
        rng = np.random.default_rng(1000 + seed)
        d = rng.normal(0, 0.03, 500) + np.repeat(rng.normal(0, 0.01, 50), 10)  # day effects
        days = [i // 10 for i in range(500)]
        rejected += stats.diebold_mariano(d, days)["p"] < 0.05
    assert 0.02 <= rejected / runs <= 0.085


def test_block_bootstrap_coverage():
    covered = 0
    runs = 150
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        d = 0.002 + rng.normal(0, 0.03, 400) + np.repeat(rng.normal(0, 0.01, 40), 10)
        out = stats.block_bootstrap(d, [i // 10 for i in range(400)], reps=500, seed=seed)
        covered += out["lo"] <= 0.002 <= out["hi"]
    assert 0.88 <= covered / runs <= 0.99


def test_weekly_ranges_hold_their_level_with_few_game_days():
    """A week's 95% range (the score-over-time band) covers the true mean about 95% of the time with 3 or 7 game days
    (a percentile bootstrap over so few days covered 49-88%), and a 2-day week shows no range at all."""
    shapes = {"nfl": [1, 13, 1], "nba": [7] * 7, "mlb": [14] * 7, "epl3": [4, 5, 1]}
    for name, per_day in shapes.items():
        covered = 0
        runs = 400
        for seed in range(runs):
            rng = np.random.default_rng(seed)
            days = [f"d{i}" for i, k in enumerate(per_day) for _ in range(k)]
            out = stats.t_interval(rng.normal(0, 0.03, len(days)), days)
            covered += out["lo"] <= 0 <= out["hi"]
        assert 0.92 <= covered / runs <= 0.98, (name, covered / runs)
    two = stats.t_interval(np.random.default_rng(0).normal(0, 0.03, 10), ["sat"] * 6 + ["sun"] * 4)
    assert two["mean"] is not None and two["lo"] is None and two["hi"] is None and two["days"] == 2
    # With ten or more days the range is day-clustered: a shared day effect widens it
    rng = np.random.default_rng(1)
    d = rng.normal(0, 0.03, 300) + np.repeat(rng.normal(0, 0.02, 30), 10)
    days = [i // 10 for i in range(300)]
    clustered = stats.t_interval(d, days)
    plain = stats.t_interval(d, list(range(300)))
    assert clustered["hi"] - clustered["lo"] > plain["hi"] - plain["lo"]
    assert stats.t_ppf(0.975, 1) == pytest.approx(12.706, abs=1e-3)
    assert stats.t_ppf(0.975, 30) == pytest.approx(2.042, abs=1e-3)


def test_holm_textbook():
    adjusted, rejected = stats.holm([0.01, 0.04, 0.03, 0.005], 0.05)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rejected == [True, False, False, True]
    adjusted, rejected = stats.holm([None, 0.2], 0.05)
    assert adjusted == [None, 0.2] and rejected == [False, False]


def test_cusum_fires_on_drift_not_on_noise():
    rng = np.random.default_rng(3)
    quiet = stats.cusum(rng.normal(0, 1, 60), 0.25, 5.0)
    drift = stats.cusum(rng.normal(0.8, 1, 60), 0.25, 5.0)
    assert not quiet["fired"] and drift["fired"] and drift["at"] < 20
    assert stats.cusum([6.0], 0.25, 5.0)["fired"]


def test_calibration_slope_and_intercept():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.15, 0.85, 4000)
    y = (rng.random(4000) < p).astype(float)
    good = stats.calibration(p, y)
    assert good["slope_lo"] < 1 < good["slope_hi"] and good["intercept_lo"] < 0 < good["intercept_hi"]
    over = stats.sigmoid(1.6 * stats.logit(p))  # too confident
    bad = stats.calibration(over, y)
    assert bad["slope_hi"] < 1
    assert stats.calibration(p[:20], y[:20])["slope"] is None


def test_reliability_curve_band_holds_a_calibrated_forecaster():
    rng = np.random.default_rng(6)
    p = rng.uniform(0.2, 0.8, 3000)
    y = (rng.random(3000) < p).astype(float)
    curve = stats.reliability(p, y, points=11, reps=100)
    inside = sum(lo - 0.02 <= x <= hi + 0.02 for x, lo, hi in zip(curve["x"][1:-1], curve["lo"][1:-1],
                                                                   curve["hi"][1:-1]))
    assert inside >= 7
    assert curve["mcb"] < 0.01 and curve["dsc"] > 0
    assert curve["mcb"] - curve["dsc"] + curve["unc"] == pytest.approx(float(np.mean((p - y) ** 2)), abs=1e-9)


# --- the leakage checklist -----------------------------------------------------------------------------------------

def test_weekly_folds_are_clean():
    rows = sim.consensus_world(1200, seed=1)
    folds = list(ev.weekly_folds(rows, min_train=300))
    assert folds
    for week, train, test in folds:
        assert len(train) >= 300
        assert ev.leakage_problems(train, test) == []
        assert all(stats.week_of(r["start"]) == week for r in test)
        assert {stats.day_of(r["start"]) for r in train}.isdisjoint({stats.day_of(r["start"]) for r in test})


def test_leakage_checklist_catches_planted_leaks():
    rows = sim.consensus_world(400, seed=2)
    train, test = rows[:300], rows[300:]
    late = dict(test[0], logged_at=test[0]["start"] + 1)
    assert any("at or after its start" in p for p in ev.leakage_problems(train, [late] + test[1:]))
    shuffled = list(test)
    shuffled[0], shuffled[5] = shuffled[5], shuffled[0]
    assert any("shuffled" in p for p in ev.leakage_problems(train, shuffled))
    same_day = [r for r in rows if r["day"] == rows[299]["day"]]
    split_train = [r for r in rows if r["start"] <= rows[299]["start"]]
    split_test = [r for r in rows if r["start"] > rows[299]["start"]]
    if any(r["day"] == rows[299]["day"] for r in split_test):
        assert any("day split" in p for p in ev.leakage_problems(split_train, split_test))
    assert len(same_day) > 1
    assert any("train" in p for p in ev.leakage_problems(test, train))
    ai = [{"game_key": "x", "start": 100.0, "asked_at": 150.0}]
    assert any("AI forecast" in p for p in ev.leakage_problems(ai=ai))
    proposals = [{"id": "p1", "created_at": 200.0, "judged_on": [150.0, 250.0]}]
    assert any("before it was made" in p for p in ev.leakage_problems(proposals=proposals))
    match = sim.matches_world(1, seed=1)[0]
    match.lines_as_of = match.start
    assert ev.leakage_problems([], [match])


# --- records, replays, scores --------------------------------------------------------------------------------------

def test_records_from_a_real_store(tmp_path):
    from telegram_betting_bot_simple.data.store import Store

    store = Store(str(tmp_path / "betting.db"))
    rows = sim.consensus_world(30, seed=3, start=2_000_000_000.0)
    for r in rows:
        written = store.log_consensus([{"league": r["league"], "game_key": r["game_key"], "home": "H", "away": "A",
                                        "start": consensus._iso(r["start"]), "sources": r["sources"],
                                        "blend": r["blend"]}], now=r["logged_at"])
        assert written == 1
        assert store.grade_consensus(r["league"], r["game_key"], "graded", r["outcome"], r["result"],
                                     now=r["start"] + 3 * 3600)
    with store._connect() as db:
        got = ev.consensus_records(db)
    assert [g["game_key"] for g in got] == [r["game_key"] for r in rows]
    assert got[0]["sport"] == "nba" and got[0]["params_version"] == 0
    assert got[0]["blend"]["p"] == pytest.approx(rows[0]["blend"]["p"])
    # Replaying at the defaults reproduces the stored blend exactly
    for r in got:
        assert ev.reblend(r, {}) == r["blend"]["p"]


def test_replays_match_the_logged_rows():
    rows = sim.consensus_world(200, seed=4)
    ll, br = ev.replay_results(rows, "nba", registry.defaults("nba"))
    stored = np.array([consensus.score(r["blend"]["p"], r["outcome"])["log_loss"] for r in rows])
    assert np.allclose(ll, stored, atol=1e-15)
    ctx = sim.context_world(300, seed=4)
    replay = ev.replay_totals(ctx, "nfl", registry.defaults("nfl"))
    # both chances are logged to 4 decimals, so the replay agrees to about 1e-4
    assert np.max(np.abs(replay - np.array([r["p_after"] for r in ctx]))) < 1.2e-4


def test_vectorised_rates_match_the_goals_model():
    matches = sim.matches_world(25, seed=5)
    for rho in (-0.13, -0.02):
        lam, mu = ev.rates_many(matches, rho)
        for m, a, b in zip(matches, lam, mu):
            real = soccer.rates_from_odds(m.p_home, m.p_draw, m.p_away, m.p_over, rho)
            assert a == pytest.approx(real[0], abs=1e-7) and b == pytest.approx(real[1], abs=1e-7)
    scored = ev.goals_scores(matches, -0.13, soccer.BTTS_NUDGE)
    m = matches[0]
    lam0, mu0 = soccer.rates_from_odds(m.p_home, m.p_draw, m.p_away, m.p_over)
    g = soccer.grid(lam0, mu0)
    assert scored["exact_ls"][0] == pytest.approx(-math.log(g[m.home_goals][m.away_goals]), rel=1e-7)
    derived = soccer.derive(m.p_home, m.p_draw, m.p_away, m.p_over)
    assert scored["btts"][0] == pytest.approx(derived["markets"]["btts"]["yes"], abs=1e-7)


def test_residual_slope_recovers_the_share():
    ctx = sim.context_world(3000, seed=6, adjusted_share=0.5, beta=1.0, weight=0.5)
    slope = ev.residual_slope([r for r in ctx if r["applied"]])
    # The results move by the full raw change; half of it was applied, so beta is about 2
    assert abs(slope["beta"] - 2.0) < 3 * slope["se"]


def test_summaries_and_weekly_scores(tmp_path):
    rows = sim.consensus_world(800, seed=7)
    summary = ev.summarise_results(rows, reps=300)
    assert summary["n"] == 800 and summary["n_market"] == 800
    assert summary["gap_lo"] <= summary["gap"] <= summary["gap_hi"]
    assert summary["details"]["versions"]["0"]["n"] == 800
    assert "reliability" in summary["details"] and summary["slope"] is not None
    ctx = sim.context_world(400, seed=7)
    totals = ev.summarise_totals(ctx, reps=300)
    assert totals["n"] == 400 and totals["details"]["adjusted"] > 0 and totals["gap"] is not None
    matches = sim.matches_world(700, seed=7)
    corpus = ev.Corpus(consensus=rows, context=ctx, matches=matches)
    week = matches[-1].week  # a week with soccer matches, NBA games and NFL totals
    db = sqlite3.connect(":memory:", isolation_level=None)
    bs.init_brain_schema(db)
    written = ev.write_week_scores(db, corpus, week, reps=200)
    stored = bs.scores(db)
    assert {(r["sport"], r["target"]) for r in written} == {(r["sport"], r["target"]) for r in stored}
    targets = {r["target"] for r in stored if r["sport"] == "soccer"}
    assert {"goals", "btts", "corners", "cards"} <= targets
    nba = next(r for r in stored if r["sport"] == "nba" and r["target"] == "result")
    assert nba["n"] > 0 and nba["market_ll"] is not None and nba["details"]["versions"]


def test_soccer_weekly_scores_use_the_version_live_that_week():
    """Exact score, BTTS, corners and cards are replayed with the soccer values that were live in the scored week
    (the champion's once it is promoted, the defaults before), and the row records that version."""
    matches = sim.matches_world(700, seed=12)
    corpus = ev.Corpus(matches=matches)
    db = sqlite3.connect(":memory:", isolation_level=None)
    bs.init_brain_schema(db)
    weeks = sorted({m.week for m in matches})
    early, late = weeks[2], weeks[-2]
    promoted_at = stats.week_bounds(late)[0] - 3 * 86400  # the week before ``late``
    tuned = {**registry.defaults("soccer"), "soccer.rho": -0.11, "soccer.btts_nudge": 0.01,
             "counts.corners.intercept": 1.7}
    version = bs.insert_version(db, "soccer", tuned, status="challenger", now=promoted_at - 10)
    bs.promote(db, version, promoted_at)
    rows = {r["target"]: r for r in ev.write_week_scores(db, corpus, late, reps=100)}
    test = [m for m in matches if m.week == late]
    want = ev.goals_scores(test, -0.11, 0.01)
    assert rows["goals"]["params_version"] == version and rows["btts"]["params_version"] == version
    assert rows["goals"]["brain_ll"] == pytest.approx(float(want["exact_ls"].mean()))
    assert rows["btts"]["brain_ll"] == pytest.approx(float(want["btts_ll"].mean()))
    assert rows["goals"]["brain_ll"] != pytest.approx(float(ev.goals_scores(test, soccer.DC_RHO,
                                                                            soccer.BTTS_NUDGE)["exact_ls"].mean()))
    train = [m for m in matches if m.start < stats.week_bounds(late)[0]]
    corners = ev.count_target_scores("corners", registry.Params("soccer", tuned).counts("corners"), train, test, -0.11)
    assert rows["corners"]["brain_ll"] == pytest.approx(corners["model_ls"])
    before = {r["target"]: r for r in ev.write_week_scores(db, corpus, early, reps=100)}
    assert before["goals"]["params_version"] == 0
    first = [m for m in matches if m.week == early]
    assert before["goals"]["brain_ll"] == pytest.approx(float(ev.goals_scores(first, soccer.DC_RHO,
                                                                              soccer.BTTS_NUDGE)["exact_ls"].mean()))
    # In safe mode since before the week: the defaults served, and are scored
    bs.set_safe_mode(db, True, "test", now=promoted_at + 1)
    assert ev.params_at(db, "soccer", stats.week_bounds(late)[0] + 3600).version == 0


def test_headline_words():
    rows = sim.consensus_world(400, seed=8)
    head = ev.headline(rows, 200, reps=300)
    assert head["n"] == 200 and head["lo"] <= head["gap"] <= head["hi"]
    assert head["word"] in ("about level with the sharp market", "behind the sharp market",
                            "ahead of the sharp market")
    assert ev.headline([], 200)["word"] == "no games yet"


def test_count_target_scores_beat_climatology_when_inputs_matter():
    strong = {"intercept": 1.6, "home": 0.2, "supremacy": 0.45, "total": 0.2, "size": 30.0}
    matches = sim.matches_world(1600, seed=9, corners=strong)
    out = ev.count_target_scores("corners", strong, matches[:1200], matches[1200:])
    assert out["n"] == 400 and out["model_ls"] < out["baseline_ls"]
    assert 0.8 < out["variance_ratio"] < 1.25


def test_climatology_is_each_leagues_own():
    """Two leagues that differ only in their corner rate (and goal markets), with corners unrelated to the game
    inside each: a count model that "beats" a pooled climatology through the league difference alone doesn't beat
    each league's own climatology; a league with too few training matches falls back to the pooled fit."""
    rng = np.random.default_rng(1)

    def league(matches, name, corners, r=15.0):
        out = []
        for m in matches:
            made = (int(rng.negative_binomial(r, r / (r + corners * 1.08))),
                    int(rng.negative_binomial(r, r / (r + corners / 1.08))))
            out.append(dataclasses.replace(m, league=name, game_key=name + m.game_key, stats={"corners": made}))
        return out

    a = league(sim.matches_world(1500, seed=21), "soccer_epl", 4.6)
    b = [dataclasses.replace(m, p_over=min(m.p_over + 0.12, 0.9))
         for m in league(sim.matches_world(1500, seed=22), "soccer_germany_bundesliga", 5.8)]
    matches = sorted(a + b, key=lambda m: (m.start, m.game_key))
    cut = matches[len(matches) // 2].start
    train, test = [m for m in matches if m.start < cut], [m for m in matches if m.start >= cut]
    # The best this model form can do on the training half: it leans on the goal markets as a league proxy
    yh = np.array([m.stats["corners"][0] for m in train], float)
    ya = np.array([m.stats["corners"][1] for m in train], float)
    sup, tot = ev.count_inputs(train)
    terms = registry.COUNT_TERMS

    def nll(x):
        model = dict(zip(terms, x.tolist()))
        model["size"] = float(np.exp(x[-1]))
        return float(-ev.count_loglik(model, yh, ya, sup, tot).sum())

    x, _, _ = stats.newton(nll, np.array([1.6, 0.15, 0.0, 0.0, np.log(15.0)]), iters=40)
    model = dict(zip(terms, x.tolist()))
    model["size"] = float(np.exp(x[-1]))
    out = ev.count_target_scores("corners", model, train, test)
    assert set(out["climatology"]["leagues"]) == {"soccer_epl", "soccer_germany_bundesliga"}
    assert out["climatology"]["leagues"]["soccer_epl"]["home"] < out["climatology"]["leagues"][
        "soccer_germany_bundesliga"]["home"]
    assert float(np.mean(out["d"])) > 0  # the model is worse than each league's own climatology
    # A league with fewer than CLIMATOLOGY_MIN training matches is scored against all leagues' climatology
    few = [m for m in train if m.league == "soccer_epl"][:ev.CLIMATOLOGY_MIN - 1] + [
        m for m in train if m.league != "soccer_epl"]
    ll, fits = ev.climatology_loglik("corners", few, test)
    assert fits["leagues"]["soccer_epl"] == fits["pooled"]
    assert fits["leagues"]["soccer_germany_bundesliga"] != fits["pooled"]


def test_history_readers_point_in_time_and_one_source_per_game():
    from telegram_betting_bot_simple.learn import history_store as hs

    db = sqlite3.connect(":memory:", isolation_level=None)
    hs.init_history_schema(db)
    start = 1_790_000_000.0
    hs.upsert_game(db, {"league": "soccer_epl", "game_key": "fd:1", "start": start, "home": "A", "away": "B",
                        "home_score": 2, "away_score": 1, "status": "final", "source": "football-data"}, now=start)
    close = start - 60
    lines = [{"kind": "close", "book": "pinnacle", "market": "h2h", "side": s, "price": p, "as_of": close}
             for s, p in (("home", 2.0), ("draw", 3.6), ("away", 4.2))]
    lines += [{"kind": "close", "book": "pinnacle", "market": "totals", "side": s, "point": 2.5, "price": p,
               "as_of": close} for s, p in (("over", 1.9), ("under", 2.0))]
    hs.add_lines(db, "fd:1", start, lines, "football-data")
    # A price stamped after the kickoff never counts (written raw: the writer refuses it)
    db.execute("INSERT INTO hist_lines VALUES ('fd:1', 'close', 'avg', 'h2h', 'home', NULL, 1.2, ?, 'x')",
               (start + 10,))
    for source, corners in (("football-data", (6, 3)), ("espn", (6, 3))):
        hs.add_stats(db, "fd:1", [{"team_side": "home", "stat": "corners", "value": corners[0]},
                                  {"team_side": "away", "stat": "corners", "value": corners[1]},
                                  {"team_side": "home", "stat": "yellow", "value": 2},
                                  {"team_side": "away", "stat": "yellow", "value": 1},
                                  {"team_side": "away", "stat": "red", "value": 1}], source)
    matches = ev.match_records(db)
    assert len(matches) == 1
    m = matches[0]
    total = 1 / 2.0 + 1 / 3.6 + 1 / 4.2
    assert m.p_home == pytest.approx(0.5 / total) and m.p_over == pytest.approx((1 / 1.9) / (1 / 1.9 + 1 / 2.0))
    assert m.stats == {"corners": (6, 3), "cards": (2, 2)}  # one source, not both added up
    assert m.lines_as_of < m.start and ev.leakage_problems([], [m]) == []
    hs.upsert_game(db, {"league": "americanfootball_nfl", "game_key": "nflverse:1", "start": start, "home": "H",
                        "away": "V", "home_score": 24, "away_score": 20, "status": "final", "source": "nflverse"})
    hs.add_lines(db, "nflverse:1", start, [
        {"kind": "close", "book": "nflverse", "market": "spreads", "side": "home", "point": -3.0, "price": 1.91,
         "as_of": close},
        {"kind": "close", "book": "nflverse", "market": "totals", "side": "over", "point": 45.5, "price": 1.91,
         "as_of": close}], "nflverse")
    lines_out = ev.line_records(db)
    assert len(lines_out) == 1
    row = lines_out[0]
    assert row["sport"] == "nfl" and row["market_margin"] == 3.0 and row["line"] == 45.5
    assert row["margin_final"] == 4.0 and row["outcome"] == "home" and row["logged_at"] < row["start"]
