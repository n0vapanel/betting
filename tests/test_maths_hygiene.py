"""P0 maths hygiene (plan section 5.19): CORP pools tied forecasts, Dixon-Coles rho stays inside its valid range
(the grids raise, serving narrows it visibly, the registry's bounds are checked against the rates seen), the power
de-vig solves on (1, 50) with a bracket check, the NB1 / NB2 conventions are named, and the calibration fit reports
both intercepts.

Goldens are the worked numbers of research_notes/Event level sports prediction deep dive/maths_formula_verification.md
(re-computed with scipy while writing this). The property checks are written by hand over seeded sweeps (no
Hypothesis dependency)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import learn_sim as sim
from telegram_betting_bot_simple.data import context as data_context
from telegram_betting_bot_simple.engine import calls, context, odds_math, soccer
from telegram_betting_bot_simple.learn import evaluate as ev
from telegram_betting_bot_simple.learn import registry, stats, tune

# --- CORP reliability: tied forecasts are pooled --------------------------------------------------------------------

# [MFV 6]: ten cases with one tie (two forecasts of 0.2, one hit and one miss)
CORP_X = [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CORP_Y = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]


def test_corp_golden_pools_the_tie():
    """The verified decomposition: MCB 0.0523, DSC 0.1233, UNC 0.2400, mean Brier 0.1690. Without pooling the tie
    (c5deb8f) it was MCB 0.069 and DSC 0.140."""
    curve = stats.reliability(CORP_X, CORP_Y)
    assert curve["mcb"] == pytest.approx(0.0523, abs=5e-5) and curve["mcb"] == pytest.approx(157 / 3000, abs=1e-12)
    assert curve["dsc"] == pytest.approx(0.1233, abs=5e-5) and curve["dsc"] == pytest.approx(370 / 3000, abs=1e-12)
    assert curve["unc"] == pytest.approx(0.24, abs=1e-12)
    assert curve["mcb"] - curve["dsc"] + curve["unc"] == pytest.approx(0.169, abs=1e-12)


def test_corp_cep_golden():
    """The conditional event probabilities (0, 1/3, 1/3, 1/3, 1/2, 1/2, 1, 1, 1, 1): both 0.2 forecasts get 1/3."""
    cep = stats.corp_cep(CORP_X, CORP_Y)
    assert np.allclose(cep, [0, 1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 1, 1, 1, 1], atol=1e-15)
    # in the input order, whatever the order of the cases
    order = np.random.default_rng(0).permutation(10)
    assert np.allclose(stats.corp_cep(np.array(CORP_X)[order], np.array(CORP_Y)[order]), cep[order], atol=1e-15)
    assert stats.corp_cep([], []).size == 0


def test_corp_does_not_depend_on_the_order_of_tied_cases():
    """Swapping the outcomes of the two tied 0.2 forecasts changes nothing (c5deb8f gave MCB 0.069 one way round
    and 0.0523 the other)."""
    swapped = list(CORP_Y)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    a, b = stats.reliability(CORP_X, CORP_Y), stats.reliability(CORP_X, swapped)
    for key in ("mcb", "dsc", "unc", "y", "lo", "hi"):
        assert a[key] == pytest.approx(b[key], abs=1e-12), key


def test_corp_a_forecaster_calibrated_per_value_has_no_miscalibration():
    """Two forecast values, each hit exactly as often as it says: the CEP is the forecast itself, so MCB is 0 and the
    curve sits on the diagonal (unpooled, PAV staircased each block by the order of its outcomes)."""
    rng = np.random.default_rng(5)
    p = np.repeat([0.3, 0.7], 50)
    y = np.concatenate([rng.permutation(np.repeat([1.0, 0.0], [15, 35])),
                        rng.permutation(np.repeat([1.0, 0.0], [35, 15]))])
    curve = stats.reliability(p, y, points=11, reps=50)
    assert curve["mcb"] == pytest.approx(0.0, abs=1e-15)
    assert curve["dsc"] == pytest.approx(curve["unc"] - float(np.mean((p - y) ** 2)), abs=1e-12)
    # read at the quantiles of p (the median one falls between the two values, at 0.5): the CEP is 0.3, then 0.7
    assert np.allclose(curve["y"], [0.3 if x < 0.7 else 0.7 for x in curve["x"]], atol=1e-12)
    assert np.allclose(stats.corp_cep(p, y), p, atol=1e-12)


def test_corp_properties_over_a_sweep():
    """Hand-written invariants over seeded random cases with and without ties: the CEP is one value per forecast value,
    non-decreasing in the forecast, unchanged by reordering the cases; MCB and DSC are at least 0 and S = MCB - DSC +
    UNC; with no ties it is the plain PAV of the sorted outcomes (today's numbers, unchanged)."""
    rng = np.random.default_rng(11)
    cases = []
    for case in range(60):
        n = int(rng.integers(10, 120))
        p = rng.uniform(0.02, 0.98, n)
        if case % 2:
            p = np.round(p, int(rng.integers(0, 3)))  # 0-2 decimals: many ties
        cases.append((p, (rng.random(n) < p).astype(float), rng.permutation(n)))
    for case, (p, y, shuffle) in enumerate(cases):
        curve = stats.reliability(p, y, reps=20)
        moved = stats.reliability(p[shuffle], y[shuffle], reps=20)
        for key in ("mcb", "dsc", "y"):
            assert curve[key] == pytest.approx(moved[key], abs=1e-12), (case, key)
        assert curve["mcb"] >= -1e-12 and curve["dsc"] >= -1e-12
        assert curve["mcb"] - curve["dsc"] + curve["unc"] == pytest.approx(float(np.mean((p - y) ** 2)), abs=1e-12)
    for p, y, shuffle in cases:
        cep = stats.corp_cep(p, y)
        for value in np.unique(p):
            assert np.ptp(cep[p == value]) == 0.0
        order = np.argsort(p, kind="mergesort")
        assert np.all(np.diff(cep[order]) >= -1e-12)
        assert np.allclose(stats.corp_cep(p[shuffle], y[shuffle]), cep[shuffle], atol=1e-12)
        if np.unique(p).size == p.size:
            assert np.array_equal(cep[order], stats.pav(y[order]))


# --- Dixon-Coles: the valid range of rho ---------------------------------------------------------------------------

def _cells(matrix):
    return np.array(matrix, dtype=float)


def test_dc_golden():
    """[MFV 1] lambda 1.5 (home), mu 1.1 (away), rho -0.10: tau 1.165 / 0.850 / 0.890 / 1.100, P(0-0) 0.08653,
    the 1X2 0.4520 / 0.2822 / 0.2658, valid range [-0.667, 0.606]; both grids, same numbers."""
    lam, mu, rho = 1.5, 1.1, -0.10
    dc, plain = _cells(soccer.grid(lam, mu, rho, max_goals=30)), _cells(soccer.grid(lam, mu, 0.0, max_goals=30))
    taus = [dc[0, 0] / plain[0, 0], dc[0, 1] / plain[0, 1], dc[1, 0] / plain[1, 0], dc[1, 1] / plain[1, 1]]
    assert taus == pytest.approx([1.165, 0.850, 0.890, 1.100], abs=1e-12)
    assert (dc[0, 0], dc[1, 0], dc[0, 1], dc[1, 1]) == pytest.approx((0.08653, 0.09916, 0.06945, 0.13481), abs=5e-6)
    assert soccer.outcome_probs(dc.tolist()) == pytest.approx((0.4520, 0.2822, 0.2658), abs=5e-5)
    many = ev.grids_many(np.array([lam]), np.array([mu]), rho, max_goals=30)[0]
    assert np.allclose(many, dc, atol=1e-15)
    assert soccer.rho_bounds(lam, mu) == pytest.approx((-2 / 3, 1 / 1.65), abs=1e-15)


def test_dc_swap_trap_golden():
    """[MFV 1] penaltyblog's helper swaps tau(1,0) and tau(0,1); the paper's mapping at (1.9, 0.8, -0.13) gives
    P(1-0) 0.11441, P(0-1) 0.04048, P(home) 0.62015 (the swap would give 0.09615 / 0.04817 and sum to 0.98943)."""
    for g in (_cells(soccer.grid(1.9, 0.8, -0.13, max_goals=30)),
              ev.grids_many(np.array([1.9]), np.array([0.8]), -0.13, max_goals=30)[0]):
        assert g[1, 0] == pytest.approx(0.11441, abs=5e-6) and g[0, 1] == pytest.approx(0.04048, abs=5e-6)
        assert soccer.outcome_probs(g.tolist())[0] == pytest.approx(0.62015, abs=5e-6)


def test_dc_rho_outside_its_range_raises():
    """rho 0.9 at (1.5, 1.1) is past min(1/(lambda mu), 1) = 0.606: both grids raise (c5deb8f clipped tau(0,0) to 0 and
    returned P(0-0) = 0 without a word)."""
    with pytest.raises(ValueError, match="valid range"):
        soccer.grid(1.5, 1.1, 0.9)
    with pytest.raises(ValueError, match="valid range"):
        ev.grids_many(np.array([1.5]), np.array([1.1]), 0.9)
    with pytest.raises(ValueError, match="valid range"):
        ev.grids_many(np.array([1.4, 1.5]), np.array([1.2, 1.1]), np.array([-0.1, 0.9]))
    with pytest.raises(ValueError):
        soccer.grid(1.5, 1.1, float("nan"))
    # -0.25, the learning loop's lowest rho, is invalid once a rate passes 4
    with pytest.raises(ValueError, match="valid range"):
        soccer.grid(4.5, 0.8, -0.25)
    with pytest.raises(ValueError, match="valid range"):
        ev.grids_many(np.array([1.2, 4.5]), np.array([1.0, 0.8]), -0.25)


def test_dc_rho_bounds_sweep():
    """Over a sweep of rates, rho just inside either end of the valid range is accepted and just outside raises, by
    both grids; ``narrow_rho`` returns a value inside unchanged and anything else at the nearest end."""
    rng = np.random.default_rng(3)
    for lam, mu in np.exp(rng.uniform(math.log(0.05), math.log(6.0), size=(150, 2))):
        lam, mu = float(lam), float(mu)
        low, high = max(-1 / lam, -1 / mu), min(1 / (lam * mu), 1.0)  # Dixon & Coles (1997), p. 270
        for rho in (low - 1e-9, high + 1e-9):
            with pytest.raises(ValueError):
                soccer.grid(lam, mu, rho)
            with pytest.raises(ValueError):
                ev.grids_many(np.array([lam]), np.array([mu]), rho)
        for rho in (low, high, low + 1e-9, high - 1e-9, 0.0):
            soccer.grid(lam, mu, rho)
            ev.grids_many(np.array([lam]), np.array([mu]), rho)
            assert soccer.narrow_rho(rho, lam, mu) == rho
        assert soccer.rho_bounds(lam, mu) == (low, high) and low < 0 < high
        assert soccer.narrow_rho(low - 0.5, lam, mu) == low and soccer.narrow_rho(high + 0.5, lam, mu) == high


def test_dc_grids_sum_to_one_and_keep_their_mass_over_a_sweep():
    """Both grids sum to 1 within 1e-12 for any valid rho (the ends included), every cell is at least 0, the
    correction leaves every cell but the four low scores at its independent-Poisson value (its terms cancel), and
    the vectorised grid equals ``soccer.grid``."""
    rng = np.random.default_rng(4)
    lams = np.exp(rng.uniform(math.log(0.05), math.log(6.0), 120))
    mus = np.exp(rng.uniform(math.log(0.05), math.log(6.0), 120))
    rhos = []
    for i, (lam, mu) in enumerate(zip(lams, mus)):
        low, high = max(-1 / lam, -1 / mu), min(1 / (lam * mu), 1.0)
        rhos.append(low if i % 7 == 0 else high if i % 7 == 1 else float(rng.uniform(low, high)))
    rhos = np.array(rhos)
    for max_goals in (10, 20, 30):
        many = ev.grids_many(lams, mus, rhos, max_goals=max_goals)
        assert np.allclose(many.sum(axis=(1, 2)), 1.0, atol=1e-12, rtol=0) and many.min() >= 0
        for lam, mu, rho, g in zip(lams, mus, rhos, many):
            dc = _cells(soccer.grid(float(lam), float(mu), float(rho), max_goals))
            plain = _cells(soccer.grid(float(lam), float(mu), 0.0, max_goals))
            assert math.fsum(dc.ravel()) == pytest.approx(1.0, abs=1e-12) and dc.min() >= 0
            untouched = np.ones_like(dc, dtype=bool)
            untouched[:2, :2] = False
            assert np.allclose(dc[untouched], plain[untouched], rtol=1e-12, atol=0)
            assert np.allclose(g, dc, rtol=1e-12, atol=1e-300)


def test_derive_narrows_rho_and_says_so():
    """A tuned -0.25 with a home rate of 4.56 is outside the valid range: ``derive`` serves the grid at the nearest
    valid rho (-1 / 4.56) and records rho_clamped, rho_requested and the rho used; at the defaults nothing is added."""
    derived = soccer.derive(0.93, 0.05, 0.02, 0.90, rho=-0.25)
    lam, mu = derived["rates"]["home"], derived["rates"]["away"]
    # The grid served is a real Dixon-Coles grid: its correction moves no mass, so every cell but the four low
    # scores keeps its independent-Poisson value (c5deb8f clipped tau(0,1) at 0 and so inflated the whole grid)
    served, plain = _cells(soccer.grid(lam, mu, derived["rho"])), _cells(soccer.grid(lam, mu, 0.0))
    untouched = np.ones_like(served, dtype=bool)
    untouched[:2, :2] = False
    assert np.allclose(served[untouched], plain[untouched], rtol=1e-12, atol=0)
    assert lam > 4 and derived["rho_clamped"] is True and derived["rho_requested"] == -0.25
    assert derived["rho"] == soccer.rho_bounds(lam, mu)[0] == -1 / lam
    assert derived["markets"] == soccer.markets(soccer.grid(lam, mu, derived["rho"]), soccer.BTTS_NUDGE)
    assert derived["fit_ok"]
    # inside the range (a 3.99 home rate still allows -0.25), and at today's default, the result is as before
    inside = soccer.derive(0.90, 0.07, 0.03, 0.85, rho=-0.25)
    assert inside["rates"]["home"] < 4 and inside["rho"] == -0.25 and "rho_clamped" not in inside
    plain = soccer.derive(0.45, 0.27, 0.28, 0.52)
    assert plain["rho"] == soccer.DC_RHO and "rho_clamped" not in plain and "rho_requested" not in plain
    # a rho no rates could use is still refused, as before
    for bad in (2.0, -1.5, float("nan")):
        with pytest.raises(ValueError, match="between -1 and 1"):
            soccer.derive(0.45, 0.27, 0.28, 0.52, rho=bad)
        with pytest.raises(ValueError, match="between -1 and 1"):
            soccer.rates_from_odds(0.45, 0.27, 0.28, 0.52, rho=bad)


def _favourites():
    """Matches whose closing prices are Dixon-Coles grids at big rates (a 4-6 goal side), plus ordinary ones."""
    out = []
    pairs = [(5.0, 0.6), (4.4, 1.0), (1.4, 1.1), (0.6, 4.6), (4.2, 0.4), (2.0, 1.2), (5.6, 0.9), (1.1, 0.9),
             (3.0, 2.5), (0.8, 5.2), (4.8, 1.6), (1.7, 1.4)]
    for i, (lam, mu) in enumerate(pairs):
        g = soccer.grid(lam, mu)
        h, d, a = soccer.outcome_probs(g)
        start = sim.T0 + i * 3600
        out.append(ev.MatchRecord("soccer_epl", f"fav{i}", start, h, d, a, soccer.over_prob(g, 2.5), i % 5, i % 2,
                                  {}, start - 120))
    return out


def test_goals_scores_narrow_rho_per_match_like_derive():
    """The learning loop's replay at a tuned -0.25 scores each match with rho narrowed into its own range, flags
    those matches, and gives the same both-teams-to-score chance ``derive`` would serve."""
    matches = _favourites()
    scored = ev.goals_scores(matches, -0.25, soccer.BTTS_NUDGE)
    lam, mu = scored["rates"]
    lows = np.maximum(-1 / lam, -1 / mu)
    assert np.array_equal(scored["rho_clamped"], lows > -0.25) and 0 < scored["rho_clamped"].sum() < len(matches)
    assert np.array_equal(ev.rho_bounds_many(lam, mu)[0], lows)
    for m, btts, clamped in zip(matches, scored["btts"], scored["rho_clamped"]):
        derived = soccer.derive(m.p_home, m.p_draw, m.p_away, m.p_over, rho=-0.25)
        assert btts == pytest.approx(derived["markets"]["btts"]["yes"], abs=1e-6)
        assert bool(derived.get("rho_clamped", False)) == bool(clamped)
    assert not ev.goals_scores(matches, soccer.DC_RHO, soccer.BTTS_NUDGE)["rho_clamped"].any()


def test_rate_fits_narrow_each_candidate_and_stay_pinned():
    """The fits try rates up to 6 goals: at -0.25 they price each candidate with rho narrowed into its range (so they
    never raise), and the vectorised fit still matches ``soccer.rates_from_odds``."""
    matches = _favourites()
    for rho in (-0.25, -0.19, soccer.DC_RHO, 0.05):
        lam, mu = ev.rates_many(matches, rho)
        for m, a, b in zip(matches, lam, mu):
            real = soccer.rates_from_odds(m.p_home, m.p_draw, m.p_away, m.p_over, rho)
            assert a == pytest.approx(real[0], abs=1e-6) and b == pytest.approx(real[1], abs=1e-6)


def test_registry_rho_bounds_are_checked_against_the_rates_seen():
    """The knob's bounds are [-0.25, 0.05]; checked against the rates seen they narrow to the range valid for every
    match (-0.25 needs both rates at most 4, 0.05 a product at most 20), so a value inside never needs narrowing."""
    assert registry.rho_bounds() == registry.KNOBS["soccer.rho"].bounds("soccer") == (-0.25, 0.05)
    assert registry.rho_bounds(([1.5, 4.5], [1.1, 0.8])) == (-1 / 4.5, 0.05)
    assert registry.rho_bounds(([6.0, 1.0], [3.5, 1.0])) == (-1 / 6.0, 1 / 21.0)
    assert registry.rho_bounds(([], [])) == (-0.25, 0.05)
    for bad in (([0.0], [1.0]), ([1.0, 2.0], [1.0]), (["x"], [1.0]), ([float("inf")], [1.0]), ([-1.0], [1.0])):
        with pytest.raises(ValueError):
            registry.rho_bounds(bad)
    # Ordinary league rates leave the knob's bounds whole: inside every match's own range
    world = sim.matches_world(200, seed=21)
    lam, mu = ev.rates_many(world, soccer.DC_RHO)
    assert registry.rho_bounds((lam, mu)) == (-0.25, 0.05)
    lows, highs = ev.rho_bounds_many(lam, mu)
    assert np.all(lows <= -0.25) and np.all(highs >= 0.05)
    # Big rates narrow it, and a rho inside the narrowed bounds needs no narrowing on those matches
    matches = _favourites()
    rates = ev.rates_many(matches, -0.25)
    low, high = registry.rho_bounds(rates)
    assert -0.25 < low < -0.1 and high == 0.05
    assert not ev.goals_scores(matches, low, soccer.BTTS_NUDGE, rates=rates)["rho_clamped"].any()
    assert ev.goals_scores(matches, -0.25, soccer.BTTS_NUDGE, rates=rates)["rho_clamped"].any()


def test_the_rho_refit_runs_at_the_lowest_bound_with_big_rates():
    """The grids now raise, but the profile likelihood at the knob's lowest value (with 4-6 goal favourites in the
    window) still runs: the replay narrows per match instead."""
    r_hat, se = tune.fit_rho(_favourites(), -0.23, -0.25, 0.05)
    assert -0.25 <= r_hat <= 0.05 and not math.isnan(se)


def test_the_rho_refit_searches_inside_the_bounds_the_rates_allow(monkeypatch):
    """learn/tune's rho refit takes its bounds from ``registry.rho_bounds`` on the training window's rates: with 4-6
    goal favourites in the window the lower bound rises from -0.25 to -1 / (the biggest rate), so the profile never
    asks for a rho some match can't use, and the fitted rho stays inside (c5deb8f searched the static [-0.25, 0.05])."""
    rows = sim.matches_world(300, seed=31) + _favourites()
    champion = registry.defaults("soccer")
    want = registry.rho_bounds(ev.rates_many(rows, champion["soccer.rho"]))
    assert -0.25 < want[0] < -0.1 and want[1] == 0.05
    seen = []
    real = tune.fit_rho

    def spy(matches, r0, low, high, *args, **kwargs):
        seen.append((low, high))
        return real(matches, r0, low, high, *args, **kwargs)

    monkeypatch.setattr(tune, "fit_rho", spy)
    fit = tune.refit("rho", "soccer", rows, champion)
    assert seen == [want]
    assert want[0] <= fit.details["fit"] <= want[1]
    # an ordinary league window keeps the knob's own bounds, so its refit is unchanged
    seen.clear()
    tune.refit("rho", "soccer", sim.matches_world(300, seed=32), champion)
    assert seen == [(-0.25, 0.05)]


# --- The weather layer's soccer searches (engine/context.py) ---------------------------------------------------------

ANFIELD = {"id": "anfield", "name": "Anfield", "elevation_m": 40, "roof": "open"}

WEATHER_ORDINARY = [  # c5deb8f's outputs, bit for bit
    (("over", 0.5, 0.3, "heat"), 0.3672204156003783),
    (("over", 0.5, 0.55, "heat"), 0.5920187386374665),
    (("over", 0.5, 0.8, "heat"), 0.817776942528112),
    (("over", 1.5, 0.3, "heat"), 0.33897472046678545),
    (("over", 1.5, 0.55, "heat"), 0.5804948144460288),
    (("over", 1.5, 0.8, "heat"), 0.8155225759124534),
    (("over", 2.5, 0.3, "heat"), 0.3280509315115995),
    (("over", 2.5, 0.55, "heat"), 0.5737310366016688),
    (("over", 2.5, 0.8, "heat"), 0.8127924611278841),
    (("over", 3.5, 0.3, "heat"), 0.3230898518088944),
    (("over", 3.5, 0.55, "heat"), 0.5704144980441391),
    (("over", 3.5, 0.8, "heat"), 0.8113848036958763),
    (("over", 4.5, 0.3, "heat"), 0.31999046063752407),
    (("over", 4.5, 0.55, "heat"), 0.5681887089629678),
    (("over", 4.5, 0.8, "heat"), 0.8103613090770065),
    (("over", 0.5, 0.3, "cold"), 0.27384760247704415),
    (("over", 0.5, 0.55, "cold"), 0.5336865494309599),
    (("over", 0.5, 0.8, "cold"), 0.7931258575514771),
    (("over", 1.5, 0.3, "cold"), 0.28579578851947257),
    (("over", 1.5, 0.55, "cold"), 0.538593544459947),
    (("over", 1.5, 0.8, "cold"), 0.7941068107801019),
    (("over", 2.5, 0.3, "cold"), 0.2898791245390115),
    (("over", 2.5, 0.55, "cold"), 0.5412271475854566),
    (("over", 2.5, 0.8, "cold"), 0.795195962719736),
    (("over", 3.5, 0.3, "cold"), 0.2916947226917483),
    (("over", 3.5, 0.55, "cold"), 0.5424918844846128),
    (("over", 3.5, 0.8, "cold"), 0.7957474095106128),
    (("over", 4.5, 0.3, "cold"), 0.2928182371948306),
    (("over", 4.5, 0.55, "cold"), 0.5433306336707959),
    (("over", 4.5, 0.8, "cold"), 0.7961433682055845),
    (("chance_over", 2.5, 0.55, "heat"), 0.5738008020417643),
    (("chance_under", 2.5, 0.45, "heat"), 0.4261991979582357),
    (("chance_over", 3.5, 0.3, "cold"), 0.2915841863917811),
    (("chance_under", 1.5, 0.25, "cold"), 0.2572544522165532),
    (("supremacy", 0.45, None, None), 0.3941842937984352),
    (("supremacy", 0.62, 0.22, 3.1), 1.1122733612381968),
    (("supremacy", 0.2, None, 2.4), -0.651038694476932),
    (("margin", 0.6, 2.6), 0.49798810483440975),
    (("margin", -1.1, None), 0.1433918583400134),
]
WEATHER_ABSURD = [  # c5deb8f's outputs (a clipped tau); now from a narrowed rho, instead of raising
    (("over", 2.5, 0.9999, "heat"), 0.9999077297818936),
    (("over", 3.5, 0.9995, "heat"), 0.9995357003433296),
    (("over", 5.5, 0.995, "heat"), 0.9952961326806072),
    (("chance_over", 2.5, 0.9999, "heat"), 0.9999077519542298),
    (("chance_under", 2.5, 0.0001, "heat"), 9.224804577023793e-05),
    (("chance_under", 5.5, 0.005, "heat"), 0.00470300895094411),
    (("supremacy", 0.9, None, 9.0), 4.3804752960568365),
    (("margin", 7.5, 9.0), 0.9942941539378609),
]


@pytest.fixture
def unbound_registry():
    registry.bind(None)
    yield
    registry.bind(None)


def test_the_weather_layer_survives_absurd_over_chances(unbound_registry):
    """engine/context's soccer searches (the total behind an over chance, the supremacy behind a home chance) try
    totals up to 15 goals, where the default rho -0.13 leaves its valid range (a rate above 7.69) and soccer.grid now
    raises. With real weather adjustments (Anfield: heat and rain since the odds, or a cold snap) every output on
    real prices is c5deb8f's bit for bit, through engine.context and data.context; and an absurd over chance (99.99%
    over 2.5, 99.5% over 5.5) still gets its adjusted chance, within 5e-5 of c5deb8f's clipped answer, instead of an
    error (without context._soccer_grid the shared tree raised ValueError)."""
    heat = context.weather_adjustment("soccer", ANFIELD, {"temp_c": 28, "precip_mm": 2}, {"temp_c": 15},
                                      market_total=2.5)
    cold = context.weather_adjustment("soccer", ANFIELD, {"temp_c": 1}, {"temp_c": 12}, market_total=2.5)
    assert heat.applied > 0 > cold.applied
    adjustments = {"heat": heat, "cold": cold}

    def run(kind, *args):
        if kind == "over":
            line, p, adj = args
            return context.adjusted_over("soccer", line, p, adjustments[adj].applied)
        if kind in ("chance_over", "chance_under"):
            line, p, adj = args
            return data_context.adjusted_chance("soccer_epl", adjustments[adj].to_dict(), kind.split("_")[1], line, p)
        if kind == "supremacy":
            p_home, p_draw, total = args
            return context.soccer_supremacy(p_home, p_draw=p_draw, total=total)
        margin, total = args
        return context.margin_to_prob("soccer", margin, total=total)

    for case, want in WEATHER_ORDINARY:
        assert run(*case) == want, case
    for case, want in WEATHER_ABSURD:
        got = run(*case)
        assert got == pytest.approx(want, abs=5e-5), case
        assert case[0] == "supremacy" or 0 < got < 1


# --- Power de-vig: bracket (1, 50) with a check ----------------------------------------------------------------------

def test_power_golden():
    """[MFV 7] 1X2 at 1.90 / 3.50 / 4.20: p = 0.50980 / 0.26848 / 0.22172 with k = 1.04966 (``implied`` reports
    0.95269 = 1/k); Clarke's tennis 1.22 / 4.33: 0.802 / 0.198 with k = 1.10665 (Clarke's own 0.904 = 1/k)."""
    implied = [1 / 1.90, 1 / 3.50, 1 / 4.20]
    assert odds_math.power_probabilities(implied) == pytest.approx([0.50980, 0.26848, 0.22172], abs=5e-6)
    assert odds_math.remove_vig([1.90, 3.50, 4.20], "power") == pytest.approx([0.50980, 0.26848, 0.22172], abs=5e-6)
    assert odds_math.power_exponent(implied) == pytest.approx(1.04966, abs=5e-6)
    assert 1 / odds_math.power_exponent(implied) == pytest.approx(0.95269, abs=5e-6)
    assert odds_math.remove_vig([1.22, 4.33], "power") == pytest.approx([0.802, 0.198], abs=5e-4)
    assert odds_math.power_exponent([1 / 1.22, 1 / 4.33]) == pytest.approx(1.10665, abs=5e-6)
    assert odds_math.POWER_BRACKET == (1.0, 50.0)


def test_power_recovers_fair_odds_over_a_sweep():
    """Prices built as q_i = p_i^(1/k) from fair chances p and an exponent k anywhere in (1, 45) give back p and k
    (c5deb8f's bracket stopped at 10, so every k above 10 came back wrong)."""
    rng = np.random.default_rng(8)
    for case in range(300):
        n = int(rng.integers(2, 9))
        fair = rng.dirichlet(np.ones(n))
        k = float(rng.uniform(1.01, 45.0)) if case % 3 else float(rng.uniform(10.0, 45.0))
        implied = [float(p) ** (1 / k) for p in fair]
        assert odds_math.power_probabilities(implied) == pytest.approx(list(fair), abs=1e-9), (case, k)
        got = odds_math.remove_vig([1 / q for q in implied], "power")
        assert got == pytest.approx(list(fair), abs=1e-9) and sum(got) == pytest.approx(1.0, abs=1e-12)
        assert odds_math.power_exponent(implied) == pytest.approx(k, rel=1e-9)
    # a heavily juiced two-way market whose exponent is 20
    book = [1 / 0.6 ** (1 / 20), 1 / 0.4 ** (1 / 20)]
    assert odds_math.remove_vig(book, "power") == pytest.approx([0.6, 0.4], abs=1e-12)


def test_power_bracket_check():
    """No root inside (1, 50) is an error, not the end of the bracket: no margin, or prices so short that even
    k = 50 leaves sum(q^k) at 1 or more (c5deb8f returned the k = 10 answer for both)."""
    with pytest.raises(ValueError, match="up to 50"):
        odds_math.power_probabilities([1 / 1.001, 1 / 1.01, 0.5])
    with pytest.raises(ValueError, match="up to 50"):
        odds_math.remove_vig([1.001, 1.01, 2.0], "power")
    with pytest.raises(ValueError, match="margin"):
        odds_math.power_probabilities([0.5, 0.4])
    with pytest.raises(ValueError, match="margin"):
        odds_math.power_exponent([0.5, 0.5])
    # remove_vig still answers a book without a margin by dividing through, as before
    assert odds_math.remove_vig([2.0, 2.0], "power") == [0.5, 0.5]


# --- NB1 and NB2, named ---------------------------------------------------------------------------------------------

def _variance(pmf):
    k = np.arange(len(pmf))
    mean = float(np.dot(k, pmf))
    return float(np.dot(k * k, pmf)) - mean * mean


def test_nb_conventions():
    """engine/calls.py's negative binomial is NB1 (a variance ratio) and learn/stats.py's is NB2 (a size). With the
    same alpha 0.125 the variances are 5.625 / 11.25 / 22.5 (NB1) against 8.125 / 22.5 / 70 (NB2) at means 5, 10, 20
    [MFV 3]; NB2 at mean 10, size 8 has pmf(10) = 0.082926."""
    alpha, kmax = 0.125, 400
    for mean, nb1, nb2 in ((5.0, 5.625, 8.125), (10.0, 11.25, 22.5), (20.0, 22.5, 70.0)):
        cdf = np.array([calls._negative_binomial_cdf(k, mean, 1 + alpha) for k in range(kmax + 1)])
        pmf1 = np.diff(np.concatenate([[0.0], cdf]))
        assert _variance(pmf1) == pytest.approx(nb1, rel=1e-9)
        pmf2 = stats.nb_pmf_table(mean, 1 / alpha, kmax)[0]
        assert _variance(pmf2) == pytest.approx(nb2, rel=1e-9)
    assert math.exp(float(stats.nb_logpmf(10, 10.0, 8.0))) == pytest.approx(0.082926, abs=5e-7)
    # the conventions are named where the numbers live (the old names stay for engine/scan.py)
    assert calls.NB_FAMILY == "nb1" and stats.NB_FAMILY == "nb2"
    assert calls.STAT_DISPERSION is calls.STAT_VARIANCE_RATIO and calls.BENCH_DISPERSION == calls.BENCH_VARIANCE_RATIO


# --- Calibration: two intercepts ------------------------------------------------------------------------------------

def test_calibration_names():
    """[MFV 6] truth logit eta ~ N(0, 1), forecast logit 1.4 eta - 0.3, 20,000 cases (seed 0): the joint fit gives
    slope 0.715 (theory 1/1.4) and intercept cal_a_joint 0.203 (theory 0.3/1.4 = 0.214); calibration-in-the-large is
    a different fit, citl 0.286, kept under the old name "intercept" too."""
    rng = np.random.default_rng(0)
    eta = rng.normal(0, 1, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-eta))).astype(float)
    cal = stats.calibration(stats.sigmoid(1.4 * eta - 0.3), y)
    assert cal["slope"] == pytest.approx(0.715025, abs=1e-5) and cal["cal_a_joint"] == pytest.approx(0.203435, abs=1e-5)
    assert cal["citl"] == pytest.approx(0.286156, abs=1e-5)
    se_b, se_a = (cal["slope_hi"] - cal["slope"]) / stats.Z95, (cal["cal_a_joint_hi"] - cal["cal_a_joint"]) / stats.Z95
    assert abs(cal["slope"] - 1 / 1.4) < 3 * se_b and abs(cal["cal_a_joint"] - 0.3 / 1.4) < 3 * se_a
    assert cal["citl"] > cal["cal_a_joint"]
    assert (cal["citl"], cal["citl_lo"], cal["citl_hi"]) == (cal["intercept"], cal["intercept_lo"], cal["intercept_hi"])
    assert cal["cal_a_joint_lo"] < cal["cal_a_joint"] < cal["cal_a_joint_hi"]
    few = stats.calibration(stats.sigmoid(eta[:20]), y[:20])
    assert all(few[k] is None for k in ("citl", "citl_lo", "citl_hi", "cal_a_joint", "cal_a_joint_lo",
                                        "cal_a_joint_hi", "intercept", "slope"))
