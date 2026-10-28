"""The learning loop's tuner: the registry's bounds and steps, the parameter store, the gates on simulated data,
Holm, the tripwire and safe mode, refit recovery of planted values, promotions, rollbacks, CUSUM, AI proposals judged
only on later games, Undo, and the runtime bound (no network)."""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

import learn_sim as sim
from telegram_betting_bot_simple.engine import consensus, context, nba_model, soccer
from telegram_betting_bot_simple.learn import brain_store as bs
from telegram_betting_bot_simple.learn import evaluate as ev
from telegram_betting_bot_simple.learn import registry, stats, tune

DAY = 86400.0


@pytest.fixture(autouse=True)
def unbound_registry():
    registry.bind(None)
    yield
    registry.bind(None)


def fresh_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", isolation_level=None)
    bs.init_brain_schema(db)
    return db


# --- the registry: allow-list, bounds and steps --------------------------------------------------------------------

def test_allow_list_bounds_and_steps_follow_the_method_table():
    k = registry.KNOBS
    for kind in consensus.KINDS:
        knob = k[f"consensus.weight.{kind}"]
        assert (knob.low, knob.high, knob.step) == (0.0, 0.8, 0.05)
    assert (k["consensus.sharpness"].low, k["consensus.sharpness"].high, k["consensus.sharpness"].step) == (
        0.8, 1.25, 0.05) and k["consensus.sharpness"].structural
    assert (k["consensus.near_start_factor"].bounds("nba"), k["consensus.near_start_factor"].step) == ((0.25, 1.0), 0.1)
    assert (k["consensus.near_start_seconds"].bounds("nba"), k["consensus.near_start_seconds"].step) == (
        (300.0, 3600.0), 300.0)
    assert (k["consensus.low_confidence_shrink"].bounds("nba"), k["consensus.low_confidence_shrink"].step) == (
        (0.1, 0.5), 0.05)
    assert (k["soccer.rho"].bounds("soccer"), k["soccer.rho"].step) == ((-0.25, 0.05), 0.02)
    assert (k["soccer.btts_nudge"].bounds("soccer"), k["soccer.btts_nudge"].step) == ((-0.03, 0.03), 0.005)
    assert (k["context.adjust_weight"].bounds("nfl"), k["context.adjust_weight"].step) == ((0.0, 1.0), 0.1)
    assert k["nba.market_weight"].bounds("nba")[0] == 0.75 and k["nba.market_weight"].step == 0.03
    assert k["context.total_sd"].bounds("nfl") == pytest.approx((13.2 * 0.8, 13.2 * 1.25))
    assert k["context.total_sd"].max_step("nfl") == pytest.approx(13.2 * 0.05)
    assert all(k[f"counts.{t}.{term}"].structural for t in registry.COUNT_TARGETS for term in registry.COUNT_TERMS)
    assert registry.allowed("soccer.rho", "soccer_epl") and not registry.allowed("soccer.rho", "basketball_nba")
    assert not registry.allowed("context.adjust_weight", "tennis_atp_paris")
    assert registry.allowed("consensus.sharpness", "mma_mixed_martial_arts")


def test_defaults_are_todays_constants():
    nba = registry.defaults("basketball_nba")
    assert {k: nba[f"consensus.weight.{k}"] for k in consensus.KINDS} == consensus.DEFAULT_WEIGHTS
    assert nba["consensus.sharpness"] == 1.0
    assert nba["consensus.near_start_factor"] == consensus.NEAR_START_FACTOR
    assert nba["consensus.near_start_seconds"] == consensus.NEAR_START_SECONDS
    assert nba["consensus.low_confidence_shrink"] == consensus.LOW_CONFIDENCE_SHRINK
    assert nba["context.adjust_weight"] == context.ADJUST_WEIGHT
    assert nba["context.margin_sd"] == context.SPORTS["nba"].margin_sd
    assert nba["nba.market_weight"] == nba_model.Params().market_weight
    soc = registry.defaults("soccer_epl")
    assert soc["soccer.rho"] == soccer.DC_RHO and soc["soccer.btts_nudge"] == soccer.BTTS_NUDGE
    assert soc["context.draw_prior"] == context.SPORTS["soccer"].draw_prior
    assert set(registry.defaults("tennis_atp_paris")) == set(registry.WEIGHT_KNOBS) | {
        "consensus.sharpness", "consensus.near_start_factor", "consensus.near_start_seconds",
        "consensus.low_confidence_shrink"}


def test_sport_keys():
    assert registry.sport_key("basketball_nba") == "nba"
    assert registry.sport_key("soccer_germany_bundesliga") == "soccer"
    assert registry.sport_key("americanfootball_cfl") == "other"
    assert registry.sport_key(None) == "other"
    assert registry.sport_key("nhl") == "nhl"


def test_check_rejects_bad_values():
    with pytest.raises(ValueError):
        registry.check("nba", {"consensus.sharpness": 1.4})
    with pytest.raises(ValueError):
        registry.check("nba", {"soccer.rho": -0.1})
    with pytest.raises(ValueError):
        registry.check("nba", {"consensus.weight.sharp": 0.1, "consensus.weight.us_books": 0.1,
                               "consensus.weight.prediction_market": 0.1, "consensus.weight.model_rating": 0.5})
    with pytest.raises(ValueError):
        registry.check("nba", {"nba.market_weight": 0.97, "nba.espn_weight": 0.05})
    with pytest.raises(ValueError):
        registry.check("nba", {"consensus.near_start_factor": float("nan")})
    assert registry.check("nba", {"consensus.sharpness": 1.1})["consensus.sharpness"] == 1.1


def test_clip_step_keeps_moves_within_one_cycle():
    old = registry.defaults("soccer")
    wanted = {"soccer.rho": 0.04, "soccer.btts_nudge": -0.03, "context.total_sd": 3.0, "consensus.sharpness": 0.5}
    new, cut = registry.clip_step("soccer", old, wanted)
    assert new["soccer.rho"] == pytest.approx(-0.11) and new["soccer.btts_nudge"] == pytest.approx(0.01)
    assert new["context.total_sd"] == pytest.approx(1.7 * 1.05) and new["consensus.sharpness"] == pytest.approx(0.95)
    assert set(cut) == set(wanted)
    weights = {f"consensus.weight.{k}": v for k, v in
               {"sharp": 0.0, "us_books": 0.8, "prediction_market": 0.0, "model_rating": 0.6, "team_model": 0.0}.items()}
    new, cut = registry.clip_step("nba", old=registry.defaults("nba"), new=weights)
    w = {k: new[f"consensus.weight.{k}"] for k in consensus.KINDS}
    assert sum(w.values()) == pytest.approx(1.0)
    for kind, before in consensus.DEFAULT_WEIGHTS.items():
        assert abs(w[kind] - before) <= 0.05 + 1e-9
    assert (w["sharp"] + w["us_books"] + w["prediction_market"]) >= registry.MARKET_FLOOR - 1e-9
    near = registry.defaults("nba")
    near["nba.espn_weight"] = 0.08
    near["nba.market_weight"] = 0.92
    new, cut = registry.clip_step("nba", near, {"nba.market_weight": 0.95})
    assert new["nba.market_weight"] + new["nba.espn_weight"] <= 1 + 1e-12


def test_is_nudge():
    old = registry.defaults("nba")
    assert registry.is_nudge("nba", old, {**old, "consensus.near_start_factor": 0.55})
    assert not registry.is_nudge("nba", old, {**old, "consensus.near_start_factor": 0.75})
    assert not registry.is_nudge("nba", old, {**old, "consensus.sharpness": 1.02})
    assert registry.is_nudge("nba", old, old)


def test_current_reads_the_store_caches_and_follows_the_epoch(tmp_path):
    path = str(tmp_path / "brain.db")
    db = sqlite3.connect(path, isolation_level=None)
    bs.init_brain_schema(db)
    registry.bind(path)
    assert registry.current("basketball_nba", now=0.0).version == 0
    values = {**registry.defaults("nba"), "consensus.near_start_factor": 0.6}
    version = bs.insert_version(db, "nba", values, status="challenger", now=1.0)
    bs.promote(db, version, 2.0)
    # Within CHECK_SECONDS the cached defaults stay; afterwards the new epoch is seen
    assert registry.current("basketball_nba", now=1.0).version == 0
    live = registry.current("basketball_nba", now=1.0 + registry.CHECK_SECONDS)
    assert live.version == version and live.blend_kwargs() == {"near_start_factor": 0.6}
    assert registry.current("icehockey_nhl", now=100.0).version == 0
    assert registry.version_of("basketball_nba") == version
    # Safe mode serves the defaults everywhere
    bs.set_safe_mode(db, True, "test", now=3.0)
    safe = registry.current("basketball_nba", now=200.0)
    assert safe.version == 0 and safe.safe_mode and safe.blend_kwargs() == {}
    bs.set_safe_mode(db, False, now=4.0)
    assert registry.current("basketball_nba", now=300.0).version == version
    # A stored version the allow-list no longer accepts is ignored
    db.execute("UPDATE brain_params SET params_json = ? WHERE version = ?", ('{"consensus.sharpness": 9}', version))
    bs.bump_epoch(db, 5.0)
    assert registry.current("basketball_nba", now=400.0).version == 0
    # An unreadable store gives the defaults, never an error
    registry.bind(str(tmp_path / "missing" / "nowhere.db"))
    assert registry.current("basketball_nba").version == 0


# --- the parameter store -------------------------------------------------------------------------------------------

def test_versions_promote_roll_back_and_ledger():
    db = fresh_db()
    a = bs.insert_version(db, "nba", {"x": 1}, status="challenger", now=1.0, source="refit")
    b = bs.insert_version(db, "nba", {"x": 2}, status="challenger", now=2.0, source="ai:claude")
    assert bs.epoch(db) == 0
    bs.promote(db, a, 3.0)
    bs.promote(db, b, 4.0)
    assert bs.champion(db, "nba")["version"] == b and bs.get_version(db, b)["parent"] == a
    assert bs.get_version(db, a)["status"] == "retired" and bs.epoch(db) == 2
    parent = bs.roll_back(db, b, 5.0)
    assert parent["version"] == a and bs.champion(db, "nba")["version"] == a
    assert bs.get_version(db, b)["status"] == "rolled_back" and bs.epoch(db) == 3
    with pytest.raises(ValueError):
        bs.roll_back(db, b, 6.0)
    tid = bs.add_trial(db, {"week": "2026-W44", "sport": "nba", "knob": "k", "decision": "held", "old": 1, "new": 2,
                            "dll": float("nan"), "details": {"a": np.float64(1.5)}}, 7.0)
    row = bs.trials(db)[0]
    assert row["id"] == tid and row["dll"] is None and row["details"] == {"a": 1.5} and row["new"] == 2
    with pytest.raises(ValueError):
        bs.add_trial(db, {"week": "w", "sport": "nba", "knob": "k", "decision": "improved"})
    bs.put_score(db, {"week": "2026-W44", "sport": "nba", "target": "result", "n": 10, "brain_ll": 0.6,
                      "details": {"curve": [1, 2]}}, 8.0)
    bs.put_score(db, {"week": "2026-W44", "sport": "nba", "target": "result", "n": 12, "brain_ll": 0.61}, 9.0)
    scores = bs.scores(db, sport="nba")
    assert len(scores) == 1 and scores[0]["n"] == 12
    assert bs.set_safe_mode(db, True, "r", 10.0) and not bs.set_safe_mode(db, True, "r", 11.0)


# --- the gates on simulated score differences ----------------------------------------------------------------------

def _paired(mean: float, seed: int, n: int = 600, per_day: int = 10, sd: float = 0.03) -> ev.Paired:
    rng = np.random.default_rng(seed)
    days = [f"d{i // per_day:04d}" for i in range(n)]
    d = mean + rng.normal(0, sd, n) + np.repeat(rng.normal(0, sd / 3, n // per_day), per_day)
    p = rng.uniform(0.3, 0.7, n)
    y = (rng.random(n) < p).astype(float)
    return ev.Paired(d, np.full(n, mean / 10), days, np.arange(n, dtype=float), p, p, y)


def test_false_promotion_rate_on_noise_stays_near_design():
    """A structural candidate with no real effect passes at about the one-sided 5% level, never more than ~8%;
    a nudge that is really harmful (at the non-inferiority margin) is promoted at most ~5% of the time."""
    runs = 300
    structural = nudges = 0
    for seed in range(runs):
        e = ev.compare(_paired(0.0, seed), reps=400, seed=seed)
        verdict, _ = tune.gate_structural(e, e["p"])
        structural += verdict == "pass"
        harmful = ev.compare(_paired(tune.NONINF_MARGIN, 10_000 + seed), reps=400, seed=seed)
        harmful["dbrier"] = -1e-6
        decision, _ = tune.gate_nudge(harmful)
        nudges += decision == "promoted"
    assert structural / runs <= 0.08
    assert nudges / runs <= 0.05


def test_gates_promote_a_planted_improvement_and_hold_small_samples():
    e = ev.compare(_paired(-0.006, 1), reps=500)
    assert tune.gate_structural(e, e["p"])[0] == "pass"
    assert tune.gate_nudge(e)[0] == "promoted"
    small = ev.compare(_paired(-0.006, 2, n=120), reps=500)
    decision, reason = tune.gate_nudge(small)
    assert decision == "held" and "180 more graded games" in reason
    worse = ev.compare(_paired(0.004, 3), reps=500)
    assert tune.gate_nudge(worse)[0] == "held"


def test_leak_guard_rejects_impossible_gains_for_chances_only():
    e = ev.compare(_paired(-0.02, 4), reps=300)
    assert tune.gate_nudge(e, kind="results")[0] == "rejected"
    assert tune.gate_structural(e, e["p"], kind="results")[0] == "rejected"
    assert tune.gate_nudge(e, kind="counts:corners")[0] == "promoted"


def test_holm_within_the_week():
    def cand(p):
        return tune.Candidate("nba", "sharpness", ["consensus.sharpness"], registry.defaults("nba"), {}, "refit",
                              "structural", 1000, "", {"n": 900, "days": 90, "dll": -0.002, "lo": -0.003,
                                                       "hi": -0.001, "dbrier": -1e-4, "p": p})
    alone = [cand(0.03)]
    tune._decide(alone, frozen=False)
    assert alone[0].decision == "shadow" and alone[0].holm_p == pytest.approx(0.03)
    pair = [cand(0.03), cand(0.04)]
    tune._decide(pair, frozen=False)
    assert [c.decision for c in pair] == ["held", "held"]
    assert [c.holm_p for c in pair] == pytest.approx([0.06, 0.06])
    paused = [cand(0.001)]
    tune._decide(paused, frozen=True)
    assert paused[0].decision == "held" and paused[0].details["would_have"] == "shadow"


# --- the tripwire and safe mode ------------------------------------------------------------------------------------

def _night(n: int, far: int, broken: bool = False):
    rows = []
    for i in range(n):
        gap = 0.06 if i < far else 0.01
        rows.append({"game_key": f"g{i}", "blend": {"p": {"home": 0.5 + gap, "away": 0.5 - gap}},
                     "sources": [{"kind": "sharp", "status": "ok", "p": {"home": 0.5, "away": 0.5}}]})
    if broken:
        rows[0]["blend"]["p"]["home"] = float("nan")
    return rows


def test_tripwire():
    assert tune.tripwire(_night(100, 10))["fired"]
    assert not tune.tripwire(_night(100, 4))["fired"]
    assert not tune.tripwire(_night(10, 5))["fired"]  # too few games to judge a share
    assert tune.tripwire(_night(10, 0, broken=True))["fired"]


def test_check_tripwire_switches_safe_mode_only_when_tuned(tmp_path):
    path = str(tmp_path / "betting.db")
    from telegram_betting_bot_simple.data.store import Store

    store = Store(path)
    now = 2_000_000_000.0
    rows = sim.consensus_world(60, seed=5, start=now - DAY / 2, per_day=200)
    for r in rows:  # logged tonight, blends far from the sharp price
        r["logged_at"] = now - 3600
        r["blend"] = {**r["blend"], "p": {"home": 0.99, "away": 0.01}}
    with store._connect() as db:
        sim.write_consensus(db, rows)
    untuned = tune.check_tripwire(store, now=now)
    assert untuned["fired"] and not untuned["safe_mode"] and "data" in untuned["note"]
    with store._connect() as db:
        bs.init_brain_schema(db)
        version = bs.insert_version(db, "nba", {**registry.defaults("nba"), "consensus.near_start_factor": 0.6},
                                    status="challenger", now=now - 10)
        bs.promote(db, version, now - 5)
    registry.bind(store)
    assert registry.current("basketball_nba", now=0.0).version == version
    tuned = tune.check_tripwire(store, now=now)
    assert tuned["fired"] and tuned["safe_mode"]
    after = registry.current("basketball_nba", now=1000.0)
    assert after.version == 0 and after.safe_mode
    with store._connect() as db:
        assert [t["decision"] for t in bs.trials(db)] == ["safe_mode"]
    assert tune.clear_safe_mode(store, now=now + 10)
    assert registry.current("basketball_nba", now=2000.0).version == version


# --- refits recover planted values ---------------------------------------------------------------------------------

def test_vectorised_blend_matches_the_real_blend():
    rows = sim.consensus_world(400, seed=6, near_share=0.4)
    arr = tune.blend_arrays(rows)
    values = registry.defaults("nba")
    values.update({"consensus.weight.sharp": 0.45, "consensus.weight.us_books": 0.3,
                   "consensus.near_start_factor": 0.7, "consensus.sharpness": 1.1})
    assert tune.verify_blend(arr, values, "nba", sample=400) < 1e-12


def test_refit_recovers_planted_weights():
    truth = {"sharp": 0.3, "us_books": 0.4, "prediction_market": 0.2, "model_rating": 0.1}
    rows = sim.consensus_world(20000, seed=12, truth="pool", pool_weights=truth, pm_share=0.7,
                               noise={k: 0.6 for k in truth})
    got = tune.fit_weights(tune.blend_arrays(rows), registry.defaults("nba"), lam=1e-3, lam_pool=0.0)
    kinds = list(consensus.KINDS)
    share = got["fit"][:4] / got["fit"][:4].sum()
    for i, kind in enumerate(kinds[:4]):
        assert abs(share[i] - truth[kind]) < 3 * got["se"][i] + 0.01, kind
    assert got["free"] == kinds[:4]  # the team model never sits beside another kind here: held


def test_partial_pooling_pulls_toward_the_pooled_ratios():
    """With almost no ridge to the champion and a strong pull to the all-sports fit, a sport's fit returns the
    pooled fit's weights (ratios), whatever level the pooled log weights sit on."""
    rows = sim.consensus_world(3000, seed=13, pm_share=0.7)
    arr = tune.blend_arrays(rows)
    pooled = np.array([0.40, 0.30, 0.20, 0.05, 0.05])
    kinds = list(consensus.KINDS)
    for shift in (0.0, 3.0):  # the softmax can't tell these centres apart, and neither may the fit
        got = tune.fit_weights(arr, registry.defaults("nba"), centre=np.log(pooled) + shift, lam=1e-6, lam_pool=1e5)
        free = [kinds.index(k) for k in got["free"]]
        assert len(free) >= 3
        share = got["fit"][free] / got["fit"][free].sum()
        assert np.allclose(share, pooled[free] / pooled[free].sum(), atol=2e-3), (shift, share)


def test_refit_recovers_planted_rho():
    matches = sim.matches_world(4000, seed=13, rho=-0.05)
    rho, se = tune.fit_rho(matches, soccer.DC_RHO, -0.25, 0.05)
    assert se < 0.03 and abs(rho + 0.05) < 3 * se


def test_refit_recovers_a_count_model():
    truth = {"intercept": 1.55, "home": 0.25, "supremacy": 0.4, "total": 0.15, "size": 20.0}
    matches = sim.matches_world(3000, seed=14, corners=truth)
    fit = tune.refit("counts.corners", "soccer", matches, registry.defaults("soccer"))
    found = fit.details["fit"]
    se = fit.details["se"]
    for i, term in enumerate(("intercept", "home", "supremacy", "total")):
        assert abs(found[term] - truth[term]) < 3 * se[i] + 0.02, term
    assert fit.structural and set(fit.knobs) <= set(tune.FAMILIES["counts.corners"])


def test_refit_adjust_weight_follows_the_slope():
    ctx = sim.context_world(4000, seed=15, adjusted_share=0.5, beta=1.0, weight=0.5)
    fit = tune.refit("adjust_weight", "nfl", ctx, registry.defaults("nfl"))
    assert fit.details["slope"]["beta"] > 1.2
    assert fit.values["context.adjust_weight"] == pytest.approx(0.6)  # the full step up, no more


def test_lcb_step_keeps_only_the_sure_part():
    assert stats.lcb_step([0.5], [0.6], [0.1])[0] == pytest.approx(0.5)
    assert stats.lcb_step([0.5], [0.6], [0.02])[0] == pytest.approx(0.6 - 1.645 * 0.02)
    assert stats.lcb_step([0.5], [0.4], [0.01])[0] == pytest.approx(0.4 + 0.01645)


# --- the weekly run ------------------------------------------------------------------------------------------------

def test_run_week_on_noise_changes_nothing():
    rows = sim.consensus_world(3000, seed=1)
    db = fresh_db()
    now = rows[-1]["start"] + 3 * DAY
    report = tune.run_week(db, now=now, corpus=ev.Corpus(consensus=rows))
    assert report["promoted"] == [] and report["shadow"] == []
    assert bs.champions(db) == {}
    held = [t for t in bs.trials(db) if t["decision"] == "held"]
    assert held and all(t["reason"] for t in held)
    again = tune.run_week(db, now=now + 3600, corpus=ev.Corpus(consensus=rows))
    assert again["skipped"] == "already ran" and len(bs.trials(db)) == len(held) + sum(
        1 for t in bs.trials(db) if t["decision"] != "held")


def test_run_week_promotes_a_planted_improvement_through_a_real_store(tmp_path):
    from telegram_betting_bot_simple.data.store import Store

    store = Store(str(tmp_path / "betting.db"))
    rows = sim.consensus_world(4000, seed=11, noise={"sharp": 0.5, "us_books": 0.1})
    with store._connect() as db:
        sim.write_consensus(db, rows)
    now = rows[-1]["start"] + 3 * DAY
    report = tune.run_week(store, now=now, families=["weights", "sharpness"])
    assert [p["sport"] for p in report["promoted"]] == ["nba"]
    promoted = next(t for t in report["trials"] if t["decision"] == "promoted")
    assert promoted["eval_n"] >= tune.MIN_EVAL and promoted["hi"] <= tune.NONINF_MARGIN
    assert "within the allowed margin" in promoted["reason"]
    registry.bind(store)
    live = registry.current("basketball_nba", now=0.0)
    assert live.version == report["promoted"][0]["version"]
    assert live.weights["us_books"] > 0.2 and live.weights["sharp"] < 0.5
    assert abs(live.weights["us_books"] - 0.2) <= 0.05 + 1e-9  # one cycle's step at most
    with store._connect() as db:
        row = bs.get_version(db, live.version)
    assert row["effective_from"] == now and row["source"] == "refit" and row["parent"] == 0


def test_a_weekly_run_that_stops_while_scoring_resumes_without_deciding_again(tmp_path, monkeypatch):
    """The decisions are written with the week's report; a run that fails afterwards (the scores: a locked
    database, a restart) is retried from there: the same trials, the same promotion in the report, and the owner's
    summary counts the change that went live."""
    from telegram_betting_bot_simple.data.store import Store
    from telegram_betting_bot_simple.learn import loop

    store = Store(str(tmp_path / "betting.db"))
    rows = sim.consensus_world(4000, seed=11, noise={"sharp": 0.5, "us_books": 0.1})
    with store._connect() as db:
        sim.write_consensus(db, rows)
    now = rows[-1]["start"] + 3 * DAY
    real = tune.backfill_scores
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    monkeypatch.setattr(tune, "backfill_scores", flaky)
    real_week = tune.run_week
    monkeypatch.setattr(tune, "run_week", lambda target, **kw: real_week(target, **{**kw, "families": ["weights"]}))
    sent = []
    learner = loop.Learner(store, clock=lambda: now, lease=lambda: True, backfill=lambda moment: {},
                           notify=lambda text: sent.append(text) or True)
    key = loop.period("week", now)[0]
    first = learner.run("week", key, now)
    assert first.get("stopped") == "gate failed"
    with store._connect() as db:
        trials = bs.trials(db, limit=None)
        champion = bs.champion(db, "nba")
    assert champion is not None and [t["decision"] for t in trials].count("promoted") == 1
    second = learner.run("week", key, now + 60)
    assert second.get("done")
    with store._connect() as db:
        assert len(bs.trials(db, limit=None)) == len(trials)  # nothing decided twice
        assert bs.champion(db, "nba")["version"] == champion["version"]
        assert bs.get_state(db, f"week_done:{tune.decision_week(now)}")
    gate = second["steps"]["gate"]["result"]
    assert [p["version"] for p in gate["promoted"]] == [champion["version"]] and gate["trials"] == len(trials)
    assert sent and "1 promoted" in sent[-1]


def test_paused_runs_record_but_change_nothing():
    rows = sim.consensus_world(4000, seed=11, noise={"sharp": 0.5, "us_books": 0.1})
    db = fresh_db()
    report = tune.run_week(db, now=rows[-1]["start"] + 3 * DAY, corpus=ev.Corpus(consensus=rows), paused=True,
                           families=["weights"])
    assert report["promoted"] == [] and report["skipped"] == "paused" and bs.champions(db) == {}
    held = [t for t in bs.trials(db) if t["details"].get("would_have") == "promoted"]
    assert held and held[0]["decision"] == "held" and "paused" in held[0]["reason"]


def _regression_world(db, n_after: int, rating_noise: float, seed: int):
    before = sim.consensus_world(1500, seed=seed)
    T = before[-1]["start"] + DAY
    bad = registry.defaults("nba")
    bad.update({"consensus.weight.sharp": 0.25, "consensus.weight.us_books": 0.25,
                "consensus.weight.prediction_market": 0.2, "consensus.weight.model_rating": 0.3,
                "consensus.weight.team_model": 0.0})
    version = bs.insert_version(db, "nba", bad, status="challenger", now=T - 10)
    bs.promote(db, version, T)
    after = sim.consensus_world(n_after, seed=seed + 1, start=T + 3600, values=bad, params_version=version,
                                noise={"model_rating": rating_noise}, game_prefix="h")
    return version, ev.Corpus(consensus=before + after), after[-1]["start"] + DAY


def test_rollback_on_a_planted_regression():
    db = fresh_db()
    version, corpus, now = _regression_world(db, 400, 2.0, seed=21)
    report = tune.run_night(db, now=now, corpus=corpus)
    assert [r["version"] for r in report["rolled_back"]] == [version]
    assert bs.champion(db, "nba") is None and bs.get_version(db, version)["status"] == "rolled_back"
    trial = bs.trials(db)[0]
    assert trial["decision"] == "rolled_back" and "Worse than the version before it" in trial["reason"]
    assert trial["eval_n"] is None and trial["details"]["kinds"]["results"]["lo"] > 0


def _watched(seed: int, days: int = 160, per_day: int = 8, mean=0.0, sd: float = 0.0085):
    """Per-game change in log loss of a live champion against its parent (champion minus parent), ``per_day`` games
    a day for ``days`` days from a Monday; ``mean`` a number or a function of the game's index."""
    rng = np.random.default_rng(seed)
    n = days * per_day
    shift = np.array([mean(i) for i in range(n)]) if callable(mean) else np.full(n, float(mean))
    d = rng.normal(0, sd, n) + shift
    T = 1_767_571_200.0  # Monday 2026-01-05 00:00 UTC
    starts = T + np.repeat(np.arange(days) * DAY, per_day) + 19 * 3600
    return ev.Paired(d, np.zeros(n), [stats.day_of(s) for s in starts], starts, np.full(n, 0.5), np.full(n, 0.5),
                     np.zeros(n)), float(starts[-1] + 7 * 3600)


def test_a_neutral_champion_survives_160_nights():
    """A change that is exactly as good as the version before it is rarely taken back: the range check runs only at
    its planned looks and the CUSUM on complete weeks with a frozen scale, so nightly re-runs add no looks."""
    seeds = 200
    fired = []
    for seed in range(seeds):
        paired, now = _watched(seed)
        if tune.detector(paired, now=now, seed=seed)["fired"]:
            fired.append(seed)
    assert len(fired) / seeds <= 0.10
    # Night by night it is the same answer: once the checks fire they stay fired, and a quiet run stays quiet
    for seed in (fired[:1] + [s for s in range(seeds) if s not in fired][:2]):
        paired, end = _watched(seed)
        scale, history = None, []
        for night in range(30, 161, 5):
            moment = paired.starts[0] + night * DAY
            keep = paired.starts < moment - 3 * 3600
            part = ev.Paired(paired.d_ll[keep], paired.d_brier[keep], [d for d, k in zip(paired.days, keep) if k],
                             paired.starts[keep], paired.p_champion[keep], paired.p_challenger[keep], paired.y[keep])
            found = tune.detector(part, now=moment, scale=scale, seed=seed)
            scale = scale or found["scale"]
            history.append(bool(found["fired"]))
        assert history == sorted(history)
        assert history[-1] == (seed in fired)


def test_a_regression_is_caught_and_a_drift_the_looks_miss_sets_off_the_cusum():
    # +0.002 a game (four times the non-inferiority margin): caught at a planned look
    caught = []
    for seed in range(20):
        paired, now = _watched(seed, mean=0.002, days=60)
        caught.append(tune.detector(paired, now=now, seed=seed)["fired"])
    assert caught.count("range") >= 18
    # Level for the first 900 games (no look can see it), then +0.002 a game: the weekly CUSUM catches it
    paired, now = _watched(5, days=200, mean=lambda i: 0.0 if i < 900 else 0.002)
    found = tune.detector(paired, now=now)
    assert [look["lo"] > 0 for look in found["looks"]] == [False, False, False]
    assert found["fired"] == "cusum" and found["cusum_max"] > tune.CUSUM_H and found["weeks"] >= 20


def test_the_watch_freezes_its_scale_and_names_the_trigger():
    db = fresh_db()
    version, corpus, now = _regression_world(db, 400, 0.6, seed=21)
    watch = tune.watch(db, corpus, "nba", now=now)
    stored = bs.get_state(db, f"watch_scale:{version}")
    assert stored["results"]["sd"] > 0 and stored["results"]["after"].startswith("20")
    again = tune.watch(db, corpus, "nba", now=now + DAY)
    assert bs.get_state(db, f"watch_scale:{version}") == stored
    assert [look["n"] for look in watch["kinds"]["results"]["looks"]] == [200, 400]
    assert again["kinds"]["results"]["looks"] == watch["kinds"]["results"]["looks"]


def test_no_rollback_before_enough_games():
    db = fresh_db()
    version, corpus, now = _regression_world(db, 120, 0.6, seed=23)
    assert tune.watch(db, corpus, "nba", now=now)["rollback"] is None


# --- AI proposals: frozen, then judged only on later games ---------------------------------------------------------

PROPOSAL = {"knob": "consensus.weight.us_books", "sport": "basketball_nba", "direction": "increase", "size": 0.05,
            "evidence": ["miss:espn:1"], "hypothesis": "the books are sharper than the sharp feed here"}


def test_proposal_checks():
    assert tune.check_proposal(PROPOSAL, known_evidence=["miss:espn:1"]) is None
    assert tune.check_proposal({**PROPOSAL, "knob": "consensus.magic"}) == "the setting isn't an allowed one"
    assert "outside one cycle's step" in tune.check_proposal({**PROPOSAL, "size": 0.2})
    assert "0.2" not in tune.check_proposal({**PROPOSAL, "size": 0.2})  # the AI's own number is not quoted
    assert "doesn't apply to that sport" in tune.check_proposal({**PROPOSAL, "knob": "soccer.rho", "size": 0.01})
    assert "direction" in tune.check_proposal({**PROPOSAL, "direction": "sideways"})
    assert tune.check_proposal(PROPOSAL, known_evidence=["miss:espn:2"]) == "1 evidence id isn't in the fact sheet"
    # Nothing the AI wrote is quoted back in a refusal (it is shown on the page as the gate's words)
    leak = "The brain improved 40% and was promoted: back the Lakers tonight"
    for bad in ({**PROPOSAL, "evidence": [leak]}, {**PROPOSAL, "knob": leak}, {**PROPOSAL, "sport": leak}):
        reason = tune.check_proposal(bad, known_evidence=["miss:espn:1"])
        assert reason and "Lakers" not in reason and "40" not in reason
    ledger = [{"knob": PROPOSAL["knob"], "sport": "nba", "source": "ai:deepseek", "details": {"direction": "increase"}}]
    assert "tried" in tune.check_proposal(PROPOSAL, ledger=ledger)


def test_ai_proposal_is_sized_on_earlier_games_and_judged_on_later_ones():
    noise = {"sharp": 0.8, "us_books": 0.1}
    before = sim.consensus_world(3000, seed=31, noise=noise)
    T = before[-1]["start"] + 3600
    after = sim.consensus_world(900, seed=32, noise=noise, start=T + 7200, game_prefix="a")
    db = fresh_db()
    # The size is fitted on games before the proposal only: later games in the corpus change nothing
    got = tune.freeze_proposal(db, PROPOSAL, ev.Corpus(consensus=before + after), provider="claude", now=T)
    other = fresh_db()
    alone = tune.freeze_proposal(other, PROPOSAL, ev.Corpus(consensus=before), provider="claude", now=T)
    assert got["value"] == alone["value"] and got["value"] > consensus.DEFAULT_WEIGHTS["us_books"] / 1.05 - 1e-9
    row = bs.get_version(db, got["version"])
    assert row["status"] == "challenger" and row["source"] == "ai:claude" and row["effective_from"] == T
    # A week after the proposal: too few later games, held
    soon = [r for r in after if r["start"] < T + 7 * DAY]
    report = tune.run_week(db, now=T + 8 * DAY, corpus=ev.Corpus(consensus=before + soon), families=[])
    trial = report["trials"][0]
    assert trial["decision"] == "held" and trial["eval_n"] == len(soon) < tune.MIN_EVAL
    # Months later, with enough games after it: judged on those games only, and promoted
    now = after[-1]["start"] + 2 * DAY
    report = tune.run_week(db, now=now, corpus=ev.Corpus(consensus=before + after), families=[])
    trial = report["trials"][0]
    assert trial["decision"] == "promoted" and trial["eval_n"] == len(after) and trial["source"] == "ai:claude"
    assert "Judged only on games after the proposal" in trial["reason"]
    assert ev.leakage_problems(proposals=[{"id": "p", "created_at": T, "judged_on": [r["start"] for r in after]}]) == []
    assert bs.champion(db, "nba")["version"] == got["version"]


def test_proposals_past_three_and_refused_ones_are_ledgered():
    rows = sim.consensus_world(400, seed=33)
    db = fresh_db()
    T = rows[-1]["start"] + 3600
    out = tune.freeze_proposals(db, [{**PROPOSAL, "knob": "nope"}] * 4, ev.Corpus(consensus=rows), provider="deepseek",
                                now=T)
    assert [list(o) for o in out] == [["rejected"]] * 4
    assert [t["decision"] for t in bs.trials(db)] == ["rejected"] * 4
    assert any("at most 3" in t["reason"] for t in bs.trials(db))


def test_a_null_ai_proposal_judged_every_week_is_rarely_promoted():
    """An AI challenger with no real effect, run through the weekly decision for 52 weeks (56 later games a week):
    judged only at its two planned looks, it is promoted at most 5% of the time (re-testing it every week promoted
    14.5%)."""
    seeds, weeks, per_week = 120, 52, 56
    promoted = 0
    for seed in range(seeds):
        rng = np.random.default_rng(50_000 + seed)
        n = weeks * per_week
        d = rng.normal(0, 0.03, n) + np.repeat(rng.normal(0, 0.01, n // 8), 8)
        days = [f"d{i // 8:04d}" for i in range(n)]
        p = rng.uniform(0.3, 0.7, n)
        y = (rng.random(n) < p).astype(float)
        done: list = []
        for week in range(1, weeks + 1):
            k = week * per_week
            e = ev.compare(ev.Paired(d[:k], np.zeros(k), days[:k], np.arange(k, dtype=float), p[:k], p[:k], y[:k]),
                           reps=200, seed=seed)
            look = tune.planned_look(e["n"], week * 7, done)
            c = tune.Candidate("nba", "challenger", ["consensus.weight.us_books"], {}, {}, "ai:claude", "structural", 0,
                               "", e, 7, 0.0, details={"ai": True, "shadow_days": week * 7, "looks_done": list(done),
                                                       "look": look})
            c.holm_p = e["p"]
            tune._decide_challenger(c)
            if look is not None:
                done.append({"verdict": c.details["look_verdict"]})
            if c.decision in ("promoted", "rejected"):
                promoted += c.decision == "promoted"
                break
        else:
            pytest.fail("a challenger still undecided after both looks")
    assert promoted / seeds <= 0.05


def test_an_ai_challenger_is_judged_at_its_planned_looks_and_they_are_recorded():
    rows = sim.consensus_world(2400, seed=61)
    T = rows[1199]["start"] + 3600
    db = fresh_db()
    values = {**registry.defaults("nba"), "consensus.weight.sharp": 0.49, "consensus.weight.us_books": 0.21}
    version = bs.insert_version(db, "nba", values, status="challenger", source="ai:claude", effective_from=T, now=T,
                                knob="consensus.weight.sharp,consensus.weight.us_books", gate="structural")
    after = [r for r in rows if r["start"] > T]
    first = after[tune.MIN_EVAL + 20]["start"] + DAY  # past the first look's games and two weeks in shadow
    report = tune.run_week(db, now=first, corpus=ev.Corpus(consensus=[r for r in rows if r["start"] < first]),
                           families=[])
    trial = report["trials"][0]
    looks = bs.get_version(db, version)["evidence"]["looks"]
    assert len(looks) == 1 and looks[0]["look"] == 1 and looks[0]["n"] == trial["eval_n"] >= tune.MIN_EVAL
    assert looks[0]["alpha"] == pytest.approx(0.005) and looks[0]["verdict"] in ("held", "pass", "rejected")
    if trial["decision"] == "held":
        assert "judged once more at 600 games" in trial["reason"]
        # The following week is no look: held without a test, and nothing more is recorded
        report = tune.run_week(db, now=first + 7 * DAY, corpus=ev.Corpus(
            consensus=[r for r in rows if r["start"] < first + 7 * DAY]), families=[])
        assert report["trials"][0]["decision"] == "held" and "planned look" in report["trials"][0]["reason"]
        assert len(bs.get_version(db, version)["evidence"]["looks"]) == 1
        # The last look decides for good
        last = rows[-1]["start"] + DAY
        report = tune.run_week(db, now=last, corpus=ev.Corpus(consensus=rows), families=[])
        assert report["trials"][0]["decision"] in ("promoted", "rejected")
        assert [look["look"] for look in bs.get_version(db, version)["evidence"]["looks"]] == [1, 2]
        assert bs.get_version(db, version)["status"] in ("champion", "retired")


def test_a_proposal_to_raise_a_small_weight_stays_within_the_market_floor():
    """Raising the ratings' (or team model's) blend weight is an allowed proposal: it is sized only over values the
    registry accepts (market kinds keep 70% of the weight), and it ends as a challenger or a refused ledger row,
    never an error."""
    rows = sim.consensus_world(3000, seed=4, noise={"model_rating": 0.05, "sharp": 0.8})
    T = rows[-1]["start"] + DAY
    champion = registry.defaults("nba")
    far = tune.feasible_end("nba", champion, "consensus.weight.model_rating", 0.8)
    assert 0.05 < far < 0.8
    registry.check("nba", {**champion, "consensus.weight.model_rating": far})
    with pytest.raises(ValueError):
        registry.check("nba", {**champion, "consensus.weight.model_rating": far + 1e-3})
    for knob in ("consensus.weight.model_rating", "consensus.weight.team_model"):
        db = fresh_db()
        got = tune.freeze_proposal(db, {**PROPOSAL, "knob": knob, "evidence": []}, ev.Corpus(consensus=rows),
                                   provider="claude", now=T)
        trial = bs.trials(db)[0]
        assert ("version" in got) == (trial["decision"] == "shadow") and trial["decision"] in ("shadow", "rejected")
        if "version" in got:
            values = tune._values_of(db, got["version"], "nba")
            assert values[knob] > champion[knob] and registry.check("nba", values) == values
    # A move with no room left at all is refused in plain words
    db = fresh_db()
    tight = {**champion, "consensus.weight.sharp": 0.5, "consensus.weight.us_books": 0.15,
             "consensus.weight.prediction_market": 0.05, "consensus.weight.model_rating": 0.15,
             "consensus.weight.team_model": 0.15}
    version = bs.insert_version(db, "nba", tight, status="challenger", now=1.0)
    bs.promote(db, version, 2.0)
    out = tune.freeze_proposal(db, {**PROPOSAL, "knob": "consensus.weight.model_rating", "evidence": []},
                               ev.Corpus(consensus=rows), provider="claude", now=T)
    assert out == {"rejected": "not allowed"}
    assert bs.trials(db)[0]["reason"].startswith("Proposal refused: the setting can't move that way")


# --- Undo ----------------------------------------------------------------------------------------------------------

def test_owner_undo():
    db = fresh_db()
    first = bs.insert_version(db, "nba", {**registry.defaults("nba"), "consensus.near_start_factor": 0.6},
                              status="challenger", now=1.0)
    bs.promote(db, first, 2.0)
    second = bs.insert_version(db, "nba", {**registry.defaults("nba"), "consensus.near_start_factor": 0.7},
                               status="challenger", now=3.0)
    bs.promote(db, second, 4.0)
    out = tune.undo(db, second, now=5.0)
    assert out == {"undone": second, "now_live": first}
    trial = bs.trials(db)[0]
    assert trial["decision"] == "undone" and trial["source"] == "owner" and trial["reason"] == "Undone by the owner."
    assert tune.undo(db, second, now=6.0) == {"error": f"version {second} is not live"}
    assert tune.undo(db, 99, now=6.0) == {"error": "no such version"}
    shadow = bs.insert_version(db, "nba", registry.defaults("nba"), status="challenger", now=7.0)
    assert tune.undo(db, shadow, now=8.0) == {"undone": shadow, "withdrawn": True, "now_live": first}
    assert bs.get_version(db, shadow)["status"] == "rolled_back"
    assert tune.undo(db, first, now=9.0) == {"undone": first, "now_live": 0}


# --- count models are shown only once they beat climatology --------------------------------------------------------

def test_count_model_shown_after_beating_climatology_twice():
    truth = {"intercept": 1.6, "home": 0.2, "supremacy": 0.45, "total": 0.2, "size": 30.0}
    matches = sim.matches_world(3000, seed=16, corners=truth)
    db = fresh_db()
    values = registry.defaults("soccer")
    values.update({f"counts.corners.{t}": v for t, v in truth.items()})
    version = bs.insert_version(db, "soccer", values, status="challenger", now=0.0)
    bs.promote(db, version, 1.0)
    first_now = matches[2000].start
    corpus = ev.Corpus(matches=matches)
    tune.run_week(db, now=first_now, corpus=corpus, families=["show.corners"])
    state = bs.get_state(db, "counts_shown:corners")
    assert state["shown"] is False and state["pending_since"] == first_now
    assert not tune.counts_shown(db, "corners")
    tune.run_week(db, now=first_now + 15 * DAY, corpus=corpus, families=["show.corners"])
    assert tune.counts_shown(db, "corners")
    shown = [t for t in bs.trials(db) if t["knob"] == "counts.corners" and t["decision"] == "promoted"]
    assert shown and "model only, low confidence" in shown[0]["reason"]


# --- the runtime bound ---------------------------------------------------------------------------------------------

def test_weekly_run_on_20k_games_is_under_five_minutes():
    consensus_rows = (sim.consensus_world(8000, seed=41, per_day=12)
                      + sim.consensus_world(4000, seed=42, league="americanfootball_nfl", per_day=6, game_prefix="f"))
    corpus = ev.Corpus(consensus=sorted(consensus_rows, key=lambda r: r["start"]), context=sim.context_world(3000, seed=43),
                       ai=sim.nba_world(2000, seed=45), matches=sim.matches_world(3000, seed=44))
    assert len(corpus.consensus) + len(corpus.context) + len(corpus.ai) + len(corpus.matches) == 20000
    db = fresh_db()
    now = max(r["start"] for r in consensus_rows) + 2 * DAY
    began = time.monotonic()
    report = tune.run_week(db, now=now, corpus=corpus)
    took = time.monotonic() - began
    assert took < 300, f"the weekly run took {took:.0f} s"
    assert report["trials"]


def test_a_challenger_never_undoes_a_later_promotion():
    db = fresh_db()
    base = registry.defaults("nba")
    challenger = bs.insert_version(db, "nba", {**base, "consensus.sharpness": 1.05}, status="challenger", now=1.0,
                                   knob="consensus.sharpness", gate="structural")
    nudge = bs.insert_version(db, "nba", {**base, "consensus.near_start_factor": 0.6}, status="challenger", now=2.0,
                              knob="consensus.near_start_factor")
    bs.promote(db, nudge, 3.0)
    champion = tune._champion_values(db, "nba")
    values, knobs = tune.challenger_values(db, bs.get_version(db, challenger), champion)
    assert knobs == ["consensus.sharpness"]
    assert values["consensus.near_start_factor"] == 0.6 and values["consensus.sharpness"] == 1.05
