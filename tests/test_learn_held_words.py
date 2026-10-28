"""A week the gate ran while learning was paused (or in safe mode) must not read "none passed" on the Brain progress
page, and the owner's Telegram line must read well in safe mode."""
from contextlib import closing

from telegram_betting_bot_simple.data.store import Store
from telegram_betting_bot_simple.learn import brain_store as bs
from telegram_betting_bot_simple.learn import loop, progress, registry
from tests import learn_world as world


def test_a_paused_week_says_so_on_the_page_and_in_telegram(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    registry.bind(store)
    try:
        tl = world.LearningTimeline(store)
        tl.world(); tl.first_week(); tl.collect(); tl.pause()
        with closing(store._connect()) as db:
            week = [t for t in bs.trials(db, limit=50) if (t.get("details") or {}).get("family")][0]["week"]
            passed_but_paused = [t for t in bs.trials(db, week=week, limit=500)
                                 if (t.get("details") or {}).get("would_have") in ("promoted", "shadow")
                                 and str(t.get("reason") or "").startswith("Learning is paused")]
        held = progress.progress(store, tl.learner, owner=True, now=tl.clock.t)["held"]
        print(held["words"], "| gate passed but paused:", len(passed_but_paused))
        assert any("learning is paused, so nothing changed" in m for m in tl.sent)
        assert held["words"].startswith("This week: learning was paused, so nothing changed (")
        assert "none passed" not in held["words"]
    finally:
        registry.bind(None)


def test_the_safe_mode_summary_reads_well():
    text = loop.weekly_summary("2026-W44", {"trials": 12, "skipped": "safe mode"}, {}, None)
    print(text.splitlines()[0])
    assert text.startswith("Brain, week 2026-W44: the brain is in safe mode, so nothing changed (12 candidates scored).")
    text = loop.weekly_summary("2026-W44", {"trials": 12, "skipped": "paused"}, {}, None)
    assert text.startswith("Brain, week 2026-W44: learning is paused, so nothing changed (12 candidates scored).")
