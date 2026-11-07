"""The bet slip exposes the evidence behind each selected game, not only a label and price."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "telegram_betting_bot_simple" / "dashboard" / "static"


def test_shell_and_build_slips_show_chances_and_research_routes() -> None:
    sources = [
        (STATIC / "js/app.js").read_text(),
        (STATIC / "js/views/build/slip.js").read_text(),
    ]

    for source in sources:
        assert 'class="slip-probs"' in source
        assert 'class="slip-research"' in source
        assert "Matchup & recent form" in source
        assert "Player projections" in source
        assert "Chance unavailable" in source
        assert "?section=players" in source


def test_slip_research_routes_use_the_selected_leg() -> None:
    for relative in ("js/app.js", "js/views/build/slip.js"):
        source = (STATIC / relative).read_text()
        assert "leg.ref?.league" in source
        assert "leg.ref?.id" in source
        assert "encodeURIComponent" in source
