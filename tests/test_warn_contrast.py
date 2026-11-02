"""Warning text stays readable in every theme.

`--warn` is the warning hue for icons, borders and fills. As text in the light themes (and Telegram's light theme,
which uses them) it falls under WCAG AA's 4.5:1: 4.42:1 on a warning notice over white, 3.71:1 on --card-3. Text
uses `--warn-ink`, the darker ink every light theme defines (the dark theme's is `--warn` itself).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pytest

STATIC = Path(__file__).resolve().parent.parent / "telegram_betting_bot_simple" / "dashboard" / "static"
STYLES = STATIC / "styles.css"
SURFACES = ("--bg", "--card", "--card-2", "--card-3")
AA_TEXT = 4.5
AA_GRAPHICS = 3.0


def _block(css: str, opener: str) -> str:
    """The declarations of the first rule whose selector line is ``opener`` (braces balanced)."""
    start = css.index(opener) + len(opener)
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(css[i], 0)
        i += 1
    return css[start:i - 1]


def _tokens(block: str) -> Dict[str, str]:
    return {name: value.strip() for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block)}


def themes() -> Dict[str, Dict[str, str]]:
    css = STYLES.read_text()
    dark = _tokens(_block(css, ":root {"))
    light_media = {**dark, **_tokens(_block(css, ':root:not([data-theme="dark"]) {'))}
    light_attr = {**dark, **_tokens(_block(css, ':root[data-theme="light"] {'))}
    return {"dark": dark, "light (system)": light_media, "light (chosen, and Telegram's light)": light_attr}


def _resolve(tokens: Dict[str, str], value: str) -> str:
    seen = 0
    while value.startswith("var(") and seen < 5:
        value = tokens[value[4:-1].strip()]
        seen += 1
    return value


def _rgba(value: str) -> Tuple[Tuple[float, float, float], float]:
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)), 1.0  # type: ignore[return-value]
    parts = [p.strip() for p in re.match(r"rgba?\(([^)]*)\)", value).group(1).split(",")]
    return tuple(float(p) for p in parts[:3]), float(parts[3]) if len(parts) > 3 else 1.0  # type: ignore[return-value]


def _over(fg: Tuple[Tuple[float, float, float], float], bg: Tuple[float, float, float]) -> Tuple[float, float, float]:
    (r, g, b), a = fg
    return (a * r + (1 - a) * bg[0], a * g + (1 - a) * bg[1], a * b + (1 - a) * bg[2])


def _luminance(rgb: Tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def surfaces(tokens: Dict[str, str]) -> Iterator[Tuple[str, Tuple[float, float, float]]]:
    """Each page surface, plain and under the warning tint (a notice, badge or pill is --warn-soft over one)."""
    soft = _rgba(_resolve(tokens, tokens["--warn-soft"]))
    for name in SURFACES:
        base = _rgba(_resolve(tokens, tokens[name]))[0]
        yield name, base
        yield f"--warn-soft over {name}", _over(soft, base)


@pytest.mark.parametrize("theme", list(themes()))
def test_warning_text_ink_reads_at_aa_on_every_surface_of_every_theme(theme: str) -> None:
    tokens = themes()[theme]
    ink = _rgba(_resolve(tokens, tokens["--warn-ink"]))[0]
    low = {name: round(contrast(ink, bg), 2) for name, bg in surfaces(tokens) if contrast(ink, bg) < AA_TEXT}
    assert not low, f"{theme}: --warn-ink under {AA_TEXT}:1 on {low}"


@pytest.mark.parametrize("theme", list(themes()))
def test_the_warning_hue_left_on_icons_still_reads_as_a_graphic(theme: str) -> None:
    tokens = themes()[theme]
    hue = _rgba(_resolve(tokens, tokens["--warn"]))[0]
    low = {name: round(contrast(hue, bg), 2) for name, bg in surfaces(tokens) if contrast(hue, bg) < AA_GRAPHICS}
    assert not low, f"{theme}: --warn under {AA_GRAPHICS}:1 (icons) on {low}"


def test_the_warning_hue_itself_is_too_light_for_text_in_the_light_themes() -> None:
    """Why the rule below exists: as text, --warn misses AA in the light themes."""
    tokens = themes()["light (chosen, and Telegram's light)"]
    hue = _rgba(_resolve(tokens, tokens["--warn"]))[0]
    assert min(contrast(hue, bg) for _, bg in surfaces(tokens)) < AA_TEXT


# Rules whose `color: var(--warn)` paints an icon (an svg), never words: graphics need 3:1, which --warn has
ICON_SELECTOR = re.compile(r"\.icon\b|-icon\b|\.lt-pin\b")
RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
TEXT_WARN = re.compile(r"(?<![\w-])color\s*:\s*var\(\s*--warn\s*\)")


def _style_sources() -> Iterator[Path]:
    yield STYLES
    yield from sorted((STATIC / "js").rglob("*.css"))
    yield from sorted((STATIC / "js").rglob("*.js"))  # the views' style modules carry their CSS as text


def test_no_rule_paints_words_in_the_warning_hue() -> None:
    offenders = []
    for path in _style_sources():
        for selector, body in RULE.findall(path.read_text()):
            if not TEXT_WARN.search(body):
                continue
            selectors = [s.strip() for s in selector.strip().splitlines()[-1].split(",")]
            for one in selectors:
                if not ICON_SELECTOR.search(one):
                    offenders.append(f"{path.relative_to(STATIC)}: {one}")
    assert not offenders, "text in var(--warn) (use var(--warn-ink)): " + "; ".join(offenders)


def test_the_warning_utilities_use_the_ink() -> None:
    css = STYLES.read_text()
    for rule in (r"^\.warn \{ color: var\(--warn-ink\); \}", r"^\.pill\.low \{ color: var\(--warn-ink\);",
                 r"^\.slip-text \.warn \{ color: var\(--warn-ink\);", r"^\.o-mark\.warn \{ color: var\(--warn-ink\); \}"):
        assert re.search(rule, css, flags=re.M), rule
