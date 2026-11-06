"""Pre-fight win streak.

The value on a row is the number of consecutive wins strictly before that
bout. The bout's own result is not an input, and a run never carries from
one fighter into the next.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def _known(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def prior_win_streak(won: Sequence[object], known: Optional[Sequence[object]] = None) -> list:
    """Win streak before each bout. ``known`` false skips the update (a card with no result yet)."""
    run = 0
    out = []
    for index, value in enumerate(won):
        out.append(run)
        if known is not None and not _known(known[index]):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        try:
            won_this = int(value) == 1
        except (TypeError, ValueError):
            continue
        run = run + 1 if won_this else 0
    return out


def assign_prior_streak(frame, group: str = "fighter", won: str = "won", known: Optional[str] = None):
    """Set ``streak_pre`` on ``frame`` from prior results only. Returns the same frame."""
    import pandas as pd

    streaks = pd.Series(index=frame.index, dtype="float")
    for _, idx in frame.groupby(group, sort=False).groups.items():
        rows = frame.loc[idx]
        known_vals = None if known is None or known not in frame.columns else rows[known].tolist()
        streaks.loc[idx] = prior_win_streak(rows[won].tolist(), known_vals)
    frame["streak_pre"] = streaks
    return frame
