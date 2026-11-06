"""football-data.co.uk odds columns.

The notes (football-data.co.uk/notes.txt) say the plain bookmaker columns are
pre-closing prices. The closing price is the same column with a C after the
book name: AvgH is the pre-close, AvgCH is the close. A file that only has the
plain columns does not have a close, and those matches are not settled as if
it did.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# (source name, closing columns, matching pre-close columns or None)
CLOSE_1X2 = (
    ("AvgC", ("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA")),
    ("PSC", ("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA")),
    ("B365C", ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")),
)
CLOSE_OU = (
    ("AvgC", ("AvgC>2.5", "AvgC<2.5"), ("Avg>2.5", "Avg<2.5")),
    ("B365C", ("B365C>2.5", "B365C<2.5"), ("B365>2.5", "B365<2.5")),
)


def _probs(odds: Sequence[float]) -> np.ndarray:
    try:
        prices = np.array(list(odds), dtype=float)
    except (TypeError, ValueError):
        return np.full(len(odds), np.nan)
    if prices.size == 0 or np.any(~np.isfinite(prices)) or np.any(prices <= 1.0):
        return np.full(max(prices.size, 1), np.nan)
    implied = 1.0 / prices
    return implied / implied.sum()


def _pick(columns, chain):
    have = set(columns)
    for name, close, earlier in chain:
        if all(column in have for column in close):
            opened = earlier if earlier and all(column in have for column in earlier) else None
            return name, close, opened
    return None, None, None


def assign_market_odds(frame: pd.DataFrame):
    """Add close, pre-close and CLV columns. Returns (frame, sources).

    ``c_*`` is the closing decimal price. ``m_open_*`` is the pre-close
    no-vig probability. ``clv_*`` is close probability minus pre-close
    probability. A source is None when that file has no closing column.
    """
    df = frame.copy()
    columns = list(df.columns)
    close_name, close_cols, open_cols = _pick(columns, CLOSE_1X2)
    sources = {"close_1x2": close_name, "open_1x2": None, "close_ou": None, "open_ou": None}
    if close_cols:
        df["c_H"], df["c_D"], df["c_A"] = (df[column] for column in close_cols)
    else:
        df["c_H"] = df["c_D"] = df["c_A"] = np.nan

    n = len(df)
    if open_cols and close_cols:
        sources["open_1x2"] = "preclose"
        m_open = np.vstack([
            _probs((h, d, a)) for h, d, a in zip(df[open_cols[0]], df[open_cols[1]], df[open_cols[2]])
        ]) if n else np.zeros((0, 3))
        m_close = np.vstack([
            _probs((h, d, a)) for h, d, a in zip(df["c_H"], df["c_D"], df["c_A"])
        ]) if n else np.zeros((0, 3))
        df["m_open_H"], df["m_open_D"], df["m_open_A"] = m_open[:, 0], m_open[:, 1], m_open[:, 2]
        clv = m_close - m_open
        df["clv_H"], df["clv_D"], df["clv_A"] = clv[:, 0], clv[:, 1], clv[:, 2]
        df["clv_mag"] = np.nansum(np.abs(clv), axis=1)
    else:
        df["m_open_H"] = df["m_open_D"] = df["m_open_A"] = np.nan
        df["clv_H"] = df["clv_D"] = df["clv_A"] = np.nan
        df["clv_mag"] = np.nan

    ou_name, ou_close, ou_open = _pick(columns, CLOSE_OU)
    sources["close_ou"] = ou_name
    if ou_close:
        df["c_Over"], df["c_Under"] = (df[column] for column in ou_close)
    else:
        df["c_Over"] = df["c_Under"] = np.nan
    if ou_open and ou_close:
        sources["open_ou"] = "preclose"
        opened = [_probs((over, under))[0] for over, under in zip(df[ou_open[0]], df[ou_open[1]])]
        closed = [_probs((over, under))[0] for over, under in zip(df["c_Over"], df["c_Under"])]
        df["m_open_Over"] = opened
        df["clv_OU"] = [c - o for c, o in zip(closed, opened)]
        df["clv_ou_avail"] = 1
    else:
        df["m_open_Over"] = np.nan
        df["clv_OU"] = np.nan
        df["clv_ou_avail"] = 0
    return df, sources
