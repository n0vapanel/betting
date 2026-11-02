"""The history backfill's sources (v3 learning loop, Contract 1): football-data.co.uk (soccer), nflverse (NFL) and
ESPN's past games plus MLB StatsAPI's season schedule. Each reads through the shared TTLCache via ``http.HistHttp``
(gentle, counted, refusable) and gives history records:
``{"game": {...hist_games...}, "lines": [...], "stats": [...], "weather": [...]}``."""

from .espn_hist import EspnHistory
from .football_data import FootballData
from .http import HistHttp, HttpError, Refused
from .nflverse import NflVerse

__all__ = ["EspnHistory", "FootballData", "HistHttp", "HttpError", "NflVerse", "Refused"]
