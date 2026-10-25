"""What an analyst is asked and the forecast it must answer with.

Claude and DeepSeek get the same instructions and the same game dossier and
answer in the same JSON shape, so their records can be compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, ValidationError


class AnalystError(Exception):
    """An analyst couldn't give a forecast; the message says why (never contains a key).

    ``cost`` is what the failed attempt still cost, in US dollars.
    """

    def __init__(self, message: str, cost: float = 0.0) -> None:
        super().__init__(message)
        self.cost = cost


@dataclass
class Reply:
    forecast: Dict[str, Any]
    model: str  # the model that answered (a refusal fallback can differ from the one asked)
    cost: float  # estimated US dollars
    input_tokens: int
    output_tokens: int
    news: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)


SYSTEM_PROMPT = """\
You are an NBA analyst in a prediction lab. Before each game you give your forecast. People compare \
their own calls with yours, and after the final buzzer every forecast is graded with the Brier score, \
next to the bookmakers' chances and our statistical model's. Your record over many games is what \
counts, not any single pick.

You get a JSON dossier for one game:
- bookmakers: the best prices and their margin-free chances (the market)
- ai_model: our model's prediction; it leans 90% on the market and adds injuries, rest and team form
- espn_home_win: ESPN's matchup predictor
- teams: injuries, players ruled out, back-to-backs, points scored and allowed per game
- players: projected minutes and stats for each team's rotation
- your_record: how your earlier forecasts were graded, once there are any
- latest_news: recent injury and lineup news from a web search, when there is any

How to forecast:
- Start from the bookmakers. Their prices pick about 68% of NBA winners and are very hard to beat; \
a forecast that copies them scores well. Move away from them only for a specific reason they may not \
have priced yet, such as news newer than the odds or a lineup detail, and name it. Moves of 1-3 \
percentage points are normal; more than 8 needs strong, concrete evidence.
- Be calibrated: of the games you call at 70%, about 7 in 10 should go your way. Never give 0% or 100%.
- Keep the numbers consistent: home_margin, total and home_win_prob describe the same game (a \
5-point home favourite wins roughly 2 times in 3).
- Use your record: if your moves away from the market have not paid off, stay closer to it.
- Use only facts in the dossier or the news. If a player's status is unclear, say so instead of guessing.

Answer with one JSON object, for example:
{"home_win_prob": 0.64, "home_margin": 4.5, "total": 226.5, "confidence": "low", \
"summary": "Two or three sentences.", "factors": ["First factor", "Second factor"], \
"edge": "None: I agree with the market"}
- home_win_prob: the chance the home team wins, from 0 to 1
- home_margin: expected home points minus away points
- total: expected combined points
- confidence: how likely you are to beat the market on this game: low (you stay close to it), \
medium, or high (rare)
- summary: two or three plain sentences for fans on how you see the game
- factors: up to five short key factors
- edge: where, if anywhere, you think the bookmakers are off (winner, spread or total) and why, \
or "None: I agree with the market"

Your forecast is shown as an opinion, never as a tip: no betting advice or guarantees.
"""

NEWS_PROMPT = """\
You check the latest news for one NBA game before tip-off, for an analyst who will forecast it. \
Search for injury reports, players ruled out or questionable, starting lineup changes, rest days and \
other late news about both teams. Use reliable sources (team and league sites, major outlets, beat \
reporters) and prefer the most recent reports.

Reply in plain text, at most 120 words: the key facts with their dates, each with the source's name. \
If you find nothing newer than the injury report you were given, say "No late news found." Do not \
forecast the game or suggest bets.
"""

# The same shape as SYSTEM_PROMPT's example; Claude's structured output enforces it
FORECAST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "home_win_prob": {"type": "number", "description": "Chance the home team wins, 0 to 1"},
        "home_margin": {"type": "number", "description": "Expected home points minus away points"},
        "total": {"type": "number", "description": "Expected combined points"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string"},
        "factors": {"type": "array", "items": {"type": "string"}},
        "edge": {"type": "string"},
    },
    "required": ["home_win_prob", "home_margin", "total", "confidence", "summary", "factors", "edge"],
    "additionalProperties": False,
}


class Forecast(BaseModel):
    home_win_prob: float = Field(ge=0.0, le=1.0)
    home_margin: float = Field(ge=-45.0, le=45.0)
    total: float = Field(ge=150.0, le=320.0)
    confidence: Literal["low", "medium", "high"]
    summary: str
    factors: List[str]
    edge: str


def parse_forecast(data: Any) -> Dict[str, Any]:
    """Validate an analyst's answer; texts are trimmed and the chance kept within 2-98%."""
    try:
        forecast = Forecast.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(sorted({str(error["loc"][0]) for error in exc.errors() if error["loc"]}))
        raise AnalystError(f"The answer didn't match the forecast format ({fields or 'shape'})") from None
    return {
        "home_win_prob": round(min(max(forecast.home_win_prob, 0.02), 0.98), 4),
        "home_margin": round(forecast.home_margin, 1),
        "total": round(forecast.total, 1),
        "confidence": forecast.confidence,
        "summary": forecast.summary.strip()[:900],
        "factors": [factor.strip()[:200] for factor in forecast.factors if factor.strip()][:5],
        "edge": forecast.edge.strip()[:400],
    }
