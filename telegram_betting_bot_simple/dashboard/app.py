"""The web dashboard: a betting-app style page backed by a small JSON API.

The page itself holds no data and shows its own sign-in form; every API call
needs the session cookie that signing in with the dashboard password sets.
Bet slips send only which picks were chosen; prices and probabilities are
looked up on the server.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .service import CallRejected, DashboardService, NotFound

STATIC = Path(__file__).parent / "static"
LEAGUE_KEY = r"^[a-z0-9_]{2,64}$"
SESSION_COOKIE = "sportsbet_session"
SESSION_SECONDS = 30 * 24 * 3600


class Pick(BaseModel):
    league: str = Field(pattern=LEAGUE_KEY)
    id: str = Field(max_length=64)
    market: Literal["h2h", "spreads", "totals"]
    side: Literal["home", "away", "draw", "over", "under"]


class SlipRequest(BaseModel):
    mode: Literal["single", "multi"] = "multi"
    stake: float = Field(0, ge=0, le=1_000_000)
    stakes: Dict[int, float] = Field(default_factory=dict)
    picks: List[Pick] = Field(default_factory=list, max_length=20)


class AskRequest(BaseModel):
    league: str = Field(pattern=LEAGUE_KEY)
    id: str = Field(max_length=64)
    analyst: Literal["claude", "deepseek"]


class Login(BaseModel):
    password: str = Field(max_length=256)


class PropCall(BaseModel):
    player_id: str = Field(pattern=r"^[0-9]{1,12}$")
    stat: Literal["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra"]
    line: float = Field(ge=0, le=150)
    side: Literal["over", "under"]
    confidence: float = Field(0.6, ge=0.5, le=1.0)


class Call(BaseModel):
    """Someone's prediction for a game: a winner with a confidence, a score and/or player lines."""

    winner: Optional[Literal["home", "away"]] = None
    confidence: float = Field(0.6, ge=0.5, le=1.0)
    home_score: Optional[int] = Field(None, ge=40, le=250)
    away_score: Optional[int] = Field(None, ge=40, le=250)
    props: List[PropCall] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def consistent(self) -> "Call":
        if (self.home_score is None) != (self.away_score is None):
            raise ValueError("Give both teams' scores, or neither")
        if self.home_score is not None:
            if self.home_score == self.away_score:
                raise ValueError("NBA games can't end in a tie")
            winner = "home" if self.home_score > self.away_score else "away"
            if self.winner and self.winner != winner:
                raise ValueError("Your score has the other team winning")
            self.winner = winner
        if not self.winner and not self.props:
            raise ValueError("Pick a winner, a score or a player line")
        return self


class JudgeRequest(BaseModel):
    league: str = Field(pattern=LEAGUE_KEY)
    id: str = Field(max_length=64)
    call: Call


class CallRequest(JudgeRequest):
    player: str = Field(pattern=r"^[A-Za-z0-9-]{8,64}$")  # a random id this browser keeps
    name: str = Field(pattern=r"^[\w .'-]{1,24}$")


# The browser re-checks the page and its files on every load (a quick 304 when unchanged),
# so after an update it never keeps running old code against the new API
NO_CACHE = {"Cache-Control": "no-cache"}


class Static(StaticFiles):
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


def session_key(password: str) -> bytes:
    """Sessions are signed with a key derived from the password, so changing it signs everyone out."""
    return hmac.new(password.encode(), b"sportsbet-ai session", hashlib.sha256).digest()


def session_token(key: bytes, expires: int) -> str:
    return f"{expires}.{hmac.new(key, str(expires).encode(), hashlib.sha256).hexdigest()}"


def valid_session(key: bytes, token: Optional[str], now: Optional[float] = None) -> bool:
    expires = (token or "").partition(".")[0]
    if not re.fullmatch(r"[0-9]{1,12}", expires) or int(expires) < (now or time.time()):
        return False
    return hmac.compare_digest(session_token(key, int(expires)).encode(), token.encode())


def create_app(service: Optional[DashboardService], password: str, login_delay: float = 1.0) -> FastAPI:
    app = FastAPI(title="SportsBet AI", docs_url=None, redoc_url=None, openapi_url=None)
    key = session_key(password)

    def require_session(request: Request) -> None:
        # A plain 401 (no WWW-Authenticate header) so the browser doesn't open its own login pop-up
        if not valid_session(key, request.cookies.get(SESSION_COOKIE)):
            raise HTTPException(401, "Sign in required")

    def get_service() -> DashboardService:
        if service is None:
            raise HTTPException(503, "Add THE_ODDS_API_KEY to the .env file and restart the dashboard")
        return service

    signed_in = [Depends(require_session)]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    @app.post("/api/login")
    async def login(body: Login, request: Request, response: Response) -> Dict[str, bool]:
        if not hmac.compare_digest(body.password.encode(), password.encode()):
            await asyncio.sleep(login_delay)  # slows down password guessing
            raise HTTPException(401, "Wrong password")
        response.set_cookie(
            SESSION_COOKIE, session_token(key, int(time.time()) + SESSION_SECONDS), max_age=SESSION_SECONDS,
            httponly=True, samesite="strict", secure=request.url.scheme == "https",
        )
        return {"ok": True}

    @app.post("/api/logout")
    def logout(response: Response) -> Dict[str, bool]:
        response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="strict")
        return {"ok": True}

    @app.get("/api/leagues", dependencies=signed_in)
    def leagues(svc: DashboardService = Depends(get_service)) -> List[Dict[str, Any]]:
        try:
            return svc.leagues()
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.get("/api/games", dependencies=signed_in)
    def games(
        league: str = Query(pattern=LEAGUE_KEY),
        refresh: bool = False,
        svc: DashboardService = Depends(get_service),
    ) -> Dict[str, Any]:
        try:
            return svc.games(league, refresh)
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.get("/api/game", dependencies=signed_in)
    def game(
        league: str = Query(pattern=LEAGUE_KEY),
        id: str = Query(max_length=64),
        svc: DashboardService = Depends(get_service),
    ) -> Dict[str, Any]:
        try:
            return svc.game(league, id)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.post("/api/slip", dependencies=signed_in)
    def slip(body: SlipRequest, svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        try:
            return svc.price_slip([p.model_dump() for p in body.picks], body.mode, body.stake, body.stakes)
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    def calls_errors(action: Any) -> Any:
        try:
            return action()
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except CallRejected as exc:
            raise HTTPException(400, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.post("/api/judge", dependencies=signed_in)
    def judge(body: JudgeRequest, svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        """The AI's view of a call while it's being made (nothing is saved)."""
        return calls_errors(lambda: svc.judge(body.league, body.id, body.call.model_dump()))

    @app.post("/api/calls", dependencies=signed_in)
    def save_call(body: CallRequest, svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        return calls_errors(lambda: svc.save_call(
            f"web:{body.player}", body.name.strip(), body.league, body.id, body.call.model_dump()
        ))

    @app.get("/api/calls", dependencies=signed_in)
    def my_calls(player: str = Query(pattern=r"^[A-Za-z0-9-]{8,64}$"),
                 svc: DashboardService = Depends(get_service)) -> List[Dict[str, Any]]:
        return calls_errors(lambda: svc.my_calls(f"web:{player}"))

    @app.delete("/api/calls/{call_id}", dependencies=signed_in)
    def delete_call(call_id: int, player: str = Query(pattern=r"^[A-Za-z0-9-]{8,64}$"),
                    svc: DashboardService = Depends(get_service)) -> Dict[str, bool]:
        if not svc.delete_call(f"web:{player}", call_id):
            raise HTTPException(404, "No pending call with that id")
        return {"ok": True}

    @app.get("/api/ai-bets", dependencies=signed_in)
    def ai_bets(
        league: str = Query(pattern=LEAGUE_KEY),
        source: Literal["all", "mine"] = "all",
        player: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9-]{8,64}$"),
        svc: DashboardService = Depends(get_service),
    ) -> Dict[str, Any]:
        return calls_errors(lambda: svc.ai_bets(league, source, f"web:{player}" if player else None))

    @app.get("/api/leaderboard", dependencies=signed_in)
    def leaderboard(svc: DashboardService = Depends(get_service)) -> List[Dict[str, Any]]:
        return calls_errors(svc.leaderboard)

    @app.post("/api/analysts/ask", dependencies=signed_in)
    def ask_analyst(body: AskRequest, svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        """Ask one AI analyst about a game now (it can take a minute; each call costs money)."""
        return calls_errors(lambda: svc.ask_analyst(body.league, body.id, body.analyst))

    @app.get("/api/track-record", dependencies=signed_in)
    def track_record(league: str = Query("basketball_nba", pattern=LEAGUE_KEY),
                     svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        return calls_errors(lambda: svc.track_record(league))

    # Page, stylesheet and script only; no data is served from here
    app.mount("/static", Static(directory=STATIC), name="static")
    return app
