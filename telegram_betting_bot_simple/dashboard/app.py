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
from pydantic import BaseModel, Field

from .service import DashboardService, NotFound

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


class Login(BaseModel):
    password: str = Field(max_length=256)


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
        return FileResponse(STATIC / "index.html")

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

    # Page, stylesheet and script only; no data is served from here
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
