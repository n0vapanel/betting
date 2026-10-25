"""The web dashboard: a betting-app style page backed by a small JSON API.

Every page and API call needs the dashboard password (HTTP Basic auth;
any username). Bet slips send only which picks were chosen; prices and
probabilities are looked up on the server.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import DashboardService, NotFound

STATIC = Path(__file__).parent / "static"
LEAGUE_KEY = r"^[a-z0-9_]{2,64}$"


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


def create_app(service: Optional[DashboardService], password: str) -> FastAPI:
    app = FastAPI(title="SportsBet AI", docs_url=None, redoc_url=None, openapi_url=None)
    basic = HTTPBasic(realm="SportsBet AI")

    def require_login(credentials: HTTPBasicCredentials = Depends(basic)) -> None:
        if not secrets.compare_digest(credentials.password.encode(), password.encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Wrong password",
                headers={"WWW-Authenticate": 'Basic realm="SportsBet AI"'},
            )

    def get_service() -> DashboardService:
        if service is None:
            raise HTTPException(503, "Add THE_ODDS_API_KEY to the .env file and restart the dashboard")
        return service

    login = [Depends(require_login)]

    @app.get("/", dependencies=login)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/leagues", dependencies=login)
    def leagues(svc: DashboardService = Depends(get_service)) -> List[Dict[str, Any]]:
        try:
            return svc.leagues()
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.get("/api/games", dependencies=login)
    def games(
        league: str = Query(pattern=LEAGUE_KEY),
        refresh: bool = False,
        svc: DashboardService = Depends(get_service),
    ) -> Dict[str, Any]:
        try:
            return svc.games(league, refresh)
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.get("/api/game", dependencies=login)
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

    @app.post("/api/slip", dependencies=login)
    def slip(body: SlipRequest, svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        try:
            return svc.price_slip([p.model_dump() for p in body.picks], body.mode, body.stake, body.stakes)
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    # Stylesheet and script only; no data is served from here
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
