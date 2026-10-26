"""The web dashboard: a betting-app style page backed by a small JSON API.

The page itself holds no data. In a browser it shows its own sign-in form and
the dashboard password starts a session; inside Telegram (as a Mini App) the
page sends Telegram's signed initData instead, so each friend gets their own
session without a password. The app is private: only Telegram users the owner
listed (TELEGRAM_ALLOWED_IDS) get in that way, since anyone on Telegram can
open the bot, and the API includes paid AI calls. Every API call needs that
session cookie (or, for Telegram clients that block cookies, the initData in a
header), and routes learn who is asking from the current_uid dependency. Bet
slips send only which picks were chosen; prices and probabilities are looked
up on the server.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Any, Collection, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .auth import (
    MAX_INIT_DATA_LENGTH,
    WEB_UID,
    InitDataError,
    private_app_message,
    session_key,
    session_token,
    session_uid,
    telegram_uid,
    uid_allowed,
    valid_session,
    verify_init_data,
)
from .service import CallRejected, DashboardService, NotFound

# The session helpers now live in auth.py; tests and older code still import them from here
__all__ = ["create_app", "current_uid", "SESSION_COOKIE", "session_key", "session_token", "valid_session"]

STATIC = Path(__file__).parent / "static"
LEAGUE_KEY = r"^[a-z0-9_]{2,64}$"
SESSION_COOKIE = "sportsbet_session"
SESSION_SECONDS = 30 * 24 * 3600
# The Mini App signs in again each time it opens, so a Telegram session needn't last as long
TELEGRAM_SESSION_SECONDS = 7 * 24 * 3600
# Telegram Web shows the Mini App in an iframe, where browsers may drop the cookie: the page then
# sends its initData with every call in this header instead
INIT_DATA_HEADER = "X-Telegram-Init-Data"


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


class TelegramLogin(BaseModel):
    init_data: str = Field(min_length=1, max_length=MAX_INIT_DATA_LENGTH)  # Telegram.WebApp.initData


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


class AsciiJSONResponse(JSONResponse):
    """JSON with every non-ASCII character escaped, so any Python string can be sent back."""

    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


async def invalid_request(request: Request, exc: RequestValidationError) -> Response:
    """FastAPI's usual 422 reply, which repeats the bad input back.

    JSON can carry a lone surrogate ("\\ud800") that UTF-8 can't encode, so the stock reply would crash
    with a 500 on such input; escaping it keeps it a plain 422.
    """
    return AsciiJSONResponse({"detail": jsonable_encoder(exc.errors())}, status_code=422)


def request_uid(request: Request) -> Optional[str]:
    """Who sent the request: "tg:<id>" or "web", or None when nobody is signed in.

    Fresh initData in the header wins over the cookie: it is Telegram's word for who has the app
    open right now, even if the webview still holds an older session. A genuine Telegram user who
    isn't on the owner's list gets a 403 that tells them their id.
    """
    state = request.app.state
    init_data = request.headers.get(INIT_DATA_HEADER)
    if init_data and state.bot_token:
        try:
            user = verify_init_data(init_data, state.bot_token)
        except InitDataError:
            user = None  # e.g. the app was left open over a day: the cookie may still be good
        if user is not None:
            uid = telegram_uid(user)
            if not uid_allowed(uid, state.telegram_ids):
                raise HTTPException(403, private_app_message(user))
            return uid
    uid = session_uid(state.session_key, request.cookies.get(SESSION_COOKIE))
    if uid is not None and not uid_allowed(uid, state.telegram_ids):
        return None  # signed in before the owner took them off the list
    return uid


def current_uid(request: Request) -> str:
    """Dependency for signed-in routes: the user id, or a 401.

    A plain 401 (no WWW-Authenticate header) so the browser doesn't open its own login pop-up.
    """
    uid = request_uid(request)
    if uid is None:
        raise HTTPException(401, "Sign in required")
    return uid


def set_session(response: Response, request: Request, key: bytes, uid: str, seconds: int) -> None:
    response.set_cookie(
        SESSION_COOKIE, session_token(key, int(time.time()) + seconds, uid), max_age=seconds,
        httponly=True, samesite="strict", secure=request.url.scheme == "https",
    )


def create_app(service: Optional[DashboardService], password: str, login_delay: float = 1.0,
               bot_token: Optional[str] = None, telegram_ids: Collection[int] = ()) -> FastAPI:
    """The dashboard app.

    Without bot_token, Telegram sign-in is off and only the password works. With it, only the
    Telegram users in telegram_ids may sign in (none when it's empty).
    """
    app = FastAPI(title="SportsBet AI", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_exception_handler(RequestValidationError, invalid_request)  # type: ignore[arg-type]
    key = session_key(password, bot_token)
    allowed = frozenset(telegram_ids)
    # Read by current_uid, so routes defined in other modules can depend on it too
    app.state.session_key = key
    app.state.bot_token = bot_token
    app.state.telegram_ids = allowed

    def get_service() -> DashboardService:
        if service is None:
            raise HTTPException(503, "Add THE_ODDS_API_KEY to the .env file and restart the dashboard")
        return service

    signed_in = [Depends(current_uid)]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    @app.post("/api/login")
    async def login(body: Login, request: Request, response: Response) -> Dict[str, bool]:
        # "surrogatepass": JSON can carry a lone surrogate ("\ud800"), which plain UTF-8 can't encode.
        # Usually the length limit already refuses it, but that depends on the pydantic version
        if not hmac.compare_digest(body.password.encode("utf-8", "surrogatepass"), password.encode()):
            await asyncio.sleep(login_delay)  # slows down password guessing
            raise HTTPException(401, "Wrong password")
        set_session(response, request, key, WEB_UID, SESSION_SECONDS)
        return {"ok": True}

    @app.post("/api/tg-login")
    def tg_login(body: TelegramLogin, request: Request, response: Response) -> Dict[str, Any]:
        """Sign in from inside Telegram with the Mini App's initData (no password)."""
        if not bot_token:
            raise HTTPException(503, "Telegram sign-in needs TELEGRAM_BOT_TOKEN in the .env file")
        try:
            user = verify_init_data(body.init_data, bot_token)
        except InitDataError as exc:
            raise HTTPException(401, str(exc)) from None
        uid = telegram_uid(user)
        if not uid_allowed(uid, allowed):
            # The owner can find a friend's id here as well as in the message the friend sees
            logging.info("Telegram user %s isn't in TELEGRAM_ALLOWED_IDS; sign-in refused", user["id"])
            raise HTTPException(403, private_app_message(user))
        set_session(response, request, key, uid, TELEGRAM_SESSION_SECONDS)
        return {"ok": True, "uid": uid, "name": str(user.get("first_name") or "")[:64]}

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
