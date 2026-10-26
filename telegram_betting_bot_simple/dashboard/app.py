"""The web dashboard: a betting-app style page backed by a small JSON API.

The page itself holds no data. In a browser it shows its own sign-in form and
the dashboard password starts a session; inside Telegram (as a Mini App) the
page sends Telegram's signed initData instead, so each friend gets their own
session without a password. The app is private: only Telegram users the owner
listed (TELEGRAM_ALLOWED_IDS) get in that way, since anyone on Telegram can
open the bot, and the API includes paid AI calls. Every API call needs that
session cookie (or, for Telegram clients that block cookies, the initData in a
header), and routes learn who is asking from the current_uid dependency. Routes
that keep something per person (settings, saved bets, plans, chat) use person():
a Telegram user is "tg:<id>", and everyone on the shared password is told apart
by the random id their browser keeps and sends as X-Player ("web:<id>"), which
is why no route shows one person's id to another (the leaderboard marks rows
"mine" instead). The password is the owner's alone, though, so for the
responsible-gambling gates (a break, the pause on picks after a warning sign,
the weekly budget) every password browser is one person: a new X-Player id
can't shed them (dashboard/lab.py's Gates). Bet
slips send only which picks were chosen; prices and probabilities are looked
up on the server.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import math
import re
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Callable, Collection, Dict, FrozenSet, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .auth import (
    INIT_DATA_EXPIRED,
    MAX_INIT_DATA_LENGTH,
    WEB_UID,
    InitDataError,
    InitDataExpired,
    private_app_message,
    session_key,
    session_token,
    session_uid,
    telegram_uid,
    uid_allowed,
    valid_session,
    verify_init_data,
)
from .lab import COOLOFF_CHOICES, Lab
from .service import CallRejected, DashboardService, NotFound

# The session helpers now live in auth.py; tests and older code still import them from here
__all__ = ["create_app", "current_uid", "person", "own_ids", "SESSION_COOKIE", "session_key", "session_token",
           "valid_session"]

STATIC = Path(__file__).parent / "static"
LEAGUE_KEY = r"^[a-z0-9_]{2,64}$"
SESSION_COOKIE = "sportsbet_session"
SESSION_SECONDS = 30 * 24 * 3600
# The Mini App signs in again each time it opens, so a Telegram session needn't last as long
TELEGRAM_SESSION_SECONDS = 7 * 24 * 3600
# Telegram Web shows the Mini App in an iframe, where browsers may drop the cookie: the page then
# sends its initData with every call in this header instead
INIT_DATA_HEADER = "X-Telegram-Init-Data"
# Password sessions are shared, so each browser sends the random id it keeps (in localStorage) here
PLAYER_HEADER = "X-Player"
PLAYER_ID = re.compile(r"[A-Za-z0-9-]{8,64}")
NO_ODDS_KEY = "Add THE_ODDS_API_KEY to the .env file and restart the dashboard"
# The largest real request is a 2,000-character chat message with a 12-leg slip: a few kilobytes
MAX_BODY_BYTES = 64 * 1024
TOO_LARGE = "The request is too large"
LOGIN_FAILURES = 10  # wrong passwords from one address (an IPv6 /64) before it is locked out ...
LOGIN_LOCKOUT = 15 * 60  # ... for this many seconds (failures older than this are forgotten)
LOGIN_QUEUE = 20  # password checks waiting their turn; past this, more are refused straight away ...
LOGIN_KEPT = 5  # ... except for these few more, kept for addresses with no recent wrong password, and as
# many again for those of them that signed in before (a flood from many fresh addresses can take the first few)
LOGIN_TRACKED = 10_000  # addresses remembered at most, so a flood of addresses can't fill the memory


def login_key(address: str) -> str:
    """Whose wrong passwords add up: an IPv4 address, or an IPv6 address's /64. A home line or a server has a
    whole /64, so a new address inside it mustn't mean a fresh count."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:  # not an address ("unknown", a test client's name)
        return address
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network((int(ip), 64), strict=False))
    return str(ip)


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


# --- version 2 request bodies: every amount and count is checked here before the Lab sees it ---------

LeagueKey = Annotated[str, Field(pattern=LEAGUE_KEY)]
Money = Annotated[Decimal, Field(ge=Decimal("0.01"), le=Decimal("100000"), decimal_places=2)]


class Ref(BaseModel):
    """One pick, by name only: the server looks up its price and chances (a price sent here is ignored)."""

    league: LeagueKey
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    market: Literal["h2h", "spreads", "totals"]
    side: Literal["home", "away", "draw", "over", "under"]
    point: Optional[float] = Field(None, ge=-1000, le=1000)  # the line the person saw, to flag a move


class TicketBody(BaseModel):
    type: Literal["single", "multi", "round_robin", "system"] = "multi"
    stake: Money  # per bet: each single, the multi, each bet of a round robin or system bet
    k: Optional[int] = Field(None, ge=2, le=7)
    system: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_ ]{2,20}$")
    legs: List[Ref] = Field(min_length=1, max_length=12)


class BuildBody(BaseModel):
    leagues: List[LeagueKey] = Field(default_factory=list, max_length=8)
    stake: Money
    target: Decimal = Field(ge=Decimal("0.01"), le=Decimal("10000000"), decimal_places=2)
    max_legs: int = Field(3, ge=1, le=6)
    tier: Optional[Literal["safe", "risky", "crazy"]] = None


class RolloverBody(BaseModel):
    start: Money
    days: int = Field(ge=1, le=60)
    min_odds: Decimal = Field(ge=Decimal("1.2"), le=Decimal("20"), decimal_places=2)
    legs_per_day: int = Field(1, ge=1, le=3)


class PlanBody(RolloverBody):
    strategy: Literal["classic", "bank_half", "ladder", "flat"] = "ladder"
    mode: Literal["paper", "tracked"] = "paper"


class PlanDayBody(BaseModel):
    legs: List[Ref] = Field(min_length=1, max_length=3)


class MeUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=64)
    odds_format: Optional[Literal["decimal", "american", "fractional"]] = None
    favourite_sports: Optional[List[LeagueKey]] = Field(None, max_length=8)
    risk_tier: Optional[Literal["safe", "risky", "crazy"]] = None
    weekly_budget: Optional[Decimal] = Field(None, ge=0, le=Decimal("1000000"), decimal_places=2)
    reality_check_minutes: Optional[Literal[15, 30, 60]] = None
    confirm_age: Optional[bool] = None
    # The person's IANA time zone (the app sends Intl.DateTimeFormat().resolvedOptions().timeZone); null clears it
    timezone: Optional[str] = Field(None, min_length=1, max_length=64)


class CooloffBody(BaseModel):
    hours: int

    @model_validator(mode="after")
    def known_length(self) -> "CooloffBody":
        if self.hours not in COOLOFF_CHOICES:
            raise ValueError(f"A break lasts {', '.join(map(str, COOLOFF_CHOICES))} hours")
        return self


class ChatRequest(BaseModel):
    """The Ask AI request bar. Any part may be missing: the assistant asks for what it still needs."""

    leagues: List[LeagueKey] = Field(default_factory=list, max_length=8)
    stake: Optional[Money] = None
    target: Optional[Decimal] = Field(None, ge=Decimal("0.01"), le=Decimal("10000000"), decimal_places=2)
    max_legs: Optional[int] = Field(None, ge=1, le=6)
    tier: Optional[Literal["safe", "risky", "crazy"]] = None


class ChatBody(BaseModel):
    thread_id: Optional[int] = Field(None, ge=1)
    message: Optional[str] = Field(None, min_length=1, max_length=2000)
    request: Optional[ChatRequest] = None
    slip: Optional[TicketBody] = None  # "ask about my slip"

    @model_validator(mode="after")
    def something_to_answer(self) -> "ChatBody":
        if not (self.message and self.message.strip()) and self.request is None and self.slip is None:
            raise ValueError("Send a message, a request or a slip")
        return self


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


class LimitBody:
    """ASGI middleware: a request body over ``limit`` bytes gets a 413 before anything reads it.

    FastAPI reads and parses a route's JSON body before its sign-in dependency runs, so without this
    anyone, signed in or not, could make the process parse (and hold) megabytes of JSON. A declared
    Content-Length over the limit is refused at once; a chunked body is counted as it arrives.
    """

    def __init__(self, app: Any, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Dict[str, Any], receive: Callable[[], Any], send: Callable[[Any], Any]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = next((value for name, value in scope.get("headers") or () if name == b"content-length"), None)
        if declared is not None and (not declared.isdigit() or int(declared) > self.limit):
            await JSONResponse({"detail": TOO_LARGE}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def counted() -> Any:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > self.limit:
                    # FastAPI passes an HTTPException from the body read straight through: a plain 413
                    raise HTTPException(413, TOO_LARGE)
            return message

        await self.app(scope, counted, send)


class LoginGuard:
    """Slows down password guessing for everyone, not only inside each request.

    Every password check gets its own turn, ``delay`` seconds after the one before, and is made only
    when its turn comes, so guesses sent side by side still go through at most one per ``delay`` in
    all (a right guess waits its turn like the rest), and a wrong one is answered ``delay`` later. The
    turns are kept by the clock, so this holds across threads and event loops. When the line is
    ``queue`` turns long, more attempts are refused at once, except that ``kept`` more turns are kept
    for an address with no recent wrong password and nothing in the line, and ``kept`` more again for
    such an address that has signed in before, so a flood of wrong guesses (even from a fresh address
    each time) can't shut the owner out. An address with ``failures`` wrong passwords within ``lockout``
    seconds is refused for ``lockout`` seconds, right password or not. An IPv6 address counts as its /64
    for all of this (login_key). Behind Caddy the address is the real client's: uvicorn takes
    X-Forwarded-For from 127.0.0.1.
    """

    def __init__(self, delay: float, failures: int = LOGIN_FAILURES, lockout: float = LOGIN_LOCKOUT,
                 queue: int = LOGIN_QUEUE, now: Callable[[], float] = time.monotonic, kept: int = LOGIN_KEPT) -> None:
        self.delay = delay
        self.failures = failures
        self.lockout = lockout
        self.queue = queue
        self.kept = kept
        self.now = now
        self._failed: Dict[str, List[float]] = {}  # address -> times of its recent wrong passwords
        self._locked: Dict[str, float] = {}  # address -> when its lockout ends
        self._waiting: Dict[str, int] = {}  # address -> its checks in the line now
        self._known: Dict[str, float] = {}  # address -> when it last signed in (the right password)
        self._free_at = 0.0  # when the next check's turn comes
        self._mutex = threading.Lock()  # held only for bookkeeping, never across an await

    def _refuse_if_locked(self, address: str) -> None:
        until = self._locked.get(address)
        if until is None:
            return
        left = until - self.now()
        if left <= 0:
            del self._locked[address]
            return
        minutes = max(1, math.ceil(left / 60))
        raise HTTPException(429, f"Too many wrong passwords: try again in {minutes} minute{'s' if minutes != 1 else ''}")

    def _failed_from(self, address: str) -> None:
        now = self.now()
        recent = [t for t in self._failed.get(address, ()) if now - t < self.lockout] + [now]
        if len(recent) >= self.failures:
            self._failed.pop(address, None)
            self._locked[address] = now + self.lockout
        else:
            self._failed[address] = recent
        if len(self._failed) + len(self._locked) > LOGIN_TRACKED:
            self._locked = {a: until for a, until in self._locked.items() if until > now}
            self._failed = {a: times for a, times in self._failed.items() if now - times[-1] < self.lockout}
            if len(self._failed) > LOGIN_TRACKED // 2:  # still a flood: keep the most recent half
                newest = sorted(self._failed.items(), key=lambda item: item[1][-1])[-(LOGIN_TRACKED // 2):]
                self._failed = dict(newest)

    def _clean(self, address: str, now: float) -> bool:
        """No wrong password from the address lately and none of its checks in the line."""
        return not self._waiting.get(address) and not any(now - t < self.lockout for t in self._failed.get(address, ()))

    async def attempt(self, address: str, right: Callable[[], bool]) -> bool:
        """Whether the password is right, checked in its turn; a 429 when the address is locked out or the
        line is full."""
        address = login_key(address)
        with self._mutex:
            self._refuse_if_locked(address)
            now = self.now()
            turn = max(now, self._free_at)
            line = self.queue
            if self._clean(address, now):
                line += self.kept * (2 if address in self._known else 1)
            if self.delay > 0 and turn - now >= line * self.delay:
                raise HTTPException(429, "Too many sign-in attempts right now: try again in a minute")
            self._free_at = turn + self.delay
            self._waiting[address] = self._waiting.get(address, 0) + 1
        try:
            if turn > now:
                await asyncio.sleep(turn - now)
            with self._mutex:
                self._refuse_if_locked(address)  # it may have been locked out while this waited
                ok = right()
                if ok:
                    self._failed.pop(address, None)
                    self._known[address] = self.now()
                    if len(self._known) > LOGIN_TRACKED:  # only the password's holder adds to it; still bounded
                        self._known = dict(sorted(self._known.items(), key=lambda item: item[1])[-LOGIN_TRACKED // 2:])
                    if self._free_at == turn + self.delay:
                        self._free_at = turn  # nobody queued behind a right password: give its turn back
                else:
                    self._failed_from(address)
        finally:
            with self._mutex:
                waiting = self._waiting.pop(address, 1) - 1
                if waiting > 0:
                    self._waiting[address] = waiting
        if not ok:
            await asyncio.sleep(self.delay)
        return ok


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
    isn't on the owner's list gets a 403 that tells them their id. Bad initData falls back to the
    cookie; when that fails too and the initData was genuine but too old (the app was left open over
    a day), the 401 says so (INIT_DATA_EXPIRED), since only reopening the app from Telegram renews it.
    """
    state = request.app.state
    init_data = request.headers.get(INIT_DATA_HEADER)
    expired = False
    if init_data and state.bot_token:
        try:
            user = verify_init_data(init_data, state.bot_token)
        except InitDataExpired:
            user, expired = None, True  # the session cookie may still be good
        except InitDataError:
            user = None
        if user is not None:
            uid = telegram_uid(user)
            if not uid_allowed(uid, state.telegram_ids):
                raise HTTPException(403, private_app_message(user))
            return uid
    uid = session_uid(state.session_key, request.cookies.get(SESSION_COOKIE))
    if uid is not None and not uid_allowed(uid, state.telegram_ids):
        uid = None  # signed in before the owner took them off the list
    if uid is None and expired:
        raise HTTPException(401, INIT_DATA_EXPIRED)
    return uid


def current_uid(request: Request) -> str:
    """Dependency for signed-in routes: the user id, or a 401.

    A plain 401 (no WWW-Authenticate header) so the browser doesn't open its own login pop-up.
    """
    uid = request_uid(request)
    if uid is None:
        raise HTTPException(401, "Sign in required")
    return uid


def person(request: Request) -> str:
    """Dependency for routes that keep something per person: "tg:<id>", or "web:<X-Player>" on the password.

    The password is shared, so a password session alone can't say whose bets these are: the
    browser's own random id (X-Player) does. Missing or malformed, it's a 400. (The Lab still treats
    every password browser as one person for a break, the pause on picks and the weekly budget.)
    """
    uid = current_uid(request)
    if uid != WEB_UID:
        return uid
    player = request.headers.get(PLAYER_HEADER) or ""
    if not PLAYER_ID.fullmatch(player):
        raise HTTPException(400, "Missing player id")
    return f"{WEB_UID}:{player}"


def own_ids(request: Request) -> FrozenSet[str]:
    """Dependency: every id the caller's own records may be under, for marking them "mine" in shared lists.

    That is their person id and, when the browser sends one, "web:<X-Player>" (calls are saved under
    the browser's id, inside Telegram too). Unlike person() a missing X-Player is no error here: the
    list is shared and nothing in it is marked.
    """
    uid = current_uid(request)
    ids = set() if uid == WEB_UID else {uid}
    player = request.headers.get(PLAYER_HEADER) or ""
    if PLAYER_ID.fullmatch(player):
        ids.add(f"{WEB_UID}:{player}")
    return frozenset(ids)


def set_session(response: Response, request: Request, key: bytes, uid: str, seconds: int) -> None:
    response.set_cookie(
        SESSION_COOKIE, session_token(key, int(time.time()) + seconds, uid), max_age=seconds,
        httponly=True, samesite="strict", secure=request.url.scheme == "https",
    )


def create_app(service: Optional[DashboardService], password: str, login_delay: float = 1.0,
               bot_token: Optional[str] = None, telegram_ids: Collection[int] = (),
               lab: Optional[Lab] = None) -> FastAPI:
    """The dashboard app.

    Without bot_token, Telegram sign-in is off and only the password works. With it, only the
    Telegram users in telegram_ids may sign in (none when it's empty). The per-person routes use
    ``lab``, else the service's own (services.build_service sets one up, with the AI assistant).
    """
    app = FastAPI(title="SportsBet AI", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_exception_handler(RequestValidationError, invalid_request)  # type: ignore[arg-type]
    app.add_middleware(LimitBody)  # before routing, so before any body is read or parsed
    key = session_key(password, bot_token)
    allowed = frozenset(telegram_ids)
    # Read by current_uid, so routes defined in other modules can depend on it too
    app.state.session_key = key
    app.state.bot_token = bot_token
    app.state.telegram_ids = allowed
    guard = app.state.login_guard = LoginGuard(login_delay)

    def get_service() -> DashboardService:
        if service is None:
            raise HTTPException(503, NO_ODDS_KEY)
        return service

    if lab is None and service is not None:
        lab = getattr(service, "lab", None) or Lab(service)

    def get_lab() -> Lab:
        if lab is None:
            raise HTTPException(503, NO_ODDS_KEY)
        return lab

    signed_in = [Depends(current_uid)]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    @app.post("/api/login")
    async def login(body: Login, request: Request, response: Response) -> Dict[str, bool]:
        # "surrogatepass": JSON can carry a lone surrogate ("\ud800"), which plain UTF-8 can't encode.
        # Usually the length limit already refuses it, but that depends on the pydantic version
        given = body.password.encode("utf-8", "surrogatepass")
        address = request.client.host if request.client else "unknown"
        # One check at a time, a wrong one holding the line, and a lockout per address or IPv6 /64 (LoginGuard)
        if not await guard.attempt(address, lambda: hmac.compare_digest(given, password.encode())):
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

    def lab_call(action: Callable[[], Any]) -> Any:
        """Run a Lab call, turning its refusals into plain answers: 400 with the reason, 404, or 502 when
        the odds feed or ESPN failed."""
        try:
            return action()
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except (CallRejected, ValueError) as exc:
            raise HTTPException(400, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    def gated(lab: Lab, uid: str, action: Callable[[], Any], picks: bool = False) -> Any:
        """The first dashboard's routes behind the same gates as the Lab's (age confirmed, not on a break
        and, with ``picks``, not in the pause after a warning sign in the chat): a 400 with the reason."""
        def run() -> Any:
            lab.ready(uid, picks=picks)
            return action()
        return lab_call(run)

    @app.get("/api/games")
    def games(
        league: str = Query(pattern=LEAGUE_KEY),
        refresh: bool = False,
        uid: str = Depends(person),
        svc: DashboardService = Depends(get_service),
        lab: Lab = Depends(get_lab),
    ) -> Dict[str, Any]:
        return gated(lab, uid, lambda: svc.games(league, refresh))

    @app.get("/api/game")
    def game(
        league: str = Query(pattern=LEAGUE_KEY),
        id: str = Query(max_length=64),
        uid: str = Depends(person),
        svc: DashboardService = Depends(get_service),
        lab: Lab = Depends(get_lab),
    ) -> Dict[str, Any]:
        return gated(lab, uid, lambda: svc.game(league, id))

    @app.post("/api/slip")
    def slip(body: SlipRequest, uid: str = Depends(person), svc: DashboardService = Depends(get_service),
             lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return gated(lab, uid, lambda: svc.price_slip([p.model_dump() for p in body.picks], body.mode, body.stake,
                                                      body.stakes))

    def calls_errors(action: Any) -> Any:
        try:
            return action()
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except CallRejected as exc:
            raise HTTPException(400, str(exc)) from None
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from None

    @app.post("/api/judge")
    def judge(body: JudgeRequest, uid: str = Depends(person), svc: DashboardService = Depends(get_service),
              lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """The AI's view of a call while it's being made (nothing is saved)."""
        return gated(lab, uid, lambda: svc.judge(body.league, body.id, body.call.model_dump()))

    @app.post("/api/calls")
    def save_call(body: CallRequest, uid: str = Depends(person), svc: DashboardService = Depends(get_service),
                  lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return gated(lab, uid, lambda: svc.save_call(
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

    @app.get("/api/ai-bets")
    def ai_bets(
        league: str = Query(pattern=LEAGUE_KEY),
        source: Literal["all", "mine"] = "all",
        player: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9-]{8,64}$"),
        uid: str = Depends(person),
        svc: DashboardService = Depends(get_service),
        lab: Lab = Depends(get_lab),
    ) -> Dict[str, Any]:
        """The AI's Safe / Risky / Crazy tickets: picks, so the pause after a warning sign applies too."""
        return gated(lab, uid, lambda: svc.ai_bets(league, source, f"web:{player}" if player else None), picks=True)

    @app.get("/api/leaderboard")
    def leaderboard(own: FrozenSet[str] = Depends(own_ids),
                    svc: DashboardService = Depends(get_service)) -> List[Dict[str, Any]]:
        """Everyone's scored calls by name. A player's id is what person() trusts on the shared password, so
        it is never shown to anyone else: each row says whether it is the caller's ("mine"), and only the
        caller's own rows keep their "player" id (for older pages that compare it)."""
        rows = calls_errors(svc.leaderboard)
        return [{**row, "player": row["player"] if row["player"] in own else None, "mine": row["player"] in own}
                for row in rows]

    @app.post("/api/analysts/ask")
    def ask_analyst(body: AskRequest, uid: str = Depends(person), svc: DashboardService = Depends(get_service),
                    lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """Ask one AI analyst about a game now (it can take a minute; each call costs money)."""
        return gated(lab, uid, lambda: svc.ask_analyst(body.league, body.id, body.analyst), picks=True)

    @app.get("/api/track-record", dependencies=signed_in)
    def track_record(league: str = Query("basketball_nba", pattern=LEAGUE_KEY),
                     svc: DashboardService = Depends(get_service)) -> Dict[str, Any]:
        return calls_errors(lambda: svc.track_record(league))

    # --- version 2: per person (settings, tickets, saved bets, plans, players, chat) -----------------

    @app.get("/api/me")
    def me(uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.me(uid))

    @app.put("/api/me")
    def update_me(body: MeUpdate, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """Settings: only the fields sent change. Lowering the weekly budget applies now; raising it waits 24 h."""
        return lab_call(lambda: lab.update_me(uid, body.model_dump(exclude_unset=True)))

    @app.post("/api/me/cooloff")
    def cooloff(body: CooloffBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """Take a break: picks, building and plans switch off until it ends (it can't be ended early)."""
        return lab_call(lambda: lab.cooloff(uid, body.hours))

    @app.get("/api/sports", dependencies=signed_in)
    def sports(lab: Lab = Depends(get_lab)) -> List[Dict[str, Any]]:
        return lab_call(lab.sports)

    @app.get("/api/today")
    def today(
        leagues: Optional[str] = Query(None, max_length=600, pattern=r"^[a-z0-9_]{2,64}(,[a-z0-9_]{2,64}){0,7}$"),
        day: Optional[str] = Query(None, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"),
        uid: str = Depends(person), lab: Lab = Depends(get_lab),
    ) -> Dict[str, Any]:
        """Games across the leagues asked for (default: the person's favourites, else the NBA)."""
        return lab_call(lambda: lab.today(uid, leagues.split(",") if leagues else None, day))

    @app.get("/api/players/search")
    def player_search(q: str = Query(min_length=2, max_length=60),
                      league: Optional[str] = Query(None, pattern=LEAGUE_KEY),
                      uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> List[Dict[str, Any]]:
        return lab_call(lambda: lab.player_search(q, league))

    @app.get("/api/players/{league}/{athlete_id}")
    def player(league: str, athlete_id: str, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        if not re.fullmatch(LEAGUE_KEY, league) or not re.fullmatch(r"[0-9]{1,12}", athlete_id):
            raise HTTPException(404, "No such player")

        def page() -> Dict[str, Any]:
            lab.ready(uid)
            return lab.player(league, athlete_id)

        return lab_call(page)

    @app.post("/api/tickets/price")
    def price_ticket(body: TicketBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """Price a ticket from the cached odds (only the picks are sent; prices come from the server)."""
        return lab_call(lambda: lab.price(uid, body.model_dump()))

    @app.post("/api/tickets/analyse")
    def analyse_ticket(body: TicketBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.analyse(uid, body.model_dump()))

    @app.post("/api/tickets/build")
    def build_ticket(body: BuildBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.build(uid, body.model_dump()))

    @app.post("/api/slips")
    def save_slip(body: TicketBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.save_slip(uid, body.model_dump()))

    @app.get("/api/slips")
    def list_slips(status: Optional[Literal["draft", "saved", "settled", "void"]] = None,
                   uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> List[Dict[str, Any]]:
        return lab_call(lambda: lab.list_slips(uid, status))

    @app.get("/api/slips/share/{code}")
    def shared_slip(code: str, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9]{4,16}", code):
            raise HTTPException(404, "No slip with that code")
        return lab_call(lambda: lab.slip_by_share(code, uid))

    @app.get("/api/slips/{slip_id}")
    def get_slip(slip_id: int, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.get_slip(uid, slip_id))

    @app.delete("/api/slips/{slip_id}")
    def delete_slip(slip_id: int, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, bool]:
        return lab_call(lambda: lab.delete_slip(uid, slip_id))

    @app.post("/api/rollover/compare")
    def rollover_compare(body: RolloverBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """Classic, bank-half, pot-and-ladder and flat stakes side by side, before any plan is started."""
        return lab_call(lambda: lab.rollover_compare(body.model_dump()))

    @app.post("/api/plans")
    def create_plan(body: PlanBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.create_plan(uid, body.model_dump()))

    @app.get("/api/plans")
    def list_plans(status: Optional[Literal["active", "done", "failed", "stopped"]] = None,
                   uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> List[Dict[str, Any]]:
        return lab_call(lambda: lab.list_plans(uid, status))

    @app.get("/api/plans/{plan_id}")
    def get_plan(plan_id: int, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.get_plan(uid, plan_id))

    @app.post("/api/plans/{plan_id}/day")
    def plan_day(plan_id: int, body: PlanDayBody, uid: str = Depends(person),
                 lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """Play the plan's next day on the person's own picks (the stake comes from the plan)."""
        return lab_call(lambda: lab.plan_day(uid, plan_id, body.model_dump()))

    @app.post("/api/plans/{plan_id}/stop")
    def stop_plan(plan_id: int, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        return lab_call(lambda: lab.stop_plan(uid, plan_id))

    @app.post("/api/chat")
    def chat(body: ChatBody, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> Dict[str, Any]:
        """One turn with the AI assistant (it can take a while: it looks things up with the brain's tools)."""
        return lab_call(lambda: lab.chat(uid, body.model_dump(mode="json")))

    @app.get("/api/chat/threads")
    def chat_threads(uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> List[Dict[str, Any]]:
        return lab_call(lambda: lab.threads(uid))

    @app.get("/api/chat/{thread_id}")
    def chat_messages(thread_id: int, uid: str = Depends(person), lab: Lab = Depends(get_lab)) -> List[Dict[str, Any]]:
        return lab_call(lambda: lab.messages(uid, thread_id))

    # Page, stylesheet and script only; no data is served from here
    app.mount("/static", Static(directory=STATIC), name="static")
    return app
