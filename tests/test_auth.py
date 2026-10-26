"""Tests for signing in: Telegram Mini App initData, session tokens, the sign-in routes and the bot's app button.

initData payloads are generated here the way Telegram documents it (HMAC-SHA256 under a key derived
from the bot token), independently of the code under test, with a made-up bot token. No network.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from telegram import MenuButtonDefault, MenuButtonWebApp
from telegram.error import BadRequest
from telegram.ext import Application, ExtBot

from telegram_betting_bot_simple import main as bot_main
from telegram_betting_bot_simple.bot import handlers
from telegram_betting_bot_simple.config import Config
from telegram_betting_bot_simple.dashboard import __main__ as dashboard_main
from telegram_betting_bot_simple.dashboard.app import (
    INIT_DATA_HEADER,
    SESSION_COOKIE,
    create_app,
    current_uid,
)
from telegram_betting_bot_simple.dashboard.auth import (
    INIT_DATA_MAX_AGE,
    InitDataError,
    session_key,
    session_token,
    session_uid,
    telegram_uid,
    uid_allowed,
    valid_session,
    verify_init_data,
)

BOT_TOKEN = "123456:made-up-token-for-tests-only"  # not a real bot
NOW = 1_800_000_000
SAM = {"id": 42, "first_name": "Sam & Jo", "last_name": "Müller", "username": "sam_j", "language_code": "en"}


def make_init_data(user: Optional[Dict[str, Any]] = None, auth_date: int = NOW, token: str = BOT_TOKEN,
                   **extra: str) -> str:
    """initData as Telegram builds it: fields, plus a hash over their sorted "key=value" lines."""
    fields = {"query_id": "AAHdF6IQAAAAAN0XohDhrOrc", "auth_date": str(auth_date), **extra}
    if user is not None:
        fields["user"] = json.dumps(user, separators=(",", ":"), ensure_ascii=False)
    check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(fields)


def edit_field(init_data: str, key: str, value: str) -> str:
    """The same initData with one field changed and the old hash kept (what a forger would send)."""
    return urlencode([(k, value if k == key else v) for k, v in parse_qsl(init_data)])


# --- initData ------------------------------------------------------------------------

def test_genuine_init_data_gives_the_telegram_user() -> None:
    init_data = make_init_data(SAM, chat_instance="-8811", chat_type="private", signature="c2lnbmF0dXJl")
    user = verify_init_data(init_data, BOT_TOKEN, now=NOW + 60)
    assert user == SAM  # URL-decoded, with "&" and "ü" intact
    assert telegram_uid(user) == "tg:42"


def test_tampered_init_data_is_refused() -> None:
    init_data = make_init_data(SAM)
    forged = [
        edit_field(init_data, "user", json.dumps({**SAM, "id": 1})),  # someone else's id
        edit_field(init_data, "auth_date", str(NOW + 3600)),  # a fresher date
        edit_field(init_data, "hash", "0" * 64),
        init_data + "&chat_type=group",  # an extra field
        "&".join(part for part in init_data.split("&") if not part.startswith("hash=")),
    ]
    for payload in forged:
        with pytest.raises(InitDataError):
            verify_init_data(payload, BOT_TOKEN, now=NOW)
    with pytest.raises(InitDataError, match="signature"):
        verify_init_data(init_data, "654321:another-bot", now=NOW)  # signed for a different bot


def test_old_init_data_is_refused() -> None:
    fresh_enough = make_init_data(SAM, auth_date=NOW - INIT_DATA_MAX_AGE)
    assert verify_init_data(fresh_enough, BOT_TOKEN, now=NOW)["id"] == 42
    expired = make_init_data(SAM, auth_date=NOW - INIT_DATA_MAX_AGE - 1)
    with pytest.raises(InitDataError, match="expired"):
        verify_init_data(expired, BOT_TOKEN, now=NOW)
    with pytest.raises(InitDataError, match="future"):
        verify_init_data(make_init_data(SAM, auth_date=NOW + 3600), BOT_TOKEN, now=NOW)


def test_malformed_init_data_is_refused() -> None:
    no_user = make_init_data(None)
    not_users = [make_init_data(user) for user in ({"id": "42"}, {"id": True}, {"id": 0}, {"id": -5})]
    repeated = make_init_data(SAM) + "&user=" + json.dumps({"id": 1})
    # Telegram percent-encodes initData, so raw non-ASCII (or a lone surrogate) can't be genuine
    raw_text = [make_init_data(SAM).replace("%C3%BC", "ü"), "a=\ud800&hash=00"]
    for payload in ("", "not a query string", "a=1&&b", repeated, no_user, *not_users, "x" * 9000, *raw_text):
        with pytest.raises(InitDataError):
            verify_init_data(payload, BOT_TOKEN, now=NOW)


# --- session tokens --------------------------------------------------------------------

def test_session_tokens_carry_the_user_and_expire() -> None:
    key = session_key("s3cret", BOT_TOKEN)
    token = session_token(key, 2_000, "tg:42")
    assert token.startswith("tg:42.2000.")
    assert session_uid(key, token, now=1_000) == "tg:42"
    assert session_uid(key, session_token(key, 2_000), now=1_000) == "web"  # password sessions
    assert session_uid(key, token, now=3_000) is None  # expired
    assert session_uid(key, token.replace("tg:42", "tg:43", 1), now=1_000) is None  # someone else
    assert session_uid(key, token.replace("2000", "9000", 1), now=1_000) is None  # longer life
    assert session_uid(session_key("s3cret", "654321:other"), token, now=1_000) is None  # new bot token
    assert session_uid(session_key("changed", BOT_TOKEN), token, now=1_000) is None  # new password
    old_format = token.split(".", 1)[1]  # "{expires}.{sig}" from before user ids
    for bad in (old_format, "tg:٤٢.2000." + "0" * 64, "", None, "a.b.c.d"):
        assert session_uid(key, bad, now=1_000) is None
    assert valid_session(key, token, now=1_000) and not valid_session(key, token, now=3_000)
    with pytest.raises(ValueError):
        session_token(key, 2_000, "tg:42.9999")  # a dot would let the uid swallow the expiry


def test_the_session_key_needs_more_than_the_password() -> None:
    # Telegram users get tokens without knowing the password: the bot token keeps the key secret
    assert session_key("s3cret", BOT_TOKEN) != session_key("s3cret")
    assert session_key("s3cret", BOT_TOKEN) != session_key("s3cret", "654321:other")


def test_only_listed_telegram_users_are_allowed() -> None:
    assert uid_allowed("tg:42", {42, 7}) and uid_allowed("tg:7", {42, 7})
    assert not uid_allowed("tg:43", {42, 7})
    assert not uid_allowed("tg:42", ())  # an empty list lets no Telegram user in
    assert uid_allowed("web", ())  # the password works either way
    assert not uid_allowed("tg:abc", {42}) and not uid_allowed("tg:", {42})


# --- routes ------------------------------------------------------------------------------

def make_client(service: Any, bot_token: Optional[str] = BOT_TOKEN, base_url: str = "http://testserver",
                telegram_ids: Any = (42, 7)) -> TestClient:
    app = create_app(service, "s3cret", login_delay=0, bot_token=bot_token, telegram_ids=telegram_ids)

    @app.get("/test/whoami")
    def whoami(uid: str = Depends(current_uid)) -> Dict[str, str]:
        return {"uid": uid}

    return TestClient(app, base_url=base_url)


def test_telegram_sign_in_sets_a_personal_session(service) -> None:
    client = make_client(service)
    assert client.get("/test/whoami").status_code == 401
    response = client.post("/api/tg-login", json={"init_data": make_init_data(SAM, auth_date=int(time.time()))})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "uid": "tg:42", "name": "Sam & Jo"}
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "secure" not in cookie
    assert client.cookies[SESSION_COOKIE].startswith("tg:42.")
    assert client.get("/test/whoami").json() == {"uid": "tg:42"}
    assert client.get("/api/leagues").status_code == 200
    client.post("/api/logout")
    assert client.get("/test/whoami").status_code == 401


def test_telegram_sign_in_refuses_forged_or_old_data(service) -> None:
    client = make_client(service)
    genuine = make_init_data(SAM, auth_date=int(time.time()))
    for init_data in (edit_field(genuine, "user", json.dumps({**SAM, "id": 1})),
                      make_init_data(SAM, auth_date=int(time.time()) - INIT_DATA_MAX_AGE - 60),
                      make_init_data(SAM, auth_date=int(time.time()), token="654321:another-bot")):
        response = client.post("/api/tg-login", json={"init_data": init_data})
        assert response.status_code == 401
        assert SESSION_COOKIE not in client.cookies
    assert client.post("/api/tg-login", json={"init_data": ""}).status_code == 422
    # Without the bot token the dashboard can't check Telegram's signature at all
    off = make_client(service, bot_token=None)
    response = off.post("/api/tg-login", json={"init_data": genuine})
    assert response.status_code == 503 and "TELEGRAM_BOT_TOKEN" in response.json()["detail"]


def test_telegram_users_not_on_the_list_are_refused(service) -> None:
    # Anyone on Telegram can open the bot; the app (with its paid AI calls) is for the owner's list
    stranger = {"id": 99, "first_name": "Stranger"}
    client = make_client(service)
    response = client.post("/api/tg-login", json={"init_data": make_init_data(stranger, auth_date=int(time.time()))})
    assert response.status_code == 403
    assert "private" in response.json()["detail"] and "99" in response.json()["detail"]  # their id, to pass on
    assert SESSION_COOKIE not in client.cookies
    header = {INIT_DATA_HEADER: make_init_data(stranger, auth_date=int(time.time()))}
    assert client.get("/test/whoami", headers=header).status_code == 403
    assert client.post("/api/analysts/ask", headers=header,
                       json={"league": "basketball_nba", "id": "g1", "analyst": "claude"}).status_code == 403
    # With no list at all, nobody gets in through Telegram (the password still works)
    closed = make_client(service, telegram_ids=())
    sam = make_init_data(SAM, auth_date=int(time.time()))
    assert closed.post("/api/tg-login", json={"init_data": sam}).status_code == 403
    assert closed.get("/test/whoami", headers={INIT_DATA_HEADER: sam}).status_code == 403
    assert closed.post("/api/login", json={"password": "s3cret"}).status_code == 200


def test_taking_someone_off_the_list_ends_their_session(service) -> None:
    signed_in = make_client(service)
    signed_in.post("/api/tg-login", json={"init_data": make_init_data(SAM, auth_date=int(time.time()))})
    cookie = signed_in.cookies[SESSION_COOKIE]
    # The same dashboard (same password and bot token) restarted without Sam on the list
    later = make_client(service, telegram_ids=(7,))
    later.cookies.set(SESSION_COOKIE, cookie)
    assert later.get("/test/whoami").status_code == 401
    assert later.get("/api/leagues").status_code == 401


def test_sign_in_survives_text_that_isnt_valid_unicode(service) -> None:
    # A lone surrogate is valid JSON but can't be encoded as UTF-8: it must be a refusal, not a crash
    client = make_client(service)
    for route, body in (("/api/tg-login", b'{"init_data": "a=\\ud800&hash=00"}'),
                        ("/api/login", b'{"password": "\\ud800"}')):
        response = client.post(route, content=body, headers={"Content-Type": "application/json"})
        assert response.status_code in (401, 422), route
    header = {INIT_DATA_HEADER: "user=%ED%A0%80&hash=00"}  # percent-encoded surrogate bytes
    assert client.get("/test/whoami", headers=header).status_code == 401


def test_password_sessions_are_the_web_user(service) -> None:
    client = make_client(service)
    assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert client.cookies[SESSION_COOKIE].startswith("web.")
    assert client.get("/test/whoami").json() == {"uid": "web"}


def test_init_data_in_a_header_works_without_cookies(service) -> None:
    # Telegram Web runs the app in an iframe, where the browser may drop the session cookie
    client = make_client(service)
    header = {INIT_DATA_HEADER: make_init_data({"id": 7, "first_name": "Kim"}, auth_date=int(time.time()))}
    assert client.get("/test/whoami", headers=header).json() == {"uid": "tg:7"}
    assert client.get("/api/leagues", headers=header).status_code == 200
    forged = {INIT_DATA_HEADER: edit_field(header[INIT_DATA_HEADER], "user", json.dumps({"id": 1}))}
    assert client.get("/test/whoami", headers=forged).status_code == 401
    # Fresh initData names who has the app open now, over an older session in the same webview
    client.post("/api/login", json={"password": "s3cret"})
    assert client.get("/test/whoami", headers=header).json() == {"uid": "tg:7"}
    assert client.get("/test/whoami", headers=forged).json() == {"uid": "web"}  # bad header: the cookie


def test_session_cookie_is_secure_over_https(service) -> None:
    client = make_client(service, base_url="https://testserver")
    response = client.post("/api/tg-login", json={"init_data": make_init_data(SAM, auth_date=int(time.time()))})
    assert "secure" in response.headers["set-cookie"].lower()


# --- config and the dashboard's start-up -------------------------------------------------

def test_public_url_and_dashboard_host_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PUBLIC_URL", "DASHBOARD_HOST", "TELEGRAM_ALLOWED_IDS"):
        monkeypatch.delenv(name, raising=False)
    plain = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None)
    assert plain.dashboard_host == "0.0.0.0" and plain.public_url is None and plain.mini_app_url is None
    assert plain.telegram_user_ids == frozenset()
    behind_caddy = Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL=" https://bets.example.com/ ",
                          DASHBOARD_HOST="127.0.0.1", _env_file=None)
    assert behind_caddy.public_url == behind_caddy.mini_app_url == "https://bets.example.com"
    assert behind_caddy.dashboard_host == "127.0.0.1"
    assert Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL="", _env_file=None).public_url is None
    # Telegram only opens HTTPS pages as Mini Apps
    insecure = Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL="http://1.2.3.4:8787", _env_file=None)
    assert insecure.public_url == "http://1.2.3.4:8787" and insecure.mini_app_url is None


def test_telegram_allowed_ids_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_IDS", "12345, 67890,")  # as it would come from .env
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None)
    assert config.telegram_user_ids == {12345, 67890} and config.telegram_allowed_ids == "12345,67890"
    for typo in ("@sam", "12345;67890", "-100123", "0", "١٢٣"):
        with pytest.raises(ValueError, match="TELEGRAM_ALLOWED_IDS"):
            Config(TELEGRAM_BOT_TOKEN="123456:test", TELEGRAM_ALLOWED_IDS=typo, _env_file=None)


def test_dashboard_listens_on_dashboard_host_with_telegram_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config(TELEGRAM_BOT_TOKEN=BOT_TOKEN, DASHBOARD_PASSWORD="s3cret", DASHBOARD_HOST="127.0.0.1",
                    TELEGRAM_ALLOWED_IDS="42", _env_file=None)
    runs: List[Dict[str, Any]] = []
    monkeypatch.setattr(dashboard_main, "load_config", lambda: config)
    monkeypatch.setattr(dashboard_main, "build_service", lambda config: None)  # no odds key: no threads
    monkeypatch.setattr(dashboard_main.uvicorn, "run", lambda app, **kwargs: runs.append({"app": app, **kwargs}))
    dashboard_main.main()
    assert runs[0]["host"] == "127.0.0.1" and runs[0]["port"] == 8787
    client = TestClient(runs[0]["app"])
    response = client.post("/api/tg-login", json={"init_data": make_init_data(SAM, auth_date=int(time.time()))})
    assert response.json()["uid"] == "tg:42"
    kim = make_init_data({"id": 7, "first_name": "Kim"}, auth_date=int(time.time()))
    assert client.post("/api/tg-login", json={"init_data": kim}).status_code == 403  # not on the list


# --- the bot's "Open app" buttons ---------------------------------------------------------

def home_screen(config: Config, chat_type: str = "private") -> List[List[Any]]:
    """The /start screen's buttons as (text, web app url or callback data) rows."""
    application = Application.builder().token("123456:test").build()
    handlers.setup_handlers(application, config)
    sent = []

    class FakeMessage:
        async def reply_text(self, text, reply_markup=None, **kwargs):
            sent.append(reply_markup)

    update = SimpleNamespace(callback_query=None, message=FakeMessage(), effective_chat=SimpleNamespace(type=chat_type))
    asyncio.run(handlers.start(update, SimpleNamespace(application=application, user_data={})))
    return [[(b.text, b.web_app.url if b.web_app else b.callback_data) for b in row]
            for row in sent[0].inline_keyboard]


def test_home_screen_opens_the_app_when_public_url_is_set() -> None:
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL="https://bets.example.com", _env_file=None)
    rows = home_screen(config)
    assert rows[0] == [("📱 Open app", "https://bets.example.com")]
    assert rows[1][0] == ("🎯 Single game", "mode:single")
    # Telegram refuses Mini App buttons outside private chats, which would lose the whole menu
    assert all(text != "📱 Open app" for row in home_screen(config, "group") for text, _ in row)
    for no_app in (Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None),
                   Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL="http://1.2.3.4:8787", _env_file=None)):
        assert home_screen(no_app)[0][0] == ("🎯 Single game", "mode:single")


def test_menu_button_opens_the_app_at_startup(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    calls = []

    async def set_chat_menu_button(self, chat_id=None, menu_button=None, **kwargs):  # no network
        calls.append(menu_button)
        return True

    monkeypatch.setattr(ExtBot, "set_chat_menu_button", set_chat_menu_button)
    config = Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL="https://bets.example.com", _env_file=None)
    application = bot_main.build_application(config)
    asyncio.run(application.post_init(application))
    assert isinstance(calls[0], MenuButtonWebApp)
    assert calls[0].text == "Open app" and calls[0].web_app.url == "https://bets.example.com"

    # Without PUBLIC_URL (or with an address Telegram won't open) the button goes back to the plain
    # one, so a button from an earlier address doesn't lead to a dead page
    for no_app in (Config(TELEGRAM_BOT_TOKEN="123456:test", _env_file=None),
                   Config(TELEGRAM_BOT_TOKEN="123456:test", PUBLIC_URL="http://1.2.3.4:8787", _env_file=None)):
        plain = bot_main.build_application(no_app)
        asyncio.run(plain.post_init(plain))
    assert len(calls) == 3
    assert all(type(button) is MenuButtonDefault for button in calls[1:])

    # A refusal from Telegram is logged; the bot still starts
    async def refuse(self, chat_id=None, menu_button=None, **kwargs):
        raise BadRequest("Url host is empty")

    monkeypatch.setattr(ExtBot, "set_chat_menu_button", refuse)
    with caplog.at_level(logging.WARNING):
        asyncio.run(application.post_init(application))
        asyncio.run(plain.post_init(plain))
    assert caplog.text.count("menu button") == 2
