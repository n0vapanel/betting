"""Who is signed in: Telegram Mini App sign-in and signed session tokens.

Inside Telegram the app is opened as a Mini App and Telegram hands the page
``initData``: a query string describing the user, signed with a key derived
from the bot token. Checking that signature on the server gives each friend a
personal login without a password (https://core.telegram.org/bots/webapps,
"Validating data received via the Mini App"). Browsers keep the shared
dashboard password.

A genuine signature only proves who someone is, and anyone on Telegram can
open the bot. The app is private (the owner and friends), so only Telegram ids
the owner listed in TELEGRAM_ALLOWED_IDS get in; with no list, nobody signs in
through Telegram and the password is the only way in.

Either way the server then sets a session cookie "{uid}.{expires}.{sig}":
the user id ("tg:<telegram id>", or "web" for the password), when the
session ends, and an HMAC over both. Nothing is stored server side; changing
the password or the bot token changes the signing key and signs everyone out.

Pure functions only (no FastAPI), so they are easy to test with made-up
payloads.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from typing import Any, Collection, Dict, Optional
from urllib.parse import parse_qsl

# Telegram signs initData when the Mini App opens; older payloads are refused so a leaked one
# stops working within a day
INIT_DATA_MAX_AGE = 24 * 3600
# Allowed difference between Telegram's clock and ours for an auth_date "from the future"
CLOCK_SKEW = 300
MAX_INIT_DATA_LENGTH = 8192
WEB_UID = "web"
# "web", "tg:12345", and room for later kinds such as "web:<random id>". ASCII only on purpose:
# Python's \d and \w would also accept other scripts' digits and letters
UID_PATTERN = re.compile(r"[a-z]{2,8}(?::[A-Za-z0-9_-]{1,64})?")
EXPIRES_PATTERN = re.compile(r"[0-9]{1,12}")
TELEGRAM_ID_PATTERN = re.compile(r"[0-9]{1,20}")


class InitDataError(ValueError):
    """Telegram's initData was missing, malformed, tampered with or too old."""


# --- Telegram Mini App initData ---------------------------------------------------

def webapp_key(bot_token: str) -> bytes:
    """The key Telegram signs initData with: HMAC-SHA256 of the bot token under the constant "WebAppData"."""
    return hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()


def parse_init_data(init_data: str) -> Dict[str, str]:
    """initData's fields (URL-decoded). A repeated field is refused: which copy was signed would be unclear."""
    if not init_data or len(init_data) > MAX_INIT_DATA_LENGTH:
        raise InitDataError("No Telegram sign-in data")
    # Telegram percent-encodes everything, so genuine initData is plain ASCII. Anything else (such as a
    # lone surrogate from a JSON body, which can't even be encoded for the HMAC) is refused here
    if not init_data.isascii():
        raise InitDataError("Telegram sign-in data is malformed")
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True, max_num_fields=64)
    except ValueError:
        raise InitDataError("Telegram sign-in data is malformed") from None
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise InitDataError("Telegram sign-in data repeats a field")
    return fields


def data_check_string(fields: Dict[str, str]) -> str:
    """What Telegram signs: every field except the hash, as sorted "key=value" lines."""
    return "\n".join(f"{key}={value}" for key, value in sorted(fields.items()) if key != "hash")


def verify_init_data(init_data: str, bot_token: str, now: Optional[float] = None,
                     max_age: int = INIT_DATA_MAX_AGE) -> Dict[str, Any]:
    """The Telegram user ({"id", "first_name", ...}) if initData is genuine and recent; else InitDataError."""
    fields = parse_init_data(init_data)
    received = fields.get("hash", "")
    expected = hmac.new(webapp_key(bot_token), data_check_string(fields).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected.encode(), received.encode()):
        raise InitDataError("Telegram sign-in data has a wrong signature")
    # The signature is good, so the remaining fields really come from Telegram
    auth_date = fields.get("auth_date", "")
    if not EXPIRES_PATTERN.fullmatch(auth_date):
        raise InitDataError("Telegram sign-in data has no date")
    age = (time.time() if now is None else now) - int(auth_date)
    if age > max_age:
        raise InitDataError("Telegram sign-in data has expired: reopen the app")
    if age < -CLOCK_SKEW:
        raise InitDataError("Telegram sign-in data is dated in the future")
    try:
        user = json.loads(fields.get("user", ""))
    except ValueError:
        raise InitDataError("Telegram sign-in data has no user") from None
    if not isinstance(user, dict) or type(user.get("id")) is not int or user["id"] <= 0:
        raise InitDataError("Telegram sign-in data has no user")
    return user


def telegram_uid(user: Dict[str, Any]) -> str:
    """The app's id for a Telegram user, the same one the bot saves calls under ("tg:<id>")."""
    return f"tg:{int(user['id'])}"


def uid_allowed(uid: str, telegram_ids: Collection[int]) -> bool:
    """Whether uid may use the app: password users always, Telegram users only if the owner listed them.

    Checked on every request, not just at sign-in, so taking someone off the list also ends the
    sessions they already have.
    """
    kind, _, ident = uid.partition(":")
    if kind != "tg":
        return True
    return TELEGRAM_ID_PATTERN.fullmatch(ident) is not None and int(ident) in telegram_ids


def private_app_message(user: Dict[str, Any]) -> str:
    """What a Telegram user who isn't on the list sees: their own id, so they can pass it to the owner."""
    return (f"This app is private. To get in, ask the owner to add your Telegram id ({int(user['id'])}) "
            "to TELEGRAM_ALLOWED_IDS.")


# --- session tokens ----------------------------------------------------------------

def session_key(password: str, secret: Optional[str] = None) -> bytes:
    """The key sessions are signed with, derived from the dashboard password and the bot token.

    Telegram users get tokens without knowing the password, so the key mixes in the bot token too:
    a token alone must not let anyone guess the password offline and then sign tokens for others.
    """
    material = password if secret is None else f"{password}\n{secret}"
    return hmac.new(material.encode(), b"sportsbet-ai session", hashlib.sha256).digest()


def _signature(key: bytes, uid: str, expires: int) -> str:
    return hmac.new(key, f"{uid}.{expires}".encode(), hashlib.sha256).hexdigest()


def session_token(key: bytes, expires: int, uid: str = WEB_UID) -> str:
    """A signed session for uid that ends at expires (Unix seconds): "{uid}.{expires}.{sig}"."""
    if not UID_PATTERN.fullmatch(uid):
        raise ValueError(f"Not a valid user id: {uid!r}")
    return f"{uid}.{expires}.{_signature(key, uid, expires)}"


def session_uid(key: bytes, token: Optional[str], now: Optional[float] = None) -> Optional[str]:
    """The signed-in user id if token is genuine and not expired, else None."""
    head, _, signature = (token or "").rpartition(".")
    uid, _, expires = head.rpartition(".")
    if not UID_PATTERN.fullmatch(uid) or not EXPIRES_PATTERN.fullmatch(expires):
        return None
    if int(expires) < (time.time() if now is None else now):
        return None
    if not hmac.compare_digest(_signature(key, uid, int(expires)).encode(), signature.encode()):
        return None
    return uid


def valid_session(key: bytes, token: Optional[str], now: Optional[float] = None) -> bool:
    return session_uid(key, token, now) is not None
