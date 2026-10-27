# telegram_betting_bot_simple/config.py
import re
from typing import Any, FrozenSet, Literal, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# The Odds API bookmakers the league board asks for (checked against its bookmaker list on 27 October 2026):
# Pinnacle and two exchanges anchor the consensus; the rest are US books people can bet at. Caesars
# (williamhill_us) is listed for paid plans only, and espnbet is now called "theScore Bet"
DEFAULT_BOOKMAKERS = ("pinnacle,betfair_ex_eu,matchbook,draftkings,fanduel,betmgm,williamhill_us,espnbet,betrivers,"
                      "bovada")
DEFAULT_SHARP_BOOKS = "pinnacle,betfair_ex_eu,matchbook"
MAX_BOOKMAKERS = 10  # every 10 bookmakers count as one region (3 credits a board for three markets)
BOOK_KEY = re.compile(r"[a-z0-9][a-z0-9_]{1,39}")

class Config(BaseSettings):
    telegram_token: str = Field(..., validation_alias="TELEGRAM_BOT_TOKEN")
    sports_api_key: Optional[str] = Field(None, validation_alias="SPORTS_API_KEY")
    api_sports_key: Optional[str] = Field(None, validation_alias="API_SPORTS_KEY")
    odds_api_key: Optional[str] = Field(None, validation_alias="THE_ODDS_API_KEY")
    news_api_key: Optional[str] = Field(None, validation_alias="NEWS_API_KEY")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
    # IANA name of the timezone game start times are shown in
    timezone: str = Field("America/New_York", validation_alias="TIMEZONE")
    # Web dashboard: login password (required to start it) and port
    dashboard_password: Optional[str] = Field(None, validation_alias="DASHBOARD_PASSWORD")
    dashboard_port: int = Field(8787, validation_alias="DASHBOARD_PORT")
    # Address the dashboard listens on: everywhere by default, 127.0.0.1 when Caddy serves it over HTTPS
    dashboard_host: str = Field("0.0.0.0", validation_alias="DASHBOARD_HOST")
    # The dashboard's public HTTPS address (e.g. https://bets.example.com). When set, the bot opens it
    # as a Telegram Mini App from its menu button and home screen
    public_url: Optional[str] = Field(None, validation_alias="PUBLIC_URL")
    # Telegram user ids (comma-separated) allowed to open the app from Telegram: the app is private,
    # and anyone on Telegram can find the bot. Empty means nobody signs in through Telegram (the
    # dashboard password still works). Kept as text: pydantic-settings would read a list as JSON
    telegram_allowed_ids: str = Field("", validation_alias="TELEGRAM_ALLOWED_IDS")
    # Minutes to keep a league's odds before fetching again (each fetch costs 3 quota units)
    odds_cache_minutes: float = Field(120, validation_alias="ODDS_CACHE_MINUTES")
    # Only used when ODDS_BOOKMAKERS is "none": the odds board then asks for whole regions (3 credits each)
    odds_regions: str = Field("us", validation_alias="ODDS_REGIONS")
    # The bookmakers each league's odds board asks for (comma-separated Odds API keys, at most 10: every 10
    # count as one region, so a board still costs 3 credits). Blank means the default list; "none" goes back to
    # ODDS_REGIONS. SHARP_BOOKS (Pinnacle and the exchanges) only anchor the markets brain's consensus: nobody is
    # offered their prices. ODDS_PRICE_BOOKS are the books a person can actually bet at, the only ones "best
    # price", the slip, the AI's tickets, the target builder and a pasted slip's "better price" use (blank: every
    # book in ODDS_BOOKMAKERS that isn't sharp or an exchange)
    odds_bookmakers: str = Field(DEFAULT_BOOKMAKERS, validation_alias="ODDS_BOOKMAKERS")
    sharp_books: str = Field(DEFAULT_SHARP_BOOKS, validation_alias="SHARP_BOOKS")
    odds_price_books: str = Field("", validation_alias="ODDS_PRICE_BOOKS")
    # The markets brain's free outside opinions (Polymarket, Kalshi, ESPN's predictor and power ratings, Torvik,
    # Tennis Abstract, eloratings.net), refreshed in the background by the dashboard's slate worker
    markets_enabled: bool = Field(True, validation_alias="MARKETS_ENABLED")
    # Below this many Odds API requests left this month, serve saved odds instead of fetching
    odds_reserve: int = Field(50, validation_alias="ODDS_RESERVE")
    # Today's all-sports slate (data/slate.py): seconds between ESPN scoreboard checks for a league with a game
    # live, with games still to play today, with none today, and for the days ahead; and how many odds
    # boards a day the app may load when someone opens a league whose odds are over 2 hours old (3 credits each)
    slate_live_seconds: float = Field(20, ge=5, validation_alias="SLATE_LIVE_SECONDS")
    slate_today_seconds: float = Field(60, ge=10, validation_alias="SLATE_TODAY_SECONDS")
    slate_idle_seconds: float = Field(600, ge=60, validation_alias="SLATE_IDLE_SECONDS")
    slate_future_seconds: float = Field(3600, ge=60, validation_alias="SLATE_FUTURE_SECONDS")
    slate_odds_per_day: int = Field(12, ge=0, validation_alias="SLATE_ODDS_PER_DAY")
    # AI analysts: each one is switched on by its API key. Their forecasts are graded, and only move the
    # AI's own chances once their record earns it
    anthropic_api_key: Optional[str] = Field(None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-opus-5-5", validation_alias="ANTHROPIC_MODEL")
    deepseek_api_key: Optional[str] = Field(None, validation_alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field("deepseek-v4-pro", validation_alias="DEEPSEEK_MODEL")
    # Calls per analyst per 24 hours (each costs money on your API account), how hard Claude thinks,
    # whether Claude checks the latest news with web search first (about a cent per search), and how
    # many hours before tip-off games get analysed
    ai_daily_limit: int = Field(25, validation_alias="AI_DAILY_LIMIT")
    ai_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field("medium", validation_alias="AI_EFFORT")
    ai_web_search: bool = Field(False, validation_alias="AI_WEB_SEARCH")
    ai_lookahead_hours: float = Field(6.0, validation_alias="AI_LOOKAHEAD_HOURS")
    # The AI chat assistant (needs ANTHROPIC_API_KEY): US cents of Claude each person may use per day,
    # the model that chats and calls the brain's tools, and the small model that reads each message
    # first (what is asked, and any warning signs of problem gambling)
    ai_chat_daily_cents: int = Field(100, ge=0, validation_alias="AI_CHAT_DAILY_CENTS")
    # Daily caps (US cents) shared by many people: AI_CHAT_WEB_DAILY_CENTS for everyone signed in with the
    # dashboard password together (each browser is told apart by an id it makes up, so a new one mustn't
    # mean a fresh allowance), and AI_CHAT_DAILY_TOTAL_CENTS for the whole app. Unset, they are 3 and 10
    # people's AI_CHAT_DAILY_CENTS; 0 switches that group's AI chat off
    ai_chat_web_daily_cents: Optional[int] = Field(None, ge=0, validation_alias="AI_CHAT_WEB_DAILY_CENTS")
    ai_chat_daily_total_cents: Optional[int] = Field(None, ge=0, validation_alias="AI_CHAT_DAILY_TOTAL_CENTS")
    ai_chat_model: str = Field("claude-opus-5-5", validation_alias="AI_CHAT_MODEL")
    ai_router_model: str = Field("claude-haiku-5-5", validation_alias="AI_ROUTER_MODEL")
    # Writable folder for the odds store and bot state shared by the bot and dashboard;
    # systemd sets STATE_DIRECTORY for services with StateDirectory=
    state_dir: Optional[str] = Field(None, validation_alias="STATE_DIRECTORY")
    # .env also holds settings for other tools (e.g. TELEGRAM_CHANNEL_ID for brain/notify.py)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("public_url")
    @classmethod
    def _tidy_public_url(cls, value: Optional[str]) -> Optional[str]:
        # "PUBLIC_URL=" in .env means unset; a trailing slash would double up when paths are added
        value = (value or "").strip().rstrip("/")
        return value or None

    @field_validator("markets_enabled", mode="before")
    @classmethod
    def _blank_is_on(cls, value: object) -> object:
        # "MARKETS_ENABLED=" in .env means the default (on), not a startup error
        return True if isinstance(value, str) and not value.strip() else value

    @field_validator("ai_chat_web_daily_cents", "ai_chat_daily_total_cents", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        # "AI_CHAT_WEB_DAILY_CENTS=" in .env means unset (the default), not a startup error
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("odds_bookmakers", "sharp_books", "odds_price_books", mode="before")
    @classmethod
    def _check_books(cls, value: object, info: Any) -> str:
        # A typo fails loudly at startup instead of quietly dropping a bookmaker (or every price) from the board
        text = "" if value is None else str(value)
        if info.field_name == "odds_bookmakers" and not text.strip():
            return DEFAULT_BOOKMAKERS  # "ODDS_BOOKMAKERS=" in .env means the default list
        if info.field_name == "sharp_books" and not text.strip():
            return DEFAULT_SHARP_BOOKS
        if info.field_name == "odds_bookmakers" and text.strip().lower() == "none":
            return "none"
        keys = parse_book_keys(text, info.field_name.upper())
        if info.field_name == "odds_bookmakers" and len(keys) > MAX_BOOKMAKERS:
            raise ValueError(f"ODDS_BOOKMAKERS can hold at most {MAX_BOOKMAKERS} bookmakers: every 10 cost one more "
                             "region (3 more credits a board)")
        return ",".join(keys)

    @field_validator("telegram_allowed_ids")
    @classmethod
    def _check_telegram_ids(cls, value: str) -> str:
        # A typo fails loudly at startup instead of quietly locking a friend out
        return ",".join(str(user_id) for user_id in sorted(parse_telegram_ids(value)))

    @property
    def telegram_user_ids(self) -> FrozenSet[int]:
        """TELEGRAM_ALLOWED_IDS as numbers."""
        return parse_telegram_ids(self.telegram_allowed_ids)

    @property
    def bookmakers(self) -> Tuple[str, ...]:
        """ODDS_BOOKMAKERS as keys; empty when it is "none" (the board then asks for ODDS_REGIONS)."""
        return () if self.odds_bookmakers == "none" else parse_book_keys(self.odds_bookmakers, "ODDS_BOOKMAKERS")

    @property
    def sharp_book_keys(self) -> Tuple[str, ...]:
        return parse_book_keys(self.sharp_books, "SHARP_BOOKS")

    @property
    def price_book_keys(self) -> Tuple[str, ...]:
        """ODDS_PRICE_BOOKS as keys; empty means every book in ODDS_BOOKMAKERS that isn't sharp or an exchange."""
        return parse_book_keys(self.odds_price_books, "ODDS_PRICE_BOOKS")

    @property
    def mini_app_url(self) -> Optional[str]:
        """PUBLIC_URL if Telegram can open it as a Mini App (HTTPS only), else None."""
        if self.public_url and self.public_url.lower().startswith("https://"):
            return self.public_url
        return None

def parse_book_keys(text: str, setting: str = "ODDS_BOOKMAKERS") -> Tuple[str, ...]:
    """Bookmaker keys from "pinnacle, draftkings" (lowercase, spaces and empty entries ignored, no repeats)."""
    keys = []
    for part in (text or "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if not BOOK_KEY.fullmatch(part):
            # The bad entry isn't repeated: something pasted into the wrong line would end up in the logs
            raise ValueError(f"{setting} should be The Odds API's bookmaker keys separated by commas, "
                             "e.g. pinnacle,draftkings,fanduel")
        if part not in keys:
            keys.append(part)
    return tuple(keys)

def parse_telegram_ids(text: str) -> FrozenSet[int]:
    """Telegram user ids from "12345, 67890" (spaces and empty entries are ignored)."""
    ids = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if not (part.isascii() and part.isdigit()) or len(part) > 20 or int(part) <= 0:
            # The bad entry isn't repeated: a token pasted into the wrong line would end up in the logs
            raise ValueError("TELEGRAM_ALLOWED_IDS should be Telegram user ids separated by commas, e.g. 12345,67890")
        ids.add(int(part))
    return frozenset(ids)

def load_config(env_file: str = ".env") -> Config:
    load_dotenv(env_file)
    return Config()  # type: ignore[call-arg]
