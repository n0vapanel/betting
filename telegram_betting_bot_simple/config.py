# telegram_betting_bot_simple/config.py
from typing import FrozenSet, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

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
    odds_regions: str = Field("us", validation_alias="ODDS_REGIONS")
    # Below this many Odds API requests left this month, serve saved odds instead of fetching
    odds_reserve: int = Field(50, validation_alias="ODDS_RESERVE")
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
    def mini_app_url(self) -> Optional[str]:
        """PUBLIC_URL if Telegram can open it as a Mini App (HTTPS only), else None."""
        if self.public_url and self.public_url.lower().startswith("https://"):
            return self.public_url
        return None

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
