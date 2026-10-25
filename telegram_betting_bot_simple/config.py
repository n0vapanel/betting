# telegram_betting_bot_simple/config.py
from typing import Literal, Optional

from pydantic import Field
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

def load_config(env_file: str = ".env") -> Config:
    load_dotenv(env_file)
    return Config()  # type: ignore[call-arg]
