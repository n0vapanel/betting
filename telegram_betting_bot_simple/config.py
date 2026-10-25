# telegram_betting_bot_simple/config.py
from typing import Optional

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
    # .env also holds settings for other tools (e.g. TELEGRAM_CHANNEL_ID for brain/notify.py)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

def load_config(env_file: str = ".env") -> Config:
    load_dotenv(env_file)
    return Config()  # type: ignore[call-arg]
