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
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

def load_config(env_file: str = ".env") -> Config:
    load_dotenv(env_file)
    return Config()  # type: ignore[call-arg]
