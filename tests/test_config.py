"""Tests for loading the bot's settings."""

from __future__ import annotations

import pytest

from telegram_betting_bot_simple.config import Config

def test_config_ignores_other_tools_settings_in_env_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=123456:test\n"
        "TELEGRAM_CHANNEL_ID=42\n"
        "# Timezone for game start times\n"
        "TIMEZONE=Europe/London\n"
    )
    for name in ("TELEGRAM_BOT_TOKEN", "TIMEZONE"):
        monkeypatch.delenv(name, raising=False)
    config = Config(_env_file=env_file)
    assert config.telegram_token == "123456:test"
    assert config.timezone == "Europe/London"
