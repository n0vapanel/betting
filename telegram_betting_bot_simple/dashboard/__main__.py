"""Run the web dashboard: python -m telegram_betting_bot_simple.dashboard"""

from __future__ import annotations

import logging
import sys

import uvicorn

from ..config import load_config
from ..services import build_service
from .app import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    if not config.dashboard_password:
        sys.exit("Set DASHBOARD_PASSWORD in the .env file to start the dashboard.")
    app = create_app(build_service(config), config.dashboard_password)
    uvicorn.run(app, host="0.0.0.0", port=config.dashboard_port, access_log=False)


if __name__ == "__main__":
    main()
