"""Run the web dashboard: python -m telegram_betting_bot_simple.dashboard"""

from __future__ import annotations

import logging
import sys
import threading
import time

import uvicorn

from ..config import load_config
from ..services import build_service
from .app import create_app
from .service import DashboardService

GRADE_EVERY_SECONDS = 600


def grade_forever(service: DashboardService) -> None:
    """Score people's calls and the AI's predictions as games finish (ESPN only, no odds quota)."""
    while True:
        try:
            service.grade_due()
        except Exception:
            logging.exception("Grading finished games failed")
        time.sleep(GRADE_EVERY_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    if not config.dashboard_password:
        sys.exit("Set DASHBOARD_PASSWORD in the .env file to start the dashboard.")
    service = build_service(config)
    if service is not None:
        threading.Thread(target=grade_forever, args=(service,), name="grader", daemon=True).start()
    app = create_app(service, config.dashboard_password)
    uvicorn.run(app, host="0.0.0.0", port=config.dashboard_port, access_log=False)


if __name__ == "__main__":
    main()
