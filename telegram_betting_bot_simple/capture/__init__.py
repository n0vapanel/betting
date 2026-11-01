"""Raw capture (v4 P1, plan section 3.3): an untrimmed copy of every game's event data, kept apart from betting.db.

- ``raw_store.RawStore``: the separate WAL SQLite file (RAW_DB_PATH, default ``<STATE_DIR>/raw.db``) with
  ``raw_payloads`` (every response, zlib-compressed, deduplicated on its sha256), ``capture_jobs`` (one per game: the
  pre-start horizons and the final's retries), the request budgets and the backfill's cursors.
- ``http.PoliteHttp``: the capture's one door to the network (urllib, the environment's proxy and CA bundle), with a
  token bucket per host and a back-off on 429 and 5xx answers. ``http.Unreachable`` is raised when a host can't be
  reached.
- ``plan``: what is captured for each sport (``KINDS``), when (the horizons and the final's retries) and the 58 ESPN
  paths.
- ``worker.Capturer``: schedules jobs from the slate and from a nightly sweep of yesterday's scoreboards, runs what is
  due within the day's budget, the static sources on their own schedule, and the backfill of older games inside its
  night window (forward jobs always first). ``worker.capture_forever`` is the dashboard's "capture" thread.
- ``sources``: the fetchers (``RawFetch``), one module per source.

Nothing here touches The Odds API: capture is ESPN, MLB StatsAPI, GitHub-hosted history files, football-data and
Kalshi's public series only.
"""

__all__: list = []
