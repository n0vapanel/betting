"""The learning loop (v3): the brain backtests old games, scores itself walk-forward against the market and simple
baselines, and tunes its own parameters behind statistical gates with automatic rollback.

- ``history``: backfill of old games, lines and match stats (point in time);
- ``registry``: the allow-listed, versioned parameters the engine reads (defaults are today's constants);
- ``evaluate``: walk-forward scores, calibration and the paired tests;
- ``tune``: the weekly refits, the gates, promotions, rollbacks and safe mode;
- ``loop`` and ``analyst``: the scheduler thread and whichever AI is linked.
"""
