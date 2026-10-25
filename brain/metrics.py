#!/usr/bin/env python3
"""Forever-learning metrics spine: append per-retrain evaluation rows, watch drift.

Usage:
    metrics.py append --sport MMA --version v2 --window 2020-2026 --n 2843 \
        --acc 0.802 --roi 0.403 --brier 0.1368 --ci-lo 0.361 --ci-hi 0.444
    metrics.py trend MMA
    metrics.py last
"""
import json, os, argparse
from datetime import datetime, timezone

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_history.jsonl")


def _load():
    if not os.path.exists(HIST):
        return []
    return [json.loads(l) for l in open(HIST) if l.strip()]


def append_metrics(sport, version, window, n, acc, roi, brier, ci_lo=None,
                   ci_hi=None, note=""):
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "sport": sport, "version": version, "window": window, "n": int(n),
           "acc": float(acc), "roi": float(roi), "brier": float(brier),
           "ci_lo": float(ci_lo) if ci_lo is not None else None,
           "ci_hi": float(ci_hi) if ci_hi is not None else None,
           "note": note}
    with open(HIST, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def trend(sport, k=10):
    rows = [r for r in _load() if r["sport"] == sport]
    rows.sort(key=lambda r: r["ts"])
    out = [f"{r['version']} {r['window']} n={r['n']} acc={r['acc']*100:.1f}% "
           f"roi={r['roi']*100:+.1f}% brier={r['brier']:.4f}"
           for r in rows[-k:]]
    return "\n".join(out) if out else f"(no metrics for {sport})"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("append")
    for k in ("sport", "version", "window"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--n", type=int, required=True)
    for k in ("acc", "roi", "brier"):
        p.add_argument(f"--{k}", type=float, required=True)
    p.add_argument("--ci-lo", type=float); p.add_argument("--ci-hi", type=float)
    p.add_argument("--note", default="")
    t = sub.add_parser("trend"); t.add_argument("sport"); t.add_argument("--k", type=int, default=10)
    sub.add_parser("last")
    a = ap.parse_args()
    if a.cmd == "append":
        append_metrics(a.sport, a.version, a.window, a.n, a.acc, a.roi, a.brier,
                       a.ci_lo, a.ci_hi, a.note)
        print("appended")
    elif a.cmd == "trend":
        print(trend(a.sport, a.k))
    elif a.cmd == "last":
        rows = _load()
        print(json.dumps(rows[-1], indent=1) if rows else "(empty)")


if __name__ == "__main__":
    main()
