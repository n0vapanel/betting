#!/usr/bin/env python3
"""Paper-trade + live-trade ledger for the brain. Stdlib only.

Usage:
    ledger.py log --sport MMA --matchup "Fiziev vs Ruffy" --side R \
        --odds-american -124 --model-prob 0.709 --fair-prob 0.533 \
        --edge 0.176 --stake 87.06 --note "q-Kelly, cap 0.25"
    ledger.py settle <bet_id> W|L   (W pays (stake * payout), L zero)
    ledger.py report [--sport MMA]
"""
import json, os, sys, time, argparse
from datetime import datetime, timezone

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")


def _load():
    if not os.path.exists(LEDGER):
        return []
    rows = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save(rows):
    with open(LEDGER, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def am_dec(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def log_bet(sport, matchup, side, odds_american, model_prob, fair_prob, edge,
            stake=1.0, note="", mode="paper"):
    rows = _load()
    bet = {
        "bet_id": f"{sport}-{int(time.time())}-{len(rows) + 1}",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode, "sport": sport, "matchup": matchup, "side": side,
        "odds_american": float(odds_american),
        "dec": round(am_dec(odds_american), 4),
        "model_prob": float(model_prob), "fair_prob": float(fair_prob),
        "edge": float(edge), "stake": float(stake),
        "status": "open", "result": None, "payout": None, "note": note,
    }
    rows.append(bet)
    _save(rows)
    return bet["bet_id"]


def settle(bet_id, result):
    assert result in ("W", "L")
    rows = _load()
    for r in rows:
        if r["bet_id"] == bet_id:
            r["status"] = "settled"
            r["result"] = result
            r["payout"] = round(r["stake"] * (r["dec"] - 1), 2) if result == "W" else 0.0
            _save(rows)
            return r
    raise KeyError(bet_id)


def report(sport=None):
    rows = _load()
    if sport:
        rows = [r for r in rows if r["sport"] == sport]
    out = {}
    for r in rows:
        k = (r["sport"], r["mode"])
        o = out.setdefault(k, {"n": 0, "open": 0, "won": 0, "staked": 0.0,
                               "staked_settled": 0.0, "returned": 0.0, "edge_sum": 0.0})
        o["staked"] += r["stake"]
        o["edge_sum"] += r["edge"]
        if r["status"] == "open":
            o["open"] += 1
        else:
            o["n"] += 1
            o["staked_settled"] += r["stake"]
            if r["result"] == "W":
                o["won"] += 1
                o["returned"] += r["payout"] + r["stake"]
    lines = []
    for (sp, mode), o in sorted(out.items()):
        roi = (o["returned"] - o["staked_settled"]) / o["staked_settled"] * 100 if o["staked_settled"] else 0.0
        wr = o["won"] / o["n"] * 100 if o["n"] else 0.0
        avg_edge = o["edge_sum"] / (o["n"] + o["open"]) if (o["n"] + o["open"]) else 0.0
        lines.append(f"{sp}/{mode}: settled={o['n']} open={o['open']} "
                     f"winrate={wr:.1f}% ROI={roi:+.2f}% avgEdge={avg_edge:+.3f}")
    return "\n".join(lines) if lines else "(empty ledger)"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("log")
    p.add_argument("--sport", required=True); p.add_argument("--matchup", required=True)
    p.add_argument("--side", required=True); p.add_argument("--odds-american", required=True)
    p.add_argument("--model-prob", type=float, required=True)
    p.add_argument("--fair-prob", type=float, required=True)
    p.add_argument("--edge", type=float, required=True)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--note", default=""); p.add_argument("--mode", default="paper")
    s = sub.add_parser("settle"); s.add_argument("bet_id"); s.add_argument("result")
    r = sub.add_parser("report"); r.add_argument("--sport", default=None)
    a = ap.parse_args()
    if a.cmd == "log":
        bid = log_bet(a.sport, a.matchup, a.side, a.odds_american, a.model_prob,
                      a.fair_prob, a.edge, a.stake, a.note, a.mode)
        print(bid)
    elif a.cmd == "settle":
        print(json.dumps(settle(a.bet_id, a.result)))
    elif a.cmd == "report":
        print(report(a.sport))


if __name__ == "__main__":
    main()
