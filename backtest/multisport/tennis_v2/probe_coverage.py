"""Probe: name-match coverage tennis-data <-> Sackmann, serve stat coverage, DOB coverage."""
import re
import pandas as pd
import numpy as np

WORK = "/Users/x/Desktop/x/Coding Projects/betting bot/backtest/multisport/tennis_v2"


def norm(name):
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- tennis-data names
td_names = set()
for y in (2021, 2022):
    df = pd.read_excel(f"{WORK}/{y}.xlsx")
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()]
    td_names |= set(df["Winner"]) | set(df["Loser"])
print("td unique names:", len(td_names))

# --- sackmann names + stats
sack = {}
for y in (2019, 2020, 2021, 2022):
    df = pd.read_csv(f"{WORK}/raw/atp_matches_{y}.csv")
    for _, r in df.iterrows():
        for side in ("w", "l"):
            nm = norm(r[f"{'winner' if side == 'w' else 'loser'}_name"])
            if nm:
                sack.setdefault(nm, 0)
                sack[nm] += 1
print("sack normalized names:", len(sack))

miss = [n for n in td_names if norm(n) not in sack]
print("td names missing in sackmann:", len(miss), "/", len(td_names))
for m in sorted(miss)[:40]:
    print("   MISS:", m)

# --- which missing names matter (match-level coverage)
tot = 0
both = 0
none = 0
one = 0
for y in (2021, 2022):
    df = pd.read_excel(f"{WORK}/{y}.xlsx")
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()]
    df = df[df["B365W"].notna() & df["B365L"].notna()]
    df = df[df["Comment"].fillna("").str.contains("Completed", na=False)]
    for _, r in df.iterrows():
        tot += 1
        w, l = norm(r["Winner"]) in sack, norm(r["Loser"]) in sack
        if w and l:
            both += 1
        elif not w and not l:
            none += 1
        else:
            one += 1
print(f"\nmatches {tot}: both {both} ({both/tot:.1%}) | one {one} ({one/tot:.1%}) | none {none} ({none/tot:.1%})")

# --- dob coverage with NEW players file
players = pd.read_csv(f"{WORK}/raw/atp_players.csv")
print("\nplayers rows:", len(players), "dob non-null:", players["dob"].notna().sum())
players["n"] = players["name_last"].fillna("").apply(norm)
sack_dob = dict(zip(players["n"], players["dob"]))
have = [n for n in td_names if n in sack_dob and pd.notna(sack_dob[n])]
print("td names with dob (new file):", len(have), "/", len(td_names))
