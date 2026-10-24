import pandas as pd
from build_data import CODE2NAME

mp = pd.read_csv('all_teams.csv', low_memory=False)
mp = mp[mp.season == 2021].copy()
mp['date'] = pd.to_datetime(mp.gameDate.astype(str), format='%Y%m%d')

ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
ar['date'] = pd.to_datetime(ar.date)
real = ~ar.home_team.str.contains('Division') & ~ar.away_team.str.contains('Division')
ar = ar[real]
ar = ar[ar.date >= '2021-10-12']

# moneypuck game table
allrows = mp[mp.situation == 'all'].copy()
rows = []
for gid, grp in allrows.groupby('gameId'):
    home = grp[grp.home_or_away == 'HOME'].iloc[0]
    away = grp[grp.home_or_away == 'AWAY'].iloc[0]
    rows.append(dict(date=home.date, home=CODE2NAME[home.team], away=CODE2NAME[away.team],
                     hg=int(home.goalsFor), ag=int(away.goalsFor), game_id=gid))
g = pd.DataFrame(rows)
print("moneypuck games:", len(g))

# archangel duplicates check
dup = ar.duplicated(subset=['date', 'home_team', 'away_team']).sum()
print("archangel dup (date,home,away):", dup)
ar = ar.drop_duplicates(subset=['date', 'home_team', 'away_team']).reset_index(drop=True)

# build a lookup: date -> list of (home, away, hs, as, idx)
from collections import defaultdict
lookup = defaultdict(list)
for i, r in ar.iterrows():
    lookup[r.date].append((r.home_team, r.away_team, r.home_score, r.away_score, i))

def scores_compat(mp_hg, mp_ag, ar_hs, ar_as):
    # exact
    if mp_hg == ar_hs and mp_ag == ar_as:
        return True
    # moneypuck excludes SO goal: ar has +1 on winner side, mp tied
    if mp_hg == mp_ag:  # OT tie in moneypuck => possible SO
        if (ar_hs == mp_hg + 1 and ar_as == mp_ag) or (ar_as == mp_ag + 1 and ar_hs == mp_hg):
            return True
    return False

matched, mult, nomatch = [], [], []
for i, r in g.iterrows():
    cands = []
    for off in (0, 1, -1, 2, -2):
        d = r.date + pd.Timedelta(days=off)
        for (h, a, hs, as_, idx) in lookup.get(d, []):
            pair_ok = {h, a} == {r.home, r.away}
            if not pair_ok:
                continue
            # align orientation
            if h == r.home and a == r.away:
                comp = scores_compat(r.hg, r.ag, hs, as_)
                cands.append((abs(off), 0 if comp else 1, idx, comp, off))
            elif a == r.home and h == r.away:
                comp = scores_compat(r.hg, r.ag, as_, hs)
                cands.append((abs(off), 0 if comp else 1, idx, comp, off))
    if not cands:
        nomatch.append((i, r.date, r.home, r.away, r.hg, r.ag))
        continue
    cands.sort(key=lambda c: (c[0], c[1]))
    best = cands[0]
    if len(cands) > 1 and (cands[0][0], cands[0][1]) == (cands[1][0], cands[1][1]):
        mult.append((i, r.date, r.home, r.away, len(cands)))
    matched.append((i, best[3], best[4], best[2]))

print("matched:", len(matched), "| multi-candidate:", len(mult), "| no match:", len(nomatch))
if nomatch:
    print("first 20 no-match:")
    for x in nomatch[:20]:
        print("  ", x[1].date(), x[2], "vs", x[3], x[4], "-", x[5])
exact = sum(1 for m in matched if m[1])
print("score-compatible on matched:", exact, "/", len(matched))
print("date offset distribution:", pd.Series([m[2] for m in matched]).value_counts().sort_index().to_dict())
