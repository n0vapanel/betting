"""NHL 2021-22 data build: moneypuck game-by-game -> game-level + team-game tables."""
import pandas as pd
import numpy as np

# ---- team code -> full name map (standard NHL abbreviations) ----
CODE2NAME = {
    'NYR': 'New York Rangers', 'SJS': 'San Jose Sharks', 'PIT': 'Pittsburgh Penguins',
    'EDM': 'Edmonton Oilers', 'WPG': 'Winnipeg Jets', 'PHI': 'Philadelphia Flyers',
    'DAL': 'Dallas Stars', 'MIN': 'Minnesota Wild', 'COL': 'Colorado Avalanche',
    'ANA': 'Anaheim Ducks', 'VGK': 'Vegas Golden Knights', 'TOR': 'Toronto Maple Leafs',
    'NYI': 'New York Islanders', 'MTL': 'Montreal Canadiens', 'SEA': 'Seattle Kraken',
    'STL': 'St. Louis Blues', 'FLA': 'Florida Panthers', 'BUF': 'Buffalo Sabres',
    'LAK': 'Los Angeles Kings', 'ARI': 'Arizona Coyotes', 'OTT': 'Ottawa Senators',
    'WSH': 'Washington Capitals', 'NJD': 'New Jersey Devils', 'TBL': 'Tampa Bay Lightning',
    'CBJ': 'Columbus Blue Jackets', 'CHI': 'Chicago Blackhawks', 'NSH': 'Nashville Predators',
    'CGY': 'Calgary Flames', 'BOS': 'Boston Bruins', 'DET': 'Detroit Red Wings',
    'VAN': 'Vancouver Canucks', 'CAR': 'Carolina Hurricanes',
}

def load_moneypuck():
    df = pd.read_csv('all_teams.csv', low_memory=False)
    df = df[df.season == 2021].copy()
    df['date'] = pd.to_datetime(df.gameDate.astype(str), format='%Y%m%d')
    return df

def build_game_table(mp):
    """One row per game (situation=='all')."""
    allrows = mp[mp.situation == 'all'].copy()
    rows = []
    for gid, grp in allrows.groupby('gameId'):
        home = grp[grp.home_or_away == 'HOME']
        away = grp[grp.home_or_away == 'AWAY']
        if len(home) != 1 or len(away) != 1:
            continue
        h, a = home.iloc[0], away.iloc[0]
        # cross-check goals
        if h.goalsFor != a.goalsAgainst or a.goalsFor != h.goalsAgainst:
            print('MISMATCH', gid, h.goalsFor, a.goalsAgainst, a.goalsFor, h.goalsAgainst)
        gtype = 'playoffs' if h.playoffGame == 1 else 'regular'
        rows.append(dict(
            game_id=gid, date=h.date, game_type=gtype,
            home=h.team, away=a.team,
            home_name=CODE2NAME.get(h.team, h.team), away_name=CODE2NAME.get(a.team, a.team),
            home_goals=int(h.goalsFor), away_goals=int(a.goalsFor),
            is_playoff=int(h.playoffGame),
        ))
    g = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    g['home_win'] = (g.home_goals > g.away_goals).astype(int)
    return g

def build_team_game_table(mp):
    """One row per team per game, with per-situation stats."""
    keep = mp[mp.situation.isin(['all', '5on5', '5on4', '4on5'])].copy()
    piv = keep.pivot_table(
        index=['gameId', 'team', 'home_or_away', 'date', 'playoffGame'],
        columns='situation',
        values=['goalsFor', 'goalsAgainst', 'xGoalsFor', 'xGoalsAgainst',
                'shotsOnGoalFor', 'shotsOnGoalAgainst', 'shotAttemptsFor', 'shotAttemptsAgainst',
                'iceTime', 'penaltiesFor', 'penalityMinutesFor',
                'faceOffsWonFor', 'faceOffsWonAgainst'],
    ).reset_index()
    piv.columns = ['_'.join(c).strip('_') for c in piv.columns.values]
    return piv

def main():
    mp = load_moneypuck()
    print('moneypuck 2021 rows:', len(mp), 'games:', mp.gameId.nunique())
    g = build_game_table(mp)
    print('game table:', len(g), '| regular:', (g.game_type == 'regular').sum(),
          '| playoff:', (g.game_type == 'playoffs').sum())
    print('date range:', g.date.min().date(), '->', g.date.max().date())
    tgt = build_team_game_table(mp)
    print('team-game table:', len(tgt))
    g.to_parquet('games.parquet', index=False)
    tgt.to_parquet('team_games.parquet', index=False)
    # test-window counts
    test = g[g.date >= '2022-03-01']
    print('moneypuck games in test window (Mar-Jun):', len(test),
          '| regular:', (test.game_type == 'regular').sum(),
          '| playoff:', (test.game_type == 'playoffs').sum())
    # reconcile vs archangel
    ar = pd.read_csv('kaggle_arch/games.csv', low_memory=False)
    ar = ar[(ar.competition == 'NHL') & (ar.season == '2021-22')].copy()
    ar['date'] = pd.to_datetime(ar.date)
    real_teams = ~ar.home_team.str.contains('Division') & ~ar.away_team.str.contains('Division')
    ar = ar[real_teams]
    ar = ar[ar.date >= '2021-10-01']
    print('archangel 2021-22 games (no preseason):', len(ar),
          '| Mar-Jun:', (ar.date >= '2022-03-01').sum(),
          '| with odds:', ar.avg_home_odds.notna().sum())
    # join check: how many moneypuck games match archangel exactly on date+names+scores
    ark = ar[['date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    gj = g.merge(ark, left_on=['date', 'home_name', 'away_name'],
                 right_on=['date', 'home_team', 'away_team'], how='left', indicator=True)
    matched = gj[gj._merge == 'both']
    print('moneypuck games matched to archangel:', len(matched))
    score_ok = (matched.home_goals == matched.home_score) & (matched.away_goals == matched.away_score)
    print('score agreement on matched:', score_ok.sum(), '/', len(matched))
    bad = matched[~score_ok]
    if len(bad):
        print(bad[['date', 'home_name', 'home_goals', 'home_score', 'away_goals', 'away_score']].head(10).to_string())
    miss = gj[gj._merge == 'left_only']
    print('moneypuck games missing in archangel:', len(miss))
    if len(miss):
        print(miss[['date', 'home_name', 'away_name']].head(15).to_string())
    extra = ar[~ar.game_id.isin(matched.game_id.astype(int).tolist()) if False else True]
    # simpler: archangel games not matched
    ar_un = ar.merge(g[['date', 'home_name', 'away_name']], left_on=['date', 'home_team', 'away_team'],
                     right_on=['date', 'home_name', 'away_name'], how='left', indicator=True)
    ar_missing = ar_un[ar_un._merge == 'left_only']
    print('archangel games missing in moneypuck:', len(ar_missing))
    if len(ar_missing):
        print(ar_missing[['date', 'home_team', 'away_team']].head(20).to_string())
        ar_missing[['date', 'home_team', 'away_team', 'home_score', 'away_score']].to_csv('arch_missing.csv', index=False)

if __name__ == '__main__':
    main()
