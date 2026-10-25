# Sports Prediction & Betting ML: GitHub Recon Report
**Research Date:** October 2026  
**Scope:** 20+ open-source repos across soccer xG, ratings systems, market tooling, sport-specific ML  
**Focus:** Production-grade models with transparent features, pipelines, and validation discipline  

---

## I. SOCCER: Expected Goals (xG) & Action Valuation

### 1. **socceraction** (ML-KULeuven/socceraction)
- **URL:** https://github.com/ML-KULeuven/socceraction
- **Stars:** 822
- **Model Types:** VAEP (Valuing Actions by Estimated Probabilities), Expected Threat (xT), SPADL event conversion
- **Features:** Event-stream probabilistic action values; possession threat models; player contribution quantification
- **Pipeline:** Ingest JSON event data (StatsBomb, Opta) → SPADL conversion → VAEP model training → player/team rankings
- **Data Sources:** StatsBomb open data, Opta (commercial); daily/per-season updates
- **Validation:** Walk-forward testing on multi-season datasets; VAEP vs. reality correlation studies
- **What to Steal:** SPADL (standardized event schema) + VAEP Bayesian update logic; framework for converting any event stream to action values

### 2. **soccer_xg** (ML-KULeuven/soccer_xg)
- **URL:** https://github.com/ML-KULeuven/soccer_xg
- **Stars:** 261
- **Model Types:** xG (Expected Goals) classifiers: logistic regression + optional deep learning
- **Features:** Shot location (x, y), distance, angle, defender proximity, shooter history, team strength
- **Pipeline:** Event log ingestion → shot feature engineering → logistic regression or neural net training → calibration
- **Data Sources:** StatsBomb, Wyscout, custom CSV; update per match
- **Validation:** Calibration curves, log loss, AUC on held-out seasons
- **What to Steal:** Shot-based xG feature engineering (angle bins, distance decay); simple but effective logistic baseline before DL

### 3. **penaltyblog** (martineastwood/penaltyblog)
- **URL:** https://github.com/martineastwood/penaltyblog
- **Stars:** 2821 (Jupyter Notebooks in separate repo)
- **Model Types:** Poisson + Bivariate Poisson, Dixon-Coles, hierarchical Bayesian, xG-based match outcome models
- **Features:** Team offensive/defensive strength, home advantage, xG per shot, league-specific adjustments
- **Pipeline:** JSON streaming (MatchFlow) from StatsBomb/Opta → team strength extraction → Poisson goal distribution → calibrated probabilities
- **Data Sources:** StatsBomb, Opta, Football Charts, Understat, FBRef; live API integration with caching
- **Validation:** RPS (Ranked Probability Score), Brier score, log loss; separate MCMC posterior sampling for uncertainty
- **What to Steal:** Cython-optimized Poisson/Dixon-Coles solver; hierarchical priors for league-wide variance learning; production-grade JSON pipeline design

### 4. **regista** (Torvaney/regista)
- **URL:** https://github.com/Torvaney/regista
- **Stars:** 94
- **Language:** R (with mezzala Python equivalent)
- **Model Types:** Dixon-Coles, extended variants (custom link functions, heteroskedastic)
- **Features:** Team offensive/defensive ratings, match importance weighting, home advantage
- **Pipeline:** CSV → Dixon-Coles model fit → prediction intervals + empirical covariance
- **Validation:** Out-of-sample log likelihood, calibration on historical seasons
- **What to Steal:** Clean R interface for Dixon-Coles; easily extensible to xG-based goals

### 5. **goalmodel** (opisthokonta/goalmodel)
- **URL:** https://github.com/opisthokonta/goalmodel
- **Stars:** 116
- **Language:** R
- **Model Types:** Poisson, Gaussian, Hurdle models, Over/Under (BTTS) prediction
- **Features:** Team strength (off/def), home advantage, match importance, prior calibration
- **Pipeline:** Fit on season matches → predict goals + draw probability → league tables
- **Validation:** RPS, Brier score, log loss; BTTS (Both Teams to Score) held-out validation
- **What to Steal:** Hurdle model for 0-0 handling; separate calibration of draw probability; p1x2 from expected goals

### 6. **understat** (amosbastian/understat)
- **URL:** https://github.com/amosbastian/understat
- **Stars:** 186
- **Language:** Python (async scraper)
- **Model Types:** Data provider (not model); xG/xA aggregation by player/team
- **Features:** Shot maps, pressure heatmaps, underlying stats
- **Pipeline:** Async HTTP scraper → event reconstruction → xG aggregation
- **Data Sources:** understat.com (web scrape, 1-2s rate limit per request)
- **Validation:** Cross-check xG vs. StatsBomb published values where overlap exists
- **What to Steal:** Async scraper pattern for rate-limited data providers; event reconstruction from partial JSON

### 7. **soccerdata** (probberechts/soccerdata)
- **URL:** https://github.com/probberechts/soccerdata
- **Stars:** 2122
- **Language:** Python
- **Model Types:** Data aggregator (7 providers: FBref, Understat, Club Elo, ESPN, SoFIFA, Football-Data.co.uk, Sofascore)
- **Features:** Unified DataFrame schema across providers; player/team stats, schedules, standings
- **Pipeline:** Provider adapters → caching layer → pandas DataFrame unification
- **Data Sources:** Multiple; daily updates on schedule
- **Validation:** Cross-provider reconciliation tests; deduplication on team/player IDs
- **What to Steal:** Multi-provider adapter pattern; fallback + cache strategy for robust pipelines; ID mapping (FBref ↔ Transfermarkt ↔ Understat)

---

## II. RATINGS SYSTEMS (Elo, Glicko, TrueSkill, OpenSkill)

### 8. **fivethirtyeight/data (soccer-spi)** 
- **URL:** https://github.com/fivethirtyeight/data/tree/master/soccer-spi
- **Model Types:** Soccer Power Index (SPI) — Bayesian hierarchical model: team strength → match outcome → goal distribution
- **Features:** Offensive SPI, Defensive SPI, home advantage, league-specific priors, importance weighting
- **Pipeline:** Season-long match log → Bayesian posterior on team strengths → projected scores + win probabilities + importance
- **Data Sources:** FBRef, FotMob, Transfermarkt; daily updates; full data archive 2016–present
- **Validation:** RPS vs. betting market comparison; published methodology on FiveThirtyEight blog
- **What to Steal:** Hierarchical Bayesian structure for multi-league calibration; importance metric (0-100 scale); adjusted score accounting for game state

### 9. **openskill.py** (vivekjoshy/openskill.py)
- **URL:** https://github.com/vivekjoshy/openskill.py
- **Stars:** 377
- **Model Types:** OpenSkill (Weng-Lin) — Bayesian multiplayer rating system; faster, open-license alternative to TrueSkill
- **Features:** Skill (μ) + uncertainty (σ) per player; team aggregation (max, sum, avg); draw probability threshold
- **Pipeline:** Per-match result → posterior update on all players → ratings convergence
- **Validation:** Head-to-head consistency; multi-player tournament calibration
- **What to Steal:** Multiplicative skill aggregation for team sports; cleaner math than TrueSkill; no closed-form likelihoods needed

### 10. **trueskill** (sublee/trueskill)
- **URL:** https://github.com/sublee/trueskill
- **Stars:** 803
- **Language:** Python
- **Model Types:** TrueSkill — Bayesian factor graph inference on player skill
- **Features:** Skill distribution (Gaussian), dynamics (decay over time), draw margin
- **Pipeline:** Match outcome → factor graph message passing → posterior skill update
- **Validation:** Tournament simulation; skill stability checks
- **What to Steal:** Skill dynamics (σ expansion per unplayed time); draw margin tuning; standard Bayesian update pattern

### 11. **glicko2** (sublee/glicko2) + implementations
- **URL:** https://github.com/sublee/glicko2 (Python)
- **Stars:** 125
- **Model Types:** Glicko-2 — time-dependent Bayesian rating with volatility
- **Features:** Rating (μ), rating deviation (RD), volatility (σ); time decay
- **Pipeline:** Period-based updates (e.g., per season) → RD expansion → skill + volatility posterior
- **Validation:** Stability across rating periods; convergence to published benchmarks
- **What to Steal:** Rating Deviation as confidence measure; volatility pinning; period-based batch updates

### 12. **skillratings** (atomflunder/skillratings)
- **URL:** https://github.com/atomflunder/skillratings
- **Stars:** 79
- **Language:** Rust (Python bindings)
- **Model Types:** Elo, Glicko, Glicko-2, TrueSkill, Weng-Lin (OpenSkill), FIFA ranking, Sticko
- **Features:** Unified API across algorithms; 1v1, team, FFA, tournament support
- **Pipeline:** Single call per match → rating update → stateless
- **Validation:** Published algorithm specs; benchmark vs. reference implementations
- **What to Steal:** Multi-algorithm comparison in one codebase; blazing-fast Rust backend with minimal dependencies

### 13. **melo-legacy** (morelandjs/melo-legacy)
- **URL:** https://github.com/morelandjs/melo-legacy
- **Stars:** 14
- **Model Types:** MELO (Margin-dependent Elo) — NFL-specific Elo that accounts for victory margin
- **Features:** Margin bins (0-3, 3-7, 7-14, 14+), K-factor decay, home advantage
- **Pipeline:** Season match log → margin-aware rating updates → spread/total distribution
- **Validation:** Calibration to actual margins; spread prediction accuracy
- **What to Steal:** Margin-dependent K-factor (Elo variant for spread prediction); point-total forecasting from ratings

### 14. **fivethirtyeight/nfl-elo-game** 
- **URL:** https://github.com/fivethirtyeight/nfl-elo-game
- **Stars:** 354
- **Model Types:** NFL Elo variant with QB adjustments, home advantage modulation, strength-of-schedule feedback
- **Features:** Team base Elo, QB strength multiplier, playoff/strength weighting, bye-week decay
- **Pipeline:** Historical game log (1920–present) → Elo updates → forecast probabilities
- **Data Sources:** Pro Football Reference, ESPN APIs
- **Validation:** Season win prediction accuracy; playoff performance calibration
- **What to Steal:** QB-adjusted Elo (multiplier at match time); recency weighting (recent games K↑); playoff modulation

---

## III. MARKET TOOLING (Devigging, Kelly, Arbitrage, CLV)

### 15. **p2w-math** (adoniscares7-sketch/p2w-math)
- **URL:** https://github.com/adoniscares7-sketch/p2w-math
- **Stars:** Open-source, MIT
- **Model Types:** Devig waterfall (Pinnacle → Circa → multiplicative-median → Shin's method), Kelly criterion, edge scorer, line shopper
- **Features:** American odds ↔ probability conversion; market-adjusted Kelly (quarter-Kelly default); EV at best book
- **Pipeline:** Poll 4+ sportsbooks → devig → edge computation vs. model forecast → Kelly sizing → bet recommendation
- **Data Sources:** Live odds APIs (Odds API); cached snapshots for testing
- **Validation:** Fail-closed (no edge = no bet); backtested vs. market closing lines
- **What to Steal:** Devig waterfall priority order (Pinnacle > Circa > median > Shin); no-vig math proofs; Kelly with bankroll caps

### 16. **flumine** (betcode-org/flumine, aka liampauling/flumine)
- **URL:** https://github.com/betcode-org/flumine
- **Stars:** 250
- **Language:** Python
- **Model Types:** Event-driven trading framework (not model); strategy execution, risk management, backtesting
- **Features:** Multi-venue (Betfair, Betdaq, Betconnect); streaming orders + market events; paper + live trading
- **Pipeline:** Market stream → custom strategy logic → risk checks → order placement
- **Data Sources:** Betting exchange APIs; custom simulation data
- **Validation:** Backtest harness with order execution simulation; paper trading shadow mode
- **What to Steal:** Modular strategy interface; streaming data architecture; risk limits (exposure, liability caps); live/paper mode toggle

### 17. **betfairlightweight** (betcode-org/betfair)
- **URL:** https://github.com/betcode-org/betfair
- **Stars:** 516
- **Language:** Python
- **Model Types:** Betfair API-NG wrapper (low-level)
- **Features:** Full API coverage (accounts, markets, orders, streaming); C/Rust backends for speed
- **Pipeline:** Authenticate → poll/stream markets → place/cancel orders → monitor positions
- **Data Sources:** Betfair exchange (live data)
- **Validation:** API contract tests; streaming reconnection logic
- **What to Steal:** Streaming market data parsing; order state machine (unmatched → matched → settled); back/lay separability

### 18. **sports-ai-bettor** (Monsterx411/sports-ai-bettor)
- **URL:** https://github.com/Monsterx411/sports-ai-bettor
- **Stars:** Active
- **Model Types:** Multi-sport ML predictor + value-bet analyzer
- **Features:** Logistic regression + neural net ensemble; implied probability from odds; EV > edge threshold
- **Pipeline:** Fetch fixtures + odds → feature engineering → predict → score value bets → CLI report
- **Data Sources:** The Odds API, API Sports
- **Validation:** Confusion matrix (precision, recall); edge threshold tuning (5% default)
- **What to Steal:** Implied probability from decimal odds (1/odds); edge threshold gating; min-confidence filter

### 19. **odds-api-python** 
- **URL:** https://github.com/odds-api-io/odds-api-python
- **Model Types:** Devig & closing-line-value (CLV) computation; fair odds notebook
- **Features:** Multiplicative and power devig methods; bookmaker margin extraction; line movement tracking
- **Pipeline:** Snapshot 4+ books → devig → fair prob → compare to model → CLV metrics
- **Data Sources:** Odds API (https://www.api.odds.to/)
- **Validation:** Historical closing-line reconciliation; beat-close rate tracking
- **What to Steal:** Closed-form devig math (multiplicative: p=sqrt(p1*p2/(1-(1-p1)*(1-p2)))); CLV in odds points vs. probability points; line movement charts

### 20. **Smart-NFL-Line-Betting** (craymichael/Smart-NFL-Line-Betting)
- **URL:** https://github.com/craymichael/Smart-NFL-Line-Betting
- **Stars:** 6
- **Model Types:** Kelly criterion line bet sizer
- **Features:** American moneyline odds → implied probability → Kelly fraction → bet recommendation
- **Pipeline:** Manual odds CSV → Kelly sizing → output wagers
- **Data Sources:** FiveThirtyEight QB-adjusted Elo probabilities (input); sportsbook lines (input CSV)
- **Validation:** Historical comparison to market odds
- **What to Steal:** Kelly criterion implementation (full + fractional); edge capping (max 25% EV); bankroll management

---

## IV. SPORT-SPECIFIC: MMA, NFL, MLB, Tennis, Basketball

### 21. **evankellener/UFC_fight_predictor**
- **URL:** https://github.com/evankellener/UFC_fight_predictor
- **Model Types:** λ-ensemble Elastic Net logistic regression (2 LRs @ λ=1.20 + λ=0.13)
- **Features:** 207 features: fighter post-fight stats, Elo (overall + style: striking, grappling, finishing), physical attributes (height/reach/weight ratios), sigmoid-weighted recency averages, matchup interactions, population priors rolling by era
- **Pipeline:** Historical fight database → feature engineering (strict walk-forward leakage check) → λ-ensemble training → Flask app + CLI
- **Data Sources:** ufcstats.com scrape; post-fight stats only (no prediction leakage)
- **Validation:** Walk-forward 4-fold: 72.21% accuracy, 0.5935 log loss, 0.7529 AUC; +14.08% ROI on 256 bets vs. Vegas (t=3)
- **What to Steal:** Recency-weighted Elo (sigmoid decay); λ-ensemble for drift resilience; symmetric (doubled) training rows; temperature-scaled calibration; leakage reference document

### 22. **lcshlr/ufc_prediction**
- **URL:** https://github.com/lcshlr/ufc_prediction
- **Model Types:** PySpark-based data pipeline; XGBoost, scikit-learn models
- **Features:** Fighter stats (striking accuracy, TD rate, submission attempts), Elo, physical attributes
- **Pipeline:** BeautifulSoup scraper (ufcstats.com) → PostgreSQL warehouse (silver/gold zones) → PySpark processing → XGBoost training → MLflow tracking → Airflow orchestration
- **Data Sources:** ufcstats.com web scrape (20+ years history); Node.js advanced odds scraping
- **Validation:** Walk-forward evaluation on multiple folds; MLflow experiment tracking
- **What to Steal:** Data lake architecture (bronze/silver/gold); Airflow DAG orchestration; PySpark scaling for large historical datasets

### 23. **nflfastR** (nflverse/nflfastR)
- **URL:** https://github.com/nflverse/nflfastR
- **Stars:** 552
- **Language:** R
- **Model Types:** Data provider + EPA (Expected Points Added) calculator
- **Features:** Play-by-play EPA, success rate, yards gained, WPA (Win Probability Added)
- **Pipeline:** NFL play-by-play JSON → EPA model fit → play-level contribution scoring
- **Data Sources:** Pro Football Reference, nflverse API (updated daily)
- **Validation:** EPA correlation to wins; calibration by field position/situation
- **What to Steal:** EPA formula (logistic regression on next points); WPA aggregation by player; play-level contribution breakdown

### 24. **nfl_data_py** (nflverse/nfl_data_py)
- **URL:** https://github.com/nflverse/nfl_data_py
- **Stars:** 442
- **Language:** Python
- **Model Types:** Data access layer (no model); play-by-play, schedule, team stats
- **Features:** EPA, yards, turnovers, defensive stats, player-level participation
- **Pipeline:** Fetch from nflverse API → caching → pandas DataFrame
- **Data Sources:** Pro Football Reference, nflfastR, official NFL data
- **Validation:** Schema consistency tests; weekly updates
- **What to Steal:** Stateless API client pattern; caching with TTL; lazy DataFrame loading

### 25. **pybaseball** (jldbc/pybaseball)
- **URL:** https://github.com/jldbc/pybaseball
- **Stars:** 1738
- **Language:** Python
- **Model Types:** Data aggregator (Baseball Reference, FanGraphs, Statcast); no built-in model but data for Pythagorean Win %
- **Features:** Statcast pitch-level (exit velo, launch angle, spin rate), season-level bWAR, ERA+
- **Pipeline:** Scrape multiple sources → unify stat names → pandas DataFrames
- **Data Sources:** MLB.com Statcast, Baseball-Reference, FanGraphs (requires key), Baseball Savant
- **Validation:** Reconciliation across sources on player IDs; ESPN reconciliation
- **What to Steal:** Statcast feature engineering (exit velo buckets, launch angle bins); lag-based feature engineering for pitcher performance

### 26. **tennis-crystal-ball** (mcekovic/tennis-crystal-ball)
- **URL:** https://github.com/mcekovic/tennis-crystal-ball
- **Stars:** 295
- **Language:** Java
- **Model Types:** Elo ratings (overall, surface-specific, set/game level), head-to-head prediction, GOAT metrics
- **Features:** Player Elo (overall + by surface + by court type), H2H, ranking, surface dominance
- **Pipeline:** Match history → Elo updates → H2H aggregation → prediction via Bradley-Terry (logistic link)
- **Data Sources:** ATP/WTA historical match data (1970–present); web scraper
- **Validation:** H2H win rate vs. predicted from Elo; Hypothetical Matchup predictions on retired players
- **What to Steal:** Multi-dimensional Elo (surface/court specificity); peak Elo tracking; GOAT points system (weighted tournament wins)

### 27. **buildoak/tennis-xgboost-autoresearch**
- **URL:** https://github.com/buildoak/tennis-xgboost-autoresearch
- **Stars:** 24
- **Language:** Python
- **Model Types:** Autonomous research loop (Codex agents) optimizing XGBoost on 245K tennis matches
- **Features:** Chess-inspired Elo (overall + surface-specific + serve/return), 400+ engineered features
- **Pipeline:** Historical ATP/WTA data → feature engineering (agent-driven) → XGBoost hyperparameter search → validation gate (ROC-AUC held-out 2026 matches)
- **Data Sources:** Tennismylife/TML-Database (same schema as Jeff Sackmann's tennis_atp)
- **Validation:** Walk-forward on 2026 unseen matches; Goodhart's Law case study (agent gaming metric)
- **What to Steal:** Chess ELO variant for tennis (serve/return separation); autonomous feature search pattern; validation gate design

### 28. **shoor003/sports-betting-model** (Tennis)
- **URL:** https://github.com/shoor003/sports-betting-model
- **Model Types:** Elo ratings + walk-forward Kelly backtester + MLR value-bet evaluator
- **Features:** Player Elo (recency-weighted), tournament importance (Slam > 500 > 250), surface effect
- **Pipeline:** ATP match log 2023-2024 → Elo updates (strict chronological) → predict test matches → Kelly sizing → compute ROI
- **Data Sources:** Tennismylife/TML-Database; real ATP match results (no synthetic odds)
- **Validation:** Walk-forward predict on actual matches (62.1% accuracy); Brier score 0.2261 vs. 0.25 (guessing)
- **What to Steal:** Walk-forward (not random split); tournament importance weighting; Brier score calibration; honest ROI framing (simulated vs. real odds)

### 29. **hoopR** (sportsdataverse/hoopR)
- **URL:** https://github.com/sportsdataverse/hoopR
- **Stars:** 152
- **Language:** R
- **Model Types:** Data provider (NCAA basketball play-by-play)
- **Features:** Play-by-play (shots, passes, fouls), player tracking
- **Pipeline:** Scrape ESPN API → tidy format → R tibbles
- **Data Sources:** ESPN college hoops API (daily during season)
- **Validation:** Schema consistency; play-by-play continuity
- **What to Steal:** Efficient sports API scraper; tidy data format; missing data imputation for live data

### 30. **kenpompy** (j-andrews7/kenpompy)
- **URL:** https://github.com/j-andrews7/kenpompy
- **Stars:** 87
- **Language:** Python
- **Model Types:** Web scraper (kenpom.com requires subscription); no model, data access only
- **Features:** Adjusted O/D ratings (AdjO, AdjD), conference stats, team rankings
- **Pipeline:** Login → scrape KenPom tables → regex parsing → pandas DataFrames
- **Data Sources:** kenpom.com (subscription required); NCAA games only
- **Validation:** Manual spot-checks against published rankings
- **What to Steal:** Captcha-aware session management; HTML parsing robustness (multi-format fallbacks)

### 31. **cfb-predictor** (euehling/cfb-predictor)
- **URL:** https://github.com/euehling/cfb-predictor
- **Stars:** 1
- **Language:** Python
- **Model Types:** Custom power-rating system (translated from Excel) + XGBoost
- **Features:** SP+ rating (Power Rating), talent composite, recency-weighted strength of schedule, pre-game snapshot (no leakage)
- **Pipeline:** CFBD API pull (SP+, AP polls) → feature engineering (pre-game) → XGBoost regression (margin, total points, home win)
- **Data Sources:** CollegeFootballData API (CFBD); 2022-2025 games
- **Validation:** Time-based split (train 2022-2023, val 2024, test 2025); power-rating stability checks vs. Excel baseline
- **What to Steal:** Power-rating translation from Excel to Python; tier-based classification (AP rank cutoffs); leakage guard (snapshot before game)

### 32. **Fantasy Football AI** (cbratkovics/fantasy-football-ai)
- **URL:** https://github.com/cbratkovics/fantasy-football-ai
- **Stars:** Active
- **Language:** Python
- **Model Types:** RandomForest ensemble per position (QB/RB/WR/TE); challenger shadow evaluation
- **Features:** 65 strictly lagged features (no future leakage); weekly projections
- **Pipeline:** dbt Core (bronze → silver → gold) on MotherDuck → feature computation → RandomForest training → residual-quantile floors/ceilings → FastAPI serving
- **Data Sources:** nflverse play-by-play data
- **Validation:** 2024 frozen test season + full 2025 out-of-sample scoring; PSI drift detection; causal trailing-mean baseline
- **What to Steal:** Data contracts (dbt tests); as-of feature engineering with strict temporal validation; GMM preseason draft tiers; stateless API serving (no cache/DB)

---

## V. KALMAN FILTERS & ONLINE LEARNING

### 33. **Extended Kalman Filter for team strength** (normal-computing/posteriors)
- **URL:** https://github.com/normal-computing/posteriors
- **Framework:** PyTorch-based uncertainty quantification; EKF tutorial on Premier League football
- **Model Types:** Extended Kalman Filter (Bradley-Terry model + Ornstein-Uhlenbeck dynamics for team strength drift)
- **Features:** Latent team skill (−3 to +3 scale), mean reversion, uncertainty (covariance matrix), home advantage (time-varying)
- **Pipeline:** Match log (chronological) → state prediction (skill decay) → observation model (match outcome) → Kalman update
- **Data Sources:** Football-data.co.uk (Premier League 2021–2023); daily updates in real season
- **Validation:** Brier score 0.2035 (2021–2023 seasons), 0.2085 (2024/25); comparison to FiveThirtyEight + betting odds
- **What to Steal:** Ornstein-Uhlenbeck skill evolution (mean reversion, not random walk); time-varying home advantage; Kalman uncertainty quantification; Davidson-Bradley-Terry for draws

### 34. **Kalman Filter Ranking (UCLA EKF)** 
- **URL:** UCLA Math Dept (research paper); implementation ideas in posteriors repo above
- **Model Types:** Extended Kalman Filter with Bradley-Terry likelihood + Fisher Information scheduling
- **Features:** Team skill latent state; covariance (uncertainty per team)
- **Pipeline:** EKF predict/update cycle with Bradley-Terry link function
- **Validation:** Fisher Information matrix tracks estimation quality; dynamic scheduling maximizes information gain
- **What to Steal:** Fisher Information as quality metric; dynamic opponent selection for maximum info gain; trace-optimal matching (Edmonds' Blossom Algorithm)

---

## VI. MISCELLANEOUS / MULTI-PURPOSE

### 35. **world-cup-2026-prediction-model** (Hicruben/world-cup-2026-prediction-model)
- **URL:** https://github.com/Hicruben/world-cup-2026-prediction-model
- **Model Types:** Elo → Dixon-Coles → Monte Carlo tournament simulation
- **Features:** Team Elo (overall), home advantage, tournament importance boost
- **Pipeline:** Historical international results → Elo updates → Dixon-Coles bivariate Poisson → 50,000-run simulation per match
- **Data Sources:** Historical international football results (100+ years)
- **Validation:** Walk-forward backtest (67/98 predictions published before kickoff, accuracy tracked publicly)
- **What to Steal:** Elo → Dixon-Coles pipeline (ratings feed into goal distribution); transparent, reproducible methodology; public record of predictions

### 36. **EFI (evxiong/efi)** 
- **URL:** https://github.com/evxiong/efi
- **Model Types:** Simplified SPI (FiveThirtyEight Soccer Power Index) clone
- **Features:** Offensive/Defensive SPI, home advantage, importance weighting
- **Pipeline:** FBRef expected goals data (2017/18–present) → Bayesian team strength model → match forecasts + league standings
- **Data Sources:** FBRef xG, Sofascore, Transfermarkt; daily updates
- **Validation:** RPS 0.2050 (vs. 0.1988 FiveThirtyEight, 0.1984 Bet365); AUC/IGN comparisons
- **What to Steal:** xG-based league model (simpler than Poisson on raw goals); Bayesian hierarchical priors per league; live website + daily predictions

---

## Summary: Top 10 Techniques to Steal

1. **SPADL + VAEP (socceraction):** Standardized event schema + probabilistic action valuation for any sport with event streams
2. **Hierarchical Bayesian Ratings (SPI, EFI):** League-specific priors + team-level posterior + importance weighting for cross-sport transferability
3. **Devig Waterfall (p2w-math):** Pinnacle → Circa → Multiplicative-Median → Shin's method; fail-closed edge detection
4. **Walk-Forward Backtesting (UFC_fight_predictor, tennis models):** Strict chronological order, no future leakage, replay market prices to measure edge
5. **Recency-Weighted Elo (UFC, NFL, tennis):** Sigmoid decay on historical results; λ-ensemble for drift resilience
6. **xG-driven Poisson Models (penaltyblog, goalmodel):** Feature-engineered xG → goal distribution → match/league forecasts; easier than Dixon-Coles from raw scores
7. **Extended Kalman Filter for Team Strength (posteriors repo):** State prediction (skill drift) + observation (match) → uncertainty quantification + dynamic scheduling
8. **Kelly Criterion with Bankroll Caps (p2w-math):** Full Kelly / 4 (quarter-Kelly) default; max exposure % of bankroll; negative-EV → $0 bet
9. **Data Contracts + Feature Validation (Fantasy Football AI):** dbt tests on data schemas; as-of features with temporal validation; PSI drift detection
10. **Multivariate Rating Systems (skillratings in Rust, openskill.py):** Unified API for Elo/Glicko/TrueSkill/Weng-Lin; easy swappability; stateless updates per match

---

## Actionable Next Steps for Your ML System

**Phase 1: Data Infrastructure**
- Implement SPADL converters for your event streams (soccer, MMA, other event-heavy sports)
- Set up soccerdata or equivalent multi-source scraper; reconcile player/team IDs across providers
- Establish walk-forward pipeline: never train on future data; validate on time-ordered folds

**Phase 2: Rating Systems**
- Swap your Elo for OpenSkill (openskill.py): faster convergence, cleaner Bayesian math, easy to add uncertainty
- Add Kalman Filter variant for drift-aware team strength (time-varying home advantage, skill decay)
- Implement λ-ensemble (recency-weighted) for model fusion, similar to UFC_fight_predictor

**Phase 3: Market Integration**
- Add devig waterfall (p2w-math) to compare model forecasts vs. sharp books (Pinnacle/Circa)
- Compute CLV (closing-line value) on past bets; beat-close rate is the true edge metric
- Implement Kelly criterion with quarter-Kelly default + bankroll cap; gating on min-confidence

**Phase 4: Sport-Specific Pipelines**
- Soccer: penaltyblog (Poisson/Dixon-Coles) + socceraction (VAEP action values)
- MMA: UFC_fight_predictor (Elo + ~200 features + λ-ensemble LR) replicates your +14% ROI structure
- NFL: nflfastR (EPA per play) + MELO (margin-dependent Elo) for spread prediction
- Tennis: multi-surface Elo (tennis-crystal-ball) + XGBoost on 400+ features

**Phase 5: Monitoring & Drift Detection**
- Daily PSI (Population Stability Index) on prediction distributions vs. baseline
- Track calibration (Brier score, RPS) on rolling windows
- Publish a public record of predictions vs. outcomes (like world-cup-2026-prediction-model)

---

## Repository Index (Alphabetical)

| Repo | URL | Sport | Type | Stars |
|------|-----|-------|------|-------|
| buildoak/tennis-xgboost-autoresearch | github.com/buildoak/tennis-xgboost-autoresearch | Tennis | ML Research | 24 |
| cbratkovics/fantasy-football-ai | github.com/cbratkovics/fantasy-football-ai | NFL | ML Pipeline | Active |
| craymichael/Smart-NFL-Line-Betting | github.com/craymichael/Smart-NFL-Line-Betting | NFL | Kelly | 6 |
| euehling/cfb-predictor | github.com/euehling/cfb-predictor | CFB | Power Ratings | 1 |
| evankellener/UFC_fight_predictor | github.com/evankellener/UFC_fight_predictor | MMA | Ensemble LR | — |
| evxiong/efi | github.com/evxiong/efi | Soccer | SPI Clone | Active |
| fivethirtyeight/data (soccer-spi) | github.com/fivethirtyeight/data/tree/master/soccer-spi | Soccer | SPI | 17480 |
| fivethirtyeight/nfl-elo-game | github.com/fivethirtyeight/nfl-elo-game | NFL | Elo | 354 |
| j-andrews7/kenpompy | github.com/j-andrews7/kenpompy | Basketball | Scraper | 87 |
| jldbc/pybaseball | github.com/jldbc/pybaseball | Baseball | Data Access | 1738 |
| lcshlr/ufc_prediction | github.com/lcshlr/ufc_prediction | MMA | Pipeline (PySpark) | — |
| liampauling/flumine | github.com/betcode-org/flumine | Betting | Framework | 250 |
| martineastwood/penaltyblog | github.com/martineastwood/penaltyblog | Soccer | Poisson/Bayes | 2821 |
| mcekovic/tennis-crystal-ball | github.com/mcekovic/tennis-crystal-ball | Tennis | Elo + GOAT | 295 |
| ML-KULeuven/goalmodel | github.com/opisthokonta/goalmodel | Soccer | Poisson | 116 |
| ML-KULeuven/soccer_xg | github.com/ML-KULeuven/soccer_xg | Soccer | xG Classification | 261 |
| ML-KULeuven/socceraction | github.com/ML-KULeuven/socceraction | Soccer | VAEP/xT | 822 |
| morelandjs/melo-legacy | github.com/morelandjs/melo-legacy | NFL | MELO | 14 |
| nflverse/nfl_data_py | github.com/nflverse/nfl_data_py | NFL | Data Access | 442 |
| nflverse/nflfastR | github.com/nflverse/nflfastR | NFL | EPA / PBP | 552 |
| normal-computing/posteriors | github.com/normal-computing/posteriors | Soccer | Kalman Filter | 386 |
| p2w-math | github.com/adoniscares7-sketch/p2w-math | Betting | Devig/Kelly | — |
| probberechts/soccerdata | github.com/probberechts/soccerdata | Soccer | Multi-Source | 2122 |
| shoor003/sports-betting-model | github.com/shoor003/sports-betting-model | Tennis | Elo + Kelly | — |
| skillratings | github.com/atomflunder/skillratings | Multi | Rating Algos | 79 |
| Torvaney/regista | github.com/Torvaney/regista | Soccer | Dixon-Coles | 94 |
| trueskill | github.com/sublee/trueskill | Multi | TrueSkill | 803 |
| vivekjoshy/openskill.py | github.com/vivekjoshy/openskill.py | Multi | OpenSkill | 377 |
| sublee/glicko2 | github.com/sublee/glicko2 | Multi | Glicko-2 | 125 |
| betcode-org/betfair | github.com/betcode-org/betfair | Betting | API Wrapper | 516 |

---

**Report Compiled:** October 2026  
**Total Repos Documented:** 36  
**Primary Focus:** Production-grade, validation-driven, multi-sport transferable  
**Recommendation:** Start with socceraction (VAEP) + penaltyblog (Poisson) for soccer; evankellener/UFC_fight_predictor as MMA blueprint; p2w-math for market tooling. All have clear README/code and published methodology.
