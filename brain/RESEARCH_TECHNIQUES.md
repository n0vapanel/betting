# Research Techniques for Multi-Sport ML Betting Brain
**Verified academic & practitioner citations for walk-forward backtester**  
Updated: 2026-10-24

---

## 1. Dixon-Coles Model (1997) — Bivariate Poisson with Tau Correction

**Summary:** Landmark parametric model fitting independent Poisson distributions for home/away goals with low-score correlation adjustment (tau function). Uses exponential time-decay weighting (ξ) to down-weight older matches. Incorporates team-specific attack/defense parameters and home advantage. Core parameters: λ (home goal expectancy), μ (away expectancy), ρ (dependence parameter, typically -0.13), γ (home effect), ξ=0.0065 per half-week. Original paper demonstrated positive expected return vs 1995-96 bookmaker odds.

**Citations:**
- Dixon & Coles 1997, *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, JRSS-C Applied Statistics 46(2): https://doi.org/10.1111/1467-9876.00065

**Brain Implementation:** Fit via MLE: grid-search ξ ∈ [0.00005, 0.0005], optimize ρ separately; apply tau adjustments to (0,0), (0,1), (1,0), (1,1) cells; weight likelihoods by exp(-ξ*days_ago/7); scipy.optimize.minimize for attack/defense parameters.

---

## 2. Maher (1982) — Independent Poisson Foundation

**Summary:** Seminal work establishing independent Poisson regression for football scores with team-specific attack (α) and defense (β) strengths. No correlation adjustment; serves as baseline for all subsequent refinements. Showed Poisson marginals adequate despite independence assumption failing at low scores (0–0, 1–0, 0–1, 1–1).

**Citations:**
- Maher 1982, *Modelling association football scores*, Statistica Neerlandica 36(3): https://doi.org/10.1111/j.1467-9574.1982.tb00782.x

**Brain Implementation:** Poisson regression: glm(goals ~ home + team + opponent, family=poisson); extract log-linear coefficients and exponentiate to λ = exp(intercept + α_home + β_away), μ = exp(α_away + β_home).

---

## 3. Expected Goals (xG) Shot Models: StatsBomb and Opta

**Summary:** xG models assign a probability (0–1) to each shot based on historical shot data and contextual factors (distance, angle, defensive pressure, goalkeeper position, shot type, assist type). StatsBomb records detailed contextual information including all defenders on-screen and keeper positioning; Opta uses machine learning on ~1M historical shots to evaluate 20+ variables including body part, play pattern, and defensive proximity. Penalties are typically assigned 0.76–0.79 xG based on historical conversion rates.

**Citations:**
- StatsBomb Blog 2014, *The Dual Life of Expected Goals (Part 2)*: https://blogarchive.statsbomb.com/articles/soccer/the-dual-life-of-expected-goals-part-2/
- Stats Perform / Opta 2024, *Expected Goals (xG): The Football Metric Changing Analysis, Betting, and Fan Engagement*: https://www.statsperform.com/insights/expected-goals-xg-the-football-metric-changing-analysis-betting-and-fan-engagement/
- Rathke 2017, *An examination of expected goals and shot efficiency in soccer*, Journal of Sports Analytics 3(2): https://www.redalyc.org/articulo.oa?id=301052437005

**Brain Implementation:** Compute xG for all shots using StatsBomb/Opta APIs or logistic regression on distance/angle/context; aggregate team xG per match and rolling window; test xG differential (xG for − xG against) as feature in win/draw/loss model.

---

## 4. Expected Threat (xT): Markov Value Iteration

**Summary:** Singh (2019) introduced Expected Threat as a Markov-based model dividing the pitch into a grid, assigning each cell a value equal to the probability of scoring within ~8 actions from that location. xT values are computed iteratively via backward induction: xT(state) = P(shot|state)·xG(state) + Σ T(state→state')·xT(state'). Unlike xG which rates shots in isolation, xT captures buildup value by recognizing that moves to high-threat zones have value even if they don't immediately yield shots.

**Citations:**
- Singh 2019, *Introducing Expected Threat (xT)*: https://karun.in/blog/expected-threat.html
- Van Roy, Robberechts, Decroos & Davis 2020, *Valuing On-the-Ball Actions in Soccer: A Critical Comparison of xT and VAEP*: https://www.semanticscholar.org/paper/Valuing-On-the-Ball-Actions-in-Soccer%3A-A-Critical-Roy-Robberechts/99a1ac4279cdab37f4021f924481ba0517c10c54

**Brain Implementation:** Discretize pitch into 12×8 or finer grid; estimate transition matrix T and shot probability from event data; solve xT = g + T·xT via value iteration; compute per-action Δ xT = xT(end) − xT(start); aggregate team xT generated per match and rolling season; use cumulative xT differential in walk-forward prediction model.

---

## 5. VAEP: Valuing Actions by Estimating Probabilities

**Summary:** Decroos et al. (KDD 2019) values every on-ball action (pass, dribble, shot, tackle) by its impact on immediate scoring and conceding probability. For each action, VAEP = ΔP(score|next 10 events) − ΔP(concede|next 10 events), computed by comparing game states before/after the action using supervised learning on possession sequences. Unlike xG/xT which focus on buildup or end product, VAEP contextualizes all actions and naturally handles defensive actions, turnovers, and multi-event sequences.

**Citations:**
- Decroos, Bransen, Van Haaren & Davis 2019, *Actions Speak Louder Than Goals: Valuing Player Actions in Soccer*, KDD 2019: https://doi.org/10.1145/3292500.3330758
- Decroos et al. 2019, *Actions Speak Louder Than Goals: Valuing Player Actions in Soccer* (arXiv): https://arxiv.org/abs/1802.07127

**Brain Implementation:** Train binary classifiers for P(score) and P(concede) on event sequences; compute VAEP per action as diff in probabilities pre/post action; aggregate per team and per player per match; use rolling VAEP differential (team VAEP per 90 − opponent VAEP per 90) as offensive/defensive quality feature in multi-sport backtest.

---

## 6. Market Efficiency: Near-Efficient NFL/NBA Closers vs Soft Soccer/Tennis Markets

**Summary:** Comprehensive literature shows NFL and NBA main-market closing lines (spreads, moneylines) are near-efficient; documented biases (home-team, favorite-longshot in some sports) are small, decaying, or already priced by closing time. MLB is efficient post-commission. Tennis, by contrast, shows the strongest favorite-longshot bias across sports (Whelan 2025, Forrest & McHale 2007); soccer long-tail/lower-league lines are demonstrably soft (Kaunitz et al. 2017 proved a consensus-relative strategy profitable at scale over 10 years). MMA/UFC closers are near-efficient overall but contain exploitable angles (youth, travel, women's contests; Miller & Nichols 2026). Player props are unverified academically but show practitioner-reported line dispersion. **Strategic implication:** concentrate stake and model effort on MMA angles, props, tennis odds-structure, and soccer long-tail; use NFL/NBA/MLB closers as near-fair benchmarks.

**Citations:**
- Sauer 1998, *The Economics of Wagering Markets*, Journal of Economic Literature 36(4): https://ideas.repec.org/a/aea/jeclit/v36y1998i4p2021-2064.html
- Thaler & Ziemba 1988, *Parimutuel Betting Markets: Racetracks and Lotteries*, Journal of Economic Perspectives 2(2): https://www.aeaweb.org/articles?id=10.1257/jep.2.2.161
- Snowberg & Wolfers 2010, *Explaining the Favorite-Longshot Bias: Is it Risk-Love or Misperceptions?*, Journal of Political Economy 118(4): https://www.nber.org/papers/w15923
- Levitt 2004, *Why are Gambling Markets Organised so Differently from Financial Markets?*, Economic Journal 114(495): https://academic.oup.com/ej/article-abstract/114/495/223/5086012
- Gandar, Zuber, O'Brien & Russo 1988, *Testing Rationality in the Point Spread Betting Market*, Journal of Finance 43(4): https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1988.tb02617.x
- Golec & Tamarkin 1991, *The Degree of Inefficiency in the Football Betting Market: Statistical Tests*, Journal of Financial Economics 30(2): https://doi.org/10.1016/0304-405X(91)90034-H
- Gray & Gray 1997, *Testing Market Efficiency: Evidence From The NFL Sports Betting Market*, Journal of Finance 52(4): https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1997.tb01129.x
- Paul & Weinbach 2002, *Market Efficiency and a Profitable Betting Rule: Evidence from Totals on Professional Football*, Journal of Sports Economics 3(3): https://ideas.repec.org/a/sae/jospec/v3y2002i3p256-263.html
- Paul, Weinbach & Wilson 2004, *Efficient Markets, Fair Bets, and Profitability in NBA Totals 1995-96 to 2001-02*, Quarterly Review of Economics and Finance 44(4): https://ideas.repec.org/a/eee/quaeco/v44y2004i4p624-632.html
- Feddersen, Humphreys & Soebbing 2018, *Sentiment Bias in National Basketball Association Betting*, Journal of Sports Economics 19(4): https://journals.sagepub.com/toc/jsea/19/4
- Davis & Krieger 2016, *Preseason Bias in the NFL and NBA Betting Markets*, Applied Economics 49(12): https://www.tandfonline.com/doi/full/10.1080/00036846.2016.1213367
- Woodland & Woodland 1994, *Market Efficiency and the Favorite-Longshot Bias: The Baseball Betting Market*, Journal of Finance 49(1): https://econpapers.repec.org/article/blajfinan/v_3a49_3ay_3a1994_3ai_3a1_3ap_3a269-79.htm
- Forrest & McHale 2007, *Anyone for Tennis (Betting)?*, European Journal of Finance 13(8): https://www.tandfonline.com/doi/abs/10.1080/13518470701705736
- Whelan 2025, *Estimating Expected Loss Rates in Betting Markets: Theory and Evidence*, Applied Economics: https://www.karlwhelan.com/Papers/Overround.pdf
- Lyócsa & Výrost 2018, *To Bet or Not to Bet: A Reality Check for Tennis Betting Market Efficiency*, Applied Economics 50(20): https://www.tandfonline.com/doi/abs/10.1080/00036846.2017.1394973
- Kaunitz, Zhong & Kreiner 2017, *Beating the Bookies with Their Own Numbers - and How the Online Sports Betting Market is Rigged*, arXiv:1710.02824: https://arxiv.org/abs/1710.02824
- Miller & Nichols 2026, *Efficient Market Insights and Favorite-Longshot Bias in Mixed Martial Arts Betting Markets*, Journal of Economics and Finance 50(1): https://doi.org/10.1007/s12197-026-09757-x

**Brain Implementation:** Concentrate stake and model effort on MMA/props/tennis-early lanes; use NFL/NBA/MLB closers as near-fair benchmarks for edge estimates.

---

## 7. Closing Line Value (CLV) as Edge Measure

**Summary:** Closing line value (CLV) is the price you got relative to the price the market settles at just before the event (decimal odds taken / closing odds - 1); the closing line is treated as the market's best aggregate estimate of true probability because it reflects the most information, liquidity and lowest margin. Croxson & Reade (2014) use high-frequency soccer data to show prices update 'swiftly and fully' to major news (goals on the cusp of half-time), implying the final price is the most informationally complete price in the sequence. Kaunitz et al. (2017) show a strategy that exploits the probability information implicit in bookmaker odds (rather than building a competing forecast) was profitable over 10 years of simulations and 5 months of real betting. Buchdahl's analysis of 132,645 football matches found the opening/closing odds ratio tracks realized returns almost one-for-one, validating closing odds as a yardstick for expected value.

**Citations:**
- Croxson & Reade 2014, *Information and Efficiency: Goal Arrival in Soccer Betting*, Economic Journal 124(575): https://academic.oup.com/ej/article-abstract/124/575/62/5076978
- Kaunitz, Zhong & Kreiner 2017, *Beating the bookies with their own numbers - and how the online sports betting market is rigged*, arXiv:1710.02824: https://arxiv.org/abs/1710.02824
- Buchdahl 2016, *Squares & Sharps, Suckers & Sharks: The Science, Psychology & Philosophy of Gambling*: https://www.lovereading.co.uk/book/9781843448587/isbn/Squares-Sharps-Suckers-Sharks-by-Joseph-Buchdahl.html
- Buchdahl (n.d.), *Using the closing line to test your skill in betting*, Sports Trading Network: https://www.sportstradingnetwork.com/article/using-the-closing-line-to-test-your-skill-in-betting/

**Brain Implementation:** Grade every backtested bet against de-vigged closing odds and store realized CLV per bet as the primary edge metric across all sport lanes, using CLV-vs-ROI slope as the model validation gate.

---

## 8. Kelly Criterion & Fractional Kelly Sizing

**Summary:** Kelly (1956) showed that a gambler with a noisy private wire who bets to maximize the expected log of wealth achieves the maximum exponential growth rate of capital. Thorp (2006) operationalizes this for gambling: for even-money bets the growth-optimal fraction is f* = p - q, generalizing to f* = (bp - q)/b with net odds b. MacLean, Thorp & Ziemba (2011) document the overbetting result: betting exactly twice the Kelly wager drives the growth rate to zero (plus the risk-free rate), and any stake beyond 2x Kelly produces a negative expected log growth rate. This motivates hard caps at a fraction of Kelly when model probabilities are uncertain. Standard practice: use 0.25-Kelly to 0.5-Kelly sizing, hard-cap per-event exposure, and treat realized CLV (not win rate) as the skill metric.

**Citations:**
- Kelly 1956, *A New Interpretation of Information Rate*, Bell System Technical Journal 35(4): https://archive.org/details/bstj35-4-917
- Thorp 2006, *The Kelly Criterion in Blackjack, Sports Betting and the Stock Market*, in Zenios & Ziemba (eds.), Handbook of Asset and Liability Management, Vol. 1: https://books.google.com.sg/books?id=SWFkCIRMNtoC
- MacLean, Thorp & Ziemba 2011, *Good and Bad Properties of the Kelly Criterion*, in The Kelly Capital Growth Investment Criterion: Theory and Practice: https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf

**Brain Implementation:** Default to 0.25-Kelly sizing on model edge vs closing odds, hard-cap every stake at <= 0.5x full Kelly and cap total per-event exposure, since stakes above 2x full Kelly ruin expected log growth.

---

## 9. Bayesian Hierarchical Team-Strength Models

**Summary:** Baio & Blangiardo (2010) model football scores with bivariate Poisson likelihoods whose log-intensities are team-specific attack/defense strengths pooled toward league-level means. The canonical PyMC example reproduces their model in modern probabilistic programming with NUTS sampling, and Gelman's *Stan goes to the World Cup* post is the canonical Stan workflow for estimating team abilities from score differences with partial pooling. These models yield full posterior distributions over match probabilities, useful both as standalone predictors and as prior anchors for other lanes, and are the standard Bayesian alternative to frequentist Dixon-Coles.

**Citations:**
- Baio & Blangiardo 2010, *Bayesian hierarchical model for the prediction of football results*, Journal of Applied Statistics 37(2): https://discovery.ucl.ac.uk/id/eprint/16040/
- Coyle 2022, *A Hierarchical model for Rugby prediction*, PyMC example notebook: https://www.pymc.io/projects/examples/en/2021.11.0/case_studies/rugby_analytics.html
- Gelman 2014, *Stan goes to the World Cup*: https://statmodeling.stat.columbia.edu/2014/07/13/stan-analyses-world-cup-data/

**Brain Implementation:** Implement a per-league Poisson hierarchical model in PyMC with partial-pooled attack/defense strengths (non-centered parameterization, NUTS); use posterior-predictive match probabilities as a base lane and as priors for other model lanes.

---

## 10. Kalman Filter / State-Space Time-Varying Team Ratings

**Summary:** Glickman & Stern (1998) model NFL score differences with team-strength parameters following a first-order autoregressive process (plus team-varying home-field advantage), estimated by MCMC, and report beating the Las Vegas line on a 110-game test set. Knorr-Held (2000) formalizes dynamic paired-comparison ratings with a constrained random-walk prior and extended Kalman filter/smoother, applied to Bundesliga and NBA data. Glickman (1999) describes the Glicko system, a computationally cheap non-iterative dynamic Bradley-Terry update with rating uncertainty that scales to tens of thousands of competitors. The common thread is that recent results should dominate current strength estimates, and Kalman-style updates give closed-form weekly rating updates with uncertainty.

**Citations:**
- Glickman & Stern 1998, *A State-Space Model for National Football League Scores*, Journal of the American Statistical Association 93(441): http://www.glicko.net/research/nfl.pdf
- Knorr-Held 2000, *Dynamic rating of sports teams*, Journal of the Royal Statistical Society: Series D (The Statistician) 49(2): https://academic.oup.com/jrsssd/article-abstract/49/2/261/7123397
- Glickman 1999, *Parameter estimation in large dynamic paired comparison experiments*, Journal of the Royal Statistical Society: Series C (Applied Statistics) 48(3): https://academic.oup.com/jrsssc/article-abstract/48/3/377/6990661
- Koopman & Lit 2015, *A dynamic bivariate Poisson model for analysing and forecasting match results in the English Premier League*, Journal of the Royal Statistical Society: Series A (Statistics in Society) 178(1): https://academic.oup.com/jrsssa/article-abstract/178/1/167/7058470

**Brain Implementation:** Implement Knorr-Held/Glickman-Stern style state-space ratings per league with weekly Kalman-filter updates (constrained random-walk transition, team-varying home-field advantage); expose current ratings and their variances as features to downstream models and as a standalone closing-line benchmark.

---

## 11. Ensemble & Stacking for Sports Probabilities

**Summary:** Wolpert (1992) introduced stacked generalization: train a meta-learner on the out-of-sample outputs of base models rather than averaging them directly. Breiman (1996) showed bagging improves unstable predictors via bootstrap aggregation. Dietterich (2000) surveys why ensembles work (statistical, computational, representational reasons) and the conditions for diversity. In sports, Hübáček, Šourek & Železný (2019) report their winning 2017 Soccer Prediction Challenge solution using gradient-boosted tree models over hand-built features. Bunker & Thabtah (2019) review ML for sport result prediction and propose a structured framework. Practitioner consensus: average in log-odds space rather than probability space, calibrate the blend (Platt/isotonic), and ensure base-model diversity across model families.

**Citations:**
- Wolpert 1992, *Stacked generalization*, Neural Networks 5(2): https://www.sciencedirect.com/science/article/pii/S0893608005200231
- Breiman 1996, *Bagging predictors*, Machine Learning 24(2): https://link.springer.com/article/10.1007/BF00058655
- Dietterich 2000, *Ensemble methods in machine learning*, in Multiple Classifier Systems: https://link.springer.com/chapter/10.1007/3-540-45014-9_1
- Hübáček, Šourek & Železný 2019, *Learning to predict soccer results from relational data with gradient boosted trees*, Machine Learning 108(1): https://dl.acm.org/doi/10.1007/s10994-018-5704-6
- Bunker & Thabtah 2019, *A machine learning framework for sport result prediction*, Applied Computing and Informatics 15(1): https://www.sciencedirect.com/science/article/pii/S2210832717301485

**Brain Implementation:** Stack heterogeneous base models (state-space ratings, hierarchical Bayesian, GBDT on team/player features) with a logistic meta-learner trained on out-of-fold predictions; average/combine in log-odds space, apply isotonic/Platt calibration, and monitor ensemble diversity.

---

## 12. Online Learning & Concept-Drift Adaptation

**Summary:** Gama et al. (2014) survey concept-drift adaptation, cataloging detection methods (statistical tests, windowing) and adaptation strategies (retraining, ensemble weighting). Gama et al. (2004) introduced DDM, which tracks error-rate increase as a drift signal; Bifet & Gavaldà (2007) introduced ADWIN, an adaptive sliding window with guaranteed bounds that automatically grows when data is stable and shrinks on drift. McMahan et al. (2013) document FTRL-Proximal, the online logistic-regression optimizer behind Google's CTR system with per-coordinate adaptive learning rates. For sports, drift is structural (off-season roster changes, rule changes, team form); Dixon & Coles (1997) pioneered recency weighting via exponential decay when fitting football models.

**Citations:**
- Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia 2014, *A survey on concept drift adaptation*, ACM Computing Surveys 46(4): https://dl.acm.org/doi/10.1145/2523813
- Gama, Medas, Castillo & Rodrigues 2004, *Learning with drift detection*, in Brazilian Symposium on Artificial Intelligence: https://dblp.org/rec/conf/sbia/GamaMCR04.html
- Bifet & Gavaldà 2007, *Learning from time-changing data with adaptive windowing*, in SIAM International Conference on Data Mining: https://epubs.siam.org/doi/abs/10.1137/1.9781611972771.42
- McMahan et al. 2013, *Ad click prediction: a view from the trenches*, in Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining: https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
- Dixon & Coles 1997, *Modelling association football scores and inefficiencies in the football betting market*, JRSS-C 46(2): https://doi.org/10.1111/1467-9876.00065

**Brain Implementation:** Run ADWIN/DDM drift detectors per league-lane to trigger walk-forward retraining; warm-start GBDT/linear models from prior-window parameters; use FTRL-Proximal online logistic updates for fast-changing markets; apply Dixon-Coles-style exponential recency decay in feature aggregation.

---

## 13. Monte Carlo Match Simulation — Per-Minute Poisson Goal Process

**Summary:** Generates full probability distributions over correct scores and match outcomes via N_sim repeated samples. For each iteration: draw home goals ~ Poisson(λ), away goals ~ Poisson(μ), apply Dixon-Coles tau adjustment if ρ fitted. Alternatively, per-minute decomposition: segment 90 min into 1-min bins, sample goals ~ Poisson(λ/90) per bin, accumulate to final scoreline. Aggregates to correct-score probabilities, totals (over/under 2.5), and in-play state vectors. Foundation for tournament bracket and in-play betting models.

**Citations:**
- Sumpter 2016, *Soccermatics: Mathematical Adventures in the Beautiful Game*: https://books.google.com/books?id=B64yEAAAQBAJ
- Koopman & Lit 2015, *A Dynamic Bivariate Poisson Model for Analysing and Forecasting Match Results in the English Premier League*, Journal of the Royal Statistical Society: Series A 178(1): https://doi.org/10.1111/rssa.12042

**Brain Implementation:** `for i in range(10000): x ~ Poisson(λ); y ~ Poisson(μ); apply tau_dc(x, y, λ, μ, ρ); tally (x, y) → histogram; normalize → P(X=x, Y=y)`. For per-minute: aggregate 10k paths to correct-score matrix and in-play state probabilities.

---

## 14. Injury & Lineup Modeling from Official Reports

**Summary:** Academic literature on injury pricing is thinner than for other topics. Sauer (2005) shows point spreads contain efficient estimates of a missing player's value—biased during short absences but consistent with efficient pricing under partial anticipation. Dare, Dennis & Paul (2015) show NBA opening lines have significant errors when a high-value player is absent but all bias is removed by closing time—i.e., closing lines already price official absence information, so injury alpha lives at the open. Fischer & Schmal (2025) find initial inertia then lagged reaction in odds after elite-soccer absence announcements. **Official sources:** NFL Personnel (Injury) Report Policy (practice participation and Out/Doubtful/Questionable designations, reported Wed-Fri with 4 p.m. ET deadlines); SportsDataIO provides machine-readable structured injury feeds. **No peer-reviewed paper quantifies direct NFL injury-report win probability effects**—the raw signal is official, but direct modeling is practitioner-grade.

**Citations:**
- Sauer 2005, *Player injuries and price responses in the point spread wagering market*, in Vaughan Williams (ed.), The Economics of Gambling, Routledge: https://www.taylorfrancis.com/chapters/edit/10.4324/9780203986936-19/player-injuries-price-responses-point-spread-wagering-market-raymond-sauer
- Dare, Dennis & Paul 2015, *Player absence and betting lines in the NBA*, The Quarterly Review of Economics and Finance 57: https://www.sciencedirect.com/science/article/abs/pii/S1544612315000227
- Fischer & Schmal 2025, *Pricing in response to new information: The case of betting markets*, Economic Inquiry 63(1): https://ideas.repec.org/a/bla/ecinqu/v63y2025i1p236-264.html
- NFL 2015, *2015 Personnel (Injury) Report Policy*: https://operations.nfl.com/media/1818/2015-injury-report-policy.pdf
- SportsDataIO, *NFL Database - Football Statistics Data (injuries data dictionary)*: https://sportsdata.io/developers/data-dictionary/nfl?subfeed_id=630

**Brain Implementation:** Ingest structured injury feeds (official NFL practice/status reports; SportsDataIO-style NBA/NFL availability APIs) and encode position-weighted injury impact (e.g., QB/star-weighted absence scores by severity tier) as features; evaluate any edge only against opening lines, since verified evidence says closers already absorb this information.

---

## 15. Benter Practitioner Template: Log-Odds Linear Model + Kelly Sizing

**Summary:** Benter's canonical report documents the full production system that returned roughly a billion dollars in Hong Kong parimutuel pools: a conditional logit (log-odds linear) fundamental model of win probabilities over engineered factors, combined with the public's implied probabilities to correct for public information the model misses. The model is bet only where its probability exceeds the market's implied probability by a threshold, and stakes are sized by Kelly on the resulting edge, with data partitioning to control overfitting (500-1,000+ race development samples with out-of-sample testing). It remains the canonical template for multi-market machine-learning betting systems judged against market prices.

**Citations:**
- Benter 1994, *Computer Based Horse Race Handicapping and Wagering Systems: A Report*: https://www.gwern.net/docs/statistics/decision/1994-benter.pdf
- Benter 1995, *Computer based horse race handicapping and wagering systems*, in Hausch, Lo & Ziemba (eds.), Efficiency of Racetrack Betting Markets, Academic Press: https://eprints.soton.ac.uk/51684/

**Brain Implementation:** Per-sport, fit a conditional logit model on engineered features, blend model probabilities with de-vigged market implied probabilities, bet when the blend exceeds the market by a threshold, and size by Kelly on the blend edge.

---

## Implementation Roadmap

### Phase 1: Soft-Market Edge Concentration
1. **MMA/UFC (primary):** Apply Miller & Nichols fighter-characteristic angles (youth, travel, women's contests) + Markov-chain fight simulation (Holmes, McHale & Zychaluk) on opening lines (Miller & Nichols 2026 efficiency finding); validate walk-forward vs closing, target 25-30% ROI.
2. **Tennis (secondary):** Model FLB-adjusted odds with surface-form Elo, serve/return EWMA, fatigue (Whelan 2025 strong bias finding); test against early-line closes; gate on high-confidence threshold.
3. **Soccer long-tail:** Implement Kaunitz-style consensus-relative strategy (compare candidate book odds against market consensus); focus lower leagues and long-tail fixtures.
4. **Props (experimental):** Encode position-weighted injury impact, run vs Pinnacle prop closers, measure realized CLV per bet.

### Phase 2: Efficient-Market Benchmarking
5. **NFL/NBA (close-only):** Deploy state-space Kalman ratings (Knorr-Held style) + hierarchical Bayesian (Baio & Blangiardo); use these as near-fair CLV benchmarks to estimate edge on boundary niches (highest totals, preseason underdogs, sentiment).
6. **All sports:** Compute realized CLV (de-vigged odds taken / closing odds - 1) per bet as the primary edge metric; discard win-rate focus.

### Phase 3: Advanced Features
7. **xG/xT for soccer:** Pull understat `/main/getLeagueData/<slug>/<season>` gzip JSON; compute rolling xG/xT differentials; test additive signal.
8. **VAEP for soccer:** On event-sequence data availability, train binary P(score) and P(concede) classifiers; aggregate team VAEP per 90 min.
9. **Injury integration:** Ingest official NFL/NBA injury feeds (SportsDataIO API or ESPN snapshot); encode as position-weighted absence severity; validate against opening lines only (closers efficient per Dare et al. 2015).
10. **Online learning:** Run ADWIN drift detectors per league; trigger walk-forward retrains on detected shifts; warm-start from prior-window parameters; apply Dixon-Coles exponential recency decay.

### Phase 4: Ensemble & Staking
11. **Combine base models:** Stack state-space ratings + hierarchical Bayesian + GBDT on features + naive log-odds blender, calibrate with isotonic regression, average in log-odds space.
12. **Kelly sizing:** Default 0.25-Kelly per bet, hard-cap at 0.5x, never exceed 2x (MacLean et al. 2011); cap total per-event exposure.
13. **Walk-forward validation:** Pre-register gates and feature sets; evaluate every lane against closing-line CLV; report CIs from bootstrap resampling.

---

## Summary: Top 10 Techniques

1. **Dixon-Coles + Monte Carlo** — Soccer probabilistic baseline; tau-corrected bivariate Poisson with time-decay.
2. **State-Space Kalman Ratings (Knorr-Held)** — Per-league dynamic team strengths; weekly updates with uncertainty bands.
3. **Bayesian Hierarchical Pooling (Baio & Blangiardo)** — Partial-pooled attack/defense in PyMC/Stan; posterior-predictive match probabilities.
4. **xG/xT/VAEP Features** — Soccer match-stat inputs; test walk-forward edge vs closers.
5. **Market Efficiency per Sport** — NFL/NBA/MLB/MMA near-efficient; tennis/soccer-long-tail/props soft; concentrate model effort accordingly.
6. **Closing Line Value (CLV)** — Primary edge metric; validate all bets against de-vigged closers (not win rate).
7. **Kelly Criterion (0.25–0.5x)** — Log-optimal staking; hard-cap at 0.5x, never exceed 2x; real-money implementation critical.
8. **Ensemble Stacking in Log-Odds** — Combine heterogeneous base models; isotonic calibration; validate per-sport.
9. **Online Learning / Drift Detection (ADWIN)** — Trigger walk-forward retrains on structural shifts; warm-start parameters; recency-decay features.
10. **Injury Modeling (Official Reports)** — Encode position-weighted absence severity; test against opening lines only (closers already efficient per Dare et al. 2015).

---

**Key Principle:** Close lines in major markets (NFL, NBA, MLB, MMA) are near-efficient. Exploitable edge lives in soft niches (MMA angles, tennis FLB, soccer long-tail, props) and hard-to-price dimensions (injury timing, line-movement timing, multi-sport ensemble). Validate ALL work walk-forward against CLV, not in-sample accuracy. Size with Kelly. Report CIs. Admit uncertainty.
