// Help's words: the glossary the Help tab shows, and the guides (Start here, How the AI decides,
// Rollover explained, Responsible play, FAQ).
//
// The glossary builds on ui.js's GLOSSARY, which the ⓘ sheets on every screen read, and adds what a
// help page needs on top: a worked example step by step, the other names people search for ("vig",
// "juice", "parlay", "acca"...), a group, related terms, where "See it live" goes, and a few terms the
// ⓘ sheets don't have yet (implied probability, Kelly, ROI, AI chance vs market chance). Keys are the
// same as GLOSSARY's, so every ⓘ sheet's "More in Help" (#/help/<key>) lands on its entry here.
//
// Every entry needs a worked example and a "See it live" link, GLOSSARY's terms included: a term added
// there needs its EXTRAS entry here too (group, other names, steps, related terms), or it shows without
// them under "Odds and chances".
//
// Every number on these pages is fixed teaching text, worked out by hand and checked (the arithmetic is
// written into the steps so it can be checked again). The app's own numbers always come from the server.

import { GLOSSARY } from "../../ui.js";

// --- the glossary ---------------------------------------------------------------------------------------

export const GROUPS = [
  { key: "odds", title: "Odds and chances" },
  { key: "bets", title: "Bet types and results" },
  { key: "money", title: "Money, staking and plans" },
  { key: "ai", title: "The AI and its record" },
];

// Terms the ⓘ sheets don't have (yet): same shape as GLOSSARY's entries
const NEW_TERMS = {
  implied_probability: {
    title: "Implied probability",
    short: "The chance a price suggests: 1 ÷ the decimal odds.",
    body: "It still includes the bookmaker's margin, so both sides of a market add up to more than 100%. Take the margin out and you get the market chance the app shows.",
    example: "2.50 implies 40%; 1.91 implies 52.4%.",
  },
  ai_vs_market: {
    title: "AI chance vs market chance",
    short: "Two views of the same outcome, shown side by side and never blended in secret.",
    body: "The market chance is the bookmakers' view with their margin removed: the hardest benchmark there is (closing prices pick about 68% of NBA winners). The AI chance is the brain's own estimate. Where they differ the app shows both, and a big gap gets a warning rather than a cheer.",
    example: "Thunder: market 64%, AI 68%. The AI likes them 4 points more than the bookmakers do.",
  },
  kelly: {
    title: "Kelly stake",
    short: "The stake that grows a bankroll fastest when you truly have an edge. With no edge it's zero.",
    body: "Kelly = (chance × odds − 1) ÷ (odds − 1), as a share of your bankroll. It needs a real edge, which almost nobody has against the market, and an overconfident chance makes it overbet, so many professionals use half of it or less. This app never sizes stakes from the AI's confidence.",
    example: "A true 55% chance at 2.00: Kelly says 10% of the bankroll, half-Kelly 5%.",
  },
  paste_slip: {
    title: "Paste a slip",
    short: "Paste a bet slip from any bookmaker (its text or a screenshot) and the app checks it at your own odds.",
    body: "In Ask AI, tap 📋 Paste a slip. The AI only copies what the slip says; the app's code checks that copy, finds each leg among the games it has odds for and works out the honest chance, what the bookmaker keeps, the weakest leg, better prices elsewhere and cheaper ways to play it. Then you can talk it through: \"what if I drop leg 3?\". "
      + "It reads moneylines, spreads, totals and NBA player points, rebounds, assists, threes, steals, blocks and turnovers. It can't model player props outside the NBA or unusual markets (both teams to score, correct score, first scorer): those legs are priced at your odds with an estimated chance, and the card says so. "
      + "Each leg's chance comes from the market (the bookmakers' prices with their margin taken out), from the NBA model, or is estimated from your own price minus a typical bookmaker margin. When a read looks doubtful (your bookmaker's return doesn't match, or some odds couldn't be read) the app asks you to check it before any numbers. "
      + "Screenshots aren't kept: only what was read from them, in your own history that nobody else can see. With too little AI allowance left, or while picks are paused, the free reader (code only) reads pasted text at no cost; screenshots need the AI, so paste the slip's text then.",
    example: "Celtics −3.5 at −110 and Nuggets to win at −400 for $10: $23.86 back if both win, about 1 in 2.6.",
  },
  roi: {
    title: "Net profit and ROI",
    short: "What you're up or down after every stake, and that as a share of everything you staked.",
    body: "Net profit is everything that came back minus everything staked. ROI (return on investment) is net profit ÷ total staked. The app never shows \"total won\" on its own: winnings without the losing stakes look better than the truth.",
    example: "Staked $40, $25 came back: net −$15, ROI −37.5%.",
  },
};

// What the Help tab adds to each term: group, other names, worked steps, related terms, "See it live"
const EXTRAS = {
  decimal_odds: {
    group: "odds", aka: ["European odds"], aliases: ["odds", "decimal", "european odds", "odds format", "price"],
    worked: ["$10 at 2.50 returns $10 × 2.50 = $25.", "That's your $10 back plus $15 profit.", "If it loses, nothing comes back: you're $10 down."],
    related: ["american_odds", "implied_probability", "payout"],
  },
  american_odds: {
    group: "odds", aka: ["US odds", "moneyline odds"], aliases: ["us odds", "moneyline odds", "+150", "-110", "odds format", "plus minus"],
    worked: ["+150: a $100 bet wins $150 profit. In decimal that's 1 + 150 ÷ 100 = 2.50.",
      "−150: stake $150 to win $100. In decimal that's 1 + 100 ÷ 150 = 1.67.",
      "−110 is 1.91 decimal (1 + 100 ÷ 110): the usual price on each side of a spread."],
    related: ["decimal_odds", "implied_probability"], seeLabel: "Switch the format in Settings",
  },
  implied_probability: {
    group: "odds", aliases: ["implied odds", "implied chance", "break even", "breakeven", "probability"],
    worked: ["1 ÷ 2.50 = 0.40, so a price of 2.50 implies a 40% chance.",
      "A spread at 1.91 on both sides implies 1 ÷ 1.91 = 52.4% each: 104.7% in all.",
      "The extra 4.7% is the bookmaker's margin. Take it out and each side is 50%."],
    related: ["market_chance", "book_cut", "fair_odds"], see: "#/today", seeLabel: "See the chances on Today",
  },
  market_chance: {
    group: "odds", aka: ["no-vig chance", "fair chance"], aliases: ["no vig", "no-vig", "fair chance", "consensus", "market", "mkt"],
    worked: ["Best prices: Thunder 1.50, Lakers 2.70.",
      "Implied chances: 1 ÷ 1.50 = 66.7% and 1 ÷ 2.70 = 37.0%, which add up to 103.7%.",
      "Divide each by 103.7% to take the margin out: Thunder 64%, Lakers 36%."],
    related: ["implied_probability", "ai_vs_market", "book_cut"], seeLabel: "See \"Mkt\" chances on Today",
  },
  ai_chance: {
    group: "ai", aliases: ["ai", "model", "prediction", "brain"],
    worked: ["The NBA model blends the bookmakers' consensus, ESPN's matchup predictor and a team model (form, rest, injuries).",
      "It says Thunder 64%: it expects them to win about 64 games in 100 like this one.",
      "Other sports have no model yet, so their screens show the market's chance, marked \"Mkt\"."],
    related: ["ai_vs_market", "market_chance", "brier"],
  },
  ai_vs_market: {
    group: "ai", aliases: ["edge", "disagree", "value", "gap", "far from market"],
    worked: ["Market: Thunder 64% (bookmakers, margin removed). AI: Thunder 68%.",
      "At odds of 1.50 a $10 bet profits $5. By the AI's chance the expected result is 0.68 × $15 − $10 = +$0.20; by the market's, 0.64 × $15 − $10 = −$0.40.",
      "A gap of 8 points or more gets a \"Far from market\" warning: it often means news the model hasn't caught, like a late injury."],
    related: ["ai_chance", "market_chance", "expected_result", "brier"], see: "#/bets?tab=record", seeLabel: "See the AI's record",
  },
  value_mark: {
    group: "ai", aka: ["value mark", "far-from-market warning"],
    aliases: ["value", "value mark", "triangle", "up arrow", "exclamation", "far from market", "warning", "marks", "flag"],
    worked: ["A price of 2.00 with the market at 48% and the AI at 55%: the AI is 7 points above the market.",
      "By the AI's chance $10 returns 0.55 × $20 = $11 on average, +$1; by the market's, 0.48 × $20 = $9.60, −$0.40. A gap of 3 to 8 points that gains more than 3% of the stake by the AI's numbers gets ▲.",
      "Even by the AI's own chance that pick loses 45 times in 100.",
      "Had the AI said 57%, 9 points above the market, the price would show ! instead: a gap that big usually means news the model hasn't caught, like a late injury."],
    related: ["ai_vs_market", "expected_result", "ai_chance"], seeLabel: "See the marks on Today",
  },
  fair_odds: {
    group: "odds", aliases: ["true odds", "break even odds", "no-vig odds"],
    worked: ["A 40% chance has fair odds of 1 ÷ 0.40 = 2.50.",
      "At 2.50, $10 returns $25 four times in ten: $10 back on average, break-even.",
      "At 2.30 it returns $23 four times in ten: 0.40 × $23 = $9.20 on average, an expected result of −$0.80."],
    related: ["implied_probability", "expected_result", "kelly"], see: "#/build", seeLabel: "See fair odds in Build",
  },
  book_cut: {
    group: "odds", aka: ["vig", "vigorish", "juice", "margin", "overround"], aliases: ["vig", "vigorish", "juice", "margin", "overround", "house edge", "the cut", "bookmaker's cut", "hold"],
    worked: ["Both sides at 1.91: one person stakes $100 on each side, $200 in all.",
      "Whichever side wins is paid $191, so the bookmaker keeps $9: 4.5% of the money staked.",
      "In a multi the cut is paid on every leg: 5 legs at 4.5% each keep about 20.6% (1 − 0.955⁵), 10 legs about 36.9%."],
    related: ["implied_probability", "multi", "expected_result"], seeLabel: "See it on your slip",
  },
  expected_result: {
    group: "money", aka: ["expected value", "EV"], aliases: ["expected value", "ev", "+ev", "average return", "expected loss", "expected cost"],
    worked: ["A 40% chance at 2.30, $10 staked.",
      "Four times in ten it returns $23, six times in ten nothing: 0.40 × $23 = $9.20 back on average.",
      "Expected result: $9.20 − $10 = −$0.80. It's positive only when the price beats the fair odds (2.50 here)."],
    related: ["fair_odds", "book_cut", "kelly"], see: "#/build", seeLabel: "See it on your slip",
  },
  one_in: {
    group: "odds", aliases: ["1 in x", "one in", "odds against", "chance"],
    worked: ["14% is about 1 in 7 (1 ÷ 0.14 = 7.1).",
      "A 3-leg multi of coin flips wins 0.5 × 0.5 × 0.5 = 12.5% of the time: 1 in 8.",
      "So it loses about 7 times in every 8."],
    related: ["multi", "most_likely"], see: "#/build", seeLabel: "See it on your slip",
  },
  single: {
    group: "bets", aka: ["straight bet"], aliases: ["straight", "straight bet", "singles", "single bet"],
    worked: ["Three singles at $10 each: 1.91, 1.80 and 2.10. They cost $30.",
      "Two win (1.91 and 2.10), one loses: $19.10 + $21.00 = $40.10 comes back, a net +$10.10.",
      "The same three picks as a $10 multi would have lost the whole $10."],
    related: ["multi", "round_robin", "stake"],
  },
  multi: {
    group: "bets", aka: ["parlay", "accumulator", "acca"], aliases: ["parlay", "accumulator", "acca", "accum", "combo", "multiple", "treble", "double"],
    worked: ["Three legs at 1.91 each: 1.91 × 1.91 × 1.91 = 6.968, shown as 6.97.",
      "$10 pays $69.68 if all three win: a $59.68 profit.",
      "If each leg is a true coin flip, all three win 1 time in 8 (12.5%). On average $10 returns 0.125 × $69.68 = $8.71: an expected result of −$1.29.",
      "That −$1.29 is the bookmaker's cut, taken three times."],
    related: ["single", "round_robin", "book_cut", "same_game"],
  },
  round_robin: {
    group: "bets", aliases: ["round robin", "rr", "by 2s", "by twos", "doubles"],
    worked: ["4 picks \"by 2s\" make 6 doubles: every pair (4 × 3 ÷ 2).",
      "$5 each costs $30. With every pick at 1.91, a winning double pays $5 × 3.6481 = $18.24.",
      "3 of 4 win: 3 doubles pay $54.72, a $24.72 profit. 2 of 4 win: one double pays $18.24, an $11.76 loss.",
      "With coin flips the chance of any profit is 31% (3 or 4 winners), against 6% (1 in 16) for the same picks as a 4-leg multi. Each double pays the bookmaker's cut twice, not four times: on average it returns 91% of its stake, the 4-leg multi 83%."],
    related: ["multi", "system", "single"],
  },
  system: {
    group: "bets", aka: ["full-cover bets"], aliases: ["system bet", "trixie", "patent", "yankee", "lucky 15", "lucky15", "canadian", "super yankee", "heinz", "goliath", "full cover"],
    worked: ["Trixie: 3 picks make 3 doubles and 1 treble, 4 bets. At $2 a bet it costs $8.",
      "With every pick at 2.00, each double pays $2 × 4 = $8 and the treble $2 × 8 = $16.",
      "Two picks win: one double pays $8, your $8 back (net $0). All three win: 3 × $8 + $16 = $40, a $32 profit.",
      "Bigger ones: Patent 7 bets (a Trixie plus the singles), Yankee 11 (4 picks), Lucky 15, Heinz 57 (6 picks), Goliath 247 (8 picks)."],
    related: ["round_robin", "stake", "multi"],
  },
  moneyline: {
    group: "bets", aka: ["match winner", "head to head"], aliases: ["winner", "ml", "h2h", "head to head", "match result", "1x2"],
    worked: ["Thunder to win at 1.48: $10 returns $14.80 if they win, a $4.80 profit.",
      "Lakers to win at 2.75: $10 returns $27.50.",
      "In soccer there are three outcomes (home, draw, away), so a draw loses both team bets."],
    related: ["spread", "total", "decimal_odds"], see: "#/today", seeLabel: "See the Winner column on Today",
  },
  spread: {
    group: "bets", aka: ["handicap", "point spread", "run line", "puck line"], aliases: ["handicap", "point spread", "ats", "against the spread", "cover", "run line", "puck line"],
    worked: ["Thunder −6.5 at 1.91: they must win by 7 or more.",
      "Final 110–100 (by 10): the bet wins and $10 returns $19.10.",
      "Final 105–100 (by 5): the bet loses, and Lakers +6.5 would have won.",
      "On a whole-number line like −7, winning by exactly 7 is a push: the stake comes back."],
    related: ["push", "line", "total"], see: "#/today", seeLabel: "See the Spread column on Today",
  },
  total: {
    group: "bets", aka: ["over/under"], aliases: ["over under", "over/under", "o/u", "totals", "points total"],
    worked: ["Over 224.5 at 1.91: the two scores together must reach 225.",
      "A 115–112 final makes 227, so the over wins: $10 returns $19.10.",
      "Over 7 runs on a 4–3 game lands exactly on 7: a push, so the stake comes back."],
    related: ["push", "line", "spread"], see: "#/today", seeLabel: "See the Total column on Today",
  },
  push: {
    group: "bets", title: "Push and void",
    aliases: ["void", "refund", "tie", "cancelled", "canceled", "postponed", "no action", "dead heat"],
    more: "A void leg (a game called off, or postponed past the bookmaker's limit) counts the same way: the stake on it comes back, and in a multi it drops out.",
    worked: ["A single on Over 7 runs and the game ends 4–3: a push, your $10 comes back.",
      "A multi of three legs at 1.91 with one push: that leg counts as 1.00, so the odds fall to 1.91 × 1.91 = 3.65 and $10 pays $36.48.",
      "Every leg pushed or void: the whole stake comes back."],
    related: ["total", "spread", "multi"], see: "#/bets?tab=settled", seeLabel: "See your settled bets",
  },
  line: {
    group: "bets", aliases: ["prop line", "number", "props", "player prop"],
    worked: ["LeBron James over 24.5 points: 25 or more wins, 24 or fewer loses.",
      "Half-point lines (.5) can't push; whole-number lines can.",
      "Lines move as money and news come in. Your picks are priced again at the current line before anything is saved."],
    related: ["hit_rate", "projection", "push"], see: "#/today", seeLabel: "Open a game, then a player",
  },
  hit_rate: {
    group: "ai", aliases: ["hit rate", "trend", "last 10", "l10", "last 5", "streak"],
    worked: ["Over 24.5 points went over in 7 of the last 10 games: \"7/10\".",
      "Ten games is a small sample. A player who truly goes over 55% of the time still shows 7/10 or better about 1 time in 4 (27%).",
      "So treat 7/10 as a hint, check the minutes and injuries, and remember the line already reflects most of it."],
    related: ["line", "projection"], see: "#/today", seeLabel: "Open a game, then a player",
  },
  projection: {
    group: "ai", aliases: ["projection", "player model", "range", "p(over)"],
    worked: ["Mean 24.1 points, range 15–33 (10% to 90%).",
      "About 1 game in 5 lands outside the range: 1 in 10 below 15, 1 in 10 above 33.",
      "55% over 23.5 means fair odds of 1 ÷ 0.55 = 1.82. A book price of 1.91 is a little above fair; 1.80 a little below."],
    related: ["hit_rate", "line", "fair_odds"], see: "#/today", seeLabel: "Open an NBA game, then a player",
  },
  rollover: {
    group: "money", aliases: ["roll over", "rollover challenge", "2 odds", "sure 2 odds", "daily double", "compounding"],
    worked: ["$10 at 2.00 every day for 10 days would end at $10 × 2¹⁰ = $10,240.",
      "Once the bookmaker's cut is out, the market gives a 2.00 bet about 47%: 0.47¹⁰ is about 1 in 1,900.",
      "47% of runs survive day 1, 22% day 2, 10% day 3, 2.3% day 5: about 78% are over by day 2.",
      "On average a run returns (0.47 × 2)¹⁰ = 0.94¹⁰ = 54% of its stake: an expected loss of 46%."],
    related: ["kelly", "bankroll", "multi"], seeLabel: "Open the rollover planner", topic: "rollover",
  },
  bankroll: {
    group: "money", aliases: ["budget", "bank", "units", "weekly budget"],
    worked: ["$200 set aside for the season; 2% a bet is $4.",
      "Ten losses in a row happen to everyone sooner or later: at $4 a bet they cost $40, leaving $160.",
      "At 25% a bet, four losses in a row leave about $63 ($200 × 0.75⁴)."],
    related: ["kelly", "stake", "roi"], see: "#/settings?focus=budget", seeLabel: "Set a weekly budget in Settings",
  },
  stake: {
    group: "money", aliases: ["wager", "bet size", "unit stake", "amount"],
    worked: ["Multi: the stake is the whole bet, e.g. $10.",
      "Singles: the stake is per pick, so 3 singles at $10 cost $30.",
      "Round robins and systems: the stake is per bet. A $2 Yankee is 11 bets, $22 in all."],
    related: ["payout", "system", "bankroll"], see: "#/build", seeLabel: "Set a stake in Build",
  },
  payout: {
    group: "money", aliases: ["return", "returns", "to win", "potential payout", "winnings", "potential win"],
    worked: ["$10 at 4.60 pays $46: your $10 back plus $36 profit.",
      "\"To win\" on many US sites means the profit ($36); \"payout\" includes the stake ($46).",
      "The app always shows what comes back next to the chance of it happening."],
    related: ["stake", "decimal_odds", "most_likely"], see: "#/build", seeLabel: "See it on your slip",
  },
  most_likely: {
    group: "odds", aliases: ["most likely outcome", "likely result"],
    worked: ["A 3-leg multi with a 14% chance pays $70 on $10.",
      "The most likely outcome is losing the $10: 86% of the time.",
      "That's why the app shows it next to the payout, not tucked away."],
    related: ["one_in", "multi", "payout"], see: "#/build", seeLabel: "See it on your slip",
  },
  paste_slip: {
    group: "ai", aka: ["slip reader", "slip checker"],
    aliases: ["paste", "pasted slip", "bet slip", "betslip", "screenshot", "bookmaker slip", "check my slip", "read my slip",
      "slip reader", "free reader", "photo of slip", "upload slip"],
    worked: ["You paste: \"Boston Celtics -3.5 -110\", \"Denver Nuggets ML -400\", \"Stake $10\", \"To win $13.86\".",
      "Code converts the odds: −110 is 1 + 100 ÷ 110 = 1.909 and −400 is 1 + 100 ÷ 400 = 1.25.",
      "$10 × 1.909 × 1.25 = $23.86 back, a profit of $13.86: it matches the bookmaker's \"To win\", so the read is right.",
      "Both legs are on tonight's board, so their chances come from the market with the margin taken out: say 50% and 77%.",
      "Both must win: 0.50 × 0.77 = 38.5%, about 1 in 2.6. On average $23.86 × 0.385 = $9.19 comes back, so you expect to lose about $0.81 of the $10 (about 8%): the bookmaker's cut on this slip.",
      "A leg the app can't find is priced at your odds with an estimated chance: at 1.80 with a typical 4.5% margin, (1 ÷ 1.80) ÷ 1.045 = 53.2%."],
    related: ["book_cut", "market_chance", "one_in"], see: "#/ask?paste=1", seeLabel: "Paste a slip in Ask AI",
  },
  analysts: {
    group: "ai", aliases: ["claude", "deepseek", "ai analysts", "forecast", "analyst"],
    worked: ["Claude says Thunder 66%, the AI 64%, the market 62%.",
      "After the game each forecast is scored with a Brier score, next to the AI's and the market's.",
      "An analyst moves nothing until it has 40 graded games; after that it earns weight only if its record beats the AI's."],
    related: ["brier", "ai_chance", "ai_vs_market"], see: "#/bets?tab=record", seeLabel: "See the analysts' record",
  },
  tiers: {
    group: "money", aliases: ["safe", "risky", "crazy", "risk level", "tier", "value", "for fun"],
    worked: ["Safe: the AI gives the ticket 60% or more. At 62% it still loses about 4 times in 10.",
      "Risky: 25% to 60%. Crazy: 3% to 25%, long shots with big payouts.",
      "\"Value\" appears only when a cautious expected result is positive; every other ticket says \"For fun\" and shows its expected cost."],
    related: ["one_in", "expected_result", "ai_vs_market"], see: "#/ask", seeLabel: "Try it in Ask AI",
  },
  same_game: {
    group: "bets", aka: ["same-game parlay", "bet builder"], aliases: ["same game parlay", "sgp", "correlated", "bet builder", "same-game"],
    worked: ["Thunder −6.5 and Over 224.5 in the same game are linked: a blowout makes both more likely.",
      "Multiplying their chances as if they were unrelated gets the price wrong, and bookmakers keep about 20–35% on same-game bets.",
      "So this app blocks them in multis: play them as singles instead."],
    related: ["multi", "single"], see: "#/build", seeLabel: "Check your slip in Build",
  },
  quota: {
    group: "ai", aliases: ["odds left", "requests", "odds api", "credits"],
    worked: ["The odds feed allows about 500 requests a month, and each league fetch costs 3.",
      "Odds are kept for 2 hours, so keeping one league fresh all day costs 12 fetches × 3 = 36 requests.",
      "With fewer than 50 left the app shows the last saved odds instead of fetching."],
    related: ["market_chance"], see: "#/today", seeLabel: "See how fresh the odds are on Today",
  },
  brier: {
    group: "ai", aliases: ["brier score", "accuracy", "calibration", "log loss"],
    worked: ["Saying 70% for a team that wins: (1 − 0.70)² = 0.09.",
      "Saying 70% for a team that loses: (0 − 0.70)² = 0.49.",
      "Always saying 50% scores 0.25 every game; the closing market averages about 0.20 on the NBA."],
    related: ["analysts", "ai_vs_market", "calls"], see: "#/bets?tab=record", seeLabel: "See the AI's record",
  },
  calls: {
    group: "ai", aliases: ["my calls", "predictions", "leaderboard", "points"],
    worked: ["You give the Thunder 70% and they win: 25 − 100 × (1 − 0.70)² = +16 points.",
      "They lose (you gave the winner 30%): 25 − 100 × (1 − 0.30)² = −24 points.",
      "A 50% call always scores 0, so you only gain by being bold and right."],
    related: ["brier", "analysts"], see: "#/bets?tab=calls", seeLabel: "See your calls",
  },
  kelly: {
    group: "money", aka: ["Kelly criterion"], aliases: ["kelly criterion", "staking", "bet sizing", "fractional kelly", "half kelly"],
    worked: ["Suppose you truly had a 55% chance at 2.00. The edge is 0.55 × 2.00 − 1 = 0.10.",
      "Kelly = 0.10 ÷ (2.00 − 1) = 10% of your bankroll; half-Kelly is 5%.",
      "With a 2- to 5-point edge, staking everything (a rollover) is 10 to 25 times the Kelly stake.",
      "At the market's 47% the edge is negative (0.47 × 2.00 − 1 = −0.06), so Kelly says: bet nothing."],
    related: ["bankroll", "rollover", "expected_result"], see: "#/build/rollover", seeLabel: "Compare staking plans",
  },
  roi: {
    group: "money", aka: ["profit and loss", "yield"], aliases: ["return on investment", "profit and loss", "p&l", "pnl", "net", "net profit", "yield", "total won"],
    worked: ["Four bets of $10: one returns $25, three lose.",
      "$25 came back and $40 was staked: net −$15.",
      "ROI = −$15 ÷ $40 = −37.5%. \"Total won: $25\" would hide the $30 lost on the other three.",
      "A void bet (every pick pushed or called off) gave its whole stake back and was never at risk, so it counts in neither the staked nor the came-back total."],
    related: ["expected_result", "bankroll"], see: "#/bets?tab=settled", seeLabel: "See your results",
  },
};

/**
 * Every glossary term for Help, sorted by title:
 * {key, title, short, body, example, worked: [step], aliases (search words), aka (other names to show),
 *  group, related: [key], see ("#/route" or null), seeLabel, topic (a guide to link) | null, live?: fn}
 */
export function terms() {
  const all = { ...GLOSSARY, ...NEW_TERMS };
  return Object.entries(all).map(([key, entry]) => {
    const extra = EXTRAS[key] || {};
    return {
      key,
      ...entry,
      title: extra.title || entry.title,
      body: extra.more ? `${entry.body} ${extra.more}` : entry.body,
      group: extra.group || "odds",
      aliases: extra.aliases || [],
      aka: extra.aka || [],
      worked: extra.worked || [],
      related: (extra.related || []).filter((k) => all[k]),
      see: extra.see || entry.see || null,
      seeLabel: extra.seeLabel || "See it live",
      topic: extra.topic || null,
    };
  }).sort((a, b) => a.title.localeCompare(b.title));
}

export function term(key) {
  return terms().find((t) => t.key === key) || null;
}

// --- the guides -----------------------------------------------------------------------------------------

export const TOPICS = [
  { key: "start", title: "Start here", blurb: "How the app works, in 60 seconds", iconName: "book",
    words: "start begin overview how it works tour basics new" },
  { key: "ai", title: "How the AI decides", blurb: "The brain counts, Claude explains", iconName: "bot",
    words: "ai brain claude model analysts trust numbers decide explain how" },
  { key: "rollover", title: "Rollover explained", blurb: "Why long runs almost always fail", iconName: "trend",
    words: "rollover roll over plan days ladder bank half flat classic chance finish" },
  { key: "responsible", title: "Responsible play", blurb: "Limits, breaks and free help", iconName: "lifebuoy",
    words: "responsible gambling limits budget break cool off cooloff reality check help helpline addiction problem stop" },
  { key: "faq", title: "FAQ", blurb: "Odds, grading, credits, privacy", iconName: "help",
    words: "faq questions answers" },
];

export function topic(key) {
  return TOPICS.find((t) => t.key === key) || null;
}

// "Start here": the whole app in five steps
export const START_STEPS = [
  { iconName: "today", title: "Today", href: "#/today", link: "Open Today",
    text: "Each game shows who's likely to win: the AI's chance for the NBA, the bookmakers' chance (margin removed) for other sports. Tap any price to add that pick to your slip." },
  { iconName: "bets", title: "Your slip", href: "#/build", link: "Open your slip",
    text: "The bar above the tabs shows your picks. Open it for the chance it wins as \"1 in X\", what it pays, what the bookmaker keeps and the expected result in money." },
  { iconName: "build", title: "Build", href: "#/build", link: "Open Build",
    text: "Make it singles, a multi, a round robin or a system bet. Ask the AI to analyse it, or let it build one for a stake and a target. The rollover planner shows four ways to play a run of days, real chances first." },
  { iconName: "ask", title: "Ask AI", href: "#/ask", link: "Open Ask AI",
    text: "Ask in your own words, paste a bet slip from any bookmaker (text or a screenshot), or fill in the request bar (sports, stake, what you'd like back, legs, Safe · Risky · Crazy). Code works out every number; Claude explains them." },
  { iconName: "chart", title: "My bets", href: "#/bets", link: "Open My bets",
    text: "Saved bets settle by themselves from the final scores. You see your net profit and ROI, your plans day by day, your calls against the AI and the AI's own graded record." },
];

// "Rollover explained": the blueprint's computed numbers (p = the chance of winning one day at 2.00)
export const ROLLOVER_TABLE = {
  columns: [["5 days", "32×"], ["7 days", "128×"], ["10 days", "1,024×"], ["14 days", "16,384×"]],
  rows: [
    { label: "48%", note: "market, 4% cut", cells: [["1 in 39", "−18.5%"], ["1 in 170", "−24.9%"], ["1 in 1,540", "−33.5%"], ["1 in 29,015", "−43.5%"]] },
    { label: "47%", note: "market, 6% cut", cells: [["1 in 44", "−26.6%"], ["1 in 197", "−35.2%"], ["1 in 1,901", "−46.1%"], ["1 in 38,961", "−57.9%"]] },
    { label: "52%", note: "a real 4% edge", cells: [["1 in 26", "+21.7%"], ["1 in 97", "+31.6%"], ["1 in 692", "+48.0%"], ["1 in 9,461", "+73.2%"]] },
    { label: "55%", note: "a real 10% edge", cells: [["1 in 20", "+61.1%"], ["1 in 66", "+94.9%"], ["1 in 395", "+159%"], ["1 in 4,314", "+280%"]] },
  ],
};

// Survival at p = 0.47: 0.47, 0.47² = 0.221, 0.47³ = 0.104, 0.47⁴ = 0.049, 0.47⁵ = 0.023
export const SURVIVAL = [
  { day: 1, share: 0.47, text: "47%" },
  { day: 2, share: 0.221, text: "22.1%" },
  { day: 3, share: 0.104, text: "10.4%" },
  { day: 4, share: 0.049, text: "4.9%" },
  { day: 5, share: 0.023, text: "2.3%" },
];

// 30-day runs, one bet a day at 2.00, bankroll starting at 1: [chance of a profit, typical (median) end,
// chance of ending at half or less], worked out exactly by engine/rollover.py's outcomes(strategy, 30, 2.0, p),
// the same maths as the rollover planner (pots are 10% of the start), at p = 0.47 (market) and 0.53 (edge)
export const STRATEGIES = [
  { title: "Classic all-in rollover", market: ["≈0%", "0", "≈100%"], edge: ["≈0%", "0", "≈100%"] },
  { title: "Roll, but bank half of each day's profit", market: ["22.1%", "0", "77.9%"], edge: ["28.1%", "0.50", "71.9%"] },
  { title: "Pot and ladder: roll a pot of 10% of the start for at most 2 wins, bank it, start a new pot; stop below half the start", market: ["30.2%", "0.40", "55.6%"], edge: ["53.4%", "1.10", "34.1%"] },
  { title: "Flat 5% of the start each day", market: ["30.4%", "0.90", "9.3%"], edge: ["55.9%", "1.10", "2.4%"] },
];

export const HELPLINES = [
  { where: "US", name: "1-800-GAMBLER (National Council on Problem Gambling)", tel: "18004262537", phone: "1-800-426-2537", url: "https://www.ncpgambling.org/", note: "Call or text, 24/7, free and confidential." },
  { where: "UK", name: "National Gambling Helpline (GamCare)", tel: "08088020133", phone: "0808 8020 133", url: "https://www.gamcare.org.uk/", note: "Phone or live chat, 24/7, free." },
  { where: "UK", name: "BeGambleAware", url: "https://www.begambleaware.org/", note: "Advice, self-assessment and local support." },
  { where: "Anywhere", name: "Gamblers Anonymous", url: "https://gamblersanonymous.org/", note: "Meetings with people who've been there." },
];

export const WARNING_SIGNS = [
  "Chasing losses: betting again to win back what you lost.",
  "Needing bigger stakes to feel the same buzz.",
  "Using money meant for rent, bills or debts, or borrowing to bet.",
  "Hiding how much you bet, or feeling anxious or irritable when you can't.",
  "Long late-night sessions, or starting rollover after rollover.",
  "Raising your limits again and again.",
];

// Questions people ask (q, then the answer as HTML: static text written here, nothing from outside)
export const FAQ = [
  { q: "Is any money bet here?",
    a: "<p>No. The app prices picks and keeps a record of them; saved bets and plans are a record only (plans start in paper mode). If you bet, you do it yourself with a licensed bookmaker.</p>" },
  { q: "Where do the odds come from, and how fresh are they?",
    a: "<p>From The Odds API: several US bookmakers, with the best price for each outcome. To save the monthly allowance they're kept for up to 2 hours (the pill in the header shows how many requests are left). Before anything is saved, the server prices your picks again from the latest odds it has.</p>" },
  { q: "Why are the odds here different from my bookmaker's?",
    a: "<p>The app shows the best price across several bookmakers at the last fetch. Your bookmaker may offer less, and prices move as money and news come in. Always check the price where you bet.</p>" },
  { q: "What's the difference between \"AI 64%\" and \"Mkt 61%\"?",
    a: "<p>\"Mkt\" is the bookmakers' chance with their margin removed; \"AI\" is the brain's own estimate (NBA only for now). See <a href=\"#/help/ai_vs_market\">AI chance vs market chance</a>.</p>" },
  { q: "Why can't two picks from the same game go in one multi?",
    a: "<p>Their results are linked, so multiplying their chances would overstate the chance and understate the bookmaker's cut. Play them as singles. See <a href=\"#/help/same_game\">same-game picks</a>.</p>" },
  { q: "How are my bets settled?",
    a: "<p>From the final scores on ESPN, usually within an hour of the end. A result exactly on the line is a push (the leg counts as 1.00); a game called off is void. A leg with no result long after its start is voided rather than guessed.</p>" },
  { q: "Can I delete a bet?",
    a: "<p>Yes, until its first game starts (My bets → Upcoming → Delete). Bets under way, settled bets and a plan's days stay in your history so the record stays honest.</p>" },
  { q: "Who can see my bets?",
    a: "<p>Only you. A share code lets someone load the same picks (and the stake) into their own slip; it doesn't show your results or your other bets. In a browser your bets belong to that browser's sign-in; inside Telegram they belong to your Telegram account. Use a share code to move a slip between them.</p>" },
  { q: "Can I paste a slip from my bookmaker?",
    a: "<p>Yes. In <a href=\"#/ask?paste=1\">Ask AI</a>, tap 📋 Paste a slip and paste its text or add a screenshot (in Telegram, send it to the bot). The AI copies what it says; the app's code checks the copy, finds each leg among its games and works out the real chance, the bookmaker's cut, better prices and cheaper ways to play it. Player props outside the NBA and unusual markets get a chance estimated from your odds, and say so. Screenshots aren't kept, and nobody else can see your slips. See <a href=\"#/help/paste_slip\">Paste a slip</a>.</p>" },
  { q: "Why does the AI need credits?",
    a: "<p>Each AI chat or written analysis costs money on the server's account, so everyone has a daily allowance (Settings shows what's left). Prices, chances, slips and plans don't use it.</p>" },
  { q: "Can the AI give me a sure bet?",
    a: "<p>No. Nothing in betting is sure, and the AI is built to refuse words like \"lock\" or \"guaranteed\". \"Safe\" is only the name for tickets the AI gives 60% or more, and those still lose about 4 times in 10.</p>" },
  { q: "Why is there an AI model only for the NBA?",
    a: "<p>A model has to be tested on graded games before it deserves trust. Other sports show the market's chance (margin removed), the honest baseline, and say so.</p>" },
  { q: "How do I set limits or take a break?",
    a: "<p>In <a href=\"#/settings?focus=limits\">Settings → Limits</a>: a weekly budget, a reality-check reminder, and breaks from 24 hours to 3 months. A break can't be cut short.</p>" },
];
