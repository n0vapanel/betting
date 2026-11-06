// Help's words: the plain-English guide on Help's home page (SECTIONS: fourteen short sections, each ending with
// CLOSER), the glossary the Help tab shows, and the guides (Start here, How the AI decides, How the brain learns,
// The Builder, Venue & conditions, Rollover explained, Responsible play, FAQ).
//
// No promises anywhere in Help: never "guaranteed", "sure", "lock", "can't lose" or "beat the bookies" (the one
// exception is the FAQ's question "Is this a sure thing?", which answers no). tests/test_help_plain.py checks it.
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
  { key: "markets", title: "Markets say: the blend and its sources" },
  { key: "bets", title: "Bet types and results" },
  { key: "money", title: "Money, staking and plans" },
  { key: "ai", title: "The AI and its record" },
  { key: "builder", title: "The Builder" },
  { key: "context", title: "Venue & conditions" },
  { key: "players", title: "Players and props" },
  { key: "lingo", title: "Bettor talk" },
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
  pra: {
    title: "PRA (points + rebounds + assists)",
    short: "One prop on a player's points, rebounds and assists added together.",
    body: "P+R, P+A and R+A add two of them; stocks are steals + blocks. The Player finder and Ask AI scan them like any other stat: the projection is the parts' projections added up, and the history counts games where the sum reached the number.",
    example: "28 points, 9 rebounds and 6 assists is 43 PRA: over 39.5 wins.",
  },
  l10: {
    title: "L5, L10, L20",
    short: "A player's last 5, 10 or 20 games.",
    body: "\"9/10 L10\" means he reached the number in 9 of his last 10 games: his hit rate over that stretch. The Player finder shows the last 5, the last 10 and this season side by side, so a hot week stands out against the whole season. Ten games is a small sample, which is why the chance comes from the projection, not from the count alone.",
    example: "4/5 L5, 7/10 L10, 25/45 season: 80%, 70% and 56%, hotter lately than over the season.",
  },
  sgp: {
    title: "SGP (same-game parlay)",
    short: "A parlay of picks from one game.",
    body: "The picks are linked (a blowout makes the over likelier), so bookmakers price the link themselves. The app doesn't put two picks from one game in a multi: multiplying linked chances as if they were unrelated would overstate the chance.",
    example: "Thunder −6.5 and Over 224.5 in one ticket is an SGP; here they go in as two singles.",
  },
  attd: {
    title: "ATTD (anytime touchdown)",
    short: "A bet that a player scores a touchdown at any point in the game.",
    body: "Rushing and receiving touchdowns count; a quarterback's passing touchdowns don't. It's a yes-or-no bet, so the Player finder ranks it by the chance of 1 or more.",
    example: "+120 (2.20) with a 45% chance: 0.45 × 2.20 − 1 = −1.0%, a little under break-even.",
  },
  sog: {
    title: "SOG (shots on goal)",
    short: "Hockey shots on target: the ones the goalie saved plus the goals.",
    body: "Blocked shots and misses don't count. A common prop is 3+ SOG (over 2.5).",
    example: "4 shots saved and 1 goal is 5 SOG: over 2.5 wins.",
  },
  ks: {
    title: "Ks (strikeouts)",
    short: "A pitcher's strikeouts in a game.",
    body: "Only games he pitched count in his history. Over 5.5 Ks needs 6 or more, the same as \"6+ strikeouts\" in the Player finder.",
    example: "7 strikeouts in 6 innings: over 5.5 wins, and so does 7+; 8+ loses.",
  },
  tb: {
    title: "TB (total bases)",
    short: "The bases from a batter's hits: a single 1, a double 2, a triple 3, a home run 4.",
    body: "Walks don't count. Over 1.5 TB needs a double or better, or two singles.",
    example: "A single and a home run: 1 + 4 = 5 total bases.",
  },
  alt_line: {
    title: "Alt line",
    short: "The same bet at a different number from the main line, at a different price.",
    body: "A lower number wins more often and pays less; a higher one pays more and wins less often. Milestones like \"20+ points\" are a ladder of alt lines. The Player finder's edge uses only the price for exactly your number.",
    example: "Main over 24.5 at −110 (1.91); alt over 19.5 at −250 (1.40); alt over 29.5 at +200 (3.00).",
  },
  chalk_dog: {
    title: "Chalk and dog",
    short: "Chalk is the favourite; the dog (underdog) is the side not expected to win.",
    body: "Minus money (−150) usually marks the chalk and plus money (+150) the dog. A pick'em is a game priced about even.",
    example: "Celtics −250 (1.40) are the chalk; Hornets +200 (3.00) are the dog.",
  },
  fade_tail: {
    title: "Fade and tail",
    short: "Tail a pick: bet the same. Fade it: bet against it.",
    body: "Ask AI understands both words. Whatever you tail or fade, only you add picks to your slip.",
    example: "Someone likes the Bills −3: tailing is the Bills −3 too, fading is the Dolphins +3.",
  },
  hammer: {
    title: "Hammer and nuke",
    short: "Bettor talk for a pick someone feels strongly about.",
    body: "Ask AI reads slang like this as \"your strongest picks\" and answers with the most likely ones and their chances. No pick is certain, so the app never uses such words about its own picks.",
    example: "\"3 hammers tonight\": the three strongest picks, say at 74%, 71% and 69%.",
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
  markets_say: {
    group: "markets", aka: ["the blend", "consensus"],
    aliases: ["markets", "markets say", "blend", "blended", "consensus", "brain", "weights", "log odds", "opinion pool", "low confidence", "best estimate"],
    worked: ["Three sources price the Celtics, margins removed: Sharp books 64%, US books 62%, Kalshi 63%.",
      "Their starting weights, 50%, 20% and 20%, add up to 90%, so they're shared out as 55.6%, 22.2% and 22.2%.",
      "In log-odds: 0.556 × 0.575 + 0.222 × 0.490 + 0.222 × 0.532 = 0.547, which is a 63.3% chance: the blend shows 63%.",
      "When no market price can be counted (none prices the game, or the ones shown are thin or out of date), ratings alone are averaged, pulled a quarter of the way toward the sport's usual home-win rate (an even split at a neutral site or where there is none) and labelled \"Low confidence\"."],
    related: ["sharp_book", "prediction_market", "disagreement", "thin_market", "market_chance", "brier"], seeLabel: "See \"Markets\" on Today",
  },
  sharp_book: {
    group: "markets", aka: ["sharp bookmaker", "exchange"], aliases: ["sharp", "sharps", "pinnacle", "betfair", "matchbook", "exchange", "anchor", "smart money"],
    worked: ["Pinnacle prices the Celtics at 1.65 and the Knicks at 2.38.",
      "1 ÷ 1.65 = 60.6% and 1 ÷ 2.38 = 42.0%: 102.6% in all, a 2.6% margin.",
      "Divide each by 102.6%: Celtics 59%, Knicks 41%. That's the Sharp chip; the price itself is never offered as one to take."],
    related: ["markets_say", "market_chance", "book_cut"], seeLabel: "See the Sharp chip on a game",
  },
  prediction_market: {
    group: "markets", aka: ["event contract"], aliases: ["kalshi", "polymarket", "prediction market", "exchange", "contract", "bid", "ask", "mid", "crowd"],
    worked: ["Kalshi's Celtics contract: best bid 61¢, best ask 65¢. The mid is 63¢.",
      "The Knicks contract's mid is 39¢: 63 + 39 = 102, so each is scaled to add up to 100%: Celtics 61.8%, Knicks 38.2%.",
      "Within 15 minutes of the start the chip counts half in the blend: prices drift near expiry.",
      "Two games' contracts at 60¢ each don't make a 36¢ parlay: a product of leg prices is never a real parlay price."],
    related: ["markets_say", "thin_market", "disagreement"], seeLabel: "See Kalshi and Polymarket on a game",
  },
  thin_market: {
    group: "markets", aka: ["illiquid market"], aliases: ["thin", "illiquid", "liquidity", "spread", "wide spread", "not counted"],
    worked: ["Bid 3¢, ask 97¢: a 94-cent spread, far over the 6-cent limit.",
      "$22 waiting to trade, under the $1,000 limit.",
      "Either one makes it thin: its chip says \"thin\" and the blend leaves it out."],
    related: ["prediction_market", "markets_say"], seeLabel: "See the chips on a game",
  },
  disagreement: {
    group: "markets", aka: ["⚖ badge", "sources differ"], aliases: ["disagree", "disagreement", "scales", "balance", "differ", "spread", "split", "conflict"],
    worked: ["Sharp 64%, US books 62%, Kalshi 57%, ESPN 60% on the Celtics.",
      "Highest 64%, lowest 57%: 7 points apart. That's 5 or more, so the game shows ⚖ 7.",
      "Only sources counted in the blend take part: a thin or out-of-date one is shown but left out.",
      "Sort Today's board by \"Sources disagree most\" to see these games first."],
    related: ["markets_say", "thin_market", "ai_vs_market"], see: "#/today?sort=dis", seeLabel: "Sort Today by disagreement",
  },
  market_chance: {
    group: "odds", aka: ["no-vig chance", "fair chance"], aliases: ["no vig", "no-vig", "fair chance", "consensus", "market", "mkt"],
    worked: ["Best prices: Thunder 1.50, Lakers 2.70.",
      "Implied chances: 1 ÷ 1.50 = 66.7% and 1 ÷ 2.70 = 37.0%, which add up to 103.7%.",
      "Divide each by 103.7% to take the margin out: Thunder 64%, Lakers 36%.",
      "That's \"Mkt\" next to a price on Today and on a game, and \"Market\" in your slip. \"Markets\" (Markets say) is another number: the blend, which also counts the exchanges, Kalshi, Polymarket and ratings, so the two can differ by a point or two."],
    related: ["implied_probability", "markets_say", "ai_vs_market", "book_cut"], seeLabel: "See \"Mkt\" chances on Today",
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
    related: ["implied_probability", "expected_result", "kelly", "dixon_coles"], see: "#/build/builder", seeLabel: "See fair odds in the Builder",
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
    related: ["line", "projection", "l10", "player_finder"], see: "#/today", seeLabel: "Open a game, then a player",
  },
  projection: {
    group: "ai", aliases: ["projection", "player model", "range", "p(over)"],
    worked: ["Mean 24.1 points, range 15–33 (10% to 90%).",
      "About 1 game in 5 lands outside the range: 1 in 10 below 15, 1 in 10 above 33.",
      "55% over 23.5 means fair odds of 1 ÷ 0.55 = 1.82. A book price of 1.91 is a little above fair; 1.80 a little below."],
    related: ["hit_rate", "line", "fair_odds", "player_finder"], see: "#/today", seeLabel: "Open an NBA game, then a player",
  },
  rollover: {
    group: "money", aliases: ["roll over", "rollover challenge", "2 odds", "daily double", "compounding"],
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
    group: "builder", aka: ["presets"], aliases: ["safe", "risky", "crazy", "risk level", "tier", "preset", "presets", "custom", "value", "for fun"],
    worked: ["Safe: the brain gives the ticket 60% or more, paying 1.50x to 2.00x on 1 to 3 legs. At 62% it still loses about 4 times in 10.",
      "Risky: 25% to 60%, paying 2.00x to 5.00x on 1 to 3 legs. Crazy: 3% to 25%, paying 5.00x to 36x on up to 5 legs: expect to lose most of these.",
      "Tap Safe and change \"Chance at least\" from 60% to 55%: the run is now Custom (from Safe), and \"Back to Safe\" puts every control back."],
    related: ["one_in", "expected_result", "target_odds", "near_miss"], topic: "builder", see: "#/build/builder", seeLabel: "Open the Builder",
  },
  target_odds: {
    group: "builder", aka: ["target", "total odds"], aliases: ["target", "target odds", "win about", "want to win", "total odds", "payout target", "10x", "multiplier"],
    worked: ["$10 stake, win about $100: 100 ÷ 10 = a target of 10.00x.",
      "With ±10%: 10 × 0.90 = 9.00x up to 10 × 1.10 = 11.00x. Three legs at 2.20 each pay 2.20 × 2.20 × 2.20 = 10.65x: inside.",
      "With no edge the chance is roughly (1 − cut) for each leg, multiplied, ÷ the target: at a 4.5% cut a leg, three legs to 10x is 0.955³ ÷ 10 = 0.871 ÷ 10, about 8.7% or 1 in 11.5."],
    related: ["tolerance", "multi", "book_cut", "one_in"], topic: "builder", see: "#/build/builder", seeLabel: "Set a target in the Builder",
  },
  tolerance: {
    group: "builder", aka: ["±%", "window"], aliases: ["tolerance", "plus minus", "window", "range", "how close"],
    worked: ["Target 20x, ±10%: 20 × 0.90 = 18x up to 20 × 1.10 = 22x.",
      "A ticket paying 21.40x fits; one paying 23.10x doesn't.",
      "The chance of reaching the target counts a ticket that pays at least 18x, the window's low end.",
      "At ±0% only exactly 20.00x would fit, which legs almost never multiply to."],
    related: ["target_odds", "multi"], topic: "builder", see: "#/build/builder", seeLabel: "Open the Builder",
  },
  target_chance: {
    group: "builder", aka: ["target chance", "win chance"], aliases: ["chance you want", "target chance", "80% chance", "4 in 5", "likely", "most likely", "win chance", "probability"],
    worked: ["80% with 2 legs: each leg needs about 90%, since 0.90 × 0.90 = 0.81.",
      "Two legs at 1.10 each pay 1.10 × 1.10 = 1.21x: about what an 80% ticket pays, far from 3.5x.",
      "Ask for 3.5x and 80% together and you get the closest options: the likeliest at your odds (near 3.5x, with its real chance), the closest to your chance (about 80%, with its real odds) and the best balance of the two."],
    related: ["closest_effort", "target_odds", "one_in"], topic: "builder", see: "#/build/builder", seeLabel: "Set a chance in the Builder",
  },
  closest_effort: {
    group: "builder", aka: ["best effort", "strict"], aliases: ["closest effort", "best effort", "closest", "strict", "relaxed", "relax", "options", "best balance", "likeliest at your odds", "closest to your odds", "closest to your chance"],
    worked: ["Safe wants each leg of a multi at 72% or more, and you ask for 2 legs near 3.5x.",
      "Two 72% legs win 0.72 × 0.72 ≈ 52% of the time and pay at most about 1 ÷ 0.52 ≈ 1.92x before the bookmaker's cut: nothing lands near 3.5x.",
      "Closest effort lowers the per-leg floor 5 points at a time (67%, 62%, 57%, then 52%) until a pair lands near 3.5x; the ticket says \"Relaxed: per-leg chance lowered from 72% to 52%\".",
      "Strict leaves the run empty and lists the rules that blocked it."],
    related: ["target_chance", "near_miss", "tolerance"], topic: "builder", see: "#/build/builder", seeLabel: "Open the Builder",
  },
  edge: {
    group: "builder", aka: ["value"], aliases: ["edge", "value", "points", "pts", "overlay", "+ev", "min edge"],
    worked: ["At 2.00 the price implies 1 ÷ 2.00 = 50%. Markets say 55%: 55 − 50 = +5.0 points.",
      "At 1.75 the price implies 1 ÷ 1.75 = 57.1%: 55 − 57.1 = −2.1 points.",
      "\"Edge at least −2 pts\" keeps the first and leaves the second out (−2.1 is below −2): it becomes a near miss."],
    related: ["implied_probability", "markets_say", "book_cut", "near_miss"], topic: "builder", see: "#/build/builder", seeLabel: "Open the Builder",
  },
  data_age: {
    group: "builder", aka: ["price age", "as of"], aliases: ["age", "old price", "stale", "fresh", "data age", "price age", "updated", "as of", "max price age"],
    worked: ["The NBA prices were saved at 2:00 PM; at 5:40 PM they are 3 h 40 min old.",
      "With \"Oldest price allowed: 3 h\" every NBA leg is left out and named as a near miss.",
      "Choose 6 h to use them, then check each price at your bookmaker before you bet: it may have moved."],
    related: ["near_miss", "quota"], topic: "builder", see: "#/build/builder", seeLabel: "Open the Builder",
  },
  near_miss: {
    group: "builder", aka: ["near misses"], aliases: ["near miss", "near misses", "left out", "rejected", "why not", "blocked", "rules"],
    worked: ["Your rules: each leg at least 55%, prices up to 3 h old.",
      "Knicks to win @ 2.30 with a 44% chance fails \"chance at least 55%\": a near miss, \"chance 44% below your minimum 55%\".",
      "Lower the minimum to 40% and it can join a ticket, with a lower chance of winning.",
      "When no ticket fits at all, the Builder says which rules ruled out how many legs, e.g. \"only 4 games pass 'min chance 55%'\"."],
    related: ["data_age", "edge", "tolerance"], topic: "builder", see: "#/build/builder", seeLabel: "Open the Builder",
  },
  dixon_coles: {
    group: "builder", aka: ["goals model", "Poisson"], aliases: ["dixon coles", "dixon-coles", "poisson", "btts", "both teams to score", "draw no bet", "dnb", "double chance", "team totals", "correct score", "soccer", "goals"],
    worked: ["The market says home 52%, draw 25%, away 23%, and 50% for over 2.5 goals.",
      "The goal rates that fit those best: 1.69 expected goals at home, 1.02 away.",
      "Every score gets a chance from those rates, with the low-score correction; the scores where both teams score add up to about 54%, plus the 1.5-point nudge: about 55%.",
      "Fair odds are 1 ÷ 0.55, about 1.81. A bookmaker paying less than that is charging its margin."],
    related: ["fair_odds", "market_chance", "moneyline"], topic: "builder", see: "#/today", seeLabel: "See it on a soccer game",
  },
  explains_vs_moves: {
    group: "context", aka: ["attribution", "adjustment", "why the line is here", "what moves it"],
    aliases: ["explains", "moves", "attribution", "adjustment", "adjusted", "adj", "waterfall", "why the line is here", "what moves it", "weather", "wind", "forecast", "roof", "venue", "conditions", "context"],
    worked: ["An NFL total of 44.5 was priced when the forecast said 8 mph of wind and −2°C: the weather then explained about −1.0 point of it.",
      "Two hours later the forecast says 19 mph: the weather now explains about −6.8 points. The change since the odds is −6.8 − (−1.0) = −5.8.",
      "Half of it counts (weight 0.5): −2.9. The NFL total's cap is ±2.0, so −2.0 is applied.",
      "The over's chance moves from 50% to about 44%, and every totals leg says why. The margin, home field and the rest of \"Why the line is here\" move nothing."],
    related: ["markets_say", "market_chance", "total"], topic: "conditions", see: "#/today", seeLabel: "See it on a game",
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
    worked: ["On the odds feed's free plan that's 500 requests a month, and each league fetch costs 3.",
      "Odds are kept for 2 hours, so keeping one league fresh all day costs 12 fetches × 3 = 36 requests.",
      "The month's requests, less a reserve of 50, are shared out day by day: on the free plan (500 − 50) ÷ 30 = 15 requests, about 5 leagues a day.",
      "With fewer than 50 left the app shows the last saved odds instead of fetching."],
    related: ["market_chance"], see: "#/today", seeLabel: "See how fresh the odds are on Today",
  },
  log_loss: {
    group: "ai", aka: ["log score", "ignorance score"], aliases: ["logloss", "log-loss", "cross entropy", "log score", "ignorance", "surprise"],
    worked: ["Saying 70% for a team that wins: −ln(0.70) = 0.357.",
      "Saying 70% for a team that loses: −ln(0.30) = 1.204, more than three times as costly.",
      "Always saying 50% scores −ln(0.50) = 0.693 every game.",
      "A gap of +0.0004 against the market is 0.4 more per 1,000 games: well inside the week-to-week noise."],
    related: ["calibration", "brier"], see: "#/bets?tab=brain", seeLabel: "See the brain's progress", topic: "brain",
  },
  calibration: {
    group: "ai", aka: ["reliability"], aliases: ["reliability diagram", "reliability curve", "calibration slope", "overconfident", "underconfident", "pit", "pit histogram"],
    worked: ["The brain gave 200 games a 70% chance and 132 of them happened: 66%.",
      "With 200 games a perfectly calibrated brain lands within about 70% ± 6.5% (two standard errors of 3.2%), so 66% is no sign of a problem.",
      "A calibration slope of 0.9 says its chances sit a little too far from 50% (overconfident); 1.1 says a little too close (too timid)."],
    related: ["log_loss", "brier"], see: "#/bets?tab=brain", seeLabel: "See the brain's calibration", topic: "brain",
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
  player_finder: {
    group: "players", aka: ["player scan", "prop finder"], aliases: ["player finder", "finder", "player scan", "scan", "props", "prop finder", "20+", "projected", "projection", "who will score", "milestone", "ladder", "hit rate", "recent form", "edge", "who's hot", "who's cold", "best props", "best edges"],
    worked: ["You pick NBA, points, at least 20, tonight.",
      "A player went 20+ in 9 of his last 10, is projected for 31.2 and has an 82% chance: he ranks near the top.",
      "His 20+ milestone is posted at −250 (1.40): edge = 0.82 × 1.40 − 1 = +14.8%.",
      "Another is projected for 18.5 with a 44% chance: under 50% and under 20, so he's \"Also close\"."],
    related: ["l10", "hit_rate", "projection", "pra", "alt_line"], see: "#/players", seeLabel: "Open the Player finder",
  },
  pra: {
    group: "players", aka: ["P+R", "P+A", "R+A", "stocks"], aliases: ["pra", "pts+reb+ast", "p+r+a", "p+r", "p+a", "r+a", "combo", "combos", "stocks", "stl+blk", "points rebounds assists"],
    worked: ["28 points + 9 rebounds + 6 assists = 43 PRA.", "Over 39.5 PRA needs 40 or more: 43 wins.", "Projected 27.1 + 8.4 + 5.9 = 41.4 PRA."],
    related: ["player_finder", "line", "projection"], see: "#/players?league=basketball_nba&stat=pra&at=line&day=today", seeLabel: "PRA in the Player finder",
  },
  l10: {
    group: "players", aka: ["last 10", "last 5", "last 20"], aliases: ["l5", "l10", "l20", "last 5", "last 10", "last 20", "form", "hot", "cold", "trend", "9/10"],
    worked: ["Last 10 games at 20+ points: 24, 31, 18, 22, 27, 20, 25, 19, 30, 21.", "Eight of them are 20 or more: \"8/10 L10\".",
      "The last 5 (24, 31, 18, 22, 27) have four: \"4/5 L5\"."],
    related: ["hit_rate", "player_finder", "projection"], see: "#/players", seeLabel: "Open the Player finder",
  },
  sgp: {
    group: "lingo", aka: ["same-game parlay", "same game multi"], aliases: ["sgp", "same game parlay", "same-game parlay", "same game multi", "bet builder"],
    worked: ["Thunder −6.5 at 1.91 and Over 224.5 at 1.91.", "Multiplied as if unrelated: 1.91 × 1.91 = 3.65.",
      "But a Thunder blowout makes the over likelier, so the real chance of both differs: the app plays them as singles."],
    related: ["same_game", "multi"], see: "#/build", seeLabel: "Open your slip",
  },
  attd: {
    group: "lingo", aka: ["anytime TD", "anytime touchdown scorer"], aliases: ["attd", "anytime td", "anytime touchdown", "td scorer", "touchdown scorer", "scores a td"],
    worked: ["+120 is 1 + 120 ÷ 100 = 2.20 in decimal.", "A 45% chance: 0.45 × 2.20 = 0.99 back for each $1.",
      "Edge = 0.99 − 1 = −1.0%: a little under break-even."],
    related: ["player_finder", "implied_probability"], see: "#/players?league=americanfootball_nfl&stat=anytime_td&at=1&day=today", seeLabel: "Anytime TD in the Player finder",
  },
  sog: {
    group: "lingo", aka: ["shots on goal", "shots"], aliases: ["sog", "shots on goal", "shots", "shots on net"],
    worked: ["4 shots saved + 1 goal = 5 shots on goal.", "A blocked shot and a miss add nothing.", "Over 2.5 SOG (3+) wins."],
    related: ["player_finder", "line"], see: "#/players?league=icehockey_nhl&stat=sog&at=3&day=today", seeLabel: "SOG in the Player finder",
  },
  ks: {
    group: "lingo", aka: ["strikeouts", "pitcher strikeouts"], aliases: ["ks", "k", "strikeouts", "pitcher strikeouts", "punchouts"],
    worked: ["Over 5.5 Ks is the same as 6+ strikeouts.", "7 strikeouts: over 5.5 wins, 7+ wins, 8+ loses."],
    related: ["player_finder", "line"], see: "#/players?league=baseball_mlb&stat=k&at=6&day=today", seeLabel: "Ks in the Player finder",
  },
  tb: {
    group: "lingo", aka: ["total bases"], aliases: ["tb", "total bases", "bases"],
    worked: ["Single 1, double 2, triple 3, home run 4.", "A single and a home run: 1 + 4 = 5 TB.", "Over 1.5 TB needs 2 or more: a double alone does it, a single alone doesn't."],
    related: ["player_finder", "line"], see: "#/players?league=baseball_mlb&stat=tb&at=2&day=today", seeLabel: "TB in the Player finder",
  },
  alt_line: {
    group: "lingo", aka: ["alternate line", "ladder", "milestone"], aliases: ["alt line", "alt", "alternate", "ladder", "milestone", "alt lines"],
    worked: ["−110 is 1 + 100 ÷ 110 = 1.91; −250 is 1 + 100 ÷ 250 = 1.40; +200 is 3.00.",
      "Over 19.5 at 1.40 implies 1 ÷ 1.40 = 71%; over 29.5 at 3.00 implies 33%.",
      "The lower rung wins more often and pays less."],
    related: ["line", "implied_probability", "player_finder"], see: "#/players", seeLabel: "Open the Player finder",
  },
  chalk_dog: {
    group: "lingo", aka: ["favourite and underdog", "favorite", "underdog", "pick'em"], aliases: ["chalk", "dog", "underdog", "favorite", "favourite", "plus money", "pick em", "pickem", "pick'em"],
    worked: ["Celtics −250 is 1.40: they imply 1 ÷ 1.40 = 71%.", "Hornets +200 is 3.00: they imply 1 ÷ 3.00 = 33%.",
      "71.4% + 33.3% = 104.8%: the extra 4.8% is the bookmaker's cut."],
    related: ["american_odds", "moneyline", "book_cut"], see: "#/today", seeLabel: "See tonight's prices",
  },
  fade_tail: {
    group: "lingo", aka: ["fade", "tail"], aliases: ["fade", "tail", "tailing", "fading", "follow"],
    worked: ["A friend's pick: Bills −3.", "Tail it: Bills −3. Fade it: Dolphins +3.", "Either way, only you add it to your slip."],
    related: ["spread", "ai_vs_market"], see: "#/ask", seeLabel: "Ask AI",
  },
  hammer: {
    group: "lingo", aka: ["nuke", "best bet"], aliases: ["hammer", "hammers", "nuke", "nukes", "best bet", "strongest", "slam"],
    worked: ["\"Give me 3 hammers\" asks for the three strongest picks.", "Ask AI ranks the most likely winners: say 74%, 71% and 69%.",
      "Three such picks together win about 0.74 × 0.71 × 0.69 ≈ 36% of the time."],
    related: ["one_in", "multi", "ai_chance"], see: "#/ask", seeLabel: "Ask AI",
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
  { key: "brain", title: "How the brain learns", blurb: "It tests itself, changes slowly and shows why", iconName: "trend",
    words: "brain learn learning learns improve improving better backtest back test tune tuning gate gates promote promoted rollback rolled back undo pause paused experiment shadow calibration log loss market ai proposal proposals claude deepseek cost cap progress held" },
  { key: "builder", title: "The Builder", blurb: "Presets, every control, closest effort and why each leg", iconName: "build",
    words: "builder build presets safe risky crazy custom controls target odds chance you want tolerance legs markets near miss swap why tickets closest effort best effort strict options best balance relaxed" },
  { key: "conditions", title: "Venue & conditions", blurb: "Weather, travel and venues: what explains the line", iconName: "stadium",
    words: "venue stadium arena park weather wind rain cold heat temperature altitude elevation roof dome surface capacity park factors travel rest back-to-back b2b time zone jet lag conditions adjustment adjusted explains moves waterfall units metric imperial open-meteo" },
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
    text: "Each game shows who's likely to win: \"Markets say\", one chance blended from the bookmakers, exchanges, prediction markets and ratings, plus the AI's own chance for the NBA. ⚖ marks games whose sources disagree. Tap any price to add that pick to your slip." },
  { iconName: "bets", title: "Your slip", href: "#/build", link: "Open your slip",
    text: "The bar above the tabs shows your picks. Open it for the chance it wins as \"1 in X\", what it pays, what the bookmaker keeps and the expected result in money." },
  { iconName: "build", title: "Build", href: "#/build", link: "Open Build",
    text: "Make it singles, a multi, a round robin or a system bet and ask the AI to analyse it. Or open the Builder: Safe, Risky and Crazy fill every control (sports, legs, target odds, markets) and you can change any of them. The rollover planner shows four ways to play a run of days, real chances first." },
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

// "Venue & conditions": the caps on what an adjustment can move, and the starting effect sizes in plain words
// (engine/context.py's constants, with their sources). Design choices and starting points, refitted from the
// context log once it has enough graded games.
export const CONTEXT_CAPS = [
  ["NFL and college football total", "±2.0 points"],
  ["NFL and college football side", "±1.0 point"],
  ["MLB total", "±0.4 runs"],
  ["NBA, WNBA and college basketball total", "±1.5 points"],
  ["NHL total", "±0.15 goals"],
  ["Soccer total", "±0.15 goals a team (±0.30 for the match)"],
  ["Win chance", "±2.5 points"],
];

export const CONTEXT_EFFECTS = [
  { what: "Home field", size: "This season's home win rate from ESPN's standings, pulled toward the usual rate with 400 games' weight (NFL 53.9%, NBA 55.4%, NHL 52.2%, MLB 52.9%, WNBA 55%, college football 57%, college basketball 60%), turned into points through how widely results spread. Soccer: home 45% and draw 27% to start, as goal difference.", source: "ESPN standings, home records", certainty: "Normal" },
  { what: "Altitude", size: "NBA in Denver: about +2.6 points (Denver's home edge is 8.6 points per 100 possessions against 6.0 elsewhere); Utah scaled by its height. Soccer: about 0.25 goals per 1,000 m of height difference above 1,500 m (half a study's slope, as team quality muddies it). NFL in Denver: +0.5 points.", source: "Sportico; BMJ 2007 (1,460 South American matches)", certainty: "Normal for the NBA in Denver; small, noisy otherwise" },
  { what: "No rest", size: "NBA back-to-back: −1.25 points. NHL team on no rest against a rested one: −5.8 points of win chance (45.0% against 50.8%).", source: "The app's NBA model; nhlscraper's back-to-back study (NHL)", certainty: "Normal" },
  { what: "Flying east", size: "Two time zones or more east within 3 days: MLB home team −3.5 points of win chance; NBA −1.3 points.", source: "PNAS 2017 (MLB); Leota et al. (NBA)", certainty: "Small, noisy, untested: no study checks whether closing prices already hold it" },
  { what: "NFL wind", size: "Nothing at 8 mph or less, −2.2 points at 12 mph, −3.8 points at 16 mph or more, in a straight line between.", source: "NFL games since 2000 (CBS Sports): 44.3 points at 0–9 mph, 42.1 at 10–14, 40.5 at 15+", certainty: "Normal" },
  { what: "NFL cold", size: "−1.0 point at 32°F (0°C) or below; with 15 mph of wind or more, 2.0 points more.", source: "NFL games since 2000 (CBS Sports)", certainty: "Normal" },
  { what: "NFL rain", size: "−1.0 point with a 60% chance or more and 0.5 mm an hour or more at kick-off.", source: "Sharp Football Analysis (unders 57.4% in measurable rain)", certainty: "Small, noisy" },
  { what: "MLB temperature", size: "+0.15 runs per 10°F above 70°F (21°C), and less below it: fly balls carry about 3 feet further per 10°F.", source: "Smithsonian", certainty: "Small, noisy" },
  { what: "MLB wind", size: "±0.08 runs per mph of wind blowing out (+) or in (−) above 5 mph, from the park's direction and the forecast, or MLB's own \"Out To RF\". At most ±1 run.", source: "FanGraphs (Wrigley Field: 6.0 runs a game blowing out, 4.1 blowing in)", certainty: "Small, noisy" },
  { what: "MLB park", size: "Baseball Savant's 3-year runs index: (index − 100) ÷ 100 × the league's average total (Coors Field high, T-Mobile Park low). Home runs are shown, not added again.", source: "Baseball Savant park factors", certainty: "Normal" },
  { what: "Soccer weather", size: "Total goals ×1.068 above 22°C, ×0.97 below 6°C, ×1.014 in rain.", source: "Mišák 2025", certainty: "Small, noisy" },
  { what: "Dome or closed roof", size: "The weather counts for nothing, and the panel says \"Roof closed\".", source: "", certainty: "" },
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
    a: "<p>From The Odds API: several US bookmakers, with the best price for each outcome, plus Pinnacle and two exchanges that only feed the \"Markets say\" blend (their prices are never offered to you). To save the monthly allowance a league's odds are loaded again only once they're over 2 hours old, when someone opens the league or ahead of games in the next 18 hours, and only while the day's share of the allowance lasts, so on a busy day they can be older (the pill in the header shows how many requests are left). Before anything is saved, the server prices your picks again from the latest odds it has.</p>" },
  { q: "What does \"Markets say\" mean?",
    a: "<p>One chance per outcome, blended from every source that prices the game: sharp bookmakers and exchanges, the US bookmakers, Kalshi and Polymarket, ESPN's predictor and ratings, and the app's own model. It's the app's best estimate, and the market usually wins, so it aims to match the market and explain it rather than beat it. Each game page shows every source's chip; ⚖ means they disagree by 5 points or more. See <a href=\"#/help/markets_say\">Markets say</a> and the record of each source under My bets → AI record.</p>" },
  { q: "Why are the odds here different from my bookmaker's?",
    a: "<p>The app shows the best price across several bookmakers at the last fetch. Your bookmaker may offer less, and prices move as money and news come in. Always check the price where you bet.</p>" },
  { q: "What's the difference between \"AI 64%\" and \"Mkt 61%\"?",
    a: "<p>\"Mkt\" is the bookmakers' chance with their margin removed; \"AI\" is the brain's own estimate (NBA only for now). See <a href=\"#/help/ai_vs_market\">AI chance vs market chance</a>.</p>" },
  { q: "Why can't two picks from the same game go in one multi?",
    a: "<p>Their results are linked, so multiplying their chances would overstate the chance and understate the bookmaker's cut. Play them as singles. See <a href=\"#/help/same_game\">same-game picks</a>.</p>" },
  { q: "How does the Builder choose legs?",
    a: "<p>From what's already saved: today's games, each league's latest bookmaker prices and the \"Markets say\" blend. It fetches nothing for a build. Each leg's chance is the blend (else the bookmakers' margin-free chance), prices come only from bookmakers you can bet at, and every leg must pass your rules; the ones that nearly made it are listed as near misses with the rule they failed. Tickets are ranked by closeness to your target, a cautious expected return and the chance of winning, and up to five that share no leg are priced at your stake. When nothing fits exactly, Closest effort (on unless you pick Strict) builds the closest tickets and says what it relaxed. See <a href=\"#/help/builder\">The Builder</a>.</p>" },
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
  { q: "Can the AI promise me a winner?",
    a: "<p>No. It gives you its strongest picks with their real chances, and no outcome is certain, so it never promises a win. \"Safe\" is only the name for tickets the AI gives 60% or more, and those can still lose up to 4 times in 10.</p>" },
  { q: "What if I ask for odds and a chance no ticket can have?",
    a: "<p>You still get tickets. Ask for, say, 2 legs at about 3.5x with an 80% chance and the Builder (or Ask AI) builds the closest options: the likeliest at your odds, the closest to your chance and the best balance, each with its real odds and chance, like \"3.52x · 27% (you asked 3.5x · 80%)\". Pick Strict in the Builder if you'd rather see nothing than something close. See <a href=\"#/help/closest_effort\">Closest effort</a>.</p>" },
  { q: "Why is there an AI model only for the NBA?",
    a: "<p>A model has to be tested on graded games before it deserves trust. Other sports show the market's chance (margin removed), the honest baseline, and say so.</p>" },
  { q: "How do I set limits or take a break?",
    a: "<p>In <a href=\"#/settings?focus=limits\">Settings → Limits</a>: a weekly budget, a reality-check reminder, and breaks from 24 hours to 3 months. A break can't be cut short.</p>" },
  { q: "How do I find players projected for 20+ points tonight?",
    a: "<p>Open the <a href=\"#/players\">Player finder</a> (Build → Find players, or Find a player on Today), or just ask Ask AI in your own words: \"who's going for 10+ boards tonight?\", \"anytime TD scorers\", \"PRA overs\". Both rank each player by the chance of reaching the number, with his last 5, last 10 and season record at it, tonight's projection and the posted line. It's free: no AI allowance on the Player finder, and no odds requests either way. See <a href=\"#/help/player_finder\">Player finder</a>.</p>" },
];

// --- the plain-English guide: Help's home page ------------------------------------------------------------------
// Fourteen short sections in a fixed order for someone who has never bet or used the app, each with an example and
// ending with CLOSER. Every statement says what the app does now (checked against the code); the numbers are worked
// out by hand. Each section is plain text, and help.js escapes every string:
//   {key, title, blurb, words (more search words), parts, example, links: [[href, label]]}
//   parts: {h} a subheading · {p} a paragraph · {list: [text]} bullets · {terms: [[name, words, href?]]} names and
//   what they mean · {qa: [[question, answer]]} questions · {helplines: true} the free helplines (HELPLINES)
// #/help?s=<key> opens one section; a search finds a section by its title, words, blurb and text.

export const CLOSER = "Predictions are estimates and can be wrong. Only bet what you can afford to lose.";

export const SECTIONS = [
  {
    key: "start", title: "Getting started", blurb: "What the app is, what it isn't, and how to open it",
    words: "start begin new overview about what is this app bookmaker bets money adults adult 18 21 age telegram bot mini app browser password sign in open tabs settings",
    parts: [
      { p: "SportsBet AI looks at sports games and works out how likely each result is. It puts the bookmakers' prices next to those chances, so you can see what a bet really costs before you decide anything." },
      { p: "It is not a bookmaker. It never takes bets or money, and nothing you do here places a bet. If you choose to bet, you do that yourself with a licensed bookmaker." },
      { p: "It's for adults only. Before any prices or picks show, you confirm you're 18 or older (21 or older in the US) and that betting is legal where you live." },
      { h: "How to open it" },
      { terms: [
        ["In Telegram", "Open the bot and send /start. Tap 📱 Open app to use the full app inside Telegram. The bot's own buttons work too: games, AI bets, Ask AI, your calls and the leaderboard."],
        ["In a browser", "Go to the web address the person running the app gave you. If it asks for a password, they can give you that as well."],
      ] },
      { h: "Finding your way" },
      { terms: [
        ["Today", "Every game, with its chances and prices."],
        ["Build", "Your slip, the Builder, the Player finder and the rollover planner."],
        ["Ask AI", "Questions in your own words, and slips you paste."],
        ["My bets", "Your saved bets, plans and calls, the leaderboard and the AI's own record."],
        ["Help", "This page."],
        ["Settings", "The button with your initial (or a person) at the top: your name, odds format, sports and limits."],
      ] },
      { p: "Tap ⓘ next to a word on any screen for a short explanation." },
    ],
    example: "The app shows a team at 1.50 and says it wins about 64 times in 100. Nothing has been bet: that only happens if you go to a bookmaker yourself.",
    links: [["#/today", "Open Today"], ["#/help/start", "The 60-second tour"]],
  },
  {
    key: "odds", title: "Reading odds and chances", blurb: "Decimal and American odds, “1 in N”, the bookmaker's cut, fair chances and edge",
    words: "odds decimal american price prices plus minus 1 in n one in chance chances probability percent implied margin cut vig juice overround bookmaker keeps fair chance market chance markets say blend edge value expected",
    parts: [
      { h: "Odds" },
      { terms: [
        ["Decimal odds (2.50)", "What each $1 returns if the bet wins, your $1 included. $10 at 2.50 returns $25: your $10 back plus $15 profit. If it loses, nothing comes back."],
        ["American odds (+150 or −150)", "+150 means a $100 bet wins $150 profit. −150 means you stake $150 to win $100. +150 is the same price as 2.50, and −150 is the same as 1.67. Choose either one in Settings → Display."],
      ] },
      { h: "Chances" },
      { terms: [
        ["A chance (64%)", "How often a result is expected to happen: 64 times in 100 such games."],
        ["“1 in N”", "Small chances are written as a count, because that's easier to feel. 1 in 8 is 12.5%: the bet loses the other 7 times."],
        ["The price's chance", "1 ÷ the decimal odds. 2.00 means 50%, 1.91 means 52.4% and 1.67 means about 60%."],
      ] },
      { h: "The bookmaker's cut (margin)" },
      { p: "Add up the price's chance for every side of a game and you get more than 100%. The extra is the bookmaker's cut, also called the margin or vig. At 1.91 on both sides it's 52.4% + 52.4%, about 105% in all. On that bet the bookmaker keeps about 4.5% of what's staked, on average. The app shows this as “Bookmaker keeps”." },
      { h: "A fair chance" },
      { p: "Take the cut out and the sides add up to 100% again. That's the fair chance, which the app calls the market chance (“Market”). At 1.91 on both sides, each side's fair chance is 50%." },
      { p: "“Markets say” is the app's own chance: it blends the fair chances from many bookmakers, betting exchanges, prediction markets and ratings into one number for each result." },
      { h: "Edge" },
      { p: "Edge compares a chance with the chance the price implies. Positive means the price pays more than the chance says it should; negative means it pays less. Most prices have a small negative edge: that's the cut. A big positive edge is more often an old price, or news the prices haven't caught up with, than a gift." },
    ],
    example: "Edge, worked out: the app gives a team 55% and a bookmaker offers 2.00. The price's chance is 1 ÷ 2.00 = 50%, so the edge is 55 − 50 = +5 points (the Builder shows “+5.0 pts”). Another way to see it: on average $1 at 2.00 returns 0.55 × $2.00 = $1.10, which is 10% more than it costs (the Player finder shows “+10.0%”).",
    links: [["#/help/decimal_odds", "Decimal odds"], ["#/help/american_odds", "American odds"], ["#/help/book_cut", "What the bookmaker keeps"], ["#/help/edge", "Edge"], ["#/help/markets_say", "Markets say"]],
  },
  {
    key: "today", title: "Today", blurb: "The board of games: leagues, filters, live scores and missing prices",
    words: "today board games slate schedule leagues league sports filters filter sort search team date day live score scores clock final postponed price prices missing odds load loading allowance requests left mkt",
    parts: [
      { p: "Today lists every game the app can see, in every league, for today and the next six days. Every sport shows unless you pick some." },
      { terms: [
        ["Leagues", "Pick a sport or a league to see only those. Sports you pin in Settings come first; pinning never hides the others."],
        ["Filters", "The date strip (with each day's number of games), All · Upcoming · Live · Finished, a team search and the sort: time, league, the biggest AI vs market gap, the sources that disagree most, or the markets' or the AI's favourite."],
        ["Live scores", "Games under way show LIVE with the clock and the score. While games are live the board checks for new scores every 30 seconds."],
        ["Prices", "The best price for each side, with its chance: “AI 64%” on NBA games, otherwise “Mkt 61%”, the market's fair chance. Tap a price to add that pick to your slip, and tap it again to take it off."],
        ["Markets", "The “Markets say” chance for each side. ⚖ means the sources disagree by 5 points or more: a reason to look closer, not a signal to bet."],
        ["AI and Edge", "On NBA games: the AI's own chance, and how far it is from the market's, in points."],
        ["Context", "Small badges for wind, cold, heat, rain, a closed roof, altitude, travel and rest. “A game's page” below says what each one means."],
        ["Odds as of", "How old the prices are, like “12 min ago”."],
      ] },
      { h: "When a price is missing" },
      { terms: [
        ["Odds load when you open a league", "Tap it. To save the app's monthly allowance of odds requests, prices load for the league you open."],
        ["Loading odds…", "They're on their way. If they haven't come after 2 minutes, it says “The odds didn't arrive: try again later”."],
        ["Odds are saved for later this month", "Today's share of the allowance is used up, or the month's last requests are being kept in reserve."],
        ["No odds market for this game", "The bookmakers don't list it right now."],
        ["No prices for this game yet", "The league's prices are fresh, but this game has none yet."],
        ["In play, over or postponed", "“In play: no prices here”, “No prices: the game is over” and “No prices: postponed”: prices are only for games that haven't started."],
      ] },
      { p: "The pill at the top (“N odds left”) shows how many odds requests are left this month." },
    ],
    example: "On a busy Sunday you open the NFL. Its rows say “Loading odds…” for a moment, then the prices appear, each with its chance, like 1.80 and “Mkt 54%”.",
    links: [["#/today", "Open Today"], ["#/help/disagreement", "Disagreement (⚖)"], ["#/help/quota", "Odds requests left"]],
  },
  {
    key: "game", title: "A game's page", blurb: "Suggested picks, props, prices on their way and the context badges",
    words: "game page match suggested picks pick brain market why add to slip in your slip props prop player props likeliest best edges prices on their way allowance reserve out of date started context badges weather wind cold hot rain roof altitude travel time zones rest b2b back-to-back adj adjusted injuries",
    parts: [
      { p: "Tap a game to open it. You'll see each result's chance, every bookmaker's prices, injuries, the players and, near the top, Suggested picks." },
      { h: "Suggested picks" },
      { p: "For each market (who wins, the spread, the total, and any soccer market a bookmaker prices), the app shows the pick it thinks likeliest:" },
      { terms: [
        ["Brain 64% · Market 58%", "Brain is the app's own chance for that pick. For who wins, it's the “Markets say” blend; for a spread or a total it's the bookmakers' fair chance, adjusted when a weather forecast or a roof change moves it. Market is the bookmakers' chance with their cut taken out."],
        ["Brain higher than market", "A plain note when the brain's chance is at least 3 points above the market's. It's not a tip."],
        ["The price", "The best price you can actually bet, and the bookmaker offering it. Spreads and totals use the main line only."],
        ["Why?", "Opens up to three short reasons behind the chance."],
        ["Add to slip", "Puts the pick on your slip. It then says “In your slip”, and a second tap takes it off. The slip takes moneylines, spreads and totals, so other soccer markets show their price with the button off."],
      ] },
      { p: "Under the picks, always: “No pick is certain. Only you add bets to your slip.”" },
      { h: "Props" },
      { p: "A prop is a bet on one player, like over 24.5 points. For games today or tomorrow in the leagues the Player finder covers, the page lists the likeliest props and the best edges (up to 3 of each), with “See all in the Player finder”." },
      { h: "When prices haven't arrived" },
      { terms: [
        ["Prices for this game are on their way…", "The page asks again by itself after about 15 and 45 seconds."],
        ["Today's odds allowance is used up", "The app has a set number of odds requests each month and shares them out day by day."],
        ["The month's odds allowance is at its reserve", "The last few requests of the month are kept back."],
        ["The bookmakers don't list these games right now", "There's nothing to price yet."],
        ["This game's prices are out of date", "They're too old to suggest a pick from."],
        ["This game has started", "Picks are for games still to come."],
      ] },
      { h: "Context badges" },
      { p: "Up to 3 small badges a game, the most important first:" },
      { terms: [
        ["Wind", "Strong wind at the start of an open-air game (24 km/h, about 15 mph, or more)."],
        ["Cold or hot", "Freezing (0 °C, 32 °F, or below) or hot (30 °C, 86 °F, or above), in the open air."],
        ["Rain", "Rain is likely at the start (a 60% chance or more)."],
        ["Roof", "The roof is closed, so the weather doesn't count."],
        ["Altitude", "The venue is 1,500 m (about 4,900 ft) up or higher."],
        ["+2h", "A team flew 2 or more time zones east in the last 3 days."],
        ["3d", "An unusual rest gap, in days off, such as an NFL game on a Thursday."],
        ["Adj", "A forecast or roof change after the prices were fetched moved one of the app's numbers."],
        ["B2B", "On Today: the team played yesterday (back-to-back)."],
      ] },
    ],
    example: "“Under 44.5 · Brain 53% · Market 51%” means the app gives the under 53% and the bookmakers' fair chance is 51%. The gap is only 2 points, so there's no “Brain higher than market” note.",
    links: [["#/help/conditions", "Venue & conditions"], ["#/help/markets_say", "Markets say"]],
  },
  {
    key: "players", title: "Player finder", blurb: "Players for a stat and a number: chances, lines, hot and cold",
    words: "player finder players props prop stat stats points rebounds assists threes shots goals strikeouts touchdowns at least fewer than under over line lines posted chance projection l5 l10 season hit rate hits hot cold trend form best edges edge hidden old line no line name names last first last, first comma commas qualifies",
    parts: [
      { p: "Open it from Build (Find players) or from Today (Find a player). Pick a league, a stat, a number and today or tomorrow. The finder ranks players by their chance of reaching that number." },
      { p: "It covers the NBA, WNBA, NFL, NHL, MLB and 15 soccer leagues." },
      { terms: [
        ["At least, or Fewer than", "“At least 20” means 20 or more. “Fewer than” looks at the under instead. You can also use each player's posted line."],
        ["Chance", "How likely the player is to get there, with a small bar."],
        ["Projection", "The number the player is expected to get. “Recent form” or “NBA model” says where it comes from."],
        ["L5, L10 and season", "How often the player got there in the last 5 games, the last 10 and this season, like 7/10."],
        ["Line", "A bookmaker's posted line and price, like “DraftKings 20+ −250”. “No line posted” means there isn't one. “Old line” means the price may have moved, so it shows no edge."],
        ["Edge", "What $1 at that price returns on average by the finder's chance, minus the $1. “+4.2%” means about $1.04 back for each $1."],
        ["Edge hidden", "An edge over +50% usually means a stale or wrong price rather than a gift, so the finder hides the number and never ranks by it."],
        ["Hot and Cold", "Rank players by how far their last 5 games are above (Hot) or below (Cold) their average over the rest of the season. Early in a season they compare with last season, and the chip says so."],
        ["Best edges (all stats)", "Ranks every stat's posted lines by edge."],
      ] },
      { h: "Typing names" },
      { p: "Type one name, or up to 5 separated by commas: “LeBron James, Stephen Curry”. Names written “Last, First” work too: “James, LeBron” finds LeBron James, and “Porter Jr., Michael” finds Michael Porter Jr. A name can be up to 60 characters." },
      { p: "With names typed, each player says “Qualifies” (projected to get there, or at least 50% likely; 30% for a yes-or-no bet such as an anytime touchdown) or “Doesn't qualify”, and a player who is ruled out shows that status." },
      { p: "The finder is free: it uses no AI allowance and no odds requests." },
    ],
    example: "A player has a 62% chance of 20+ points, and the posted price for 20+ is 1.70. On average $1 returns 0.62 × 1.70 = $1.054, so the edge is +5.4%.",
    links: [["#/players", "Open the Player finder"], ["#/help/player_finder", "Player finder in the glossary"], ["#/help/l10", "L5, L10, L20"]],
  },
  {
    key: "builder", title: "The Builder", blurb: "Safe, Risky and Crazy, the chance you want, Closest effort, and when nothing fits",
    words: "builder build ticket tickets presets preset safe risky crazy custom controls odds you want target odds chance you want target chance stake win about legs closest effort strict near miss near misses nothing fits no ticket relax relaxed swap",
    parts: [
      { p: "Build → Builder makes tickets for you from the prices the app already has (it fetches nothing for a build). Pick a preset, change any control you like, then tap Build tickets. You get 3 tickets by default (up to 5) that share no leg, each with the reasons behind every leg." },
      { terms: [
        ["Safe", "Tickets that win 60% of the time or more, at total odds of about 1.50 to 2.00, with up to 3 legs. “More likely, small payout. It still loses some of the time.”"],
        ["Risky", "Tickets that win 25% to 60% of the time, at odds of about 2.00 to 5.00, with up to 3 legs. “Balanced: a real chance and a real payout.”"],
        ["Crazy", "Tickets that win 3% to 25% of the time, at odds of about 5.00 to 36, with up to 5 legs. “Long shot, big payout: expect to lose most of these.”"],
        ["Odds you want", "The total odds you'd like, like 3.5x. Or fill in Stake and Win about (total back), and the Builder works it out: win about ÷ stake."],
        ["Chance you want", "How often the whole ticket should win: Any, 50% to 90%, or your own number. Every leg has to win, so a higher chance means lower odds."],
        ["Closest effort", "On by default. When nothing fits exactly you still get the closest tickets, marked “Likeliest at your odds”, “Closest to your chance” or “Best balance”, each saying how close it gets and what it relaxed."],
        ["Strict", "Switch Closest effort off (“Strict: only exactly what I asked”) to get exact fits only, or nothing."],
      ] },
      { h: "Why a ticket can't be built" },
      { terms: [
        ["Nothing to build from yet", "No game in your sports and time window has a price to bet at right now. Prices load in the background: widen the sports or the time, or build again later."],
        ["No ticket fits these controls", "Strict only. It lists what ruled the legs out and what to relax, with “Show the closest instead”."],
        ["Rules that never relax", "Games that have started, prices older than you allow (3 hours unless you change it), your bookmakers, your markets and one leg per game."],
        ["The Builder is off for now", "You're on a break, or picks are paused for 24 hours."],
        ["Near misses", "Legs that nearly made it, each with the rule it failed, so you can see what a small change would let in."],
      ] },
    ],
    example: "You ask for 2 legs at about 3.5x with an 80% chance. No ticket can do both, so Closest effort shows, for example, a 3.52x ticket that wins about 27% of the time and a 1.24x ticket that wins about 78%.",
    links: [["#/build/builder", "Open the Builder"], ["#/help/builder", "The Builder in full"], ["#/help/closest_effort", "Closest effort"], ["#/help/target_chance", "Chance you want"]],
  },
  {
    key: "slip", title: "Your slip and your bets", blurb: "Singles and multis, payouts, pushes and voids, the rollover planner and your results",
    words: "slip bet slip single singles multi multis parlay accumulator acca round robin system combined odds payout pays stake costs chance it wins any money back bookmaker keeps expected result most likely push pushes pushed void voids voided called off rollover planner plan plans paper save saved share code delete my bets upcoming settled results net profit roi track tracking",
    parts: [
      { p: "Every pick you tap goes on your slip. The slip bar shows how many picks you have and what they'd pay; tap it, or open Build, for all the numbers. Nothing on the slip is bet: it's a calculator and a record." },
      { terms: [
        ["Single", "Each pick is its own bet with its own stake. 3 singles at $10 cost $30."],
        ["Multi (parlay)", "One bet on several picks, and every pick has to win. The odds multiply: 3 legs at 1.91 make 1.91 × 1.91 × 1.91 = 6.97."],
        ["Round robin and System", "Several smaller bets made from the same picks, so one loss doesn't sink everything. “Costs” shows how many bets that is."],
        ["Pays", "What comes back if it wins, your stake included (“Pays up to” when the ticket can pay different amounts)."],
        ["Chance it wins", "Written as “1 in N” when it's under 50%."],
        ["Bookmaker keeps", "The cut, as a share of your stake. In a multi it's paid again on every leg."],
        ["Expected result", "What the bet returns on average, in money, if it could be placed many times. It's almost always a small loss: that's the cut."],
      ] },
      { h: "Pushes and voids" },
      { p: "A push is a result exactly on the line, like Over 7 when the game ends 4–3: that bet's stake comes back. A void is a game called off. In a multi, a pushed or void leg drops out (it counts as 1.00) and the other legs still have to win. If every leg pushes or is void, the whole stake comes back. A leg with no result 4 days after its start is voided rather than guessed." },
      { h: "The rollover planner" },
      { p: "A rollover bets everything you won again the next day. Build → Rollover planner shows four ways to play the same run of days side by side (Classic, Bank half, Pot & ladder and Flat 5%), with the real chance that each one lasts and ends in profit. Plans start on paper and never roll on by themselves: you choose each day's picks." },
      { h: "Tracking results" },
      { p: "Save a slip and it shows in My bets → Upcoming. It settles by itself from the final scores and moves to Settled. “Your results” shows your net profit or loss and your ROI for 7 days, 30 days or all time, never a “total won” on its own. You can delete a bet until its first game starts; after that it stays in your history." },
    ],
    example: "$10 on a 2-leg multi at 1.91 and 2.10: the odds are 1.91 × 2.10 = 4.01, so it pays $40.11 if both win. If the second leg pushes, it counts as 1.00 and you get $10 × 1.91 = $19.10.",
    links: [["#/build", "Open your slip"], ["#/build/rollover", "Rollover planner"], ["#/bets", "Open My bets"], ["#/help/rollover", "Rollover explained"], ["#/help/push", "Push"]],
  },
  {
    key: "ask", title: "Ask AI", blurb: "What the AI can and can't do, its daily allowance, and pasting a slip",
    words: "ask ai chat assistant claude anthropic question questions explain request bar build me a ticket credits allowance limit limits daily resets midnight paused out of credits paste slip screenshot photo text check age",
    parts: [
      { p: "Ask in your own words, like “Who does the AI favour tonight?” or “Explain what the bookmaker keeps”. Or use the request bar to have a ticket built: sports, stake, what you'd like back, the number of legs, and Safe · Risky · Crazy." },
      { terms: [
        ["Code counts, the AI explains", "The app's code works out every number: chances, prices, payouts and the cut. The AI (Claude, made by Anthropic) picks from those numbers and explains them. A check reads every reply first, and a reply with a number the code didn't produce, or a word that promises a win, is sent back."],
        ["What it can do", "Answer questions about games and players, build tickets, find players for a stat, compare rollover plans and check a slip you paste."],
        ["What it can't do", "Place a bet, add a pick to your slip for you, or promise a result. It never suggests a bigger stake, and it can still be wrong."],
        ["Daily allowance", "Each AI answer costs the app money, so everyone gets an allowance each day (the “AI allowance today” meter). It resets at midnight (US Eastern). Games, odds, slip prices and the Player finder don't use it."],
        ["Adults only", "Confirm you're 18 or older (21 or older in the US) to chat with the AI."],
      ] },
      { h: "Pasting a slip" },
      { p: "Tap 📋 Paste a slip and paste a bookmaker's slip as text or a screenshot (in Telegram you can send it to the bot). The AI copies what the slip says, then the app's code checks it at your odds: the real chance, what the bookmaker keeps, the weakest leg and better prices elsewhere. Pasted text can also be read by code alone, for free; screenshots need the AI. Screenshots are never kept." },
    ],
    example: "You paste “Celtics −3.5 at −110, Nuggets to win at −400, $10”. −110 is 1.909 and −400 is 1.25, so the card works out $10 × 1.909 × 1.25 = $23.86 back if both win.",
    links: [["#/ask", "Open Ask AI"], ["#/ask?paste=1", "Paste a slip"], ["#/help/paste_slip", "Paste a slip in full"], ["#/help/ai", "How the AI decides"]],
  },
  {
    key: "brain", title: "The Brain page", blurb: "How the brain learns, its cards, and how its scores work",
    words: "brain learn learning learns predicted graded pending every game back test backtest backtested back-tested 14 days capture raw capture backfill older games seasons wave hit rate brier log loss calibration score scores market line sharp new sport few games",
    parts: [
      { p: "My bets → AI record → Brain progress. The brain is the app's own code that turns odds and stats into chances. This page shows, in the open, how well those chances do." },
      { h: "How it learns" },
      { list: [
        "Every game is predicted before it starts. From an hour before the start the brain logs its chance of each result (and the spread and total where there's a line), then freezes it at the start.",
        "After the final whistle each prediction is graded, and it's never redone.",
        "It back-tests the last 14 days: for each finished game it works out what it would have said an hour before the start, using only what was known then, and grades that at once.",
        "It scores itself every night and decides on changes once a week. Most weeks nothing changes, and the page says why.",
      ] },
      { h: "The cards" },
      { terms: [
        ["Is the brain getting better?", "The plain answer, sport by sport."],
        ["Every game", "Per sport, three lines kept apart: “Every game, live”, “Every game, last 14 days (back-tested)” and “Priced picks”. Each gives the games predicted, graded and pending, the hit rate, and the Brier score next to the line's. These are learning records, not picks."],
        ["Raw capture", "Every finished game's full record (plays, box scores, officials), kept as it came, plus free prices saved before each start."],
        ["Older games", "A fold inside Raw capture: past seasons, loaded a little each night, as “N of M seasons done”. Most leagues come in a later wave, shown as one line like “Wave 3: 12 of 43 leagues done”."],
      ] },
      { h: "The scores, in plain words" },
      { terms: [
        ["Hit rate", "How often the result the brain thought likeliest happened. 64% means about 2 games in 3."],
        ["Brier score", "The average squared miss of a chance. Saying 50% every time scores 0.250. Lower is better."],
        ["Log loss", "Like the Brier score, but it punishes being confidently wrong much harder. Lower is better."],
        ["The bar to beat", "Each score sits next to the market line's on the same games. The market is very hard to beat, so getting close to it is a good result."],
      ] },
      { h: "Why a new sport starts with few games" },
      { p: "A sport's record only counts games the brain predicted before they started, plus the 14-day back-test. A sport that has just been added, or is between seasons, has few such games, so its numbers swing a lot and say little until more games are graded." },
    ],
    example: "“Every game, live: 120 predicted · 96 graded · 24 pending · hit rate 64% · Brier 0.215 vs the line's 0.218 (96 games)”. Lower is better, so the brain's average miss was a little smaller than the line's on those 96 games.",
    links: [["#/bets?tab=record&view=brain", "Open the Brain page"], ["#/help/brain", "How the brain learns"], ["#/help/brier", "Brier score"], ["#/help/log_loss", "Log loss"]],
  },
  {
    key: "calls", title: "Make your call", blurb: "Your own predictions, how they're scored, and the leaderboard",
    words: "make your call calls call prediction predictions predict confidence confident points score scored leaderboard leaders vs ai vs books beat ai name",
    parts: [
      { p: "On an NBA game, tap Make your call (🔮 in the bot). Pick the winner and how confident you are, from 50% to 100%. In the app you can also add the final score and up to 8 player lines." },
      { p: "As you choose, the app shows the AI's view and the points you'd get. A confident call scores more when it's right and loses more when it's wrong. After the game your call is scored, and so is the AI's on the same game." },
      { terms: [
        ["My bets → Calls", "Your calls and how each one scored."],
        ["My bets → Leaders", "The leaderboard: your points; “vs AI”, your points minus the AI's on the same games (copying the AI scores 0, so above 0 means you read those games better); “vs books”, your winner calls against the bookmakers; and “Beat AI”, how many of your calls beat the AI's."],
        ["Your name", "The leaderboard shows the name you choose: letters, numbers, spaces and . ' - (up to 24)."],
      ] },
      { p: "Calls are a game of judgement: no money is staked." },
    ],
    example: "At 50% you score 0 whatever happens. At 60% you get +9 if your team wins and −11 if it loses. At 100% it's +25 or −75, so only go all the way when you really mean it.",
    links: [["#/today", "Find an NBA game"], ["#/bets?tab=calls", "My calls"], ["#/bets?tab=leaders", "Leaderboard"], ["#/help/calls", "Calls in the glossary"]],
  },
  {
    key: "safety", title: "Safety tools", blurb: "Budget and limits, the 24-hour break, the reality check, the late-night note and help",
    words: "safety responsible gambling limits limit budget weekly break cool off cooloff time out pause paused 24 hours reality check reminder late night note help helpline helplines addiction problem stop gamcare gambler",
    parts: [
      { p: "Settings → Limits holds the tools. Set them before you need them." },
      { terms: [
        ["Weekly budget", "Stakes you save count against it, and a save past what's left is refused. Lowering it works at once; raising it waits 24 hours."],
        ["Take a break", "24 hours, 3 days, 1 week, 1 month or 3 months. Picks, building, plans and the AI's picks switch off until it ends. It can't be cut short or undone, and nothing starts again by itself; you can only make it longer."],
        ["Reality check", "A reminder every 15, 30 or 60 minutes of how long the app has been open, with a button to take a break."],
        ["The late-night note", "Between 1 and 5 in the morning, the AI adds a line to its answers: decisions made tired tend to be worse ones."],
        ["The 24-hour pause", "If a message to the AI sounds like chasing losses or money worries, picks switch off for 24 hours and you're shown where to get help. Browsing games stays open."],
      ] },
      { h: "Where to get help" },
      { p: "Free and confidential:" },
      { helplines: true },
      { p: "In an emergency, or if you're thinking about harming yourself, call your local emergency number." },
    ],
    example: "Your weekly budget is $50 and you've saved $40 of bets this week. A $20 bet is refused, because only $10 is left. Raising the budget to $100 only takes effect 24 hours later.",
    links: [["#/settings?focus=limits", "Set my limits"], ["#/settings?focus=break", "Take a break"], ["#/help/responsible", "Responsible play"]],
  },
  {
    key: "data", title: "Data and privacy", blurb: "Where the data comes from, what the app keeps about you, and why predictions can be wrong",
    words: "data privacy private sources source where from odds feed sports data provider stats scores injuries weather prediction markets stored store keep keeps kept delete share code who can see screenshots chats history id telegram browser wrong",
    parts: [
      { p: "Where the data comes from: game schedules, scores, injuries and player stats come from sports data providers, and bookmakers' prices from an odds feed. The app also reads prediction markets, weather forecasts and public ratings. The app's own code turns all of it into chances." },
      { p: "Prices move all the time and can be out of date here. Always check the price where you bet." },
      { h: "What the app keeps about you" },
      { list: [
        "An ID: your Telegram account, or a random ID for your browser.",
        "Your settings: your name if you give one, your age confirmation, odds format, sports, units and time zone (a theme you choose stays on your device).",
        "Your limits and breaks.",
        "Your saved bets, plans and calls.",
        "Your chats with the AI and the slips you paste: the text and what was read from it. Screenshots are never kept.",
        "How much of your daily AI allowance you've used.",
        "A short note when a message showed a warning sign, such as chasing losses, for the person running the app to review.",
      ] },
      { p: "Other people using the app can't see your bets, chats or slips. A share code only lets someone load the same picks into their own slip. The leaderboard shows the name you choose and your calls' points. The person who runs the app can see what's stored on its server." },
      { h: "Predictions can be wrong" },
      { p: "They're estimates from data that can be late or missing, and no model reliably beats the betting market. The app's own record, misses included, is in My bets → AI record." },
    ],
    example: "You paste a screenshot of a slip. The app keeps what it read from it (“Celtics −3.5 at −110”), not the picture, and only you see it in your chats.",
    links: [["#/settings", "Settings"], ["#/bets?tab=record", "The AI's record"]],
  },
  {
    key: "glossary", title: "Glossary", blurb: "The words you'll see, one line each",
    words: "glossary terms words meaning define definition moneyline spread total over under prop parlay multi push void implied probability margin vig juice edge value clv closing line back-test backtest calibration",
    parts: [
      { terms: [
        ["Moneyline", "A bet on who wins the game.", "#/help/moneyline"],
        ["Spread", "A bet on the winning margin: −3.5 has to win by 4 or more, and +3.5 can lose by up to 3.", "#/help/spread"],
        ["Total (over/under)", "A bet on whether the combined score goes over or under a number.", "#/help/total"],
        ["Prop", "A bet on one player or one event, like a player scoring 20 or more points.", "#/help/player_finder"],
        ["Parlay or multi", "One bet on several picks: every pick has to win.", "#/help/multi"],
        ["Push", "The result lands exactly on the line, so the stake comes back.", "#/help/push"],
        ["Void", "A bet called off, for example when the game is cancelled: the stake comes back."],
        ["Implied probability", "The chance a price suggests: 1 ÷ the decimal odds.", "#/help/implied_probability"],
        ["Margin or vig", "The bookmaker's cut built into the prices (also called juice).", "#/help/book_cut"],
        ["Edge", "A chance minus the chance the price implies: positive means the price pays more than the chance says.", "#/help/edge"],
        ["Value", "A price that pays more than the chance says it should, which is the same as a positive edge.", "#/help/value_mark"],
        ["CLV (closing line value)", "How the price you took compares with the last price before the start. The app doesn't track it for your bets."],
        ["Back-test", "Testing a method on past games using only what was known before each one.", "#/help/brain"],
        ["Calibration", "Whether chances come true as often as they say: 60% picks should win about 60 times in 100.", "#/help/calibration"],
      ] },
    ],
    example: "Over 7 on a game that ends 4–3 is a push: exactly 7, so the stake comes back.",
    links: [["#/help/glossary", "Every term, with worked examples"]],
  },
  {
    key: "faq", title: "FAQ", blurb: "Short answers to the questions people ask most",
    words: "faq questions question answers certain certainty no edge prices disappear disappeared gone missing ai paused allowance credits few games brain",
    parts: [
      { qa: [
        ["Is this a sure thing?", "No. Every pick can lose, even one at 80%, and a 60% pick loses about 4 times in 10. The app shows real chances so you can see the risk: no outcome is certain."],
        ["Why does it say no edge?", "Bookmakers build their cut into every price, so most bets lose a little on average. When no ticket has a positive expected result, the app says so plainly, for example “No edge found: entertainment with an expected cost of $0.45”. That's the honest normal, not an error."],
        ["Why do prices disappear?", "Prices are only for games that haven't started, so they go when a game starts. Picks and the Builder also leave out prices that are too old, and bookmakers change or take down markets. A league's prices load again once they're over 2 hours old, when someone opens it or ahead of games in the next 18 hours, as long as today's share of the app's odds allowance lasts."],
        ["Why is the AI paused?", "One of these: today's AI allowance is used up (it resets at midnight, US Eastern); picks are paused for 24 hours after a message about losses or money worries; you're on a break; you haven't confirmed you're 18 or older yet; or the app's own AI account has run out of credits, which the person running the app has to top up. Browsing games still works."],
        ["Why only a few games in the brain?", "The brain's record only counts games it predicted an hour before the start and graded after the final, plus a back-test of the last 14 days. A sport that has just been added, or is between seasons, has few such games yet. Older seasons are also loading a little each night: Raw capture → Older games shows how far that has got."],
      ] },
    ],
    example: "A $10 ticket with an expected result of −$0.45 costs you 45 cents on average each time you play it, even though some of those tickets win.",
    links: [["#/help/faq", "More questions"]],
  },
];
