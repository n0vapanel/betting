// Brain progress: the payload of GET /api/brain/progress (learn/progress.py progress()) checked into the shape the page
// draws, and the words and number formats it uses. Every figure is the server's (the evaluator and the gate, through
// the store); this file only checks types, cuts text to size, renames and formats. Status words ("Promoted", "Rolled
// back", "Undone", "Held") come from the gate's own decisions as stored, never from the numbers.
//
// The payload, as learn/progress.py documents it (unknown keys are ignored; missing ones are shown as "–"):
//   {as_of, owner (the viewer may Undo and pause), learning: {enabled, paused, by, at, locked},
//    safe_mode: null | {since, reason},
//    sports: [{sport, label, version: {number, since, source, changed, labels}, headline: {n, gap, lo, hi, word,
//              text, since, until} | null, series: [{week, n, gap, lo, hi, base_gap, n_market}],
//              markers: [{week, at, kind: promoted|rolled_back|undone, version, knob, label}],
//              calibration: {n, slope, slope_lo, slope_hi, intercept, intercept_lo, intercept_hi, words,
//                            curve: {x, y, lo, hi} | null} | null,
//              targets: [{target, label, week, n, brain_ll, market_ll, baseline_ll, gap, gap_lo, gap_hi, vs, shown,
//                         status, pit}],
//              adjustments: {week, games, adjusted, dll, lo, hi, brier: {mean, lo, hi}, slope: {beta, lo, hi},
//                            causes: {cause: n}} | null}],
//    changes: [{id, version, sport, sport_label, knob, label, old, new, at, week, decision, gate, source, provider,
//               evidence: {n, train_n, dll, lo, hi, p, holm_p, tried}, reason, text, text_source, text_model,
//               rolled_back: null | {at, reason, decision}, undo}],
//    held: {week, tested, passed, words, items: [{id, sport, sport_label, knob, label, decision, reason, eval_n,
//           needed}]},
//    experiments: [{version, sport, sport_label, label, knobs, source, n, needed, progress, dll, lo, hi, since}],
//    ai: {mode, provider, model, reason, note: {text, hypotheses, source, model, at, run} | null,
//         proposals: {made, passed, failed, pending, refused}, cost: {spent_cents, reserved_cents, cap_cents},
//         rejected_texts, pending, memo: {text, ideas, source, model, at, month} | null},
//    health: {graded: {sport: n}, voided: {sport: {n, reasons}}, backlog, backfill: {source: {units, done, current,
//             active, todo, games, requests, errors, updated_at}}, history: {league: {games, final, void, with_close,
//             with_stats}}, runs: {night|week|month: {last, period, current}}, trials: {month, promoted, rolled_back}},
//    capture: {enabled, stored, unreadable, as_of, last_stored, coverage: {days, within_hours, target, sports: [{sport,
//              label, finals, covered, missed, pending, share, unread}], total, unread: [{league, day, tries, final}]},
//              size: {raw_bytes, disk_total, disk_free, disk_full, floor_bytes, paused} | null, requests: {day, capture,
//              prices, per_day, backfill, backfill_night, per_night, night, night_open},
//              http: {since, requests, errors, share, counted, days} | null (forward capture),
//              backfill_http: {since, requests, errors, share} | null (the older-games backfill), backfill: [{league,
//              seasons: [{season, done, games_seen, games_stored}], done, begun, planned, games_seen, games_stored,
//              updated_at}], errors: [{at, league, game_key, what, error}],
//              alarms: [{kind: http|backfill|boards|shape|disk, text}]} (v4 P1: the raw capture's health, from raw.db)}
//
//   normalizeProgress(raw) -> the checked payload (never throws)
//   ll(v) "+0.0004" · range(lo, hi) "−0.0011 to +0.0019" · moveText(old, new, unit) "0.20 → 0.22" ·
//   weekStart("2026-W43") Date · weekLabel(week) "Oct 19" · dayLabel(when) · timeLabel(when) · valueText(v, unit) ·
//   sourceWords(source) · providerName(p) · games(n) · sportName(key)

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const int = (v) => (num(v) === null ? null : Math.round(v));
const text = (v, n = 400) => (typeof v === "string" ? v.slice(0, n) : "");
const list = (v) => (Array.isArray(v) ? v : []);
const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
const WEEK = /^(\d{4})-W(\d{1,2})$/;
const week = (v) => (typeof v === "string" && WEEK.test(v) ? v : null);
/** An ISO time from an ISO string or epoch seconds; null otherwise. */
const when = (v) => {
  if (typeof v === "number" && Number.isFinite(v)) return new Date(v * 1000).toISOString();
  return typeof v === "string" && !Number.isNaN(new Date(v).getTime()) ? v : null;
};

/** {value, lo, hi}; null when there is no value. */
function iv(value, lo, hi) {
  return num(value) === null ? null : { value, lo: num(lo), hi: num(hi) };
}

const SPORT_NAMES = {
  nfl: "NFL", ncaaf: "College football", nba: "NBA", wnba: "WNBA", ncaab: "College basketball", nhl: "NHL", mlb: "MLB",
  soccer: "Soccer", other: "Other sports",
};
/** A sport key's name ("nba" -> "NBA"), for the parts of the payload that send only the key. */
export const sportName = (key) => SPORT_NAMES[key] || text(key, 40).toUpperCase();

const CAUSES = { forecast: "Forecast changed", roof_closed: "Roof closed", roof_open: "Roof opened", roof: "Roof" };
const DECISIONS = new Set(["promoted", "rolled_back", "undone", "safe_mode", "cleared", "shadow"]);

function headline(h) {
  if (!h || typeof h !== "object") return { n: 0, gap: null, lo: null, hi: null, word: "", text: "", since: null, until: null };
  return { n: int(h.n) ?? 0, gap: num(h.gap), lo: num(h.lo), hi: num(h.hi), word: text(h.word, 120), text: text(h.text, 400),
    since: when(h.since), until: when(h.until) };
}

function weekKey(w) {
  const m = WEEK.exec(w);
  return Number(m[1]) * 100 + Number(m[2]);
}

function series(raw) {
  return list(raw).map((p) => {
    const o = obj(p);
    return { week: week(o.week), n: int(o.n), gap: num(o.gap), lo: num(o.lo), hi: num(o.hi) };
  }).filter((p) => p.week && p.gap !== null).sort((a, b) => weekKey(a.week) - weekKey(b.week));
}

function markers(raw) {
  return list(raw).map((m) => {
    const o = obj(m);
    const kind = text(o.kind, 20);
    return { week: week(o.week), at: when(o.at), decision: kind, kind: kind === "promoted" ? "live" : "back", version: int(o.version),
      label: text(o.label, 120) || text(o.knob, 80) };
  }).filter((m) => m.week && ["promoted", "rolled_back", "undone"].includes(m.decision));
}

function calibration(raw) {
  if (!raw || typeof raw !== "object") return null;
  const c = obj(raw.curve);
  const xs = list(c.x);
  const ys = list(c.y);
  const los = list(c.lo);
  const his = list(c.hi);
  const curve = [];
  const band = [];
  xs.forEach((x, i) => {
    const p = num(x);
    if (p === null || p < 0 || p > 1) return;
    const obs = num(ys[i]);
    if (obs !== null && obs >= 0 && obs <= 1) curve.push({ p, obs });
    const lo = num(los[i]);
    const hi = num(his[i]);
    if (lo !== null && hi !== null) band.push({ p, lo, hi });
  });
  const slope = iv(raw.slope, raw.slope_lo, raw.slope_hi);
  const intercept = iv(raw.intercept, raw.intercept_lo, raw.intercept_hi);
  if (!curve.length && !slope && !text(raw.words)) return null;
  return { n: int(raw.n), curve, band, slope, intercept, words: text(raw.words) };
}

function targets(raw) {
  return list(raw).map((t) => {
    const o = obj(t);
    const pit = list(o.pit).map(num);
    return {
      target: text(o.target, 40), label: text(o.label, 60) || text(o.target, 40), week: week(o.week), n: int(o.n),
      brain: num(o.brain_ll), market: num(o.market_ll), baseline: num(o.baseline_ll), gap: num(o.gap), lo: num(o.gap_lo),
      hi: num(o.gap_hi), vs: text(o.vs, 60), shown: o.shown !== false, status: text(o.status, 200),
      pit: pit.length >= 2 && pit.every((b) => b !== null && b >= 0) ? pit : [],
    };
  }).filter((t) => t.label);
}

function adjustments(raw) {
  if (!raw || typeof raw !== "object") return null;
  const brier = obj(raw.brier);
  const slope = obj(raw.slope);
  return {
    week: week(raw.week), games: int(raw.games), n: int(raw.adjusted) ?? 0, dll: iv(raw.dll, raw.lo, raw.hi),
    dbrier: iv(brier.mean, brier.lo, brier.hi), beta: iv(slope.beta, slope.lo, slope.hi),
    byCause: Object.entries(obj(raw.causes)).filter(([, n]) => int(n)).map(([cause, n]) => ({ cause: text(cause, 40), label: CAUSES[cause] || text(cause, 40), n: int(n) })),
  };
}

function version(raw) {
  if (typeof raw === "number") return { number: int(raw) ?? 0, since: null, source: "", labels: [] };
  const o = obj(raw);
  return { number: int(o.number) ?? 0, since: when(o.since), source: text(o.source, 40),
    labels: list(o.labels).filter((l) => typeof l === "string").map((l) => l.slice(0, 120)) };
}

function sport(raw) {
  const o = obj(raw);
  const key = text(o.sport, 40);
  if (!key) return null;
  return {
    sport: key, label: text(o.label, 40) || sportName(key), version: version(o.version), headline: headline(o.headline),
    series: series(o.series), markers: markers(o.markers), calibration: calibration(o.calibration), targets: targets(o.targets),
    adjustments: adjustments(o.adjustments),
  };
}

/** A knob value as sent: a number, or {knob: number} for a joint change (several knobs at once). */
function values(v) {
  if (num(v) !== null) return v;
  if (v && typeof v === "object" && !Array.isArray(v)) {
    const out = {};
    for (const [k, x] of Object.entries(v)) if (num(x) !== null) out[text(k, 80)] = x;
    return Object.keys(out).length ? out : null;
  }
  return null;
}

function change(raw) {
  const o = obj(raw);
  const decision = text(o.decision, 20);
  if (!DECISIONS.has(decision)) return null;
  const e = obj(o.evidence);
  const back = o.rolled_back && typeof o.rolled_back === "object"
    ? { at: when(o.rolled_back.at), reason: text(o.rolled_back.reason, 600), decision: text(o.rolled_back.decision, 20) } : null;
  const key = text(o.sport, 40);
  return {
    id: int(o.id) ?? text(String(o.id ?? ""), 40), version: int(o.version), sport: key, sportLabel: text(o.sport_label, 40) || sportName(key),
    knob: text(o.knob, 200), label: text(o.label, 200) || text(o.knob, 200), old: values(o.old), new: values(o.new),
    at: when(o.at), week: week(o.week), decision, gate: text(o.gate, 20), source: text(o.source, 40),
    evidence: { n: int(e.n), trainN: int(e.train_n), dll: num(e.dll), lo: num(e.lo), hi: num(e.hi), p: num(e.p), holmP: num(e.holm_p),
      tried: int(e.tried) },
    reason: text(o.reason, 800), text: text(o.text, 1200), textSource: text(o.text_source, 40) || "template", textModel: text(o.text_model, 80),
    rolledBack: back, canUndo: o.undo === true,
  };
}

function held(raw) {
  if (!raw || typeof raw !== "object") return null;
  const items = list(raw.items).map((i) => {
    const o = obj(i);
    const key = text(o.sport, 40);
    return { sport: key, sportLabel: text(o.sport_label, 40) || sportName(key), knob: text(o.knob, 200),
      label: text(o.label, 200) || text(o.knob, 200), decision: o.decision === "rejected" ? "rejected" : "held",
      reason: text(o.reason, 600), n: int(o.eval_n), needs: int(o.needed) };
  }).filter((i) => i.label);
  return { week: week(raw.week), tested: int(raw.tested) ?? 0, passed: int(raw.passed) ?? 0, words: text(raw.words, 400), items };
}

function experiment(raw) {
  const o = obj(raw);
  const key = text(o.sport, 40);
  return {
    version: int(o.version), sport: key, sportLabel: text(o.sport_label, 40) || sportName(key),
    label: text(o.label, 200) || list(o.knobs).filter((k) => typeof k === "string").join(", "), source: text(o.source, 40),
    n: int(o.n) ?? 0, needed: int(o.needed), progress: num(o.progress), dll: num(o.dll), lo: num(o.lo), hi: num(o.hi), since: when(o.since),
  };
}

function ai(raw) {
  const o = obj(raw);
  const n = obj(o.note);
  const note = typeof n.text === "string" && n.text.trim()
    ? { text: text(n.text, 2400), source: text(n.source, 40) || "template", model: text(n.model, 80), at: when(n.at),
      hypotheses: list(n.hypotheses).map((h) => text(typeof h === "string" ? h : obj(h).text, 400)).filter(Boolean).slice(0, 3) }
    : null;
  const m = obj(o.memo);
  const memo = typeof m.text === "string" && m.text.trim()
    ? { text: text(m.text, 4000), source: text(m.source, 40) || "template", model: text(m.model, 80), at: when(m.at), month: text(m.month, 20),
      ideas: list(m.ideas).map((i) => text(typeof i === "string" ? i : obj(i).text || obj(i).idea, 400)).filter(Boolean).slice(0, 6) }
    : null;
  const p = obj(o.proposals);
  const c = obj(o.cost);
  return {
    mode: text(o.mode, 20), provider: text(o.provider, 40), model: text(o.model, 80), reason: text(o.reason, 200), note, memo,
    proposals: { made: int(p.made) ?? 0, passed: int(p.passed) ?? 0, failed: int(p.failed) ?? 0, pending: int(p.pending) ?? 0, refused: int(p.refused) ?? 0 },
    cost: { spentCents: num(c.spent_cents), reservedCents: num(c.reserved_cents), capCents: num(c.cap_cents) },
    rejected: int(o.rejected_texts) ?? 0, pending: int(o.pending) ?? 0,
  };
}

// Keyed by the backfill's own source names (learn/history.py SOURCES: each learn/sources module's SOURCE). ESPN's
// MLB seasons join MLB StatsAPI's schedule for results and weather inside the "espn" source
const BACKFILL_NAMES = { "football-data": "football-data.co.uk (soccer)", nflverse: "nflverse (NFL)", espn: "ESPN (NBA, NHL, MLB, college)" };

function health(raw) {
  const o = obj(raw);
  const graded = obj(o.graded);
  const voided = obj(o.voided);
  const keys = [...new Set([...Object.keys(graded), ...Object.keys(voided)])];
  const order = Object.keys(SPORT_NAMES);
  keys.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  const runs = obj(o.runs);
  const run = (k) => when(obj(runs[k]).last);
  const trials = obj(o.trials);
  return {
    sports: keys.map((k) => {
      const v = obj(voided[k]);
      return { sport: text(k, 40), label: sportName(k), graded: int(graded[k]) ?? 0, voided: int(v.n) ?? 0,
        reasons: Object.entries(obj(v.reasons)).filter(([, n]) => int(n)).map(([r, n]) => [text(r, 40), int(n)]) };
    }),
    backfill: Object.entries(obj(o.backfill)).map(([source, b]) => {
      const x = obj(b);
      return { source: text(source, 40), label: BACKFILL_NAMES[source] || text(source, 40), units: int(x.units) ?? 0,
        done: (int(x.done) ?? 0) + (int(x.current) ?? 0), todo: (int(x.todo) ?? 0) + (int(x.active) ?? 0), games: int(x.games) ?? 0,
        requests: int(x.requests) ?? 0, errors: int(x.errors) ?? 0, updated: when(x.updated_at) };
    }),
    history: Object.entries(obj(o.history)).map(([league, h]) => {
      const x = obj(h);
      return { league: text(league, 64), games: int(x.games) ?? 0, close: int(x.with_close) ?? 0, stats: int(x.with_stats) ?? 0 };
    }).filter((h) => h.games).sort((a, b) => b.games - a.games),
    runs: { night: run("night"), week: run("week"), month: run("month") },
    trials: { month: int(trials.month) ?? 0, promoted: int(trials.promoted) ?? 0, rolledBack: int(trials.rolled_back) ?? 0 },
    backlog: int(o.backlog),
  };
}

// The raw capture's health (v4 P1, learn/progress.py capture_health): shares are 0-1 and sizes are bytes. The server
// words the alarms; their kind only sets the order (disk, then shape, then HTTP), and all show as warnings
const CAPTURE_ALARMS = ["disk", "shape", "boards", "http", "backfill"];
const share01 = (v) => (num(v) === null || v < 0 || v > 1 ? null : v);

function captureCounts(raw) {
  const o = obj(raw);
  return { finals: int(o.finals) ?? 0, covered: int(o.covered) ?? 0, missed: int(o.missed) ?? 0, pending: int(o.pending) ?? 0,
    share: share01(o.share) };
}

/** A "YYYY-MM-DD" day as sent (a sweep's ESPN day, a backfill night); "" for anything else. */
const isoDay = (v) => (typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v) ? v : "");

function capture(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const cov = obj(raw.coverage);
  const size = raw.size && typeof raw.size === "object" ? obj(raw.size) : null;
  const r = obj(raw.requests);
  const h = raw.http && typeof raw.http === "object" ? obj(raw.http) : null;
  const b = raw.backfill_http && typeof raw.backfill_http === "object" ? obj(raw.backfill_http) : null;
  return {
    enabled: raw.enabled !== false, stored: raw.stored === true, unreadable: text(raw.unreadable, 40),
    lastStored: when(raw.last_stored),
    coverage: {
      days: int(cov.days) ?? 7, hours: int(cov.within_hours) ?? 36, target: share01(cov.target),
      sports: list(cov.sports).map((s) => ({ sport: text(obj(s).sport, 20), label: text(obj(s).label, 40) || text(obj(s).sport, 20), ...captureCounts(s),
        unread: int(obj(s).unread) ?? 0 }))
        .filter((s) => s.label && (s.finals || s.unread)),
      total: captureCounts(cov.total),
      // Boards the nightly sweep couldn't read: their games are left out unseen, so coverage is unknown
      unread: list(cov.unread).map((u) => ({ league: text(obj(u).league, 64), day: isoDay(obj(u).day), tries: int(obj(u).tries) ?? 0,
        final: obj(u).final === true })).filter((u) => u.league).slice(0, 60),
    },
    size: size ? { raw: int(size.raw_bytes), total: int(size.disk_total), free: int(size.disk_free), full: share01(size.disk_full),
      floor: int(size.floor_bytes), paused: size.paused === true } : null,
    requests: { day: text(r.day, 10), capture: int(r.capture) ?? 0, prices: int(r.prices) ?? 0, perDay: int(r.per_day),
      backfill: int(r.backfill) ?? 0, backfillNight: isoDay(r.backfill_night), perNight: int(r.per_night), night: isoDay(r.night),
      nightOpen: r.night_open === true },
    // Forward capture's requests (the backfill's own are backfillHttp)
    http: h ? { since: when(h.since), requests: int(h.requests) ?? 0, errors: int(h.errors) ?? 0, share: share01(h.share),
      counted: h.counted === "requests" ? "requests" : "stored" } : null,
    backfillHttp: b ? { since: when(b.since), requests: int(b.requests) ?? 0, errors: int(b.errors) ?? 0, share: share01(b.share) } : null,
    backfill: list(raw.backfill).map((b) => {
      const o = obj(b);
      return {
        league: text(o.league, 64), done: int(o.done) ?? 0, begun: int(o.begun) ?? 0, planned: int(o.planned), stored: int(o.games_stored) ?? 0,
        seen: int(o.games_seen) ?? 0, updated: when(o.updated_at),
        seasons: list(o.seasons).map((s) => ({ season: int(obj(s).season), done: obj(s).done === true, stored: int(obj(s).games_stored) ?? 0 }))
          .filter((s) => s.season !== null).slice(0, 12),
      };
    }).filter((b) => b.league).slice(0, 60),
    errors: list(raw.errors).map((e) => {
      const o = obj(e);
      return { at: when(o.at), league: text(o.league, 64), key: text(o.game_key, 80), what: text(o.what, 80), error: text(o.error, 240) };
    }).filter((e) => e.what || e.error).slice(0, 5),
    alarms: list(raw.alarms).map((a) => ({ kind: text(obj(a).kind, 20), text: text(obj(a).text, 300) })).filter((a) => a.text)
      .sort((a, b) => (CAPTURE_ALARMS.indexOf(a.kind) + 1 || 9) - (CAPTURE_ALARMS.indexOf(b.kind) + 1 || 9)).slice(0, 8),
  };
}

/** The payload checked; never throws (a malformed part is left out or shown as "–"). */
export function normalizeProgress(raw) {
  const o = obj(raw);
  const l = obj(o.learning);
  const locked = l.locked === true || l.enabled === false;
  const paused = l.paused === true;
  const safe = o.safe_mode && typeof o.safe_mode === "object" ? { since: when(o.safe_mode.since), reason: text(o.safe_mode.reason, 400) } : null;
  return {
    asOf: when(o.as_of),
    owner: o.owner === true,
    learning: { enabled: !locked && !paused, pausedBy: locked ? "env" : paused ? text(l.by, 20) || "owner" : "", pausedAt: paused ? when(l.at) : null },
    safeMode: safe,
    sports: list(o.sports).map(sport).filter(Boolean),
    changes: list(o.changes).map(change).filter(Boolean),
    held: held(o.held),
    experiments: list(o.experiments).map(experiment).filter((e) => e.label),
    ai: ai(o.ai),
    health: health(o.health),
    capture: capture(o.capture),
  };
}

// --- formats ------------------------------------------------------------------------------------------------------

const MINUS = "−";

/** A log-loss figure with its sign, 4 decimals and a real minus: "+0.0004", "−0.0011", "0.0000"; "–" when unknown. */
export function ll(v, digits = 4) {
  if (num(v) === null) return "–";
  const t = Math.abs(v).toFixed(digits);
  if (Number(t) === 0) return (0).toFixed(digits);
  return `${v > 0 ? "+" : MINUS}${t}`;
}

/** "−0.0011 to +0.0019" (a 95% range), or "" when either end is unknown. */
export function range(lo, hi, fmt = ll) {
  return num(lo) === null || num(hi) === null ? "" : `${fmt(lo)} to ${fmt(hi)}`;
}

/** A plain number with a real minus sign: fixed(−0.11, 2) "−0.11". */
export function fixed(v, digits = 2) {
  if (num(v) === null) return "–";
  const t = Math.abs(v).toFixed(digits);
  return Number(t) === 0 ? (0).toFixed(digits) : `${v < 0 ? MINUS : ""}${t}`;
}

/** The Monday an ISO week ("2026-W43") starts on, as a UTC date; null for anything else. */
export function weekStart(week) {
  const m = WEEK.exec(typeof week === "string" ? week : "");
  if (!m) return null;
  const year = Number(m[1]);
  const jan4 = new Date(Date.UTC(year, 0, 4));
  const monday = new Date(jan4.getTime() - (((jan4.getUTCDay() + 6) % 7) * 86400000));
  return new Date(monday.getTime() + (Number(m[2]) - 1) * 7 * 86400000);
}

/** The ISO week ("2026-W43") of an ISO time; null when it can't be read. */
export function isoWeek(when) {
  const d = new Date(when);
  if (Number.isNaN(d.getTime())) return null;
  const day = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  day.setUTCDate(day.getUTCDate() + 3 - ((day.getUTCDay() + 6) % 7)); // the week's Thursday decides its year
  const jan4 = new Date(Date.UTC(day.getUTCFullYear(), 0, 4));
  const week = 1 + Math.round(((day - jan4) / 86400000 - 3 + ((jan4.getUTCDay() + 6) % 7)) / 7);
  return `${day.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** "Oct 19": the Monday a week starts on. */
export function weekLabel(week) {
  const d = weekStart(week);
  return d ? d.toLocaleDateString([], { month: "short", day: "numeric", timeZone: "UTC" }) : "";
}

/** "Oct 19" from an ISO time (the viewer's own day). */
export function dayLabel(when) {
  const d = new Date(when || "");
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/** "Oct 19, 03:31" from an ISO time. */
export function timeLabel(when) {
  const d = new Date(when || "");
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** A knob's value as sent: seconds as minutes when whole ("15 min"), numbers without trailing noise. */
export function valueText(v, unit = "") {
  if (v && typeof v === "object") {
    const entries = Object.entries(v);
    if (entries.length === 1) return valueText(entries[0][1], unitOf(entries[0][0], unit));
    return entries.map(([k, x]) => `${shortKnob(k)} ${valueText(x, unitOf(k, unit))}`).join(" · ") || "–";
  }
  if (num(v) === null) return v === null || v === undefined || v === "" ? "–" : text(String(v), 40);
  if (unit === "s") return v % 60 === 0 ? `${v / 60} min` : `${v} s`;
  const t = String(Math.round(v * 1e6) / 1e6).replace("-", MINUS);
  return unit ? `${t} ${unit}` : t;
}

/** Seconds knobs are shown in minutes ("15 min"). */
const unitOf = (knob, unit) => unit || (/seconds$/.test(String(knob)) ? "s" : "");

/** "consensus.weight.prediction_market" -> "prediction market". */
const shortKnob = (k) => String(k).split(".").pop().replace(/_/g, " ");

/** The decimals a number was sent with (at most 4). */
function places(v) {
  const t = String(v);
  const i = t.indexOf(".");
  return i < 0 || t.includes("e") ? 0 : Math.min(4, t.length - i - 1);
}

/** "0.20 → 0.22", "0.015 → 0.010", "15 min → 20 min": both ends at the same precision. */
export function moveText(from, to, unit = "") {
  if (from && to && typeof from === "object" && typeof to === "object") {
    const keys = Object.keys(to).filter((k) => k in from);
    if (keys.length === 1) return moveText(from[keys[0]], to[keys[0]], unitOf(keys[0], unit));
    return keys.map((k) => `${shortKnob(k)} ${moveText(from[k], to[k], unitOf(k, unit))}`).join(" · ") || "–"; // a joint change
  }
  if (num(from) === null || num(to) === null || unit === "s") return `${valueText(from, unit)} → ${valueText(to, unit)}`;
  const digits = Math.max(places(from), places(to), Number.isInteger(from) && Number.isInteger(to) ? 0 : 2);
  const f = (v) => `${fixed(v, digits)}${unit ? ` ${unit}` : ""}`;
  return `${f(from)} → ${f(to)}`;
}

const PROVIDERS = { claude: "Claude", deepseek: "DeepSeek" };

/** "Claude", "DeepSeek", or the name as sent. */
export function providerName(p) {
  const key = text(p, 40).toLowerCase();
  return PROVIDERS[key] || text(p, 40);
}

/** Who made a change: the refit, an AI's proposal (judged on later games only) or the owner. */
export function sourceWords(source) {
  const s = text(source, 40);
  if (s === "refit") return "Refit from graded games";
  if (s === "owner") return "The owner";
  if (s === "defaults" || s === "default") return "The starting values";
  if (s.startsWith("ai:")) return `${providerName(s.slice(3))}'s proposal`;
  return s || "–";
}

/** The plural-aware "1 game" / "640 games". */
export function games(n, word = "game") {
  if (num(n) === null) return `– ${word}s`;
  return `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
}
