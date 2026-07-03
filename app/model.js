const DATA = require('./data/model_data.json');

const DATES = Object.keys(DATA).sort();

const REASON_RULES = [
  { key: 'total_events', roll: 'total_events_roll7_mean', dir: 'low', text: 'Less overall activity around the house than usual' },
  { key: 'total_events', roll: 'total_events_roll7_mean', dir: 'high', text: 'More overall activity than a typical day' },
  { key: 'room_transitions', roll: 'room_transitions_roll7_mean', dir: 'low', text: 'Less movement between rooms than usual' },
  { key: 'n_long_inactivity_gaps', roll: 'n_long_inactivity_gaps_roll7_mean', dir: 'high', text: 'A few longer stretches without any activity' },
  { key: 'routine_similarity_cos', roll: null, dir: 'low', threshold: 0.6, text: "Today's rhythm didn't match the usual pattern as closely" },
  { key: 'night_frac', roll: null, dir: 'high', threshold: 0.15, text: 'More activity than usual during typical sleep hours' },
];

function getReasons(day) {
  const reasons = [];
  for (const rule of REASON_RULES) {
    const val = day[rule.key];
    if (val === null || val === undefined) continue;
    if (rule.roll) {
      const rollVal = day[rule.roll];
      if (!rollVal) continue;
      const ratio = val / rollVal;
      if (rule.dir === 'low' && ratio < 0.75) reasons.push(rule.text);
      if (rule.dir === 'high' && ratio > 1.35) reasons.push(rule.text);
    } else {
      if (rule.dir === 'low' && val < rule.threshold) reasons.push(rule.text);
      if (rule.dir === 'high' && val > rule.threshold) reasons.push(rule.text);
    }
  }
  return reasons.slice(0, 3);
}

function statusFor(day) {
  if (day.label_statistical === 1) {
    return {
      level: 'flagged',
      word: 'A little different today',
      sub: "A few things about today's pattern stood out from the usual rhythm.",
    };
  }
  if (day.n_features_deviant >= 1) {
    return {
      level: 'quiet',
      word: 'Mostly a normal day',
      sub: 'A small thing or two looked slightly off, nothing that stands out on its own.',
    };
  }
  return {
    level: 'normal',
    word: 'A day like any other',
    sub: "Today's activity closely matched the usual pattern.",
  };
}

/** "Today" in this demo is the most recent date in the historical dataset.
 *  A live deployment would replace this with an ingestion endpoint fed by
 *  the actual smart-home sensors; that seam is intentionally isolated here
 *  (getLatestDate / getDay) so swapping the data source later doesn't
 *  require touching the status/reasons/email logic at all. */
function getLatestDate() {
  return DATES[DATES.length - 1];
}

function getDay(date) {
  const day = DATA[date];
  if (!day) return null;
  return { date, ...day, status: statusFor(day), reasons: getReasons(day) };
}

function getAllDates() {
  return DATES;
}

function getLastNDays(n) {
  return DATES.slice(-n).map(getDay);
}

module.exports = { getDay, getLatestDate, getAllDates, getLastNDays, statusFor, getReasons };
