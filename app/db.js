const fs = require('fs');
const path = require('path');

const DB_FILE = path.join(__dirname, 'hearth.db.json');

const DEFAULT_STATE = {
  settings: {
    companion_email: null,
    share_enabled: false,
    digest_enabled: true,
    nudge_enabled: true,
    updated_at: new Date().toISOString(),
  },
  // append-only audit trail of every settings change — consent state for
  // sharing personal data should be provable later, not just overwritten
  settings_history: [],
  checkins: {},       // date -> { mood, created_at }
  sent_log: {},        // `${type}:${refDate}` -> sent_at
};

/**
 * A tiny dependency-free JSON-file "database". This is intentionally
 * simple rather than a real embedded database (no native compilation
 * step, works identically on any Node host, easy to inspect by hand)
 * — appropriate for this app's actual data volume (one settings row,
 * a handful of check-ins), not a choice that would scale to many
 * concurrent users. A real multi-user deployment would swap this for
 * Postgres/SQLite behind the same function interface below.
 */
function load() {
  if (!fs.existsSync(DB_FILE)) {
    fs.writeFileSync(DB_FILE, JSON.stringify(DEFAULT_STATE, null, 2));
  }
  return JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
}

function save(state) {
  fs.writeFileSync(DB_FILE, JSON.stringify(state, null, 2));
}

function getSettings() {
  return load().settings;
}

function updateSettings({ companion_email, share_enabled, digest_enabled, nudge_enabled }) {
  const state = load();
  const now = new Date().toISOString();

  state.settings = {
    companion_email: companion_email ?? state.settings.companion_email,
    share_enabled: !!share_enabled,
    digest_enabled: !!digest_enabled,
    nudge_enabled: !!nudge_enabled,
    updated_at: now,
  };
  state.settings_history.push({ ...state.settings, changed_at: now });

  save(state);
  return state.settings;
}

function saveCheckin(date, mood) {
  const state = load();
  state.checkins[date] = { mood, created_at: new Date().toISOString() };
  save(state);
}

function getCheckin(date) {
  return load().checkins[date] || null;
}

function hasAlreadySent(type, refDate) {
  const state = load();
  return !!state.sent_log[`${type}:${refDate}`];
}

function markSent(type, refDate) {
  const state = load();
  state.sent_log[`${type}:${refDate}`] = new Date().toISOString();
  save(state);
}

module.exports = { getSettings, updateSettings, saveCheckin, getCheckin, hasAlreadySent, markSent };
