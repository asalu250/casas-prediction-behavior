require('dotenv').config();
const express = require('express');
const path = require('path');
const db = require('./db');
const model = require('./model');
const { sendEmail } = require('./email');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ---------------------------------------------------------------------
// Status endpoints
// ---------------------------------------------------------------------

app.get('/api/today', (req, res) => {
  const day = model.getDay(model.getLatestDate());
  res.json(day);
});

app.get('/api/day/:date', (req, res) => {
  const day = model.getDay(req.params.date);
  if (!day) return res.status(404).json({ error: 'No data for that date' });
  res.json(day);
});

app.get('/api/dates', (req, res) => {
  res.json(model.getAllDates());
});

app.get('/api/days', (req, res) => {
  const n = Math.min(Number(req.query.limit) || 90, 365);
  res.json(model.getLastNDays(n));
});

// ---------------------------------------------------------------------
// Settings (companion email + sharing toggles)
// ---------------------------------------------------------------------

app.get('/api/settings', (req, res) => {
  const s = db.getSettings();
  res.json({
    companion_email: s.companion_email,
    share_enabled: !!s.share_enabled,
    digest_enabled: !!s.digest_enabled,
    nudge_enabled: !!s.nudge_enabled,
    updated_at: s.updated_at,
  });
});

app.post('/api/settings', (req, res) => {
  const { companion_email, share_enabled, digest_enabled, nudge_enabled } = req.body;

  if (share_enabled && companion_email && !/^\S+@\S+\.\S+$/.test(companion_email)) {
    return res.status(400).json({ error: 'That email address doesn\'t look right' });
  }
  if (share_enabled && !companion_email) {
    return res.status(400).json({ error: 'Add a companion email before turning sharing on' });
  }

  const updated = db.updateSettings({ companion_email, share_enabled, digest_enabled, nudge_enabled });
  res.json({
    companion_email: updated.companion_email,
    share_enabled: !!updated.share_enabled,
    digest_enabled: !!updated.digest_enabled,
    nudge_enabled: !!updated.nudge_enabled,
  });
});

// ---------------------------------------------------------------------
// Check-ins
// ---------------------------------------------------------------------

app.post('/api/checkin', (req, res) => {
  const { date, mood } = req.body;
  if (!date || !['fine', 'off', 'rough'].includes(mood)) {
    return res.status(400).json({ error: 'Invalid check-in' });
  }
  db.saveCheckin(date, mood);
  res.json({ ok: true });
});

app.get('/api/checkin/:date', (req, res) => {
  const c = db.getCheckin(req.params.date);
  res.json(c || null);
});

// ---------------------------------------------------------------------
// Cron-triggered notifications
//
// These are meant to be called by an external scheduler (e.g. a free
// service like cron-job.org hitting this URL on a schedule), since the
// free tier of most simple hosts doesn't include reliable built-in cron.
// Protected with a shared secret so the mail-sending endpoints can't be
// triggered by anyone who finds the URL.
// ---------------------------------------------------------------------

function requireCronSecret(req, res, next) {
  const secret = process.env.CRON_SECRET;
  if (!secret) return next(); // no secret configured (local dev) — allow
  if (req.query.key !== secret) return res.status(401).json({ error: 'Unauthorized' });
  next();
}

app.post('/api/cron/nudge-check', requireCronSecret, async (req, res) => {
  const settings = db.getSettings();
  const today = model.getDay(model.getLatestDate());

  if (!settings.share_enabled || !settings.nudge_enabled || !settings.companion_email) {
    return res.json({ sent: false, reason: 'sharing or nudges not enabled' });
  }
  if (today.status.level !== 'flagged') {
    return res.json({ sent: false, reason: 'today was not flagged' });
  }
  if (db.hasAlreadySent('nudge', today.date)) {
    return res.json({ sent: false, reason: 'already sent for this day' });
  }

  const result = await sendEmail({
    to: settings.companion_email,
    subject: 'A gentle check-in from Hearth',
    text: `${today.date} looked a bit different from the usual routine. No action needed — just checking in when you get a chance.\n\n— Sent automatically by Hearth, on behalf of the person you're connected to.`,
  });
  db.markSent('nudge', today.date);
  res.json({ sent: result.sent, mode: result.mode, date: today.date });
});

app.post('/api/cron/digest', requireCronSecret, async (req, res) => {
  const settings = db.getSettings();
  const latest = model.getLatestDate();

  if (!settings.share_enabled || !settings.digest_enabled || !settings.companion_email) {
    return res.json({ sent: false, reason: 'sharing or digest not enabled' });
  }
  if (db.hasAlreadySent('digest', latest)) {
    return res.json({ sent: false, reason: 'already sent for this week' });
  }

  const week = model.getLastNDays(7);
  const flaggedDays = week.filter(d => d.status.level === 'flagged').map(d => d.date);
  const summary = flaggedDays.length === 0
    ? 'The past week looked consistent with the usual routine.'
    : `The past week was mostly typical, with ${flaggedDays.length} day(s) that stood out: ${flaggedDays.join(', ')}.`;

  const result = await sendEmail({
    to: settings.companion_email,
    subject: 'Your weekly Hearth digest',
    text: `${summary}\n\nThis is a brief automatic summary, not a detailed report — reach out directly if you'd like to know more.\n\n— Sent automatically by Hearth.`,
  });
  db.markSent('digest', latest);
  res.json({ sent: result.sent, mode: result.mode });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Hearth server running on port ${PORT}`));

module.exports = app;
