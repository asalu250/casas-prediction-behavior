const nodemailer = require('nodemailer');

let transporter = null;

function getTransporter() {
  if (transporter) return transporter;
  const { SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS } = process.env;
  if (!SMTP_HOST || !SMTP_USER || !SMTP_PASS) {
    return null; // not configured — caller falls back to logging
  }
  transporter = nodemailer.createTransport({
    host: SMTP_HOST,
    port: Number(SMTP_PORT) || 587,
    secure: Number(SMTP_PORT) === 465,
    auth: { user: SMTP_USER, pass: SMTP_PASS },
  });
  return transporter;
}

/**
 * Sends an email, or logs it instead if SMTP isn't configured yet.
 * This lets the rest of the app (routes, cron logic, tests) run and be
 * verified end-to-end without real credentials present — the only thing
 * that changes once SMTP_* env vars are set is that mail actually leaves
 * the server, not any of the calling logic.
 */
async function sendEmail({ to, subject, text }) {
  const t = getTransporter();
  const from = process.env.FROM_EMAIL || 'hearth-app@example.com';

  if (!t) {
    console.log('[email:DEV MODE — not sent, SMTP not configured]');
    console.log(`  to: ${to}`);
    console.log(`  subject: ${subject}`);
    console.log(`  body: ${text}`);
    return { sent: false, mode: 'dev-log' };
  }

  await t.sendMail({ from, to, subject, text });
  return { sent: true, mode: 'smtp' };
}

module.exports = { sendEmail };
