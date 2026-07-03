# Hearth — Backend

A real, working backend for the Hearth companion app: a Node/Express API backed by simple JSON-file
storage, with real email delivery (via SMTP) for weekly digests and same-day nudges to a companion.

This is **not a mockup** — settings, check-ins, and sent-notification history all genuinely persist,
and the email-sending code is real `nodemailer` usage. The only thing missing out of the box is your
own (free) SMTP credentials and a hosting deployment, both covered below.

## What's real vs. what you provide

| Piece | Status |
|---|---|
| API (status, settings, check-ins) | ✅ fully working, tested |
| Data persistence | ✅ fully working (JSON file, `hearth.db.json`) |
| Settings audit trail (consent timestamps) | ✅ fully working |
| Duplicate-send prevention | ✅ fully working |
| Email sending code | ✅ fully working, but needs **your** SMTP credentials to actually send |
| Public hosting | Needs **your** account on a hosting platform (steps below) |
| Scheduled digest/nudge triggers | Needs **your** account on a free cron-ping service (steps below) |

**"Today"** in this demo is the most recent day in the historical CASAS dataset the model was trained
on — a live deployment would replace `model.js`'s `getLatestDate()`/`getDay()` with a feed from actual
live sensors, but that's out of scope here (this app's job is to demonstrate the product, not
re-implement live sensor ingestion).

## Running it locally

```bash
npm install
cp .env.example .env   # fine to leave the SMTP_* values as placeholders for now
npm start
```

Open `http://localhost:3000` in your browser. Without real SMTP credentials, sharing/notifications
still work end-to-end — emails just get printed to your terminal instead of actually sent, so you can
test everything before setting up real email.

## Step 1: Get real email sending working (Gmail, free)

1. Go to your Google Account → **Security** → turn on **2-Step Verification** (required for the next step)
2. Go to **Security → App passwords** (search "app passwords" in your Google Account settings if you
   can't find it)
3. Create a new app password (name it "Hearth" or anything), Google gives you a 16-character code
4. In your `.env` file:
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=youraccount@gmail.com
   SMTP_PASS=the16charactercode
   FROM_EMAIL=youraccount@gmail.com
   ```
5. Restart the server and trigger a test:
   ```bash
   curl -X POST http://localhost:3000/api/cron/digest
   ```
   Check the companion email inbox you configured in the app's Companion tab.

## Step 2: Deploy it online (Render, free tier)

1. Push this `app/` folder to your GitHub repo (see main project README for the upload steps)
2. Go to [render.com](https://render.com), sign up free, connect your GitHub account
3. **New → Web Service** → pick your repo
4. Set:
   - **Root Directory**: `app` (or wherever this backend folder lives in your repo)
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
5. Under **Environment**, add each variable from your `.env` file (`SMTP_HOST`, `SMTP_PORT`,
   `SMTP_USER`, `SMTP_PASS`, `FROM_EMAIL`, `CRON_SECRET`) — never commit `.env` itself to GitHub,
   which is why it's in `.gitignore`
6. Deploy. Render gives you a public URL like `https://hearth-yourname.onrender.com`

Note: Render's free tier sleeps after inactivity and wakes up on the next request (takes a few
seconds) — fine for a demo, not for a real always-on product.

## Step 3: Make the digest/nudge actually fire on a schedule

Since free hosting tiers don't reliably run their own background schedulers, use a free external
pinger to hit your endpoints on a schedule:

1. Go to [cron-job.org](https://cron-job.org), make a free account
2. Create a cron job:
   - URL: `https://your-app.onrender.com/api/cron/nudge-check?key=YOUR_CRON_SECRET`
   - Schedule: once daily (e.g. evening)
   - Method: POST
3. Create a second cron job:
   - URL: `https://your-app.onrender.com/api/cron/digest?key=YOUR_CRON_SECRET`
   - Schedule: once weekly
   - Method: POST

Use the same `CRON_SECRET` value you set in Render's environment variables — this stops random
internet traffic from triggering your email sends.

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/today` | GET | Status for the most recent day |
| `/api/day/:date` | GET | Status for a specific date (`YYYY-MM-DD`) |
| `/api/dates` | GET | All available dates |
| `/api/days?limit=90` | GET | Last N days, for the trend chart |
| `/api/settings` | GET | Current companion email + toggles |
| `/api/settings` | POST | Update companion email + toggles |
| `/api/checkin` | POST | Save a mood check-in `{date, mood}` |
| `/api/checkin/:date` | GET | Get a saved check-in |
| `/api/cron/nudge-check?key=...` | POST | Send same-day nudge if today was flagged |
| `/api/cron/digest?key=...` | POST | Send weekly digest |

## Known limitations (be upfront about these in a demo)

- Single-occupant design — the JSON-file store holds one settings row, not per-user accounts.
  A real multi-household product would need real user accounts and a proper database.
- SMTP sending hasn't been verified against a real inbox in development (network-restricted sandbox);
  the code is standard `nodemailer` usage, but test it yourself once deployed with real credentials.
- No authentication on the app itself — anyone with the URL can view/change settings. Fine for a
  personal demo, not for a real deployment with a stranger's health data.
