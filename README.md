# $SPY Crash Entry LEAP Trigger System

**Hunt 10-baggers. Buy fear. Deploy with discipline.**

A quantitative system for entering long-dated deep-ITM SPY LEAP call options during market crashes, based on a tiered three-trigger entry framework. Includes a 10-year backtest engine (v5) and a daily email scanner.

---

## Strategy Overview

Enter only when **2 of 3 triggers align** AND the market has calmed enough to enter at good IV levels.

### The 3 Triggers

| Trigger | Conditions | Purpose |
|---------|-----------|---------|
| **T1 — Price Damage** | SPY 8–15% below 252d high · 2 consecutive red weeks · close < 50d MA on volume · daily range >2% | Structural selling |
| **T2 — Volatility** | VIX >22 · VIX spiked 20%+ from 10d low · SPY range >2% · VIX IV rank >50% | Fear expansion |
| **T3 — Fear Spike** | VVIX >120 (volatility of VIX; proxy for P/C ratio spike) | Panic options buying |

### Entry Gates (VIX cool-down)
Never enter at peak panic — wait for VIX to cool:

| Gate | Condition | Action |
|------|-----------|--------|
| **T1 entry (half size)** | VIX ≤ 25 after trigger fires | Buy 0.5 unit deep ITM call |
| **T2 entry (full size)** | VIX ≤ 20 within 120 days of T1 | Add 1.0 unit deep ITM call |

### Fed Rate Filter (Hard Block)
If the 2-year Treasury yield (FRED: DGS2) has risen ≥ 0.50% over the past 9 months → **skip the episode entirely**. This filters out Fed-driven bear markets (2022) where crash-bounce LEAPs don't work.

### Position Details
- **Option type**: Deep ITM call, delta ~0.775 (midpoint of 0.75–0.80)
- **DTE at entry**: 18 months (540 days)
- **Exit**: 150% gain OR natural expiry

### 10-Year Backtest Results (v5, 2016–2026)
| Metric | Value |
|--------|-------|
| Total episodes | 3 (all others filtered by Fed block) |
| Win rate | **100%** |
| Avg blended return | **+132%** |
| Worst trade | +113% (Aug 2024) |
| Best trade | +143% (Mar 2025, still live) |
| Avg leverage vs SPY | 3.6× |

---

## Repository Structure

```
Jim - twitter/
├── daily_scanner.py          ← Daily CI scanner (sends email report)
├── positions.json            ← Your open positions (edit manually)
├── leap_backtest_v5.py       ← Full 10-year backtest (v5, latest)
├── leap_backtest_v4.py       ← Tiered entry backtest
├── leap_backtest.py          ← v3 baseline backtest
├── README.md                 ← This file
└── LEAP trigger system.jpeg  ← Original strategy image
```

---

## Daily Scanner Setup

The scanner runs every weekday at **4:30 PM ET** via GitHub Actions and emails a full HTML report to `avin.khurana18@gmail.com`.

### Step 1 — Create a Gmail App Password

1. Go to your Google Account → **Security** → **2-Step Verification** (must be enabled)
2. Scroll to **App passwords** → create one (select "Mail" + "Other (Custom name)" → name it "LEAP Scanner")
3. Copy the **16-character password** shown (you won't see it again)

### Step 2 — Add GitHub Secrets

In your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret name | Value |
|-------------|-------|
| `GMAIL_SENDER` | Your Gmail address (e.g., `avin.khurana18@gmail.com`) |
| `GMAIL_APP_PASSWORD` | The 16-character App Password from Step 1 |

> **Note:** The existing scanner already uses `EMAIL_FROM` and `EMAIL_PASSWORD` secrets.
> Add `GMAIL_SENDER` and `GMAIL_APP_PASSWORD` as new secrets alongside them.

### Step 3 — Enable the Workflow

The workflow file is at `.github/workflows/leap_trigger_daily.yml`. It will activate automatically once pushed to `main`.

To trigger a manual test run: GitHub repo → **Actions** → **LEAP Trigger System — Daily Scanner** → **Run workflow**.

### Step 4 — Run Locally (Optional)

```bash
cd "Jim - twitter"

# Set credentials
export GMAIL_SENDER="avin.khurana18@gmail.com"
export GMAIL_APP_PASSWORD="your-16-char-app-password"

pip install pandas-datareader   # if not already installed
python daily_scanner.py
```

Without credentials set, the script runs normally but prints the email subject to stdout instead of sending.

---

## Tracking Open Positions

When a signal fires and you decide to enter, add the position to `positions.json`:

```json
[
  {
    "id": "aug24_t1",
    "tranche": "T1",
    "entry_date": "2024-08-08",
    "entry_spy": 519.7,
    "entry_vix": 23.8,
    "K_itm": 469,
    "entry_cost": 106.8,
    "expiry_date": "2026-02-04",
    "notes": "Aug 2024 yen unwind crash — half size"
  },
  {
    "id": "aug24_t2",
    "tranche": "T2",
    "entry_date": "2024-08-13",
    "entry_spy": 530.9,
    "entry_vix": 18.1,
    "K_itm": 496,
    "entry_cost": 87.3,
    "expiry_date": "2026-02-09",
    "notes": "Aug 2024 — full size add at VIX 18"
  }
]
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Short unique name for the trade |
| `tranche` | `"T1"` (half size) or `"T2"` (full size) |
| `entry_date` | Date you actually bought the option |
| `entry_spy` | SPY price on entry date |
| `entry_vix` | VIX close on entry date |
| `K_itm` | Option strike price (deep ITM) |
| `entry_cost` | Option premium paid per share |
| `expiry_date` | LEAP expiration date (18 months from entry) |
| `notes` | Free text — name the episode |

The scanner marks each position to market daily using Black-Scholes (VIX as IV proxy) and alerts you when a position is approaching the **150% gain exit target**.

---

## Understanding the Daily Email

Each day's email contains:

| Section | What it shows |
|---------|--------------|
| **Signal Banner** | Color-coded status: ALL CLEAR / MONITORING / ENTER T1 / ENTER T2 / BLOCKED |
| **Market Snapshot** | SPY price, drawdown, MA50, daily range, weekly candle status |
| **Trigger Scorecard** | T1/T2/T3 individual status with supporting data |
| **Fed Rate Filter** | 2yr Treasury yield, 9-month change, block status |
| **VIX Entry Gates** | Current VIX vs T1 (≤25) and T2 (≤20) thresholds |
| **Open Positions** | Mark-to-market P&L for all tracked positions |

### Signal States Explained

| Banner | Meaning | Action |
|--------|---------|--------|
| 🚨 **ENTER T1** | ≥2 triggers active, VIX ≤ 25, Fed OK | Buy half-size deep ITM call now |
| ✅ **ENTER T2** | T1 entered ~recently, VIX now ≤ 20 | Add full-size second tranche |
| 👀 **MONITORING** | Trigger fired but VIX still > 25 | Watch daily — waiting for VIX to cool |
| 🚫 **BLOCKED** | Fed filter active (rates rising) | No trades until filter clears |
| 📊 **ALL CLEAR** | No recent triggers | Nothing to do |

---

## Backtest Scripts

All backtest scripts are standalone and can be run directly:

```bash
cd "Jim - twitter"

# Latest version (v5: P/C proxy, Fed filter, 150% exit)
python leap_backtest_v5.py

# Tiered entry v4 (T1 half + T2 full)
python leap_backtest_v4.py

# Single deep ITM leg v3
python leap_backtest.py
```

Each script generates:
- `leap_backtest_*_results.png` — chart (SPY + entries, VIX, P&L per trade)
- `leap_backtest_*_trades.csv` — full trade log

---

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| SPY price + volume | Yahoo Finance (`yfinance`) | Auto-adjusted for splits/dividends |
| VIX | Yahoo Finance (`^VIX`) | CBOE 30-day implied volatility index |
| VVIX | Yahoo Finance (`^VVIX`) | Volatility of VIX; used as T3 (P/C proxy) |
| 2yr Treasury yield | FRED via `pandas-datareader` (`DGS2`) | Fed filter; forward-looking vs FEDFUNDS |

**Why VVIX instead of CBOE equity P/C ratio?**
The CBOE equity put/call ratio is no longer freely downloadable via API (returns 403/HTML). VVIX is the best available free proxy — it spikes when traders panic-buy options, which is the underlying signal the P/C ratio captures.

---

## Limitations & Disclaimers

- **Black-Scholes pricing** uses VIX/100 as the annualized IV input. Real LEAP implied volatility typically has a term-structure discount (long-dated IV < short-dated VIX). Results are approximate.
- **No bid-ask spread** modeled. Real fills on deep ITM LEAPs can be wide.
- **VVIX as T3 proxy** is an approximation. Actual P/C ratio data would be more precise.
- **Position sizing** in the backtest is trade-level returns only (not dollar P&L). Allocate capital according to your own risk tolerance.
- This is **not financial advice**. Past backtest results do not guarantee future performance.

---

## Requirements

```
yfinance
pandas
numpy
scipy
requests
pandas-datareader   # for FRED 2yr Treasury yield
matplotlib          # backtest charts only
```

Install: `pip install -r ../requirements.txt && pip install pandas-datareader`
