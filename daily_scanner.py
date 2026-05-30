#!/usr/bin/env python3
"""
$SPY Crash Entry LEAP Trigger System — Daily Scanner (v5)
Runs every weekday after market close. Sends HTML email report to configured address.

Signal logic (identical to leap_backtest_v5.py):
  T1  Price Damage   : SPY 8–15% below 252d high, 2 consec red weeks,
                       below 50d MA on volume, daily range >2%
  T2  Volatility     : VIX >22, spike +20% from 10d low, range >2%, IV rank >50%
  T3  Fear spike     : VVIX > 120  (put-buying proxy; CBOE P/C unavailable via free API)
  Fed filter         : 2yr Treasury (DGS2) 9-month change ≥ +0.50%  → hard block
  Entry gates        : T1 half-size when VIX ≤ 25 | T2 full-size when VIX ≤ 20

Open positions are tracked in positions.json (same directory).
"""

import os, sys, json, smtplib, warnings, requests
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import io
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
POSITIONS_FILE = SCRIPT_DIR / "positions.json"

TO_EMAIL     = "avin.khurana18@gmail.com"
FROM_EMAIL   = os.environ.get("GMAIL_SENDER", os.environ.get("EMAIL_FROM", TO_EMAIL))
APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", os.environ.get("EMAIL_PASSWORD", ""))

# Strategy parameters (must match v5 backtest)
VVIX_THRESH   = 120
FED_YIELD_CHG = 0.50
FED_LOOKBACK  = 270
T1_VIX_MAX    = 25;  T1_DD_MIN = -0.15; T1_DD_MAX = -0.05
T2_VIX_MAX    = 20;  T2_DD_MAX = -0.02
EXIT_TARGET   = 1.50   # 150% gain
RISK_FREE     = 0.05
DTE_DAYS      = 540

SIGNAL_LOOKBACK_DAYS = 30   # how far back to search for a pending trigger cluster
T2_WINDOW_DAYS       = 120  # how long after T1 entry to watch for T2


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def bs_call(S, K, T, r, σ):
    if T < 1e-6: return max(0.0, S - K)
    if σ < 1e-6: return max(0.0, S - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r + 0.5*σ**2)*T) / (σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

def bs_delta(S, K, T, r, σ):
    if T < 1e-6: return 1.0 if S >= K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*σ**2)*T) / (σ*np.sqrt(T))
    return float(norm.cdf(d1))


# ── Rate data helpers ─────────────────────────────────────────────────────────

def _fred_csv(series_id, start, end):
    """Download a FRED series as a daily pandas Series via the public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    df.columns = [series_id.lower()]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    series = pd.to_numeric(df[series_id.lower()], errors="coerce").dropna()
    return series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]


def _yf_rate(ticker, start, end):
    """Download a rate ticker from yfinance and return as a daily Series."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    series = df["Close"].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.dropna()


# ── Data download ─────────────────────────────────────────────────────────────

def download_data():
    """Download ~400 days of SPY, VIX, VVIX and 2yr Treasury yield."""
    end   = datetime.today()
    start = end - timedelta(days=400)

    print("Downloading market data …")
    spy_raw = yf.download("SPY",   start=start, end=end, auto_adjust=True, progress=False)
    vix_raw = yf.download("^VIX",  start=start, end=end, auto_adjust=True, progress=False)
    vvix_raw= yf.download("^VVIX", start=start, end=end, auto_adjust=True, progress=False)

    for d in (spy_raw, vix_raw, vvix_raw):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    data = pd.DataFrame({
        "open":   spy_raw["Open"],
        "high":   spy_raw["High"],
        "low":    spy_raw["Low"],
        "close":  spy_raw["Close"],
        "volume": spy_raw["Volume"],
        "vix":    vix_raw["Close"],
        "vvix":   vvix_raw["Close"],
    }).dropna(subset=["open","close","vix"])
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data["vvix"] = data["vvix"].ffill().fillna(100)

    # 2yr Treasury — try FRED public CSV first, then yfinance ^FVX (5yr), then ^IRX (3mo)
    dgs2_loaded = False
    for attempt, (label, fetch) in enumerate([
        ("FRED CSV (DGS2)", lambda: _fred_csv("DGS2", start, end)),
        ("yfinance ^FVX (5yr proxy)", lambda: _yf_rate("^FVX", start, end)),
        ("yfinance ^IRX (3mo proxy)", lambda: _yf_rate("^IRX", start, end)),
    ]):
        try:
            series = fetch()
            if series is not None and len(series) > 50:
                data["dgs2"] = series.reindex(data.index, method="ffill")
                print(f"  2yr yield: {label} ✓")
                dgs2_loaded = True
                break
        except Exception as e:
            print(f"  2yr yield: {label} failed — {e}")

    data["dgs2"] = pd.to_numeric(data["dgs2"], errors="coerce").ffill()
    return data


# ── Indicators & triggers ──────────────────────────────────────────────────────

def compute_signals(data):
    data = data.copy()

    data["high_252"]    = data["close"].rolling(252).max()
    data["drawdown"]    = (data["close"] - data["high_252"]) / data["high_252"]
    data["ma50"]        = data["close"].rolling(50).mean()
    data["vol_ma20"]    = data["volume"].rolling(20).mean()
    data["daily_range"] = (data["high"] - data["low"]) / data["close"].shift(1)
    data["vix_10d_min"] = data["vix"].rolling(10).min()
    data["vix_spike"]   = (data["vix"] - data["vix_10d_min"]) / data["vix_10d_min"]
    data["vix_iv_rank"] = data["vix"].rolling(252).apply(
        lambda x: (x[-1]-x.min())/(x.max()-x.min()) if x.max()>x.min() else 0.5, raw=True)

    wk_close = data["close"].resample("W-FRI").last()
    wk_open  = data["open"].resample("W-FRI").first()
    wk_red   = wk_close < wk_open
    two_red  = wk_red.shift(1) & wk_red.shift(2)
    data["two_consec_red"] = two_red.reindex(data.index, method="ffill").fillna(False)

    data["dgs2_9m_chg"] = data["dgs2"] - data["dgs2"].shift(FED_LOOKBACK)
    data["fed_block"]   = data["dgs2_9m_chg"] >= FED_YIELD_CHG

    T1 = (
        (data["drawdown"] < -0.08) & (data["drawdown"] > -0.15) &
        data["two_consec_red"] &
        (data["close"] < data["ma50"]) & (data["volume"] > data["vol_ma20"]) &
        (data["daily_range"] > 0.02)
    )
    T2 = (
        (data["vix"] > 22) & (data["vix_spike"] > 0.20) &
        (data["daily_range"] > 0.02) & (data["vix_iv_rank"] > 0.50)
    )
    T3 = data["vvix"] > VVIX_THRESH

    data["T1"] = T1; data["T2"] = T2; data["T3"] = T3
    data["n_triggers"] = T1.astype(int) + T2.astype(int) + T3.astype(int)
    data["signal"]     = data["n_triggers"] >= 2
    data.dropna(subset=["high_252","ma50"], inplace=True)
    return data


# ── Signal state determination ─────────────────────────────────────────────────

def determine_signal_state(data):
    """
    Returns a dict with today's signal state and actionable recommendation.
    Looks back SIGNAL_LOOKBACK_DAYS to detect pending trigger clusters.
    """
    today_row = data.iloc[-1]
    recent    = data.tail(SIGNAL_LOOKBACK_DAYS + 1)

    vix_now  = float(today_row["vix"])
    dd_now   = float(today_row["drawdown"])
    fed_ok   = not bool(today_row["fed_block"])
    n_trig   = int(today_row["n_triggers"])
    sig_now  = bool(today_row["signal"])

    # Find most recent trigger cluster (any signal day in last 30 days)
    recent_signals = recent[recent["signal"]]
    has_recent_trigger = len(recent_signals) > 0
    last_trigger_row   = recent_signals.iloc[-1] if has_recent_trigger else None
    last_trigger_date  = recent_signals.index[-1] if has_recent_trigger else None

    # Find if a T1 entry opportunity occurred in the last T2_WINDOW_DAYS
    # (a day where VIX ≤ 25 AND SPY in drawdown AND there was a trigger within prior 30 days)
    t1_entry_date = None
    t1_window = data.tail(T2_WINDOW_DAYS + SIGNAL_LOOKBACK_DAYS)
    for i in range(len(t1_window) - 1, -1, -1):
        r = t1_window.iloc[i]
        dt = t1_window.index[i]
        if float(r["vix"]) <= T1_VIX_MAX and T1_DD_MIN <= float(r["drawdown"]) <= T1_DD_MAX:
            # Check if a trigger fired in prior 30 days
            prior = t1_window.iloc[max(0, i-SIGNAL_LOOKBACK_DAYS):i]
            if len(prior[prior["signal"]]) > 0:
                t1_entry_date = dt
                break

    # Determine recommendation
    if not fed_ok:
        status = "BLOCKED"
        color  = "#6c757d"
        emoji  = "🚫"
        action = "Fed filter active — 2yr Treasury yield rising. No entries per strategy rules."
    elif sig_now and vix_now <= T1_VIX_MAX and T1_DD_MIN <= dd_now <= T1_DD_MAX:
        status = "ENTER T1"
        color  = "#28a745"
        emoji  = "🚨"
        action = f"Signal firing TODAY with VIX {vix_now:.1f} ≤ 25. Enter HALF-SIZE deep ITM LEAP now."
    elif t1_entry_date and vix_now <= T2_VIX_MAX and dd_now <= T2_DD_MAX:
        status = "ENTER T2"
        color  = "#20c997"
        emoji  = "✅"
        action = (f"T1 entered ~{t1_entry_date.strftime('%b %d')}. VIX {vix_now:.1f} now ≤ 20. "
                  f"Add FULL-SIZE second tranche.")
    elif has_recent_trigger and vix_now > T1_VIX_MAX:
        days_ago = (datetime.today() - last_trigger_date).days
        status = "MONITORING"
        color  = "#fd7e14"
        emoji  = "👀"
        action = (f"Trigger fired {days_ago}d ago (VIX was {float(last_trigger_row['vix']):.1f}). "
                  f"Waiting for VIX ≤ 25. Currently {vix_now:.1f}. "
                  f"Window expires {(last_trigger_date + timedelta(days=SIGNAL_LOOKBACK_DAYS)).strftime('%b %d')}.")
    elif has_recent_trigger and vix_now <= T1_VIX_MAX and not (T1_DD_MIN <= dd_now <= T1_DD_MAX):
        status = "WATCH — DRAWDOWN OUTSIDE BAND"
        color  = "#ffc107"
        emoji  = "⚠️"
        action = (f"Trigger fired recently but SPY drawdown ({dd_now*100:.1f}%) is outside the "
                  f"5–15% entry band. Monitor daily.")
    else:
        status = "ALL CLEAR"
        color  = "#007bff"
        emoji  = "📊"
        action = "No active signals. Strategy is watching for the next crash entry opportunity."

    return {
        "status"            : status,
        "color"             : color,
        "emoji"             : emoji,
        "action"            : action,
        "fed_ok"            : fed_ok,
        "n_triggers"        : n_trig,
        "T1_flag"           : bool(today_row["T1"]),
        "T2_flag"           : bool(today_row["T2"]),
        "T3_flag"           : bool(today_row["T3"]),
        "has_recent_trigger": has_recent_trigger,
        "last_trigger_date" : last_trigger_date,
        "t1_entry_date"     : t1_entry_date,
        "vix"               : vix_now,
        "vvix"              : float(today_row["vvix"]),
        "spy"               : float(today_row["close"]),
        "drawdown_pct"      : float(today_row["drawdown"]) * 100,
        "daily_range_pct"   : float(today_row["daily_range"]) * 100,
        "dgs2"              : float(today_row["dgs2"]) if not pd.isna(today_row["dgs2"]) else None,
        "dgs2_9m_chg"       : float(today_row["dgs2_9m_chg"]) if not pd.isna(today_row["dgs2_9m_chg"]) else None,
        "ma50"              : float(today_row["ma50"]),
        "spy_vs_ma50_pct"   : (float(today_row["close"]) / float(today_row["ma50"]) - 1) * 100,
        "vix_iv_rank"       : float(today_row["vix_iv_rank"]) * 100,
        "vix_spike_pct"     : float(today_row["vix_spike"]) * 100 if not pd.isna(today_row["vix_spike"]) else 0,
        "two_consec_red"    : bool(today_row["two_consec_red"]),
        "high_252"          : float(today_row["high_252"]),
    }


# ── Position mark-to-market ───────────────────────────────────────────────────

def load_positions():
    if not POSITIONS_FILE.exists():
        return []
    with open(POSITIONS_FILE) as f:
        return json.load(f)

def mark_positions(positions, spy_price, vix_now):
    """Calculate current BS value and P&L for each open position."""
    today  = datetime.today()
    marked = []
    for p in positions:
        expiry = datetime.strptime(p["expiry_date"], "%Y-%m-%d")
        T      = max((expiry - today).days / 365.0, 1/365.0)
        iv     = max(vix_now / 100.0, 0.10)
        K      = p["K_itm"]
        cost   = p["entry_cost"]
        curr   = bs_call(spy_price, K, T, RISK_FREE, iv)
        pnl    = (curr / cost - 1) * 100
        delta  = bs_delta(spy_price, K, T, RISK_FREE, iv)
        intrinsic = max(spy_price - K, 0.0)
        days_left = (expiry - today).days
        near_exit  = pnl >= EXIT_TARGET * 100 * 0.80   # within 80% of target
        near_expiry= days_left < 90

        marked.append({**p,
            "current_price": round(curr, 2),
            "pnl_pct"      : round(pnl, 1),
            "delta_now"    : round(delta, 3),
            "intrinsic"    : round(intrinsic, 2),
            "days_left"    : days_left,
            "near_exit"    : near_exit,
            "near_expiry"  : near_expiry,
        })
    return marked


# ── HTML email builder ────────────────────────────────────────────────────────

def _row(label, value, status="", highlight=False):
    bg = ' style="background:#1a2332"' if highlight else ""
    return (f'<tr{bg}><td style="padding:6px 12px;color:#8b949e;font-size:13px">{label}</td>'
            f'<td style="padding:6px 12px;color:#e6edf3;font-weight:600;font-size:13px">{value}</td>'
            f'<td style="padding:6px 12px;font-size:13px">{status}</td></tr>')

def _check(val): return "✅" if val else "❌"

def build_email(state, positions, report_date):
    s = state
    bg    = "#0d1117"; card  = "#161b22"; border = "#30363d"
    text  = "#e6edf3"; muted = "#8b949e"; green = "#3fb950"
    red   = "#f85149"; gold  = "#d29922"

    # ── Subject ────────────────────────────────────────────────────────────────
    subject = (f"{s['emoji']} [LEAPS] {s['status']} — "
               f"SPY ${s['spy']:.0f}  VIX {s['vix']:.1f}  |  "
               f"{report_date.strftime('%a %b %d, %Y')}")

    # ── Trigger rows ────────────────────────────────────────────────────────────
    t1_details = (f"DD {s['drawdown_pct']:.1f}%  |  "
                  f"{'2 red wks ✅' if s['two_consec_red'] else '2 red wks ❌'}  |  "
                  f"{'below MA50 ✅' if s['spy_vs_ma50_pct'] < 0 else 'above MA50 ❌'}  |  "
                  f"range {s['daily_range_pct']:.1f}%")
    t2_details = (f"VIX {s['vix']:.1f}  |  "
                  f"spike {s['vix_spike_pct']:.0f}%  |  "
                  f"IV rank {s['vix_iv_rank']:.0f}%")
    t3_details = f"VVIX {s['vvix']:.1f} (threshold >{VVIX_THRESH})"

    fed_chg    = f"{s['dgs2_9m_chg']:+.2f}%" if s["dgs2_9m_chg"] is not None else "n/a"
    fed_level  = f"{s['dgs2']:.2f}%" if s["dgs2"] is not None else "n/a"
    fed_status = (f'<span style="color:{red}">🚫 BLOCKING ({fed_chg} over 9mo)</span>'
                  if not s["fed_ok"]
                  else f'<span style="color:{green}">✅ CLEAR ({fed_chg} over 9mo)</span>')

    vix_vs_t1 = "✅ READY" if s['vix'] <= T1_VIX_MAX else f"🔴 Need -{s['vix'] - T1_VIX_MAX:.1f} more"
    vix_vs_t2 = "✅ READY" if s['vix'] <= T2_VIX_MAX else f"🔴 Need -{s['vix'] - T2_VIX_MAX:.1f} more"

    # ── Position rows ───────────────────────────────────────────────────────────
    pos_rows = ""
    pos_alert = ""
    if positions:
        for p in positions:
            pnl_color = green if p["pnl_pct"] > 0 else red
            alert_badge = ""
            if p["near_exit"]:
                alert_badge = f'<span style="background:#1a4731;color:{green};padding:2px 8px;border-radius:4px;font-size:11px;margin-left:6px">⚡ NEAR 150% EXIT</span>'
                pos_alert += f'<p style="color:{gold};margin:4px 0">⚡ Position <b>{p.get("id","?")}</b> is at {p["pnl_pct"]:.1f}% — approaching 150% exit target.</p>'
            if p["near_expiry"]:
                alert_badge += f'<span style="background:#3a1a1a;color:{red};padding:2px 8px;border-radius:4px;font-size:11px;margin-left:6px">⏰ &lt;90d DTE</span>'
            pos_rows += f"""
            <tr>
              <td style="padding:8px 12px;color:{text};font-size:12px">{p.get('id','—')}</td>
              <td style="padding:8px 12px;color:{muted};font-size:12px">{p.get('tranche','—')}</td>
              <td style="padding:8px 12px;color:{text};font-size:12px">{p.get('entry_date','—')}</td>
              <td style="padding:8px 12px;color:{text};font-size:12px">${p.get('entry_spy','—')}</td>
              <td style="padding:8px 12px;color:{text};font-size:12px">${p.get('K_itm','—')}</td>
              <td style="padding:8px 12px;color:{text};font-size:12px">${p.get('entry_cost','—')}</td>
              <td style="padding:8px 12px;color:{text};font-size:12px">${p['current_price']}</td>
              <td style="padding:8px 12px;color:{pnl_color};font-weight:700;font-size:13px">{p['pnl_pct']:+.1f}%{alert_badge}</td>
              <td style="padding:8px 12px;color:{muted};font-size:12px">Δ{p['delta_now']:.2f}</td>
              <td style="padding:8px 12px;color:{muted};font-size:12px">{p['days_left']}d</td>
              <td style="padding:8px 12px;color:{muted};font-size:12px">{p.get('expiry_date','—')}</td>
            </tr>"""
    else:
        pos_rows = f'<tr><td colspan="11" style="padding:16px;text-align:center;color:{muted}">No open positions tracked. Add entries to positions.json to track P&amp;L.</td></tr>'

    # ── Main HTML ───────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:{bg};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:900px;margin:0 auto;padding:24px 16px">

  <!-- Header -->
  <div style="background:{card};border:1px solid {border};border-radius:10px;padding:20px 24px;margin-bottom:16px">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <div>
        <h1 style="margin:0;color:{text};font-size:20px;font-weight:700">$SPY Crash Entry LEAP System</h1>
        <p style="margin:4px 0 0;color:{muted};font-size:13px">
          Daily Report — {report_date.strftime('%A, %B %d, %Y')} &nbsp;|&nbsp; After-Close Scan
        </p>
      </div>
      <div style="text-align:right">
        <span style="font-size:13px;color:{muted}">SPY</span>
        <span style="display:block;font-size:24px;font-weight:700;color:{text}">${s['spy']:.2f}</span>
      </div>
    </div>
  </div>

  <!-- Signal Banner -->
  <div style="background:{s['color']}22;border:2px solid {s['color']};border-radius:10px;
              padding:18px 24px;margin-bottom:16px;text-align:center">
    <div style="font-size:28px;margin-bottom:4px">{s['emoji']}</div>
    <div style="font-size:22px;font-weight:700;color:{s['color']};letter-spacing:1px">{s['status']}</div>
    <div style="font-size:14px;color:{text};margin-top:8px;line-height:1.5">{s['action']}</div>
  </div>

  {f'<div style="background:#3a1c0044;border:1px solid {gold};border-radius:8px;padding:12px 16px;margin-bottom:16px">{pos_alert}</div>' if pos_alert else ''}

  <!-- Two-column: Market Snapshot + Trigger Scorecard -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">

    <!-- Market Snapshot -->
    <div style="background:{card};border:1px solid {border};border-radius:10px;padding:0;overflow:hidden">
      <div style="background:#1c2433;padding:10px 16px;border-bottom:1px solid {border}">
        <h3 style="margin:0;color:{text};font-size:14px;font-weight:600">📈 Market Snapshot</h3>
      </div>
      <table style="width:100%;border-collapse:collapse">
        {_row("SPY Close",      f"${s['spy']:.2f}")}
        {_row("SPY vs 252d High", f"{s['drawdown_pct']:.2f}%", "🔴 In drawdown" if s['drawdown_pct'] < -1 else "🟢 Near high")}
        {_row("SPY vs 50d MA",  f"{s['spy_vs_ma50_pct']:+.2f}%", "Below MA" if s['spy_vs_ma50_pct'] < 0 else "Above MA")}
        {_row("Daily Range",    f"{s['daily_range_pct']:.2f}%", "⚠️ Elevated" if s['daily_range_pct'] > 2 else "Normal")}
        {_row("2 Red Weeks",    "Yes ✅" if s['two_consec_red'] else "No ❌")}
        {_row("252d High",      f"${s['high_252']:.2f}")}
      </table>
    </div>

    <!-- Trigger Scorecard -->
    <div style="background:{card};border:1px solid {border};border-radius:10px;padding:0;overflow:hidden">
      <div style="background:#1c2433;padding:10px 16px;border-bottom:1px solid {border}">
        <h3 style="margin:0;color:{text};font-size:14px;font-weight:600">🎯 Trigger Scorecard  ({s['n_triggers']}/3)</h3>
      </div>
      <table style="width:100%;border-collapse:collapse">
        {_row("T1 Price Damage", _check(s['T1_flag']), t1_details, s['T1_flag'])}
        {_row("T2 Volatility",   _check(s['T2_flag']), t2_details, s['T2_flag'])}
        {_row(f"T3 VVIX >{VVIX_THRESH}", _check(s['T3_flag']), t3_details, s['T3_flag'])}
        {_row("Triggers needed", "2 of 3", f"{'🟢 MET' if s['n_triggers'] >= 2 else '🔴 NOT MET'}", s['n_triggers'] >= 2)}
      </table>
    </div>
  </div>

  <!-- Fed Filter + VIX Watch -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">

    <!-- Fed Filter -->
    <div style="background:{card};border:1px solid {border};border-radius:10px;padding:0;overflow:hidden">
      <div style="background:#1c2433;padding:10px 16px;border-bottom:1px solid {border}">
        <h3 style="margin:0;color:{text};font-size:14px;font-weight:600">🏦 Fed Rate Filter</h3>
      </div>
      <table style="width:100%;border-collapse:collapse">
        {_row("2yr Treasury (DGS2)", f"{fed_level}")}
        {_row("9-month change",     f"{fed_chg}", "Threshold: ≥ +0.50%")}
        {_row("Filter status",      "", fed_status)}
      </table>
    </div>

    <!-- VIX Watch -->
    <div style="background:{card};border:1px solid {border};border-radius:10px;padding:0;overflow:hidden">
      <div style="background:#1c2433;padding:10px 16px;border-bottom:1px solid {border}">
        <h3 style="margin:0;color:{text};font-size:14px;font-weight:600">📉 VIX Entry Gates</h3>
      </div>
      <table style="width:100%;border-collapse:collapse">
        {_row("VIX now",            f"{s['vix']:.2f}")}
        {_row("VVIX now",           f"{s['vvix']:.1f}", f"{'🔴 SPIKE' if s['vvix'] > VVIX_THRESH else '🟢 Normal'} (threshold {VVIX_THRESH})")}
        {_row("T1 entry (half size)", f"VIX <= {T1_VIX_MAX}", vix_vs_t1)}
        {_row("T2 entry (full)",      f"VIX <= {T2_VIX_MAX}", vix_vs_t2)}
        {_row("VIX IV Rank",        f"{s['vix_iv_rank']:.0f}%", "Elevated" if s['vix_iv_rank'] > 50 else "Normal")}
      </table>
    </div>
  </div>

  <!-- Open Positions -->
  <div style="background:{card};border:1px solid {border};border-radius:10px;margin-bottom:16px;overflow:hidden">
    <div style="background:#1c2433;padding:10px 16px;border-bottom:1px solid {border}">
      <h3 style="margin:0;color:{text};font-size:14px;font-weight:600">
        💼 Open Positions  ({len(positions)} tracked)
        <span style="font-size:11px;color:{muted};font-weight:400;margin-left:12px">
          Exit target: +150% | Deep ITM LEAP (Δ~0.775) | 18-mo DTE
        </span>
      </h3>
    </div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;min-width:700px">
        <thead>
          <tr style="background:#1c2433">
            {''.join(f'<th style="padding:8px 12px;color:{muted};font-size:11px;font-weight:600;text-align:left;text-transform:uppercase;letter-spacing:0.5px">{h}</th>' for h in ['ID','Tranche','Entry','SPY In','Strike','Cost','Current','P&L','Delta','DTE','Expiry'])}
          </tr>
        </thead>
        <tbody>{pos_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Strategy Quick Reference -->
  <div style="background:{card};border:1px solid {border};border-radius:10px;padding:16px 20px;margin-bottom:16px">
    <h3 style="margin:0 0 10px;color:{text};font-size:14px;font-weight:600">📋 Strategy Rules Quick Reference</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div style="font-size:12px;color:{muted};line-height:1.8">
        <b style="color:{text}">Entry:</b> 2-of-3 triggers + VIX ≤ 25<br>
        <b style="color:{text}">T1 size:</b> Half position (VIX ≤ 25)<br>
        <b style="color:{text}">T2 size:</b> Full position (VIX ≤ 20, within 120d of T1)<br>
        <b style="color:{text}">Position:</b> Deep ITM call Δ~0.775, 18-month DTE
      </div>
      <div style="font-size:12px;color:{muted};line-height:1.8">
        <b style="color:{text}">Exit:</b> 150% gain OR 18-month expiry<br>
        <b style="color:{text}">Fed block:</b> 2yr yield +0.50% over 9mo<br>
        <b style="color:{text}">T3 proxy:</b> VVIX &gt; 120 (put-buying spike)<br>
        <b style="color:{text}">10yr backtest:</b> 3 trades (all winners) avg +132%
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div style="text-align:center;padding:12px">
    <p style="margin:0;color:{muted};font-size:11px">
      Generated by $SPY LEAP Scanner v5 · {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC ·
      Data: Yahoo Finance + FRED · BS pricing (VIX as IV proxy)
    </p>
    <p style="margin:4px 0 0;color:{muted};font-size:11px">
      ⚠️ Not financial advice. Past backtest results do not guarantee future performance.
    </p>
  </div>

</div>
</body></html>"""

    return subject, html


# ── Email sender ──────────────────────────────────────────────────────────────

def send_email(subject, html, to_addr, from_addr, app_password):
    if not app_password:
        print("⚠  No GMAIL_APP_PASSWORD set — printing email to stdout instead")
        print(f"\nSUBJECT: {subject}")
        print("(HTML body omitted — set GMAIL_APP_PASSWORD to send)")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_addr, app_password)
        server.sendmail(from_addr, to_addr, msg.as_string())
    print(f"✅ Email sent to {to_addr}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    report_date = date.today()
    print(f"\n{'='*60}")
    print(f"  $SPY LEAP Daily Scanner — {report_date}")
    print(f"{'='*60}\n")

    data       = download_data()
    data       = compute_signals(data)
    state      = determine_signal_state(data)
    positions  = load_positions()
    marked_pos = mark_positions(positions, state["spy"], state["vix"])

    print(f"Signal status : {state['emoji']}  {state['status']}")
    print(f"Triggers      : T1={state['T1_flag']}  T2={state['T2_flag']}  T3={state['T3_flag']}  ({state['n_triggers']}/3)")
    print(f"Fed filter    : {'BLOCKING' if not state['fed_ok'] else 'CLEAR'}  (2yr Δ {state['dgs2_9m_chg']:+.2f}% over 9mo)" if state['dgs2_9m_chg'] else "")
    print(f"SPY           : ${state['spy']:.2f}  ({state['drawdown_pct']:.1f}% from high)")
    print(f"VIX           : {state['vix']:.1f}  |  VVIX: {state['vvix']:.1f}")
    if marked_pos:
        print(f"Positions     : {len(marked_pos)} open")
        for p in marked_pos:
            flag = " ⚡NEAR EXIT" if p["near_exit"] else ""
            print(f"  {p.get('id','?')} ({p.get('tranche','?')}): {p['pnl_pct']:+.1f}%{flag}")

    subject, html = build_email(state, marked_pos, report_date)
    send_email(subject, html, TO_EMAIL, FROM_EMAIL, APP_PASSWORD)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
