#!/usr/bin/env python3
"""
$SPY Crash Entry LEAP Trigger System — v5
Based on v4 (tiered entry T1/T2). Three changes:

  1. Early exit at 150% gain  (was 300%)
  2. T3 = equity Put/Call ratio > 0.9  (replaces VIX>30 proxy)
     → fetched from CBOE; falls back to VIX>30 if unavailable
  3. Fed filter (hard block): 2-year Treasury yield 9-month change ≥ +0.50%
     → catches forward-looking rate expectations before Fed actually hikes
     → blocks entries during rate-hike regimes; allows entries during cuts/pauses

All other conditions identical to v4:
  T1 entry (half size): VIX ≤ 25, SPY 5–15% below 252d high
  T2 entry (full size): VIX ≤ 20, within 120 days of T1, SPY ≥ 2% below 252d high
  Position: deep ITM call, delta ~0.775, 18-month DTE (540 days)
  Need 2-of-3 triggers: T1 Price Damage | T2 Volatility | T3 P/C Spike
"""

import warnings, io
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import yfinance as yf


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def bs_call(S, K, T, r, σ):
    if T < 1e-6: return max(0.0, S - K)
    if σ < 1e-6: return max(0.0, S - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * np.sqrt(T))
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - σ * np.sqrt(T)))

def bs_delta_call(S, K, T, r, σ):
    if T < 1e-6: return 1.0 if S >= K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * np.sqrt(T))
    return float(norm.cdf(d1))

def strike_for_delta(S, T, r, σ, target_Δ):
    d1t = norm.ppf(target_Δ)
    return float(S * np.exp((r + 0.5 * σ**2) * T - d1t * σ * np.sqrt(T)))


# ── Parameters ────────────────────────────────────────────────────────────────

BACKTEST_YEARS  = 10
WARMUP_DAYS     = 400
DTE_DAYS        = 540
ITM_DELTA       = 0.775
RISK_FREE       = 0.05
COOLDOWN_DAYS   = 30

T1_VIX_MAX      = 25;  T1_WATCH_DAYS = 30;  T1_DD_MIN = -0.15; T1_DD_MAX = -0.05
T2_VIX_MAX      = 20;  T2_WATCH_DAYS = 120; T2_DD_MAX = -0.02

TARGET_MULT     = 2.5       # ← 150% gain exit (was 4.0 / 300%)

VVIX_THRESH     = 120       # VVIX threshold for T3 (fear/put-buying spike proxy)
FED_YIELD_CHG   = 0.50      # 2yr yield 9-month change threshold for Fed block
FED_LOOKBACK    = 270       # 9 months in calendar days


# ── Data download ─────────────────────────────────────────────────────────────

TODAY          = datetime.today()
BT_START       = TODAY - timedelta(days=BACKTEST_YEARS * 365)
DOWNLOAD_START = BT_START  - timedelta(days=WARMUP_DAYS)

print(f"Downloading SPY and VIX …")
spy_raw = yf.download("SPY",  start=DOWNLOAD_START, end=TODAY, auto_adjust=True, progress=False)
vix_raw = yf.download("^VIX", start=DOWNLOAD_START, end=TODAY, auto_adjust=True, progress=False)
for _d in (spy_raw, vix_raw):
    if isinstance(_d.columns, pd.MultiIndex): _d.columns = _d.columns.get_level_values(0)

data = pd.DataFrame({
    "open": spy_raw["Open"], "high": spy_raw["High"],
    "low":  spy_raw["Low"],  "close": spy_raw["Close"],
    "volume": spy_raw["Volume"], "vix": vix_raw["Close"],
}).dropna()
data.index = pd.to_datetime(data.index).tz_localize(None)


# ── VVIX (T3 — fear/put-buying spike) ────────────────────────────────────────
# VVIX = volatility of VIX; spikes when traders panic-buy options (≈ P/C spike proxy)
# CBOE equity P/C ratio unavailable via free API; VVIX is the best available proxy.

print("Downloading VVIX (options fear gauge) …")
vvix_raw = yf.download("^VVIX", start=DOWNLOAD_START, end=TODAY,
                        auto_adjust=True, progress=False)
if isinstance(vvix_raw.columns, pd.MultiIndex):
    vvix_raw.columns = vvix_raw.columns.get_level_values(0)

if len(vvix_raw) > 0:
    data["vvix"] = vvix_raw["Close"].reindex(data.index, method="ffill")
    pc_source    = f"VVIX > {VVIX_THRESH} (put-buying spike proxy; CBOE P/C unavailable)"
    pc_available = True
    print(f"  VVIX: {len(vvix_raw)} rows ✓  (range {vvix_raw['Close'].min():.0f}–{vvix_raw['Close'].max():.0f})")
else:
    data["vvix"] = 100.0
    pc_source    = "VVIX unavailable — T3 disabled"
    pc_available = False

data["vvix"] = pd.to_numeric(data["vvix"], errors="coerce").ffill()


# ── 2-year Treasury yield (Fed filter) ───────────────────────────────────────

def fetch_2yr_yield():
    """Fetch 2yr Treasury yield. FRED DGS2 is the primary source (forward-looking).
    Unlike ^IRX (3-month T-bill) which tracks FEDFUNDS retroactively,
    the 2yr yield prices in EXPECTED hikes — correctly signals Feb/Mar 2022 before hiking started."""
    from pandas_datareader import data as pdr
    df = pdr.get_data_fred("DGS2", start=DOWNLOAD_START, end=TODAY)
    df.columns = ["dgs2"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"  2yr yield (DGS2): {len(df)} rows from FRED ✓  "
          f"range {df['dgs2'].min():.2f}–{df['dgs2'].max():.2f}%")
    return df

print("Downloading 2-year Treasury yield …")
yr2_df = fetch_2yr_yield()
if yr2_df is not None:
    data["dgs2"] = yr2_df["dgs2"].reindex(data.index, method="ffill")
    data["dgs2_9m_chg"] = data["dgs2"] - data["dgs2"].shift(FED_LOOKBACK)
    data["fed_block"]   = data["dgs2_9m_chg"] >= FED_YIELD_CHG
    fed_source = "2yr Treasury yield 9-month Δ ≥ +0.50%"
else:
    print("  ⚠ 2yr yield unavailable — Fed filter disabled")
    data["fed_block"] = False
    fed_source = "DISABLED (data unavailable)"

data["fed_block"] = data["fed_block"].fillna(False)


# ── Indicators ────────────────────────────────────────────────────────────────

data["high_252"]    = data["close"].rolling(252).max()
data["drawdown"]    = (data["close"] - data["high_252"]) / data["high_252"]
data["ma50"]        = data["close"].rolling(50).mean()
data["vol_ma20"]    = data["volume"].rolling(20).mean()
data["daily_range"] = (data["high"] - data["low"]) / data["close"].shift(1)
data["vix_10d_min"] = data["vix"].rolling(10).min()
data["vix_spike"]   = (data["vix"] - data["vix_10d_min"]) / data["vix_10d_min"]
data["vix_iv_rank"] = data["vix"].rolling(252).apply(
    lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
    raw=True)

wk_close = data["close"].resample("W-FRI").last()
wk_open  = data["open"].resample("W-FRI").first()
two_red  = (wk_close < wk_open).shift(1) & (wk_close < wk_open).shift(2)
data["two_consec_red"] = two_red.reindex(data.index, method="ffill").fillna(False)

data.dropna(subset=["high_252","ma50","vix_iv_rank"], inplace=True)
data["vvix"] = data["vvix"].ffill()


# ── Triggers ──────────────────────────────────────────────────────────────────

SIG_T1 = (
    (data["drawdown"] < -0.08) & (data["drawdown"] > -0.15) &
    data["two_consec_red"] &
    (data["close"] < data["ma50"]) & (data["volume"] > data["vol_ma20"]) &
    (data["daily_range"] > 0.02)
)
SIG_T2 = (
    (data["vix"] > 22) & (data["vix_spike"] > 0.20) &
    (data["daily_range"] > 0.02) & (data["vix_iv_rank"] > 0.50)
)
# T3: VVIX fear spike (proxy for put/call ratio spike)
SIG_T3 = data["vvix"] > VVIX_THRESH

data["T1_flag"]    = SIG_T1
data["T2_flag"]    = SIG_T2
data["T3_flag"]    = SIG_T3
data["n_triggers"] = SIG_T1.astype(int) + SIG_T2.astype(int) + SIG_T3.astype(int)
data["signal"]     = data["n_triggers"] >= 2


# ── Simulate one LEAP ────────────────────────────────────────────────────────

def sim_leap(entry_dt, S0, iv0, full_data):
    T0    = DTE_DAYS / 365.0
    K_itm = min(round(strike_for_delta(S0, T0, RISK_FREE, iv0, ITM_DELTA)), round(S0) - 1)
    p0    = bs_call(S0, K_itm, T0, RISK_FREE, iv0)
    if p0 < 0.01: return None

    intrinsic    = max(S0 - K_itm, 0.0)
    expiry_dt    = entry_dt + timedelta(days=DTE_DAYS)
    future       = full_data[(full_data.index > entry_dt) & (full_data.index <= expiry_dt)]

    exit_dt, exit_reason = None, "expiration"
    for fdt, frow in future.iterrows():
        Sf  = float(frow["close"])
        ivf = max(float(frow["vix"]) / 100.0, 0.10)
        Tf  = max((expiry_dt - fdt).days / 365.0, 1/365.0)
        if bs_call(Sf, K_itm, Tf, RISK_FREE, ivf) >= p0 * TARGET_MULT:
            exit_dt, exit_reason = fdt, f"{int((TARGET_MULT-1)*100)}%+ target hit"
            break

    if exit_dt is None:
        if len(future) == 0: return None
        exit_dt = future.index[-1]
        days_left = (expiry_dt - exit_dt).days
        exit_reason = "expiration" if days_left <= 5 else "open / still live"

    exit_row = full_data.loc[exit_dt]
    Sf_e     = float(exit_row["close"])
    ivf_e    = max(float(exit_row["vix"]) / 100.0, 0.10)
    Tf_e     = max((expiry_dt - exit_dt).days / 365.0, 1/365.0)
    p_exit   = bs_call(Sf_e, K_itm, Tf_e, RISK_FREE, ivf_e)

    return dict(
        entry_dt=entry_dt, exit_dt=exit_dt, days_held=(exit_dt-entry_dt).days,
        exit_reason=exit_reason,
        S_entry=round(S0,2), S_exit=round(Sf_e,2),
        spy_ret_pct=round((Sf_e/S0-1)*100,1),
        vix_entry=round(iv0*100,1), K_itm=K_itm,
        delta=round(bs_delta_call(S0,K_itm,T0,RISK_FREE,iv0),3),
        cost=round(p0,2), intrinsic=round(intrinsic,2),
        time_prem=round(p0-intrinsic,2),
        time_prem_pct=round((p0-intrinsic)/p0*100,1),
        exit_price=round(p_exit,2),
        pnl_pct=round((p_exit/p0-1)*100,1),
    )


# ── Episode detection ─────────────────────────────────────────────────────────

bt          = data[data.index >= BT_START].copy()
episodes    = []
trigger_log = []
pending     = []           # (trigger_dt, expiry_dt, vix_at_trigger)
last_ep_dt  = None
fed_blocked = []           # log of dates blocked by Fed filter

for dt in bt.index:
    row         = bt.loc[dt]
    in_cooldown = last_ep_dt is not None and (dt - last_ep_dt).days < COOLDOWN_DAYS
    pending     = [(td, ed, vt) for td, ed, vt in pending if dt <= ed]

    if row["signal"]:
        trigger_log.append({
            "date"     : dt.strftime("%Y-%m-%d"),
            "vix"      : round(float(row["vix"]), 1),
            "vvix"     : round(float(row["vvix"]), 1) if not pd.isna(row["vvix"]) else "–",
            "spy"      : round(float(row["close"]), 2),
            "dd%"      : round(float(row["drawdown"])*100, 1),
            "n_trig"   : int(row["n_triggers"]),
            "T1"       : bool(row["T1_flag"]),
            "T2"       : bool(row["T2_flag"]),
            "T3"       : bool(row["T3_flag"]),
            "fed_block": bool(row["fed_block"]),
        })

    if row["signal"] and not in_cooldown:
        if bool(row["fed_block"]):
            fed_blocked.append(dt)
            continue   # hard block — skip entirely

        vix_now, dd_now = float(row["vix"]), float(row["drawdown"])

        if vix_now <= T1_VIX_MAX and T1_DD_MIN <= dd_now <= T1_DD_MAX:
            leap = sim_leap(dt, float(row["close"]), max(vix_now/100, 0.10), data)
            if leap:
                episodes.append({"trigger_dt": dt, "trigger_vix": vix_now,
                                  "trigger_ntrig": int(row["n_triggers"]),
                                  "t1": leap, "t1_type": "immediate", "t1_lag": 0,
                                  "t2": None})
                last_ep_dt = dt; pending = []
                continue
        elif vix_now > T1_VIX_MAX:
            exp = dt + timedelta(days=T1_WATCH_DAYS)
            if not any(abs((dt-td).days) <= 5 for td,_,_ in pending):
                pending.append((dt, exp, vix_now))

    if pending and not in_cooldown and not bool(row["fed_block"]):
        vix_now, dd_now = float(row["vix"]), float(row["drawdown"])
        if vix_now <= T1_VIX_MAX and T1_DD_MIN <= dd_now <= T1_DD_MAX:
            trig_dt, _, vix_trig = pending[0]
            leap = sim_leap(dt, float(row["close"]), max(vix_now/100, 0.10), data)
            if leap:
                episodes.append({"trigger_dt": trig_dt, "trigger_vix": vix_trig,
                                  "trigger_ntrig": int(bt.loc[trig_dt,"n_triggers"]),
                                  "t1": leap, "t1_type": "vix_cooled",
                                  "t1_lag": (dt - trig_dt).days,
                                  "t2": None})
                last_ep_dt = dt; pending = []


# ── Add Tranche 2 ─────────────────────────────────────────────────────────────

for ep in episodes:
    t1_dt = ep["t1"]["entry_dt"]
    win   = data[(data.index > t1_dt) &
                 (data.index <= t1_dt + timedelta(days=T2_WATCH_DAYS))]
    for dt, row in win.iterrows():
        vix_now = float(row["vix"])
        if vix_now <= T2_VIX_MAX and float(row["drawdown"]) <= T2_DD_MAX:
            leap = sim_leap(dt, float(row["close"]), max(vix_now/100, 0.10), data)
            if leap:
                ep["t2"] = leap
                ep["t2_lag"] = (dt - t1_dt).days
            break


# ── Blended P&L ───────────────────────────────────────────────────────────────

for ep in episodes:
    t1p = ep["t1"]["pnl_pct"]
    if ep["t2"]:
        ep["blended_pnl"] = round((0.5*t1p + 1.0*ep["t2"]["pnl_pct"]) / 1.5, 1)
        ep["total_units"]  = 1.5
    else:
        ep["blended_pnl"] = t1p
        ep["total_units"]  = 0.5


# ── Print ─────────────────────────────────────────────────────────────────────

sep = "=" * 130
tdf = pd.DataFrame(trigger_log)

print(f"\n{'─'*60}")
print(f"  T3 source : {pc_source}")
print(f"  Fed filter: {fed_source}")
print(f"  Exit target: {int((TARGET_MULT-1)*100)}% gain  (was 300% in v4)")
print(f"{'─'*60}")
print(f"\nBacktest: {BT_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}")
print(f"Signal days  : {bt['signal'].sum()}")
print(f"Fed-blocked  : {len(fed_blocked)}  ({', '.join(d.strftime('%Y-%m') for d in fed_blocked[:8])}{'…' if len(fed_blocked)>8 else ''})")
print(f"Episodes     : {len(episodes)}  "
      f"| T2 fired: {sum(1 for e in episodes if e['t2'])}  "
      f"| T1-only: {sum(1 for e in episodes if not e['t2'])}")

print(f"\n{sep}\nTRIGGER LOG\n{sep}")
print(tdf.to_string(index=False))

print(f"\n{sep}\nEPISODE LOG\n{sep}")

rows = []
for i, ep in enumerate(episodes, 1):
    t1, t2 = ep["t1"], ep["t2"]
    rows.append({
        "#"          : i,
        "trigger"    : ep["trigger_dt"].strftime("%Y-%m-%d"),
        "tVIX"       : ep["trigger_vix"],
        "nT"         : ep["trigger_ntrig"],
        "T1_entry"   : t1["entry_dt"].strftime("%Y-%m-%d"),
        "T1_lag"     : ep["t1_lag"],
        "T1_VIX"     : t1["vix_entry"],
        "T1_SPY"     : t1["S_entry"],
        "T1_K"       : t1["K_itm"],
        "T1_cost"    : t1["cost"],
        "T1_exit"    : t1["exit_dt"].strftime("%Y-%m-%d"),
        "T1_reason"  : t1["exit_reason"],
        "T1_SPY_exit": t1["S_exit"],
        "T1_pnl%"    : t1["pnl_pct"],
        "T2_entry"   : t2["entry_dt"].strftime("%Y-%m-%d") if t2 else "—",
        "T2_lag"     : ep.get("t2_lag","—"),
        "T2_VIX"     : t2["vix_entry"] if t2 else "—",
        "T2_SPY"     : t2["S_entry"] if t2 else "—",
        "T2_K"       : t2["K_itm"] if t2 else "—",
        "T2_cost"    : t2["cost"] if t2 else "—",
        "T2_exit"    : t2["exit_dt"].strftime("%Y-%m-%d") if t2 else "—",
        "T2_reason"  : t2["exit_reason"] if t2 else "—",
        "T2_pnl%"    : t2["pnl_pct"] if t2 else "—",
        "units"      : ep["total_units"],
        "blended%"   : ep["blended_pnl"],
    })

ep_df = pd.DataFrame(rows)
with pd.option_context("display.max_columns", None, "display.width", 260,
                       "display.float_format", "{:.1f}".format):
    print(ep_df.to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────

if episodes:
    bl   = [e["blended_pnl"] for e in episodes]
    t1p  = [e["t1"]["pnl_pct"] for e in episodes]
    t2p  = [e["t2"]["pnl_pct"] for e in episodes if e["t2"]]
    spy  = [e["t1"]["spy_ret_pct"] for e in episodes]
    n    = len(episodes)
    nw   = sum(1 for v in bl if v > 0)
    n_eh = sum(1 for e in episodes if "target" in e["t1"]["exit_reason"] or
               (e["t2"] and "target" in e["t2"]["exit_reason"]))

    print(f"\n{sep}\nSUMMARY STATISTICS\n{sep}")
    print(f"  Episodes total            : {n}")
    print(f"  Fed-blocked episodes      : {len(fed_blocked)}")
    print(f"  T2 also fired             : {sum(1 for e in episodes if e['t2'])} / {n}")
    print(f"  Early exits (150% target) : {n_eh}")
    print()
    print(f"  ── Blended P&L ─────────────────────────────────────────────────")
    print(f"  Win rate                  : {nw}/{n}  ({nw/n*100:.0f}%)")
    print(f"  Avg return                : {np.mean(bl):+.1f}%")
    print(f"  Median return             : {np.median(bl):+.1f}%")
    print(f"  Best                      : {max(bl):+.1f}%")
    print(f"  Worst                     : {min(bl):+.1f}%")
    print()
    print(f"  ── T1 standalone ───────────────────────────────────────────────")
    print(f"  Avg T1 return             : {np.mean(t1p):+.1f}%")
    print(f"  Median T1 return          : {np.median(t1p):+.1f}%")
    if t2p:
        print(f"\n  ── T2 standalone ({len(t2p)} episodes) ──────────────────────────────")
        print(f"  Avg T2 return             : {np.mean(t2p):+.1f}%")
        print(f"  Median T2 return          : {np.median(t2p):+.1f}%")
    print()
    print(f"  Avg SPY return (T1 period): {np.mean(spy):+.1f}%")
    lev = np.mean(bl) / np.mean(spy) if np.mean(spy) != 0 else float('nan')
    print(f"  Avg leverage              : {lev:.2f}x")
    print(f"\n  Avg T1 VIX at entry       : {np.mean([e['t1']['vix_entry'] for e in episodes]):.1f}")
    if t2p:
        print(f"  Avg T2 VIX at entry       : {np.mean([e['t2']['vix_entry'] for e in episodes if e['t2']]):.1f}")


# ── Chart ─────────────────────────────────────────────────────────────────────

DARK   = "#0d1117"; GRID  = "#21262d"; TEXT  = "#c9d1d9"
GREEN  = "#3fb950"; RED   = "#f85149"; BLUE  = "#58a6ff"
GOLD   = "#d29922"; AMBER = "#e3b341"; TEAL  = "#39d353"; PURPLE = "#bc8cff"

fig = plt.figure(figsize=(22, 17))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.50,
                        height_ratios=[3, 1.5, 1.5, 2])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3])

for ax in (ax1, ax2, ax3, ax4):
    ax.set_facecolor(DARK); ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.yaxis.label.set_color(TEXT); ax.xaxis.label.set_color(TEXT)

bt_plot = bt.copy()

# Panel 1: SPY + entries
ax1.plot(bt_plot.index, bt_plot["close"], color=BLUE, lw=1.4, label="SPY")
ax1.plot(bt_plot.index, bt_plot["ma50"],  color=GOLD, lw=0.9, ls="--", alpha=0.6, label="50d MA")

# Fed-blocked signals (grey diamond)
for dt in fed_blocked:
    if dt in bt_plot.index:
        ax1.scatter(dt, bt_plot.loc[dt,"close"], marker="X", s=55,
                    color="#555", zorder=5, alpha=0.7)

for ep in episodes:
    t1, t2 = ep["t1"], ep["t2"]
    clr    = GREEN if ep["blended_pnl"] > 0 else RED
    dt1    = t1["entry_dt"]
    # trigger diamond
    if ep["trigger_dt"] in bt_plot.index:
        ax1.scatter(ep["trigger_dt"], bt_plot.loc[ep["trigger_dt"],"close"],
                    marker="D", s=40, color=AMBER, zorder=6)
    # T1 triangle
    if dt1 in bt_plot.index:
        y1 = bt_plot.loc[dt1,"close"]
        ax1.axvline(dt1, color=clr, lw=0.6, alpha=0.35)
        ax1.scatter(dt1, y1, marker="^", s=80, color=clr, zorder=7)
        ax1.annotate(f"T1:{t1['pnl_pct']:+.0f}%\nVIX{t1['vix_entry']:.0f}",
                     xy=(dt1,y1), xytext=(0,12), textcoords="offset points",
                     ha="center", fontsize=6.5, color=clr, fontweight="bold")
    # T2 circle
    if t2 and t2["entry_dt"] in bt_plot.index:
        y2 = bt_plot.loc[t2["entry_dt"],"close"]
        ax1.scatter(t2["entry_dt"], y2, marker="o", s=60,
                    color=TEAL, zorder=8, edgecolors=TEXT, linewidths=0.4)
        ax1.annotate(f"T2:{t2['pnl_pct']:+.0f}%\nVIX{t2['vix_entry']:.0f}",
                     xy=(t2["entry_dt"],y2), xytext=(0,-20), textcoords="offset points",
                     ha="center", fontsize=6.5, color=TEAL, fontweight="bold")

ax1.scatter([],[],marker="X",s=50,color="#555",label="Blocked by Fed filter")
ax1.scatter([],[],marker="D",s=40,color=AMBER,label="Trigger fired")
ax1.scatter([],[],marker="^",s=60,color=GREEN,label="T1 entry (VIX≤25, ½)")
ax1.scatter([],[],marker="o",s=50,color=TEAL,label="T2 entry (VIX≤20, full)")
ax1.set_title("SPY + v5 Entries  |  ✕ Fed-blocked  ◆ trigger  ▲ T1  ● T2",
              color=TEXT, fontsize=10, pad=6)
ax1.set_ylabel("SPY ($)"); ax1.legend(facecolor=DARK, labelcolor=TEXT, fontsize=7.5, loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y")); ax1.grid(color=GRID, lw=0.5)

# Panel 2: VIX
ax2.fill_between(bt_plot.index, bt_plot["vix"], alpha=0.25, color=RED)
ax2.plot(bt_plot.index, bt_plot["vix"], color=RED, lw=1.0, label="VIX")
ax2.axhline(T1_VIX_MAX, color=GREEN,  lw=1.0, ls="--", label=f"VIX {T1_VIX_MAX} (T1 entry)")
ax2.axhline(T2_VIX_MAX, color=TEAL,   lw=1.0, ls=":",  label=f"VIX {T2_VIX_MAX} (T2 entry)")
ax2.axhline(22,         color=GOLD,   lw=0.8, ls=":",  alpha=0.5)
ax2.set_ylabel("VIX"); ax2.legend(facecolor=DARK, labelcolor=TEXT, fontsize=7.5)
ax2.grid(color=GRID, lw=0.5)

# Panel 3: P/C ratio + 2yr yield
ax3_r = ax3.twinx()
if pc_available and "vvix" in bt_plot.columns:
    vvix_series = pd.to_numeric(bt_plot["vvix"], errors="coerce")
    ax3.plot(bt_plot.index, vvix_series, color=PURPLE, lw=0.9, label="VVIX (VIX of VIX)")
    ax3.axhline(VVIX_THRESH, color=PURPLE, lw=0.9, ls="--",
                label=f"VVIX {VVIX_THRESH} (T3 spike threshold)")
    ax3.set_ylabel("VVIX", color=PURPLE)
    ax3.set_ylim(50, 230)
if "dgs2" in bt_plot.columns:
    ax3_r.plot(bt_plot.index, bt_plot["dgs2"], color=GOLD, lw=0.9, alpha=0.7, label="2yr yield")
    ax3_r.set_ylabel("2yr yield (%)", color=GOLD)
ax3.set_title("VVIX (put-buying fear proxy)  +  2yr Treasury Yield (Fed filter)",
              color=TEXT, fontsize=9, pad=5)
ax3.legend(facecolor=DARK, labelcolor=TEXT, fontsize=7.5, loc="upper left")
ax3_r.legend(facecolor=DARK, labelcolor=TEXT, fontsize=7.5, loc="upper right")
ax3.grid(color=GRID, lw=0.5)

# Panel 4: P&L bars
if episodes:
    x          = np.arange(len(episodes))
    bl_vals    = [e["blended_pnl"] for e in episodes]
    t1_vals    = [e["t1"]["pnl_pct"] for e in episodes]
    t2_vals    = [e["t2"]["pnl_pct"] if e["t2"] else np.nan for e in episodes]
    bar_colors = [GREEN if v > 0 else RED for v in bl_vals]

    ax4.bar(x, bl_vals, color=bar_colors, alpha=0.85, width=0.55, label="Blended (T1×½+T2×1)")
    ax4.bar(x-0.20, t1_vals, width=0.16, color=BLUE,  alpha=0.65, label="T1 standalone")
    t2x = [xi for xi,v in zip(x,t2_vals) if not (isinstance(v,float) and np.isnan(v))]
    t2y = [v for v in t2_vals if not (isinstance(v,float) and np.isnan(v))]
    if t2x: ax4.bar([xi+0.20 for xi in t2x], t2y, width=0.16, color=TEAL, alpha=0.65, label="T2 standalone")
    ax4.axhline(0,   color=TEXT,  lw=0.7)
    ax4.axhline(int((TARGET_MULT-1)*100), color=GREEN, lw=0.9, ls="--", alpha=0.6,
                label=f"{int((TARGET_MULT-1)*100)}% exit target")

    for xi, val in zip(x, bl_vals):
        ax4.text(xi, val+(3 if val>=0 else -14), f"{val:+.0f}%",
                 ha="center", fontsize=8, color=TEXT, fontweight="bold")

    xlabels = []
    for ep in episodes:
        t2s = f"T2+{ep.get('t2_lag',0)}" if ep["t2"] else "T1 only"
        xlabels.append(f"#{episodes.index(ep)+1} {ep['t1']['entry_dt'].strftime('%m/%y')}\n"
                       f"VIX{ep['t1']['vix_entry']:.0f} | {t2s}")
    ax4.set_xticks(x); ax4.set_xticklabels(xlabels, fontsize=7, color=TEXT)
    ax4.set_ylabel("Return %")
    ax4.set_title(
        f"Per-Episode P&L  |  150% exit  |  Deep ITM Δ~0.775  |  Fed filter: {fed_source}",
        color=TEXT, fontsize=9, pad=6)
    ax4.legend(facecolor=DARK, labelcolor=TEXT, fontsize=7.5, loc="upper left")
    ax4.grid(color=GRID, lw=0.5, axis="y")
    ax4.set_xlim(-0.6, len(episodes)-0.4)

plt.suptitle(
    f"$SPY LEAP v5  |  T3=VVIX>{VVIX_THRESH}  |  150% exit  |  Fed 2yr filter  "
    f"|  {BT_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}",
    color=TEXT, fontsize=11, fontweight="bold", y=1.005)

OUT = "leap_backtest_v5_results.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nChart saved → {OUT}")
ep_df.to_csv("leap_backtest_v5_trades.csv", index=False)
print("Trade log saved → leap_backtest_v5_trades.csv")
