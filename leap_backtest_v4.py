#!/usr/bin/env python3
"""
$SPY Crash Entry LEAP Trigger System — v4  (Tiered Entry)
All trigger conditions identical to v3. New: two-tranche sizing.

Tranche 1 (half size)  — fires when VIX cools to ≤ 25 after trigger
Tranche 2 (full size)  — fires when VIX cools further to ≤ 20
                          within 120 days of T1 entry
                          SPY must still be ≥ 2% below 252d high

Each tranche is an independent 18-month LEAP:
  • Deep ITM call  (delta ~0.775, new strike at current SPY price)
  • Own 18-month DTE from its entry date

If Tranche 2 never fires → Tranche 1 runs solo to its expiry.

Sizing / blended P&L
  • If only T1:          blended = T1 P&L
  • If T1 + T2:          blended = (0.5 × T1_pnl  +  1.0 × T2_pnl) / 1.5

Exit: 300%+ gain on a tranche triggers early close of that tranche only.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import yfinance as yf


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def bs_call(S, K, T, r, σ):
    if T < 1e-6:
        return max(0.0, S - K)
    if σ < 1e-6:
        return max(0.0, S - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * np.sqrt(T))
    d2 = d1 - σ * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_delta_call(S, K, T, r, σ):
    if T < 1e-6:
        return 1.0 if S >= K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * np.sqrt(T))
    return float(norm.cdf(d1))


def strike_for_delta(S, T, r, σ, target_Δ):
    d1_target = norm.ppf(target_Δ)
    return float(S * np.exp((r + 0.5 * σ**2) * T - d1_target * σ * np.sqrt(T)))


# ── Parameters ────────────────────────────────────────────────────────────────

BACKTEST_YEARS   = 10
WARMUP_DAYS      = 400
DTE_DAYS         = 540       # 18 months
ITM_DELTA        = 0.775
RISK_FREE        = 0.05
COOLDOWN_DAYS    = 30        # between separate trade episodes

# Tranche 1
T1_VIX_MAX       = 25        # enter T1 half-size when VIX cools below this
T1_WATCH_DAYS    = 30        # days to watch after trigger fires
T1_DD_MIN        = -0.15     # SPY must still be in drawdown at T1 entry
T1_DD_MAX        = -0.05

# Tranche 2
T2_VIX_MAX       = 20        # enter T2 full-size when VIX cools below this
T2_WATCH_DAYS    = 120       # days from T1 entry to watch for T2
T2_DD_MAX        = -0.02     # SPY still ≥ 2% below 252d high at T2 entry

TARGET_MULT      = 4.0       # 300% gain → early exit for that tranche


# ── Data ──────────────────────────────────────────────────────────────────────

TODAY          = datetime.today()
BT_START       = TODAY - timedelta(days=BACKTEST_YEARS * 365)
DOWNLOAD_START = BT_START  - timedelta(days=WARMUP_DAYS)

print(f"Downloading SPY and VIX ({DOWNLOAD_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}) …")
spy_raw = yf.download("SPY",  start=DOWNLOAD_START, end=TODAY, auto_adjust=True, progress=False)
vix_raw = yf.download("^VIX", start=DOWNLOAD_START, end=TODAY, auto_adjust=True, progress=False)

for _d in (spy_raw, vix_raw):
    if isinstance(_d.columns, pd.MultiIndex):
        _d.columns = _d.columns.get_level_values(0)

data = pd.DataFrame({
    "open":   spy_raw["Open"],
    "high":   spy_raw["High"],
    "low":    spy_raw["Low"],
    "close":  spy_raw["Close"],
    "volume": spy_raw["Volume"],
    "vix":    vix_raw["Close"],
}).dropna()
data.index = pd.to_datetime(data.index).tz_localize(None)


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
    raw=True,
)

wk_close = data["close"].resample("W-FRI").last()
wk_open  = data["open"].resample("W-FRI").first()
wk_red   = wk_close < wk_open
two_red  = wk_red.shift(1) & wk_red.shift(2)
data["two_consec_red"] = two_red.reindex(data.index, method="ffill").fillna(False)

data.dropna(inplace=True)


# ── Triggers (identical to v3) ────────────────────────────────────────────────

T1_sig = (
    (data["drawdown"] < -0.08) & (data["drawdown"] > -0.15) &
    data["two_consec_red"] &
    (data["close"] < data["ma50"]) & (data["volume"] > data["vol_ma20"]) &
    (data["daily_range"] > 0.02)
)
T2_sig = (
    (data["vix"] > 22) &
    (data["vix_spike"] > 0.20) &
    (data["daily_range"] > 0.02) &
    (data["vix_iv_rank"] > 0.50)
)
T3_sig = data["vix"] > 30

data["T1_flag"]    = T1_sig
data["T2_flag"]    = T2_sig
data["T3_flag"]    = T3_sig
data["n_triggers"] = T1_sig.astype(int) + T2_sig.astype(int) + T3_sig.astype(int)
data["signal"]     = data["n_triggers"] >= 2


# ── Simulate a single LEAP option position ────────────────────────────────────

def sim_leap(entry_dt, S0, iv0, full_data, expiry_dt=None):
    """Price and simulate one 18-month deep-ITM LEAP from entry_dt."""
    T0    = DTE_DAYS / 365.0
    K_itm = min(round(strike_for_delta(S0, T0, RISK_FREE, iv0, ITM_DELTA)),
                round(S0) - 1)
    p0    = bs_call(S0, K_itm, T0, RISK_FREE, iv0)
    if p0 < 0.01:
        return None

    intrinsic    = max(S0 - K_itm, 0.0)
    time_premium = p0 - intrinsic
    expiry_dt    = entry_dt + timedelta(days=DTE_DAYS)
    future       = full_data[(full_data.index > entry_dt) & (full_data.index <= expiry_dt)]

    exit_dt     = None
    exit_reason = "expiration"

    for fdt, frow in future.iterrows():
        Sf  = float(frow["close"])
        ivf = max(float(frow["vix"]) / 100.0, 0.10)
        Tf  = max((expiry_dt - fdt).days / 365.0, 1 / 365.0)
        pf  = bs_call(Sf, K_itm, Tf, RISK_FREE, ivf)
        if pf >= p0 * TARGET_MULT:
            exit_dt     = fdt
            exit_reason = "300%+ target hit"
            break

    if exit_dt is None:
        if len(future) == 0:
            return None
        exit_dt = future.index[-1]
        days_left = (expiry_dt - exit_dt).days
        exit_reason = "expiration" if days_left <= 5 else "open / still live"

    exit_row = full_data.loc[exit_dt]
    Sf_e     = float(exit_row["close"])
    ivf_e    = max(float(exit_row["vix"]) / 100.0, 0.10)
    Tf_e     = max((expiry_dt - exit_dt).days / 365.0, 1 / 365.0)
    p_exit   = bs_call(Sf_e, K_itm, Tf_e, RISK_FREE, ivf_e)

    return dict(
        entry_dt     = entry_dt,
        exit_dt      = exit_dt,
        days_held    = (exit_dt - entry_dt).days,
        exit_reason  = exit_reason,
        S_entry      = round(S0, 2),
        S_exit       = round(Sf_e, 2),
        spy_ret_pct  = round((Sf_e / S0 - 1) * 100, 1),
        vix_entry    = round(iv0 * 100, 1),
        K_itm        = K_itm,
        delta        = round(bs_delta_call(S0, K_itm, T0, RISK_FREE, iv0), 3),
        cost         = round(p0, 2),
        intrinsic    = round(intrinsic, 2),
        time_prem    = round(time_premium, 2),
        time_prem_pct= round(time_premium / p0 * 100, 1),
        exit_price   = round(p_exit, 2),
        pnl_pct      = round((p_exit / p0 - 1) * 100, 1),
    )


# ── Episode detection: trigger → T1 entry → T2 entry ─────────────────────────

bt          = data[data.index >= BT_START].copy()
episodes    = []          # final list of trade episodes
trigger_log = []

pending_triggers   = []   # (trigger_dt, expiry_dt, vix_at_trigger)
last_episode_dt    = None  # date of most recent T1 entry (for cooldown)

for dt in bt.index:
    row = bt.loc[dt]

    in_cooldown = (last_episode_dt is not None and
                   (dt - last_episode_dt).days < COOLDOWN_DAYS)

    # Remove expired pending triggers
    pending_triggers = [(td, ed, vt) for td, ed, vt in pending_triggers if dt <= ed]

    # Log every signal day
    if row["signal"]:
        trigger_log.append({
            "trigger_date"  : dt.strftime("%Y-%m-%d"),
            "vix"           : round(float(row["vix"]), 1),
            "spy"           : round(float(row["close"]), 2),
            "drawdown_pct"  : round(float(row["drawdown"]) * 100, 1),
            "n_triggers"    : int(row["n_triggers"]),
            "T1"            : bool(row["T1_flag"]),
            "T2"            : bool(row["T2_flag"]),
            "T3"            : bool(row["T3_flag"]),
        })

    # Check for new trigger signal
    if row["signal"] and not in_cooldown:
        vix_now = float(row["vix"])
        dd_now  = float(row["drawdown"])

        if vix_now <= T1_VIX_MAX and T1_DD_MIN <= dd_now <= T1_DD_MAX:
            # Immediate T1 entry
            S0   = float(row["close"])
            iv0  = max(vix_now / 100.0, 0.10)
            leap = sim_leap(dt, S0, iv0, data)
            if leap:
                episodes.append({
                    "trigger_dt"     : dt,
                    "trigger_vix"    : vix_now,
                    "trigger_ntrig"  : int(row["n_triggers"]),
                    "t1"             : leap,
                    "t1_entry_type"  : "immediate",
                    "t1_days_lag"    : 0,
                    "t2"             : None,
                })
                last_episode_dt = dt
                pending_triggers = []
                continue
        elif vix_now > T1_VIX_MAX:
            expiry = dt + timedelta(days=T1_WATCH_DAYS)
            if not any(abs((dt - td).days) <= 5 for td, _, _ in pending_triggers):
                pending_triggers.append((dt, expiry, vix_now))

    # Check pending triggers for T1 cool-down entry
    if pending_triggers and not in_cooldown:
        vix_now = float(row["vix"])
        dd_now  = float(row["drawdown"])

        if vix_now <= T1_VIX_MAX and T1_DD_MIN <= dd_now <= T1_DD_MAX:
            trigger_dt, _, vix_at_trig = pending_triggers[0]
            S0   = float(row["close"])
            iv0  = max(vix_now / 100.0, 0.10)
            leap = sim_leap(dt, S0, iv0, data)
            if leap:
                episodes.append({
                    "trigger_dt"     : trigger_dt,
                    "trigger_vix"    : vix_at_trig,
                    "trigger_ntrig"  : int(bt.loc[trigger_dt, "n_triggers"]),
                    "t1"             : leap,
                    "t1_entry_type"  : "vix_cooled",
                    "t1_days_lag"    : (dt - trigger_dt).days,
                    "t2"             : None,
                })
                last_episode_dt = dt
                pending_triggers = []

# ── Add Tranche 2 to each episode where VIX ≤ 20 within 120 days ─────────────

for ep in episodes:
    t1_entry_dt = ep["t1"]["entry_dt"]
    t2_window_end = t1_entry_dt + timedelta(days=T2_WATCH_DAYS)
    t2_candidates = data[
        (data.index > t1_entry_dt) & (data.index <= t2_window_end)
    ]

    for dt, row in t2_candidates.iterrows():
        vix_now = float(row["vix"])
        dd_now  = float(row["drawdown"])
        if vix_now <= T2_VIX_MAX and dd_now <= T2_DD_MAX:
            S0   = float(row["close"])
            iv0  = max(vix_now / 100.0, 0.10)
            leap = sim_leap(dt, S0, iv0, data)
            if leap:
                ep["t2"]         = leap
                ep["t2_days_lag"] = (dt - t1_entry_dt).days
            break  # take the first qualifying day


# ── Compute blended P&L ───────────────────────────────────────────────────────

for ep in episodes:
    t1_pnl = ep["t1"]["pnl_pct"]
    if ep["t2"] is not None:
        t2_pnl = ep["t2"]["pnl_pct"]
        # T1 weight = 0.5 unit, T2 weight = 1.0 unit → 1.5 total
        ep["blended_pnl"] = round((0.5 * t1_pnl + 1.0 * t2_pnl) / 1.5, 1)
        ep["total_units"]  = 1.5
    else:
        ep["blended_pnl"] = t1_pnl
        ep["total_units"]  = 0.5


# ── Print results ─────────────────────────────────────────────────────────────

trig_df = pd.DataFrame(trigger_log)
sep     = "=" * 130

print(f"\nBacktest : {BT_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}")
print(f"Episodes : {len(episodes)}  "
      f"| T2 also fired: {sum(1 for e in episodes if e['t2'])}  "
      f"| T1 only: {sum(1 for e in episodes if not e['t2'])}")

print(f"\n{sep}")
print("TRIGGER LOG")
print(sep)
print(trig_df.to_string(index=False))

print(f"\n{sep}")
print("EPISODE LOG  (each row = one trade episode, showing both tranches)")
print(sep)

rows = []
for i, ep in enumerate(episodes, 1):
    t1  = ep["t1"]
    t2  = ep["t2"]
    row = {
        "#"              : i,
        "trigger_date"   : ep["trigger_dt"].strftime("%Y-%m-%d"),
        "trig_VIX"       : ep["trigger_vix"],
        "n_trig"         : ep["trigger_ntrig"],
        # Tranche 1
        "T1_entry"       : t1["entry_dt"].strftime("%Y-%m-%d"),
        "T1_lag"         : ep["t1_days_lag"],
        "T1_type"        : ep["t1_entry_type"],
        "T1_VIX"         : t1["vix_entry"],
        "T1_SPY"         : t1["S_entry"],
        "T1_K"           : t1["K_itm"],
        "T1_Δ"           : t1["delta"],
        "T1_cost"        : t1["cost"],
        "T1_exit"        : t1["exit_dt"].strftime("%Y-%m-%d"),
        "T1_reason"      : t1["exit_reason"],
        "T1_SPY_exit"    : t1["S_exit"],
        "T1_spy_ret%"    : t1["spy_ret_pct"],
        "T1_pnl%"        : t1["pnl_pct"],
        # Tranche 2
        "T2_entry"       : t2["entry_dt"].strftime("%Y-%m-%d") if t2 else "—",
        "T2_lag_from_T1" : ep.get("t2_days_lag", "—"),
        "T2_VIX"         : t2["vix_entry"] if t2 else "—",
        "T2_SPY"         : t2["S_entry"] if t2 else "—",
        "T2_K"           : t2["K_itm"] if t2 else "—",
        "T2_cost"        : t2["cost"] if t2 else "—",
        "T2_exit"        : t2["exit_dt"].strftime("%Y-%m-%d") if t2 else "—",
        "T2_reason"      : t2["exit_reason"] if t2 else "—",
        "T2_spy_ret%"    : t2["spy_ret_pct"] if t2 else "—",
        "T2_pnl%"        : t2["pnl_pct"] if t2 else "—",
        # Combined
        "units"          : ep["total_units"],
        "blended_pnl%"   : ep["blended_pnl"],
    }
    rows.append(row)

ep_df = pd.DataFrame(rows)
with pd.option_context("display.max_columns", None, "display.width", 260,
                       "display.float_format", "{:.1f}".format):
    print(ep_df.to_string(index=False))

# ── Summary statistics ────────────────────────────────────────────────────────

blended  = [e["blended_pnl"] for e in episodes]
t1_pnls  = [e["t1"]["pnl_pct"] for e in episodes]
t2_pnls  = [e["t2"]["pnl_pct"] for e in episodes if e["t2"]]

n        = len(episodes)
n_t2     = sum(1 for e in episodes if e["t2"])
n_win    = sum(1 for v in blended if v > 0)
n_lose   = n - n_win

print(f"\n{sep}")
print("SUMMARY STATISTICS")
print(sep)
print(f"  Total episodes            : {n}")
print(f"  Episodes with T2 fired    : {n_t2}  ({n_t2/n*100:.0f}%)")
print(f"  Episodes T1-only          : {n - n_t2}")
print()
print(f"  ── Blended P&L (T1×0.5 + T2×1.0) ──────────────────────────────")
print(f"  Winners                   : {n_win}  ({n_win/n*100:.0f}%)")
print(f"  Losers                    : {n_lose}  ({n_lose/n*100:.0f}%)")
print(f"  Avg blended return        : {np.mean(blended):+.1f}%")
print(f"  Median blended return     : {np.median(blended):+.1f}%")
print(f"  Best episode              : {max(blended):+.1f}%")
print(f"  Worst episode             : {min(blended):+.1f}%")
print()
print(f"  ── Tranche 1 standalone ─────────────────────────────────────────")
print(f"  Avg T1 return             : {np.mean(t1_pnls):+.1f}%")
print(f"  Median T1 return          : {np.median(t1_pnls):+.1f}%")
print(f"  Best T1                   : {max(t1_pnls):+.1f}%")
print(f"  Worst T1                  : {min(t1_pnls):+.1f}%")
if t2_pnls:
    print()
    print(f"  ── Tranche 2 (only episodes where T2 fired) ─────────────────────")
    print(f"  Avg T2 return             : {np.mean(t2_pnls):+.1f}%")
    print(f"  Median T2 return          : {np.median(t2_pnls):+.1f}%")
    print(f"  Best T2                   : {max(t2_pnls):+.1f}%")
    print(f"  Worst T2                  : {min(t2_pnls):+.1f}%")
    print()
    # Compare blended vs T1-only for episodes where T2 fired
    fired_t1 = [e["t1"]["pnl_pct"] for e in episodes if e["t2"]]
    fired_bl = [e["blended_pnl"] for e in episodes if e["t2"]]
    print(f"  ── T2-fired episodes: T1-only vs blended ────────────────────────")
    print(f"  Avg T1-only return        : {np.mean(fired_t1):+.1f}%")
    print(f"  Avg blended return        : {np.mean(fired_bl):+.1f}%  "
          f"({'better' if np.mean(fired_bl) > np.mean(fired_t1) else 'worse'} with T2)")
print()
t1_spy = [e["t1"]["spy_ret_pct"] for e in episodes]
print(f"  ── vs SPY buy-and-hold ──────────────────────────────────────────")
print(f"  Avg SPY return (T1 period): {np.mean(t1_spy):+.1f}%")
print(f"  Avg blended leverage      : {np.mean(blended)/np.mean(t1_spy):.2f}x")
print()
lags = [e["t1_days_lag"] for e in episodes]
t2_lags = [e.get("t2_days_lag", 0) for e in episodes if e["t2"]]
print(f"  Avg T1 lag (trigger→T1)   : {np.mean(lags):.1f} days")
if t2_lags:
    print(f"  Avg T2 lag (T1→T2)        : {np.mean(t2_lags):.1f} days")
print(f"  Avg T1 VIX at entry       : {np.mean([e['t1']['vix_entry'] for e in episodes]):.1f}")
if n_t2:
    print(f"  Avg T2 VIX at entry       : {np.mean([e['t2']['vix_entry'] for e in episodes if e['t2']]):.1f}")


# ── Chart ─────────────────────────────────────────────────────────────────────

DARK   = "#0d1117"; GRID  = "#21262d"; TEXT  = "#c9d1d9"
GREEN  = "#3fb950"; RED   = "#f85149"; BLUE  = "#58a6ff"
GOLD   = "#d29922"; AMBER = "#e3b341"; PURPLE = "#bc8cff"
TEAL   = "#39d353"

fig = plt.figure(figsize=(22, 15))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.48)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(DARK); ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color(GRID)
    ax.yaxis.label.set_color(TEXT); ax.xaxis.label.set_color(TEXT)

bt_plot = bt.copy()

# ── Panel 1: SPY + entry markers ─────────────────────────────────────────────
ax1.plot(bt_plot.index, bt_plot["close"], color=BLUE, lw=1.4, label="SPY", zorder=2)
ax1.plot(bt_plot.index, bt_plot["ma50"],  color=GOLD, lw=0.9, ls="--", alpha=0.6, label="50d MA")

for ep in episodes:
    t1   = ep["t1"]
    t2   = ep["t2"]
    clr  = GREEN if ep["blended_pnl"] > 0 else RED
    dt1  = t1["entry_dt"]
    trig = ep["trigger_dt"]

    # Trigger diamond
    if trig in bt_plot.index:
        ax1.scatter(trig, bt_plot.loc[trig, "close"],
                    marker="D", s=45, color=AMBER, zorder=6)

    # T1 entry triangle
    if dt1 in bt_plot.index:
        y1 = bt_plot.loc[dt1, "close"]
        ax1.axvline(dt1, color=clr, lw=0.6, alpha=0.35)
        ax1.scatter(dt1, y1, marker="^", s=90, color=clr, zorder=7)
        ax1.annotate(
            f"T1:{t1['pnl_pct']:+.0f}%\nVIX{t1['vix_entry']:.0f}",
            xy=(dt1, y1), xytext=(0, 14),
            textcoords="offset points", ha="center",
            fontsize=7, color=clr, fontweight="bold",
        )

    # T2 entry circle
    if t2 is not None:
        dt2 = t2["entry_dt"]
        if dt2 in bt_plot.index:
            y2 = bt_plot.loc[dt2, "close"]
            ax1.scatter(dt2, y2, marker="o", s=70, color=TEAL, zorder=8,
                        edgecolors=TEXT, linewidths=0.5)
            ax1.annotate(
                f"T2:{t2['pnl_pct']:+.0f}%\nVIX{t2['vix_entry']:.0f}",
                xy=(dt2, y2), xytext=(0, -22),
                textcoords="offset points", ha="center",
                fontsize=7, color=TEAL, fontweight="bold",
            )

# Legend proxies
ax1.scatter([], [], marker="D", s=45, color=AMBER, label="Trigger fired")
ax1.scatter([], [], marker="^", s=70, color=GREEN, label="T1 entry (VIX ≤ 25, half size)")
ax1.scatter([], [], marker="o", s=55, color=TEAL,  label="T2 entry (VIX ≤ 20, full size)")
ax1.set_title(
    "SPY + Tiered LEAP Entries  |  ◆ trigger  ▲ T1 half (VIX≤25)  ● T2 full (VIX≤20)",
    color=TEXT, fontsize=10, pad=8,
)
ax1.set_ylabel("SPY ($)", color=TEXT)
ax1.legend(facecolor=DARK, labelcolor=TEXT, fontsize=8, loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.grid(color=GRID, lw=0.5)

# ── Panel 2: VIX with thresholds ──────────────────────────────────────────────
ax2.fill_between(bt_plot.index, bt_plot["vix"], alpha=0.3, color=RED)
ax2.plot(bt_plot.index, bt_plot["vix"], color=RED, lw=1.1, label="VIX")
ax2.axhline(T1_VIX_MAX, color=GREEN,  lw=1.1, ls="--",
            label=f"VIX {T1_VIX_MAX} — T1 entry (half)")
ax2.axhline(T2_VIX_MAX, color=TEAL,   lw=1.1, ls=":",
            label=f"VIX {T2_VIX_MAX} — T2 entry (full)")
ax2.axhline(22,         color=GOLD,   lw=0.8, ls=":",  alpha=0.6, label="22 (T2 trigger)")
ax2.axhline(30,         color=PURPLE, lw=0.8, ls=":",  alpha=0.6, label="30 (T3 Extreme Fear)")

# Shade T1→T2 windows
for ep in episodes:
    if ep["t2"]:
        ax2.axvspan(ep["t1"]["entry_dt"], ep["t2"]["entry_dt"],
                    alpha=0.12, color=TEAL)

ax2.set_ylabel("VIX", color=TEXT)
ax2.legend(facecolor=DARK, labelcolor=TEXT, fontsize=8)
ax2.grid(color=GRID, lw=0.5)

# ── Panel 3: P&L bar chart ────────────────────────────────────────────────────
x          = np.arange(len(episodes))
blend_vals = [e["blended_pnl"] for e in episodes]
t1_vals    = [e["t1"]["pnl_pct"] for e in episodes]
t2_vals    = [e["t2"]["pnl_pct"] if e["t2"] else np.nan for e in episodes]
bar_colors = [GREEN if v > 0 else RED for v in blend_vals]

ax3.bar(x, blend_vals, color=bar_colors, alpha=0.85, width=0.55, label="Blended (T1×½ + T2×1)")
ax3.bar(x - 0.20, t1_vals, width=0.16, color=BLUE,  alpha=0.65, label="T1 standalone")
t2_x = [xi for xi, v in zip(x, t2_vals) if not np.isnan(v)]
t2_y = [v for v in t2_vals if not np.isnan(v)]
if t2_x:
    ax3.bar([xi + 0.20 for xi in t2_x], t2_y, width=0.16,
            color=TEAL, alpha=0.65, label="T2 standalone")

ax3.axhline(0,   color=TEXT,  lw=0.7)
ax3.axhline(300, color=GREEN, lw=0.9, ls="--", alpha=0.6, label="300% target")

# Value labels on blended bars
for xi, val in zip(x, blend_vals):
    ax3.text(xi, val + (3 if val >= 0 else -14),
             f"{val:+.0f}%", ha="center", fontsize=8, color=TEXT, fontweight="bold")

# Tick labels
x_labels = []
for e in episodes:
    t1_type = "⚡" if e["t1_entry_type"] == "immediate" else f"+{e['t1_days_lag']}d"
    t2_str  = f"T2+{e.get('t2_days_lag','')}" if e["t2"] else "T1 only"
    x_labels.append(
        f"#{episodes.index(e)+1} {e['t1']['entry_dt'].strftime('%m/%y')}\n"
        f"T1 VIX{e['t1']['vix_entry']:.0f} {t1_type}\n{t2_str}"
    )

ax3.set_xticks(x)
ax3.set_xticklabels(x_labels, fontsize=7, color=TEXT)
ax3.set_ylabel("Return %", color=TEXT)
ax3.set_title(
    f"Per-Episode P&L  |  T1 half (VIX≤{T1_VIX_MAX}) + T2 full (VIX≤{T2_VIX_MAX})  "
    f"|  Deep ITM Δ~0.775  |  18-mo DTE",
    color=TEXT, fontsize=10, pad=8,
)
ax3.legend(facecolor=DARK, labelcolor=TEXT, fontsize=8, loc="upper left")
ax3.grid(color=GRID, lw=0.5, axis="y")
ax3.set_xlim(-0.6, len(episodes) - 0.4)

plt.suptitle(
    f"$SPY LEAP v4  |  Tiered Entry: T1 VIX≤{T1_VIX_MAX} (½)  +  T2 VIX≤{T2_VIX_MAX} (full)  "
    f"|  18-mo Deep ITM  |  {BT_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}",
    color=TEXT, fontsize=12, fontweight="bold", y=1.005,
)

OUT_PNG = "leap_backtest_v4_results.png"
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nChart saved → {OUT_PNG}")

ep_df.to_csv("leap_backtest_v4_trades.csv", index=False)
print("Trade log saved → leap_backtest_v4_trades.csv")
