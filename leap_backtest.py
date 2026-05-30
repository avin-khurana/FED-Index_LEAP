#!/usr/bin/env python3
"""
$SPY Crash Entry LEAP Trigger System — 2-Year Backtest v3
Changes from v2:
  - Single leg     : deep ITM call, delta 0.75–0.80 (no OTM leg)
  - VIX entry cap  : 27  →  25
  - Less IV-crush exposure: most option value is intrinsic, not time premium

Triggers
  T1 Price Damage   : SPY 8–15% below 252d high, 2 consecutive red weeks,
                      close < 50d MA on above-avg volume, daily range >2%
  T2 Volatility     : VIX >22, VIX spiked +20% from 10d low, SPY range >2%,
                      VIX IV-rank (252d) >50%
  T3 Sentiment proxy: VIX >30  (proxy for CNN Fear & Greed = Extreme Fear)

Entry
  • If ≥2 triggers AND VIX ≤ 25  →  enter immediately
  • If ≥2 triggers AND VIX > 25  →  go PENDING; watch daily for up to 30 days
    Cool-down entry fires when: VIX drops below 25 AND SPY still 5–15% below 252d high
  • 30-day trade cooldown between entries

Position
  Single deep ITM call, delta ~0.775 (midpoint of 0.75–0.80), 18-month DTE (540 days)

Exit
  300%+ gain  →  early exit; otherwise hold to expiry
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
    """Closed-form OTM call strike for a target delta."""
    d1_target = norm.ppf(target_Δ)
    return float(S * np.exp((r + 0.5 * σ**2) * T - d1_target * σ * np.sqrt(T)))


# ── Parameters ────────────────────────────────────────────────────────────────

BACKTEST_YEARS  = 10
WARMUP_DAYS     = 400
DTE_DAYS        = 540       # 18 months
OTM_DELTA       = 0.30      # 30-delta OTM strike
RISK_FREE       = 0.05
COOLDOWN_DAYS   = 30        # min days between trade entries
VIX_MAX_ENTRY   = 25        # don't enter above this VIX level
VIX_WATCH_DAYS  = 30        # days to watch for VIX cool-down after trigger
DD_WATCH_MIN    = -0.15     # SPY must still be ≥ 5% below high at cool-down entry
DD_WATCH_MAX    = -0.05
TARGET_MULT     = 4.0       # 300% gain = 4× cost  →  early exit
ITM_DELTA       = 0.775     # target delta for deep ITM call (midpoint of 0.75–0.80)


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


# ── Triggers ──────────────────────────────────────────────────────────────────

T1 = (
    (data["drawdown"] < -0.08) & (data["drawdown"] > -0.15) &
    data["two_consec_red"] &
    (data["close"] < data["ma50"]) & (data["volume"] > data["vol_ma20"]) &
    (data["daily_range"] > 0.02)
)
T2 = (
    (data["vix"] > 22) &
    (data["vix_spike"] > 0.20) &
    (data["daily_range"] > 0.02) &
    (data["vix_iv_rank"] > 0.50)
)
T3 = data["vix"] > 30

data["T1"]         = T1
data["T2"]         = T2
data["T3"]         = T3
data["n_triggers"] = T1.astype(int) + T2.astype(int) + T3.astype(int)
data["signal"]     = data["n_triggers"] >= 2


# ── Entry selection with VIX cool-down watching ───────────────────────────────

bt = data[data.index >= BT_START].copy()

entries        = []   # final list of trade dicts
pending        = []   # list of (trigger_date, expiry_date, vix_at_trigger)
last_entry_dt  = None
trigger_log    = []   # all signal days (for reporting)

for dt in bt.index:
    row = bt.loc[dt]

    in_cooldown = (last_entry_dt is not None and
                   (dt - last_entry_dt).days < COOLDOWN_DAYS)

    # Remove expired pending signals
    pending = [(td, ed, vt) for td, ed, vt in pending if dt <= ed]

    # ── Log every signal day ────────────────────────────────────────────────
    if row["signal"]:
        trigger_log.append({
            "trigger_date"    : dt.strftime("%Y-%m-%d"),
            "vix_at_trigger"  : round(float(row["vix"]), 1),
            "spy_at_trigger"  : round(float(row["close"]), 2),
            "drawdown_pct"    : round(float(row["drawdown"]) * 100, 1),
            "n_triggers"      : int(row["n_triggers"]),
            "T1"              : bool(row["T1"]),
            "T2"              : bool(row["T2"]),
            "T3"              : bool(row["T3"]),
        })

    # ── New signal: immediate entry or go pending ───────────────────────────
    if row["signal"] and not in_cooldown:
        if float(row["vix"]) <= VIX_MAX_ENTRY:
            entries.append(dict(
                trigger_date      = dt,
                entry_date        = dt,
                entry_type        = "immediate",
                days_lag          = 0,
                vix_at_trigger    = float(row["vix"]),
                SPY_entry         = float(row["close"]),
                VIX_entry         = float(row["vix"]),
                drawdown_at_entry = float(row["drawdown"]),
                n_triggers        = int(row["n_triggers"]),
                T1=bool(row["T1"]), T2=bool(row["T2"]), T3=bool(row["T3"]),
            ))
            last_entry_dt = dt
            pending = []
            continue
        else:
            expiry = dt + timedelta(days=VIX_WATCH_DAYS)
            # Don't stack multiple pending from the same cluster (within 5 days)
            if not any(abs((dt - td).days) <= 5 for td, _, _ in pending):
                pending.append((dt, expiry, float(row["vix"])))

    # ── Check pending signals for VIX cool-down entry ──────────────────────
    if pending and not in_cooldown:
        vix_now = float(row["vix"])
        dd_now  = float(row["drawdown"])
        vix_ok  = vix_now <= VIX_MAX_ENTRY
        dd_ok   = DD_WATCH_MIN <= dd_now <= DD_WATCH_MAX

        if vix_ok and dd_ok:
            trigger_dt, _, vix_at_trigger = pending[0]
            entries.append(dict(
                trigger_date      = trigger_dt,
                entry_date        = dt,
                entry_type        = "vix_cooled",
                days_lag          = (dt - trigger_dt).days,
                vix_at_trigger    = vix_at_trigger,
                SPY_entry         = float(row["close"]),
                VIX_entry         = vix_now,
                drawdown_at_entry = dd_now,
                n_triggers        = int(bt.loc[trigger_dt, "n_triggers"]),
                T1 = bool(bt.loc[trigger_dt, "T1"]),
                T2 = bool(bt.loc[trigger_dt, "T2"]),
                T3 = bool(bt.loc[trigger_dt, "T3"]),
            ))
            last_entry_dt = dt
            pending = []

trig_df = pd.DataFrame(trigger_log)

print(f"\nBacktest window : {BT_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}")
print(f"Signal days     : {bt['signal'].sum()}")
print(f"Trigger clusters: {len(trig_df)}")
print(f"Actual entries  : {len(entries)}  "
      f"({sum(1 for e in entries if e['entry_type']=='immediate')} immediate, "
      f"{sum(1 for e in entries if e['entry_type']=='vix_cooled')} after VIX cool-down)\n")


# ── Trade simulation ──────────────────────────────────────────────────────────

trades = []

for e in entries:
    entry_dt = e["entry_date"]
    S0       = e["SPY_entry"]
    iv0      = max(e["VIX_entry"] / 100.0, 0.10)
    T0       = DTE_DAYS / 365.0

    # Deep ITM strike targeting delta 0.775
    K_itm = round(strike_for_delta(S0, T0, RISK_FREE, iv0, ITM_DELTA))
    # Ensure it's actually ITM (below spot)
    K_itm = min(K_itm, round(S0) - 1)

    p0 = bs_call(S0, K_itm, T0, RISK_FREE, iv0)
    if p0 < 0.01:
        continue

    actual_delta = bs_delta_call(S0, K_itm, T0, RISK_FREE, iv0)
    intrinsic    = max(S0 - K_itm, 0.0)
    time_premium = p0 - intrinsic

    expiry_dt = entry_dt + timedelta(days=DTE_DAYS)
    future    = data[(data.index > entry_dt) & (data.index <= expiry_dt)]

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
            continue
        exit_dt = future.index[-1]
        days_left = (expiry_dt - exit_dt).days
        exit_reason = "expiration" if days_left <= 5 else "open / still live"

    exit_row = data.loc[exit_dt]
    Sf_e     = float(exit_row["close"])
    ivf_e    = max(float(exit_row["vix"]) / 100.0, 0.10)
    Tf_e     = max((expiry_dt - exit_dt).days / 365.0, 1 / 365.0)
    p_exit   = bs_call(Sf_e, K_itm, Tf_e, RISK_FREE, ivf_e)

    ret_itm = p_exit / p0 - 1
    ret_spy = Sf_e / S0 - 1

    trades.append(dict(
        trigger_date     = e["trigger_date"].strftime("%Y-%m-%d"),
        entry_date       = entry_dt.strftime("%Y-%m-%d"),
        entry_type       = e["entry_type"],
        days_lag         = e["days_lag"],
        exit_date        = exit_dt.strftime("%Y-%m-%d"),
        days_held        = (exit_dt - entry_dt).days,
        exit_reason      = exit_reason,
        T1               = e["T1"],
        T2               = e["T2"],
        T3               = e["T3"],
        n_triggers       = e["n_triggers"],
        SPY_entry        = round(S0, 2),
        SPY_exit         = round(Sf_e, 2),
        VIX_at_trigger   = round(e["vix_at_trigger"], 1),
        VIX_at_entry     = round(e["VIX_entry"], 1),
        drawdown_pct     = round(e["drawdown_at_entry"] * 100, 1),
        K_itm            = K_itm,
        delta_entry      = round(actual_delta, 3),
        option_cost      = round(p0, 2),
        intrinsic        = round(intrinsic, 2),
        time_premium     = round(time_premium, 2),
        time_prem_pct    = round(time_premium / p0 * 100, 1),
        exit_price       = round(p_exit, 2),
        pnl_pct          = round(ret_itm * 100, 1),
        spy_return_pct   = round(ret_spy * 100, 1),
    ))

res = pd.DataFrame(trades)


# ── Print results ─────────────────────────────────────────────────────────────

sep = "=" * 120
print(sep)
print(f"TRIGGER LOG  (all signal days, including those blocked by VIX > {VIX_MAX_ENTRY})")
print(sep)
if not trig_df.empty:
    print(trig_df.to_string(index=False))

print(f"\n{sep}")
print("TRADE LOG  (single deep ITM call, delta ~0.775, 18-month DTE)")
print(sep)

COLS = [
    "trigger_date", "entry_date", "entry_type", "days_lag",
    "exit_date", "days_held", "exit_reason",
    "n_triggers", "SPY_entry", "SPY_exit",
    "VIX_at_trigger", "VIX_at_entry", "drawdown_pct",
    "K_itm", "delta_entry", "option_cost", "intrinsic", "time_premium", "time_prem_pct",
    "exit_price", "pnl_pct", "spy_return_pct",
]

if res.empty:
    print("No trades executed in the backtest period.")
else:
    with pd.option_context("display.max_columns", None, "display.width", 240,
                           "display.float_format", "{:.1f}".format):
        print(res[COLS].to_string(index=False))

    n       = len(res)
    winners = res[res["pnl_pct"] > 0]
    losers  = res[res["pnl_pct"] <= 0]
    targets = res[res["exit_reason"] == "300%+ target hit"]

    print(f"\n{sep}")
    print("SUMMARY STATISTICS  —  Deep ITM LEAP  |  VIX ≤ {VIX_MAX_ENTRY} entry")
    print(sep)
    print(f"  Total trades              : {n}")
    print(f"  Winners (pnl > 0)         : {len(winners)}  ({len(winners)/n*100:.0f}%)")
    print(f"  Losers                    : {len(losers)}  ({len(losers)/n*100:.0f}%)")
    print(f"  Hit 300%+ target early    : {len(targets)}")
    print()
    print(f"  Avg return                : {res['pnl_pct'].mean():+.1f}%")
    print(f"  Median return             : {res['pnl_pct'].median():+.1f}%")
    print(f"  Best trade                : {res['pnl_pct'].max():+.1f}%")
    print(f"  Worst trade               : {res['pnl_pct'].min():+.1f}%")
    print()
    print(f"  Avg SPY spot return       : {res['spy_return_pct'].mean():+.1f}%  (same holding period)")
    print(f"  Avg option leverage       : {res['pnl_pct'].mean() / res['spy_return_pct'].mean():.2f}x  (option return ÷ SPY return)")
    print()
    print(f"  Avg time premium at entry : {res['time_premium'].mean():.2f}  ({res['time_prem_pct'].mean():.1f}% of cost)")
    print(f"  Avg intrinsic at entry    : {res['intrinsic'].mean():.2f}")
    print(f"  Avg option cost           : {res['option_cost'].mean():.2f}")
    print()
    print(f"  Avg VIX at trigger        : {res['VIX_at_trigger'].mean():.1f}")
    print(f"  Avg VIX at entry          : {res['VIX_at_entry'].mean():.1f}")
    print(f"  Avg drawdown at entry     : {res['drawdown_pct'].mean():.1f}%")
    print(f"  Avg days lag (trigger→entry): {res['days_lag'].mean():.1f}")
    print(f"  Avg days held             : {res['days_held'].mean():.0f}")
    print()
    print(f"  3-of-3 trigger trades     : {(res['n_triggers']==3).sum()}")
    print(f"  2-of-3 trigger trades     : {(res['n_triggers']==2).sum()}")
    imm  = (res['entry_type'] == 'immediate').sum()
    cool = (res['entry_type'] == 'vix_cooled').sum()
    print(f"  Immediate entries         : {imm}")
    print(f"  VIX cool-down entries     : {cool}")


# ── Charts ────────────────────────────────────────────────────────────────────

DARK  = "#0d1117"
GRID  = "#21262d"
TEXT  = "#c9d1d9"
GREEN = "#3fb950"
RED   = "#f85149"
BLUE  = "#58a6ff"
GOLD  = "#d29922"
AMBER = "#e3b341"
PURPLE = "#bc8cff"

fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.48)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color(GRID)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

bt_plot = bt.copy()

# ── Panel 1: SPY + signals ────────────────────────────────────────────────────
ax1.plot(bt_plot.index, bt_plot["close"], color=BLUE, lw=1.4, label="SPY", zorder=2)
ax1.plot(bt_plot.index, bt_plot["ma50"],  color=GOLD, lw=0.9, ls="--", alpha=0.7, label="50d MA")

if not res.empty:
    for _, tr in res.iterrows():
        trig_dt  = pd.to_datetime(tr["trigger_date"])
        entry_dt = pd.to_datetime(tr["entry_date"])
        clr      = GREEN if tr["pnl_pct"] > 0 else RED

        # Trigger marker (diamond) — shown even if different from entry
        if trig_dt in bt_plot.index:
            yt = bt_plot.loc[trig_dt, "close"]
            ax1.scatter(trig_dt, yt, marker="D", s=50, color=AMBER,
                        zorder=6, label="_nolegend_")

        # Entry marker (triangle) — where money went in
        if entry_dt in bt_plot.index:
            ye = bt_plot.loc[entry_dt, "close"]
            ax1.axvline(entry_dt, color=clr, lw=0.7, alpha=0.45)
            ax1.scatter(entry_dt, ye, marker="^", s=95, color=clr, zorder=7)
            label_text = (
                f"{tr['pnl_pct']:+.0f}%\n"
                f"{'🔄' if tr['entry_type']=='vix_cooled' else '⚡'}"
                f"VIX {tr['VIX_at_entry']:.0f}"
            )
            ax1.annotate(
                label_text,
                xy=(entry_dt, ye), xytext=(0, 16),
                textcoords="offset points", ha="center",
                fontsize=7.5, color=clr, fontweight="bold",
            )

        # Arrow from trigger to entry if there was a lag
        if tr["days_lag"] > 1 and trig_dt in bt_plot.index and entry_dt in bt_plot.index:
            yt = bt_plot.loc[trig_dt,  "close"]
            ye = bt_plot.loc[entry_dt, "close"]
            ax1.annotate("", xy=(entry_dt, ye), xytext=(trig_dt, yt),
                         arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2))

ax1.scatter([], [], marker="D", s=50, color=AMBER, label="Trigger fired")
ax1.scatter([], [], marker="^", s=70, color=GREEN, label="Entry (profit)")
ax1.scatter([], [], marker="^", s=70, color=RED,   label="Entry (loss)")
ax1.set_title(
    "SPY + LEAP Entries  |  ◆ = trigger fired  ▲ = actual entry  "
    "(⚡ immediate  🔄 after VIX cool-down)",
    fontsize=10, pad=8,
)
ax1.set_ylabel("SPY ($)")
ax1.legend(facecolor=DARK, labelcolor=TEXT, fontsize=8, loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.grid(color=GRID, lw=0.5)

# ── Panel 2: VIX ─────────────────────────────────────────────────────────────
ax2.fill_between(bt_plot.index, bt_plot["vix"], alpha=0.3, color=RED)
ax2.plot(bt_plot.index, bt_plot["vix"], color=RED, lw=1.1, label="VIX")
ax2.axhline(VIX_MAX_ENTRY, color=GREEN, lw=1.3, ls="--",
            label=f"VIX {VIX_MAX_ENTRY} — entry cap / cool-down target")
ax2.axhline(22, color=GOLD,   lw=0.9, ls=":",  label="22 (T2 trigger level)")
ax2.axhline(30, color=PURPLE, lw=0.9, ls=":",  label="30 (T3 Extreme Fear proxy)")

# Shade the pending windows
for td, ed, _ in []:  # placeholder — pending windows already consumed
    pass

if not res.empty:
    for _, tr in res.iterrows():
        if tr["entry_type"] == "vix_cooled":
            td = pd.to_datetime(tr["trigger_date"])
            ed = pd.to_datetime(tr["entry_date"])
            ax2.axvspan(td, ed, alpha=0.15, color=AMBER, label="_nolegend_")

ax2.set_ylabel("VIX")
ax2.legend(facecolor=DARK, labelcolor=TEXT, fontsize=8)
ax2.grid(color=GRID, lw=0.5)

# ── Panel 3: P&L per trade ────────────────────────────────────────────────────
if not res.empty:
    x      = np.arange(len(res))
    pnls   = res["pnl_pct"].values
    colors = [GREEN if v > 0 else RED for v in pnls]

    bars = ax3.bar(x, pnls, color=colors, alpha=0.85, width=0.6, label="Deep ITM LEAP P&L")
    ax3.axhline(0,   color=TEXT,  lw=0.7)
    ax3.axhline(300, color=GREEN, lw=0.9, ls="--", alpha=0.7, label="300% exit target")

    # SPY return comparison dots
    ax3.scatter(x, res["spy_return_pct"].values, marker="o", s=40,
                color=BLUE, zorder=5, label="SPY spot return (same period)")

    ax3.set_xticks(x)
    labels = []
    for i, tr in res.iterrows():
        etype = "⚡" if tr["entry_type"] == "immediate" else f"+{int(tr['days_lag'])}d"
        labels.append(
            f"#{i+1} {tr['entry_date'][5:]}\n"
            f"Δ{tr['delta_entry']:.2f}  VIX{tr['VIX_at_entry']:.0f}\n{etype}"
        )
    ax3.set_xticklabels(labels, fontsize=7.5, color=TEXT)

    for bar, val in zip(bars, pnls):
        ypos = bar.get_height() + (3 if val >= 0 else -14)
        ax3.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{val:+.0f}%", ha="center", fontsize=8.5, color=TEXT, fontweight="bold")

    ax3.set_ylabel("Return %")
    ax3.set_title(
        f"Per-Trade P&L  |  Deep ITM Call (Δ~0.775)  |  18-mo DTE  |  VIX ≤ {VIX_MAX_ENTRY} entry",
        fontsize=10, pad=8,
    )
    ax3.legend(facecolor=DARK, labelcolor=TEXT, fontsize=8, loc="upper left")
    ax3.grid(color=GRID, lw=0.5, axis="y")
else:
    ax3.text(0.5, 0.5, "No trades in backtest period",
             ha="center", va="center", color=TEXT, fontsize=13, transform=ax3.transAxes)

plt.suptitle(
    f"$SPY Crash Entry LEAP v3  |  Deep ITM (Δ~0.775)  |  18-mo DTE  |  VIX ≤ {VIX_MAX_ENTRY} entry  "
    f"|  {BT_START:%Y-%m-%d} → {TODAY:%Y-%m-%d}",
    color=TEXT, fontsize=12, fontweight="bold", y=1.005,
)

OUT_PNG = "leap_backtest_results.png"
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nChart saved → {OUT_PNG}")

res.to_csv("leap_backtest_trades.csv", index=False)
print("Trade log saved → leap_backtest_trades.csv")
