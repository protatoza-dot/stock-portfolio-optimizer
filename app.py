"""
Stock Portfolio Allocation Optimizer  ·  Premium Dark UI Edition
================================================================
A Streamlit app implementing Harry Markowitz's Mean-Variance Model.

Architecture
------------
1. Input Layer        -> Sidebar widgets (tickers, dates, risk-free rate, run button)
2. Data Layer         -> yfinance download + Pandas cleaning
3. Processing Layer   -> Log returns, annualised statistics, SciPy SLSQP optimisation
4. Output Layer       -> Tabbed dashboard: KPI cards, animated Plotly charts, tables
5. Error Handling     -> try/except blocks that surface friendly st.error alerts
6. Presentation       -> Injected CSS (glassmorphism, animations, hover effects)

Run with:  streamlit run app.py
(Optional but recommended: keep `.streamlit/config.toml` next to this file so that
native Streamlit widgets/popups also use the dark theme.)
"""

from datetime import date, timedelta
from html import escape as _escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TRADING_DAYS = 252          # Used to annualise daily statistics
MIN_OBSERVATIONS = 30       # Minimum number of daily return rows required
N_RANDOM_PORTFOLIOS = 2500  # Random portfolios plotted as background cloud
N_FRONTIER_POINTS = 50      # Resolution of the efficient frontier curve

# Design tokens ------------------------------------------------------------- #
EMERALD, CYAN, AMBER = "#34d399", "#22d3ee", "#fbbf24"
TEXT, MUTED, GRID = "#e2e8f0", "#94a3b8", "rgba(148,163,184,0.12)"
PALETTE = [
    "#34d399", "#22d3ee", "#818cf8", "#f472b6", "#fbbf24", "#fb7185",
    "#a78bfa", "#2dd4bf", "#60a5fa", "#f97316", "#a3e635", "#e879f9",
]
PORTFOLIO_STYLE = {
    "Max Sharpe": dict(
        color=EMERALD, accent="emerald", badge="ok", icon="⭐",
        desc="Highest risk-adjusted return (tangency portfolio)",
    ),
    "Min Variance": dict(
        color=CYAN, accent="cyan", badge="cyan", icon="🛡️",
        desc="Lowest achievable total volatility",
    ),
}


# --------------------------------------------------------------------------- #
# Custom exception for predictable, user-facing problems
# --------------------------------------------------------------------------- #
class InputError(Exception):
    """Raised when the user's input (tickers, dates, data) is unusable."""


# =========================================================================== #
#                         PRESENTATION LAYER  (CSS + HTML)
# =========================================================================== #
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bg: #0b1120;
  --card: rgba(30, 41, 59, 0.55);
  --emerald: #10b981;
  --emerald-2: #34d399;
  --cyan: #22d3ee;
  --amber: #fbbf24;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --border: rgba(148, 163, 184, 0.16);
}

/* ---------- Keyframes ---------- */
@keyframes fadeUp   { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
@keyframes shimmer  { to { background-position: 200% center; } }
@keyframes growX    { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes pulseDot {
  0%   { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.65); }
  70%  { box-shadow: 0 0 0 9px rgba(52, 211, 153, 0); }
  100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}
@keyframes floaty   { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }

/* ---------- App shell ---------- */
.stApp {
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(16, 185, 129, 0.13), transparent 60%),
    radial-gradient(1000px 500px at 100% 0%, rgba(34, 211, 238, 0.10), transparent 55%),
    linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
  color: var(--text);
  font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #f1f5f9; letter-spacing: -0.01em; }
.stApp p, .stApp label { color: var(--text); }

/* Scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.25); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(52, 211, 153, 0.5); }

/* ---------- Hero ---------- */
.hero {
  padding: 28px 32px; margin-bottom: 20px; border-radius: 22px;
  border: 1px solid transparent;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(30, 41, 59, 0.62)) padding-box,
    linear-gradient(135deg, rgba(16, 185, 129, 0.6), rgba(34, 211, 238, 0.4), rgba(148, 163, 184, 0.1)) border-box;
  backdrop-filter: blur(14px);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38);
  animation: fadeUp 0.7s ease both;
}
.hero h1 {
  margin: 0 0 6px 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15;
  background: linear-gradient(90deg, #34d399, #22d3ee, #34d399); background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 6s linear infinite;
}
.hero p { margin: 0 0 16px 0; color: var(--muted); font-size: 1rem; max-width: 760px; }

/* ---------- Badges ---------- */
.badge-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 6px; animation: fadeIn 0.8s ease both; }
.badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 5px 13px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 600; letter-spacing: 0.03em; color: var(--muted);
  border: 1px solid var(--border); background: rgba(15, 23, 42, 0.6);
  transition: all 0.25s ease;
}
.badge:hover { transform: translateY(-2px); border-color: rgba(52, 211, 153, 0.5); color: #e2e8f0; }
.badge.ok   { color: #6ee7b7; border-color: rgba(52, 211, 153, 0.38); background: rgba(16, 185, 129, 0.12); }
.badge.cyan { color: #67e8f9; border-color: rgba(34, 211, 238, 0.38); background: rgba(34, 211, 238, 0.10); }
.badge.warn { color: #fcd34d; border-color: rgba(251, 191, 36, 0.38); background: rgba(251, 191, 36, 0.10); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--emerald-2); animation: pulseDot 1.8s infinite; }

/* ---------- Section titles ---------- */
.section-title {
  display: flex; align-items: center; gap: 12px; margin: 26px 0 12px;
  font-size: 1.05rem; font-weight: 700; color: #f1f5f9; animation: fadeUp 0.6s ease both;
}
.section-title::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(148, 163, 184, 0.35), transparent); }

/* ---------- KPI cards (glassmorphism) ---------- */
.pf-head { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 18px 0 4px; animation: fadeUp 0.6s ease both; }
.pf-desc { color: var(--muted); font-size: 0.85rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 10px 0 8px; }
.kpi {
  --accent: #34d399; position: relative; overflow: hidden; padding: 20px 22px; border-radius: 18px;
  border: 1px solid transparent;
  background:
    linear-gradient(160deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.62)) padding-box,
    linear-gradient(135deg, rgba(148, 163, 184, 0.28), rgba(148, 163, 184, 0.06)) border-box;
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
  transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
  animation: fadeUp 0.65s ease both;
}
.kpi.emerald { --accent: #34d399; }
.kpi.cyan    { --accent: #22d3ee; }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--accent), transparent); }
.kpi::after  { content: ""; position: absolute; right: -40px; top: -40px; width: 120px; height: 120px; border-radius: 50%; background: radial-gradient(circle, var(--accent), transparent 70%); opacity: 0.10; transition: opacity 0.3s ease; }
.kpi:hover {
  transform: translateY(-6px) scale(1.02);
  background:
    linear-gradient(160deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.7)) padding-box,
    linear-gradient(135deg, var(--accent), rgba(148, 163, 184, 0.15)) border-box;
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.4), 0 0 28px -6px var(--accent);
}
.kpi:hover::after { opacity: 0.28; }
.kpi-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
.kpi-value { margin-top: 8px; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; font-family: 'JetBrains Mono', 'Inter', monospace; }
.kpi-sub   { margin-top: 4px; font-size: 0.8rem; color: var(--muted); }

/* ---------- Glass panels & tables ---------- */
.glass {
  border-radius: 18px; border: 1px solid var(--border); background: var(--card);
  backdrop-filter: blur(12px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
  transition: border-color 0.3s ease, box-shadow 0.3s ease; animation: fadeUp 0.7s ease both;
}
.glass:hover { border-color: rgba(52, 211, 153, 0.35); box-shadow: 0 14px 38px rgba(0, 0, 0, 0.32); }
.table-wrap { overflow-x: auto; padding: 6px; }
.dtable { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.92rem; }
.dtable th {
  text-align: left; padding: 14px 16px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); white-space: nowrap;
}
.dtable td { padding: 13px 16px; border-bottom: 1px solid rgba(148, 163, 184, 0.07); color: var(--text); font-variant-numeric: tabular-nums; white-space: nowrap; }
.dtable tbody tr { transition: background 0.2s ease; }
.dtable tbody tr:hover { background: rgba(52, 211, 153, 0.07); }
.dtable tbody tr:last-child td { border-bottom: none; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 10px; vertical-align: middle; }
.mono { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
.pos { color: #6ee7b7; } .neg { color: #fda4af; } .flat { color: var(--muted); }

/* Weight bars */
.wbar { position: relative; height: 28px; min-width: 150px; border-radius: 8px; background: rgba(148, 163, 184, 0.08); overflow: hidden; }
.wfill { position: absolute; inset: 0 auto 0 0; border-radius: 8px; transform-origin: left center; animation: growX 1s cubic-bezier(.22, 1, .36, 1) both; }
.f-emerald { background: linear-gradient(90deg, rgba(16, 185, 129, 0.75), rgba(52, 211, 153, 0.28)); }
.f-cyan    { background: linear-gradient(90deg, rgba(6, 182, 212, 0.75), rgba(34, 211, 238, 0.28)); }
.wbar span { position: relative; z-index: 1; display: block; line-height: 28px; padding: 0 10px; font-size: 0.82rem; font-weight: 600; }

/* ---------- Insight / feature cards ---------- */
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 8px 0; }
.info-card { padding: 20px 22px; border-radius: 18px; border: 1px solid var(--border); background: var(--card); backdrop-filter: blur(12px);
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease; animation: fadeUp 0.7s ease both; }
.info-card:hover { transform: translateY(-5px); border-color: rgba(34, 211, 238, 0.5); box-shadow: 0 16px 36px rgba(0, 0, 0, 0.35); }
.info-card .ic { font-size: 1.6rem; display: inline-block; animation: floaty 3.5s ease-in-out infinite; }
.info-card h4 { margin: 10px 0 6px; font-size: 1rem; font-weight: 700; }
.info-card p  { margin: 0; font-size: 0.88rem; color: var(--muted); line-height: 1.55; }
.info-card .big { font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-top: 6px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(11, 17, 32, 0.98));
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #cbd5e1; }
.brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 18px; animation: fadeIn 0.8s ease both; }
.logo { width: 42px; height: 42px; display: grid; place-items: center; font-size: 1.3rem; border-radius: 12px;
  background: linear-gradient(135deg, #10b981, #06b6d4); box-shadow: 0 8px 22px rgba(16, 185, 129, 0.35); }
.brand-name { font-weight: 800; font-size: 1.05rem; color: #f8fafc; line-height: 1.1; }
.brand-sub  { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }
.side-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 18px 0 4px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 4px; }
.chip { padding: 3px 10px; border-radius: 8px; font-size: 0.74rem; font-weight: 700; color: #a7f3d0;
  background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.3); transition: all 0.2s ease; }
.chip:hover { transform: translateY(-2px); background: rgba(16, 185, 129, 0.22); }

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], .stTextInput input, .stDateInput input {
  background-color: rgba(15, 23, 42, 0.7) !important; color: #f1f5f9 !important;
}
[data-baseweb="input"] { border-radius: 10px !important; border: 1px solid var(--border) !important; transition: border-color 0.2s ease, box-shadow 0.2s ease; }
[data-baseweb="input"]:hover { border-color: rgba(52, 211, 153, 0.45) !important; }
[data-baseweb="input"]:focus-within { border-color: var(--emerald-2) !important; box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.18) !important; }
.stSlider [role="slider"] { background: var(--emerald-2) !important; box-shadow: 0 0 0 6px rgba(52, 211, 153, 0.18); transition: box-shadow 0.2s ease; }
.stSlider [role="slider"]:hover { box-shadow: 0 0 0 10px rgba(52, 211, 153, 0.26); }
[data-testid="stSliderThumbValue"] { color: var(--emerald-2) !important; font-weight: 700; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  width: 100%; padding: 0.7rem 1rem; border-radius: 12px; border: 1px solid rgba(167, 243, 208, 0.45);
  background: linear-gradient(135deg, #10b981, #06b6d4); color: #04221c; font-weight: 700; letter-spacing: 0.02em;
  box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25); transition: all 0.25s ease;
}
.stButton > button p, .stDownloadButton > button p { color: inherit; font-weight: 700; }
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px) scale(1.01); border-color: #a7f3d0; color: #022c22;
  box-shadow: 0 14px 34px rgba(34, 211, 238, 0.35);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(0.99); }
.stDownloadButton > button { background: rgba(30, 41, 59, 0.7); color: #a7f3d0; box-shadow: none; }
.stDownloadButton > button:hover { background: linear-gradient(135deg, #10b981, #06b6d4); }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; padding: 6px; border-radius: 16px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(10px); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { height: auto; padding: 10px 18px; border-radius: 11px; background: transparent; transition: all 0.25s ease; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; font-size: 0.92rem; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(52, 211, 153, 0.10); transform: translateY(-1px); }
.stTabs [data-baseweb="tab"]:hover p { color: #e2e8f0; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(16, 185, 129, 0.9), rgba(6, 182, 212, 0.85)) !important; box-shadow: 0 6px 18px rgba(16, 185, 129, 0.3); }
.stTabs [aria-selected="true"] p { color: #04221c !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; animation: fadeUp 0.55s ease both; }

/* ---------- Native elements ---------- */
[data-testid="stPlotlyChart"] {
  border-radius: 18px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.45); padding: 8px;
  box-shadow: 0 12px 34px rgba(0, 0, 0, 0.3); overflow: hidden; animation: fadeIn 0.9s ease both;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover { border-color: rgba(52, 211, 153, 0.35); box-shadow: 0 16px 40px rgba(0, 0, 0, 0.38); }
[data-testid="stExpander"], [data-testid="stStatus"] { border: 1px solid var(--border) !important; border-radius: 14px !important; background: rgba(30, 41, 59, 0.4); }
[data-testid="stAlert"] { border-radius: 14px; backdrop-filter: blur(8px); animation: fadeUp 0.5s ease both; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }

.footer { margin-top: 40px; padding-top: 16px; text-align: center; font-size: 0.78rem; color: var(--muted); border-top: 1px solid var(--border); }

/* ---------- Accessibility: respect reduced-motion preference ---------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
/* ---------- Small screens ---------- */
@media (max-width: 640px) {
  .hero { padding: 20px; } .hero h1 { font-size: 1.6rem; } .kpi-value { font-size: 1.6rem; }
}
</style>
"""


def esc(text) -> str:
    """HTML-escape user-influenced text (also neutralises '$' so Markdown won't treat it as LaTeX)."""
    return _escape(str(text)).replace("$", "&#36;")


def render_html(markup: str) -> None:
    """
    Render an HTML snippet through st.markdown.
    Lines are flattened because Markdown would treat indented / blank-line-separated
    HTML as code blocks. Every snippet passed in must start with a block-level tag.
    """
    flat = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_title(text: str) -> None:
    render_html(f'<div class="section-title">{esc(text)}</div>')


def render_hero() -> None:
    render_html(
        """
        <div class="hero">
          <h1>Stock Portfolio Allocation Optimizer</h1>
          <p>Build an efficient portfolio with Harry Markowitz's Mean-Variance framework —
          find the allocation that maximises risk-adjusted return, or the one that minimises risk.</p>
          <div class="badge-row">
            <span class="badge ok"><span class="dot"></span>Live market data</span>
            <span class="badge cyan">Markowitz MVO</span>
            <span class="badge">SciPy SLSQP</span>
            <span class="badge">Long-only · Fully invested</span>
          </div>
        </div>
        """
    )


def render_landing() -> None:
    """Empty-state shown before the first optimisation run."""
    section_title("How it works")
    render_html(
        """
        <div class="card-grid">
          <div class="info-card" style="animation-delay:.05s"><span class="ic">🧺</span>
            <h4>1 · Choose your assets</h4>
            <p>Enter two or more ticker symbols in the sidebar, separated by commas (e.g. AAPL, MSFT, TSLA).</p></div>
          <div class="info-card" style="animation-delay:.15s"><span class="ic">🎛️</span>
            <h4>2 · Set the parameters</h4>
            <p>Pick the historical window and the annual risk-free rate used for the Sharpe ratio.</p></div>
          <div class="info-card" style="animation-delay:.25s"><span class="ic">🚀</span>
            <h4>3 · Run the optimization</h4>
            <p>Get the Max Sharpe and Minimum Variance portfolios, plus an interactive efficient frontier.</p></div>
        </div>
        """
    )


def render_footer() -> None:
    render_html(
        '<div class="footer">Built with Streamlit · yfinance · SciPy · Plotly &nbsp;|&nbsp; '
        "For educational purposes only — not financial advice. Past performance does not guarantee future results.</div>"
    )


# --------------------------------------------------------------------------- #
# Data Acquisition Layer
# --------------------------------------------------------------------------- #
def parse_tickers(raw: str) -> list[str]:
    """Turn 'aapl, msft ,TSLA' into ['AAPL', 'MSFT', 'TSLA'] (deduplicated, ordered)."""
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return list(dict.fromkeys(tickers))  # remove duplicates, keep order


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Returns a DataFrame (rows = dates, columns = tickers) with NaNs removed.
    Raises InputError for invalid tickers / empty results.
    """
    # yfinance treats `end` as exclusive, so add one day to include the chosen date.
    raw = yf.download(
        list(tickers),
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise InputError(
            "No price data was returned. Check the ticker symbols and the date range."
        )

    # `Close` is a DataFrame for multiple tickers and (in some versions) also for one.
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Identify tickers that returned no data at all (invalid / delisted symbols).
    empty_cols = [c for c in prices.columns if prices[c].isna().all()]
    if empty_cols:
        raise InputError(
            f"No data found for: {', '.join(empty_cols)}. "
            "Please check that these ticker symbols are valid for the selected dates."
        )

    # Clean: drop rows with any missing value so all assets share the same dates.
    prices = prices.dropna()
    if prices.empty:
        raise InputError(
            "The selected tickers have no overlapping trading days in this date range."
        )

    return prices[list(tickers)]  # keep the user's ticker order


# --------------------------------------------------------------------------- #
# Processing Layer – statistics
# --------------------------------------------------------------------------- #
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def annualised_stats(log_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualised mean-return vector and covariance matrix."""
    mean_returns = log_returns.mean().values * TRADING_DAYS
    cov_matrix = log_returns.cov().values * TRADING_DAYS
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix) -> tuple[float, float]:
    """Expected return and volatility (std-dev) for a given weight vector."""
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights @ cov_matrix @ weights))
    return ret, vol


# --------------------------------------------------------------------------- #
# Processing Layer – optimisation (SciPy SLSQP)
# --------------------------------------------------------------------------- #
def _base_setup(n_assets: int):
    """Common bounds, constraints and starting point for all optimisations."""
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))              # 0 <= w_i <= 1
    sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}    # sum(w) == 1
    x0 = np.repeat(1.0 / n_assets, n_assets)                         # equal-weight start
    return bounds, sum_to_one, x0


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate) -> np.ndarray:
    """Maximise the Sharpe ratio  (R_p - R_f) / sigma_p  by minimising its negative."""
    n = len(mean_returns)
    bounds, sum_to_one, x0 = _base_setup(n)

    def neg_sharpe(w):
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol if vol > 1e-12 else 1e6

    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=[sum_to_one]
    )
    if not result.success:
        raise RuntimeError(f"Max Sharpe optimisation failed: {result.message}")
    return result.x


def min_variance_portfolio(mean_returns, cov_matrix) -> np.ndarray:
    """Minimise total portfolio variance."""
    n = len(mean_returns)
    bounds, sum_to_one, x0 = _base_setup(n)

    result = minimize(
        lambda w: w @ cov_matrix @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_to_one],
    )
    if not result.success:
        raise RuntimeError(f"Minimum variance optimisation failed: {result.message}")
    return result.x


def efficient_frontier(mean_returns, cov_matrix, min_var_ret) -> tuple[list, list]:
    """
    Trace the efficient frontier: for each target return between the
    minimum-variance portfolio's return and the best single asset's return,
    find the lowest-volatility portfolio that achieves it.
    """
    n = len(mean_returns)
    bounds, sum_to_one, x0 = _base_setup(n)
    targets = np.linspace(min_var_ret, mean_returns.max(), N_FRONTIER_POINTS)

    vols, rets = [], []
    for target in targets:
        constraints = [
            sum_to_one,
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        res = minimize(
            lambda w: w @ cov_matrix @ w,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:  # silently skip any target the solver could not reach
            r, v = portfolio_performance(res.x, mean_returns, cov_matrix)
            rets.append(r)
            vols.append(v)
            x0 = res.x   # warm-start the next target for stability
    return vols, rets


def random_portfolios(mean_returns, cov_matrix, risk_free_rate, n=N_RANDOM_PORTFOLIOS):
    """Monte-Carlo cloud of long-only portfolios (for visual context)."""
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(len(mean_returns)), size=n)  # rows sum to 1
    rets = weights @ mean_returns
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix, weights))
    sharpes = (rets - risk_free_rate) / vols
    return vols, rets, sharpes


# --------------------------------------------------------------------------- #
# Output helpers – tables (data) and HTML components
# --------------------------------------------------------------------------- #
def build_summary_table(portfolios: dict, mean_returns, cov_matrix, rf) -> pd.DataFrame:
    """Return / volatility / Sharpe for each named portfolio."""
    rows = {}
    for name, w in portfolios.items():
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        rows[name] = {
            "Expected Annual Return": ret,
            "Annual Volatility": vol,
            "Sharpe Ratio": (ret - rf) / vol if vol > 1e-12 else np.nan,
        }
    return pd.DataFrame(rows).T


def build_allocation_table(portfolios: dict, tickers: list[str]) -> pd.DataFrame:
    """Asset weights side by side for each portfolio."""
    return pd.DataFrame(
        {name: np.round(w, 6) for name, w in portfolios.items()}, index=tickers
    )


def table_html(headers: list[str], rows: list[list[str]]) -> str:
    """Build a styled glass table. Row cells must already be HTML-safe strings."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        '<div class="glass table-wrap"><table class="dtable">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def summary_table_html(summary: pd.DataFrame) -> str:
    rows = []
    for name, r in summary.iterrows():
        color = PORTFOLIO_STYLE[name]["color"]
        rows.append([
            f'<span class="swatch" style="background:{color}"></span><b>{esc(name)}</b>',
            f'<span class="mono">{r["Expected Annual Return"]:.2%}</span>',
            f'<span class="mono">{r["Annual Volatility"]:.2%}</span>',
            f'<span class="mono" style="color:{color}">{r["Sharpe Ratio"]:.3f}</span>',
        ])
    return table_html(["Portfolio", "Expected Annual Return", "Annual Volatility", "Sharpe Ratio"], rows)


def weight_bar_html(weight: float, css_class: str) -> str:
    width = max(0.0, min(weight, 1.0)) * 100
    return (
        f'<div class="wbar"><div class="wfill {css_class}" style="width:{width:.2f}%"></div>'
        f"<span>{weight:.2%}</span></div>"
    )


def allocation_table_html(allocation: pd.DataFrame, color_map: dict) -> str:
    names = list(allocation.columns)
    rows = []
    for ticker, r in allocation.iterrows():
        cells = [f'<span class="swatch" style="background:{color_map[ticker]}"></span><b>{esc(ticker)}</b>']
        for name in names:
            cells.append(weight_bar_html(float(r[name]), "f-" + PORTFOLIO_STYLE[name]["accent"]))
        diff = float(r[names[0]] - r[names[1]]) * 100
        cls = "pos" if diff > 0.05 else "neg" if diff < -0.05 else "flat"
        cells.append(f'<span class="mono {cls}">{diff:+.1f} pp</span>')
        rows.append(cells)
    headers = ["Asset", *[f"{n} Weight" for n in names], f"Δ ({names[0]} − {names[1]})"]
    return table_html(headers, rows)


def kpi_card(label: str, value: str, sub: str, accent: str, delay: float) -> str:
    return (
        f'<div class="kpi {accent}" style="animation-delay:{delay:.2f}s">'
        f'<div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
    )


def render_portfolio_kpis(name, weights, mean_returns, cov_matrix, rf, base_delay=0.0) -> None:
    """Three glass KPI cards (return / volatility / Sharpe) for one portfolio."""
    style = PORTFOLIO_STYLE[name]
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - rf) / vol if vol > 1e-12 else float("nan")
    cards = "".join([
        kpi_card("Expected Return", f"{ret:.2%}", "annualised, from log returns", style["accent"], base_delay),
        kpi_card("Volatility", f"{vol:.2%}", "annualised standard deviation", style["accent"], base_delay + 0.08),
        kpi_card("Sharpe Ratio", f"{sharpe:.2f}", f"risk-free rate = {rf:.2%}", style["accent"], base_delay + 0.16),
    ])
    render_html(
        f'<div><div class="pf-head"><span class="badge {style["badge"]}">{style["icon"]} {esc(name)} Portfolio</span>'
        f'<span class="pf-desc">{esc(style["desc"])}</span></div><div class="kpi-grid">{cards}</div></div>'
    )


def render_insights(portfolios, tickers, mean_returns, cov_matrix) -> None:
    """Small 'key takeaways' cards derived from the optimised weights."""
    w_ms, w_mv = portfolios["Max Sharpe"], portfolios["Min Variance"]
    ret_ms, vol_ms = portfolio_performance(w_ms, mean_returns, cov_matrix)
    ret_mv, vol_mv = portfolio_performance(w_mv, mean_returns, cov_matrix)
    top_ms, top_mv = int(np.argmax(w_ms)), int(np.argmax(w_mv))
    eff_ms, eff_mv = 1.0 / np.sum(np.square(w_ms)), 1.0 / np.sum(np.square(w_mv))

    cards = [
        ("🏆", "Top holding · Max Sharpe", f"{tickers[top_ms]} · {w_ms[top_ms]:.1%}",
         "Largest position in the tangency portfolio."),
        ("🛡️", "Top holding · Min Variance", f"{tickers[top_mv]} · {w_mv[top_mv]:.1%}",
         "Largest position in the lowest-risk portfolio."),
        ("🧩", "Effective # of holdings", f"{eff_ms:.1f} vs {eff_mv:.1f}",
         "Max Sharpe vs Min Variance (1 / Σw²). Higher means more diversified."),
        ("⚖️", "Risk / return trade-off", f"{(vol_ms - vol_mv) * 100:+.1f} pp vol",
         f"Moving from Min Variance to Max Sharpe adds {(ret_ms - ret_mv) * 100:+.1f} pp of expected return."),
    ]
    html_cards = "".join(
        f'<div class="info-card" style="animation-delay:{i * 0.08:.2f}s"><span class="ic">{icon}</span>'
        f'<h4>{esc(title)}</h4><div class="big">{esc(big)}</div><p>{esc(text)}</p></div>'
        for i, (icon, title, big, text) in enumerate(cards)
    )
    render_html(f'<div class="card-grid">{html_cards}</div>')


# --------------------------------------------------------------------------- #
# Output helpers – Plotly figures (dark, translucent, animated)
# --------------------------------------------------------------------------- #
def _dark_layout(fig: go.Figure, height: int = 640, **kwargs) -> go.Figure:
    """Apply the shared dark / translucent / glass look to any Plotly figure."""
    # Shared defaults. Any key supplied by the caller (e.g. `legend`, `font`)
    # overrides the default instead of being passed twice, which would raise
    # "got multiple values for keyword argument".
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color=TEXT, size=13),
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.96)", bordercolor=EMERALD,
            font=dict(family="Inter, sans-serif", color="#f1f5f9", size=13),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    )
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def build_frontier_figure(
    tickers, mean_returns, cov_matrix, rf, portfolios, frontier, cloud
) -> go.Figure:
    """
    Interactive Efficient Frontier.
    * Random-portfolio cloud coloured by Sharpe ratio
    * Glowing frontier line  (a replayable draw-in animation is attached via frames)
    * Capital Market Line and individual assets
    * Optimised portfolios as stars with pulsing translucent halos
    """
    fig = go.Figure()

    # 1) Random portfolio cloud, coloured by Sharpe ratio
    c_vols, c_rets, c_sharpes = cloud
    fig.add_trace(
        go.Scatter(
            x=c_vols, y=c_rets, mode="markers", name="Random portfolios",
            marker=dict(
                size=5, color=c_sharpes, opacity=0.55,
                colorscale=[[0, "#334155"], [0.5, "#6366f1"], [1, "#c084fc"]],
                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(color=MUTED)),
                    tickfont=dict(color=MUTED), thickness=10, len=0.6, outlinewidth=0,
                ),
            ),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra>Random portfolio</extra>",
        )
    )

    # 2) Efficient frontier: soft glow underlay + crisp line
    f_vols, f_rets = frontier
    fig.add_trace(go.Scatter(
        x=f_vols, y=f_rets, mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="rgba(52,211,153,0.22)", width=14, shape="spline"),
    ))
    glow_idx = len(fig.data) - 1
    fig.add_trace(go.Scatter(
        x=f_vols, y=f_rets, mode="lines", name="Efficient Frontier",
        line=dict(color="#f8fafc", width=3.5, shape="spline"),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra>Efficient Frontier</extra>",
    ))
    line_idx = len(fig.data) - 1

    # 3) Individual assets
    asset_vols = np.sqrt(np.diag(cov_matrix))
    fig.add_trace(
        go.Scatter(
            x=asset_vols, y=mean_returns, mode="markers+text", text=tickers,
            textposition="top center", textfont=dict(color=AMBER, size=12),
            name="Individual assets",
            marker=dict(size=11, color=AMBER, symbol="diamond", line=dict(color="#0b1120", width=1.5)),
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    # 4) Capital Market Line (risk-free asset -> Max Sharpe portfolio)
    ms_ret, ms_vol = portfolio_performance(portfolios["Max Sharpe"], mean_returns, cov_matrix)
    if ms_vol > 1e-12:
        x_end = max(max(c_vols), ms_vol) * 1.05
        slope = (ms_ret - rf) / ms_vol
        fig.add_trace(
            go.Scatter(
                x=[0, x_end], y=[rf, rf + slope * x_end], mode="lines",
                name="Capital Market Line",
                line=dict(color="rgba(148,163,184,0.75)", dash="dash", width=1.5),
                hoverinfo="skip",
            )
        )

    # 5) Optimised portfolios: translucent halo (pulses) + star marker
    halo_idx = []
    for name, w in portfolios.items():
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        color = PORTFOLIO_STYLE[name]["color"]
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=34, color=color, opacity=0.25),
        ))
        halo_idx.append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers", name=f"{name} Portfolio",
            marker=dict(size=21, color=color, symbol="star", line=dict(color="#f8fafc", width=1.6)),
            hovertemplate=(
                f"<b>{name}</b><br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}"
                f"<br>Sharpe: {(ret - rf) / vol:.2f}<extra></extra>"
            ),
        ))

    # ---- Replayable animation: frontier draws in, then the halos pulse ---- #
    frames = []
    for k in range(2, len(f_vols) + 1):
        frames.append(go.Frame(
            name=f"draw{k}", traces=[glow_idx, line_idx],
            data=[go.Scatter(x=f_vols[:k], y=f_rets[:k]), go.Scatter(x=f_vols[:k], y=f_rets[:k])],
        ))
    for i in range(0, 37):                       # three smooth pulse cycles
        s = 0.5 * (1 - np.cos(2 * np.pi * i / 12))
        frames.append(go.Frame(
            name=f"pulse{i}", traces=halo_idx,
            data=[go.Scatter(marker=dict(size=30 + 20 * s, opacity=0.34 - 0.24 * s)) for _ in halo_idx],
        ))
    fig.frames = frames

    _dark_layout(
        fig,
        title=dict(text="Efficient Frontier · Mean-Variance Space", x=0.0, xanchor="left", y=0.97,
                   font=dict(size=18, color="#f1f5f9")),
        margin=dict(l=20, r=20, t=95, b=20),
        xaxis=dict(title="Annualised Volatility (Risk)", tickformat=".0%", gridcolor=GRID,
                   zeroline=False, linecolor=GRID, title_font=dict(color=MUTED)),
        yaxis=dict(title="Annualised Expected Return", tickformat=".0%", gridcolor=GRID,
                   zeroline=False, linecolor=GRID, title_font=dict(color=MUTED)),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        updatemenus=[dict(
            type="buttons", direction="left", x=1.0, xanchor="right", y=1.13, yanchor="bottom",
            bgcolor="rgba(16,185,129,0.18)", bordercolor="rgba(52,211,153,0.55)",
            font=dict(color="#a7f3d0", size=12), pad=dict(r=8, l=8, t=4, b=4),
            buttons=[dict(
                label="▶ Replay animation", method="animate",
                args=[None, dict(frame=dict(duration=28, redraw=False),
                                 transition=dict(duration=0), fromcurrent=False, mode="immediate")],
            )],
        )],
    )
    return fig


def build_donut(name: str, weights, tickers, color_map) -> go.Figure:
    """Donut chart of one portfolio's allocation (assets under 0.5% are hidden)."""
    keep = [i for i, w in enumerate(weights) if w > 0.005]
    labels = [tickers[i] for i in keep]
    values = [float(weights[i]) for i in keep]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.64, sort=False, direction="clockwise",
        marker=dict(colors=[color_map[t] for t in labels], line=dict(color="#0b1120", width=3)),
        textinfo="percent", textfont=dict(color="#0b1120", size=13),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    style = PORTFOLIO_STYLE[name]
    _dark_layout(
        fig, height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=f"{style['icon']} {name}", x=0.5, xanchor="center", font=dict(size=16, color="#f1f5f9")),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        annotations=[dict(text=f"<b>{len(labels)}</b><br><span style='font-size:11px'>assets</span>",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=style["color"]))],
    )
    return fig


def build_correlation_heatmap(log_returns: pd.DataFrame) -> go.Figure:
    corr = log_returns.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=[[0, "#f472b6"], [0.5, "#1e293b"], [1, EMERALD]],
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(color="#f1f5f9"),
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED)),
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        xgap=3, ygap=3,
    ))
    _dark_layout(
        fig, height=460, margin=dict(l=20, r=20, t=55, b=20),
        title=dict(text="Correlation of Daily Log Returns", x=0, font=dict(size=16, color="#f1f5f9")),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_performance_chart(prices: pd.DataFrame, color_map) -> go.Figure:
    """Price history rebased to 100 at the start date."""
    rebased = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for t in rebased.columns:
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased[t], mode="lines", name=t,
            line=dict(color=color_map[t], width=2),
            hovertemplate=f"<b>{esc(t)}</b><br>%{{x|%b %d, %Y}}<br>Value: %{{y:.1f}}<extra></extra>",
        ))
    _dark_layout(
        fig, height=440, margin=dict(l=20, r=20, t=55, b=20), hovermode="x unified",
        title=dict(text="Growth of 100 · Rebased Price Performance", x=0, font=dict(size=16, color="#f1f5f9")),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, title="Rebased value"),
    )
    return fig


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(tickers: list[str], start: date, end: date) -> None:
    """Raise InputError for anything we can detect before hitting the network."""
    if len(tickers) < 2:
        raise InputError("Please enter at least two distinct ticker symbols.")
    if len(tickers) > 30:
        raise InputError("Please limit the portfolio to 30 tickers or fewer.")
    if start >= end:
        raise InputError("The start date must be earlier than the end date.")
    if end > date.today():
        raise InputError("The end date cannot be in the future.")
    if (end - start).days < 60:
        raise InputError("Please select a date range of at least two months.")


# --------------------------------------------------------------------------- #
# Results dashboard (tabs)
# --------------------------------------------------------------------------- #
def render_results(res: dict) -> None:
    tickers, prices, log_returns = res["tickers"], res["prices"], res["log_returns"]
    mean_returns, cov_matrix = res["mean_returns"], res["cov_matrix"]
    portfolios, rf = res["portfolios"], res["rf"]
    color_map = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(tickers)}

    # Status badges
    render_html(
        f"""
        <div class="badge-row">
          <span class="badge ok"><span class="dot"></span>Optimization complete</span>
          <span class="badge cyan">{len(tickers)} assets</span>
          <span class="badge">{len(log_returns):,} trading days</span>
          <span class="badge">{esc(prices.index[0].date())} → {esc(prices.index[-1].date())}</span>
          <span class="badge warn">Risk-free {rf:.2%}</span>
        </div>
        """
    )

    tab_summary, tab_frontier, tab_alloc, tab_data = st.tabs([
        "📊 Executive Summary",
        "📈 Efficient Frontier Chart",
        "💼 Asset Allocation Weights",
        "📋 Raw Historical Data",
    ])

    summary = build_summary_table(portfolios, mean_returns, cov_matrix, rf)
    allocation = build_allocation_table(portfolios, tickers)

    # ---------------------------- Executive Summary --------------------------- #
    with tab_summary:
        render_portfolio_kpis("Max Sharpe", portfolios["Max Sharpe"], mean_returns, cov_matrix, rf, 0.0)
        render_portfolio_kpis("Min Variance", portfolios["Min Variance"], mean_returns, cov_matrix, rf, 0.25)
        section_title("Portfolio Summary Metrics")
        render_html(summary_table_html(summary))
        section_title("Key Takeaways")
        render_insights(portfolios, tickers, mean_returns, cov_matrix)

    # ---------------------------- Efficient Frontier -------------------------- #
    with tab_frontier:
        fig = build_frontier_figure(
            tickers, mean_returns, cov_matrix, rf, portfolios, res["frontier"], res["cloud"]
        )
        st.plotly_chart(fig, config={"displaylogo": False, "responsive": True})
        st.caption(
            "Each dot in the cloud is a random long-only portfolio. The white curve is the efficient frontier; "
            "the dashed line is the Capital Market Line. Use **▶ Replay animation** to redraw the frontier."
        )

    # ------------------------------ Allocation -------------------------------- #
    with tab_alloc:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(build_donut("Max Sharpe", portfolios["Max Sharpe"], tickers, color_map),
                            config={"displaylogo": False})
        with col2:
            st.plotly_chart(build_donut("Min Variance", portfolios["Min Variance"], tickers, color_map),
                            config={"displaylogo": False})
        section_title("Asset Allocation Comparison")
        render_html(allocation_table_html(allocation, color_map))
        st.write("")
        st.download_button(
            "⬇️ Download allocation (CSV)",
            data=allocation.to_csv(index_label="Ticker").encode("utf-8"),
            file_name="portfolio_allocation.csv",
            mime="text/csv",
        )

    # ------------------------------- Raw data --------------------------------- #
    with tab_data:
        st.plotly_chart(build_performance_chart(prices, color_map), config={"displaylogo": False})
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(build_correlation_heatmap(log_returns), config={"displaylogo": False})
        with c2:
            section_title("Per-Asset Statistics (annualised)")
            vols = np.sqrt(np.diag(cov_matrix))
            rows = [[
                f'<span class="swatch" style="background:{color_map[t]}"></span><b>{esc(t)}</b>',
                f'<span class="mono">{mean_returns[i]:.2%}</span>',
                f'<span class="mono">{vols[i]:.2%}</span>',
                f'<span class="mono">{(mean_returns[i] - rf) / vols[i]:.2f}</span>',
            ] for i, t in enumerate(tickers)]
            render_html(table_html(["Asset", "Exp. Return", "Volatility", "Sharpe"], rows))
        section_title("Adjusted Closing Prices")
        st.dataframe(prices.round(2))
        st.download_button(
            "⬇️ Download prices (CSV)",
            data=prices.to_csv().encode("utf-8"),
            file_name="historical_prices.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Streamlit application
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
    inject_css()

    # ------------------------- Input Layer (Sidebar) ------------------------ #
    with st.sidebar:
        render_html(
            """
            <div class="brand"><div class="logo">📈</div>
              <div><div class="brand-name">PortfolioLab</div>
              <div class="brand-sub">Markowitz Optimizer</div></div></div>
            """
        )
        render_html('<div class="side-label">Assets</div>')
        ticker_input = st.text_input(
            "Ticker symbols (comma-separated)", value="AAPL, MSFT, TSLA, JPM, XOM"
        )
        chips = "".join(f'<span class="chip">{esc(t)}</span>' for t in parse_tickers(ticker_input)[:30])
        if chips:
            render_html(f'<div class="chips">{chips}</div>')

        render_html('<div class="side-label">Time window</div>')
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=5 * 365))
        end_date = st.date_input("End date", value=date.today())

        render_html('<div class="side-label">Parameters</div>')
        risk_free_pct = st.slider(
            "Risk-free rate (annual, %)", min_value=0.0, max_value=10.0,
            value=4.0, step=0.1,
        )
        st.write("")
        run = st.button("Run Optimization", type="primary")

    render_hero()
    risk_free_rate = risk_free_pct / 100.0

    # ----------------------- Core logic (error-guarded) --------------------- #
    if run:
        st.session_state.pop("result", None)   # clear any previous result
        error = None                           # (kind, message) if something fails

        with st.status("Running optimization…", expanded=True) as status:
            try:
                st.write("🔎 Validating inputs…")
                tickers = parse_tickers(ticker_input)
                validate_inputs(tickers, start_date, end_date)

                st.write("📡 Downloading market data from Yahoo Finance…")
                prices = fetch_prices(tuple(tickers), start_date, end_date)

                st.write("🧮 Computing log returns and the covariance matrix…")
                log_returns = compute_log_returns(prices)
                if len(log_returns) < MIN_OBSERVATIONS:
                    raise InputError(
                        f"Only {len(log_returns)} trading days of overlapping data were found; "
                        f"at least {MIN_OBSERVATIONS} are required. Try a longer date range."
                    )
                mean_returns, cov_matrix = annualised_stats(log_returns)

                st.write("⚙️ Solving Max Sharpe & Minimum Variance with SLSQP…")
                w_sharpe = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
                w_minvar = min_variance_portfolio(mean_returns, cov_matrix)
                portfolios = {"Max Sharpe": w_sharpe, "Min Variance": w_minvar}

                st.write("📈 Tracing the efficient frontier…")
                minvar_ret, _ = portfolio_performance(w_minvar, mean_returns, cov_matrix)
                frontier = efficient_frontier(mean_returns, cov_matrix, minvar_ret)
                cloud = random_portfolios(mean_returns, cov_matrix, risk_free_rate)

                # Keep everything in session state so the dashboard survives reruns
                # (e.g. clicking a download button or switching tabs).
                st.session_state["result"] = dict(
                    tickers=tickers, prices=prices, log_returns=log_returns,
                    mean_returns=mean_returns, cov_matrix=cov_matrix,
                    portfolios=portfolios, frontier=frontier, cloud=cloud, rf=risk_free_rate,
                )
                status.update(label="Optimization complete ✔", state="complete", expanded=False)

            # Known, user-correctable problems -> friendly message
            except InputError as exc:
                error = ("input", str(exc))
                status.update(label="Input problem", state="error", expanded=False)
            # Optimiser failures (e.g. singular covariance matrix)
            except RuntimeError as exc:
                error = ("optim", str(exc))
                status.update(label="Optimization failed", state="error", expanded=False)
            # Anything unexpected (network down, yfinance changes, etc.)
            except Exception as exc:  # noqa: BLE001 – deliberate catch-all so the app never crashes
                error = ("other", f"{type(exc).__name__}: {exc}")
                status.update(label="Unexpected error", state="error", expanded=False)

        if error:
            kind, message = error
            if kind == "input":
                st.error(f"⚠️ {message}")
            elif kind == "optim":
                st.error(f"⚠️ Optimization error: {message}")
            else:
                st.error(
                    "⚠️ Something went wrong while processing your request. "
                    "Please verify your tickers, dates, and internet connection, then try again."
                )
                with st.expander("Technical details"):
                    st.code(message)
            render_footer()
            return

    # --------------------------- Output Layer --------------------------- #
    result = st.session_state.get("result")
    if result is None:
        render_landing()
    else:
        render_results(result)
    render_footer()


if __name__ == "__main__":
    main()
