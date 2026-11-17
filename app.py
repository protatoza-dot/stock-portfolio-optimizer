"""
Stock Portfolio Allocation Optimizer  ·  Premium Dark UI Edition
================================================================
A Streamlit app implementing Harry Markowitz's Mean-Variance Model.

Architecture
------------
1. Input Layer        -> Sidebar: searchable ticker multiselect (state-backed), dates, risk-free rate, run button
2. Data Layer         -> yfinance download + Pandas cleaning
3. Processing Layer   -> Log returns, annualised statistics, SciPy SLSQP optimisation
4. Output Layer       -> Tabbed dashboard: KPI cards, animated Plotly charts, tables
5. Error Handling     -> try/except blocks that surface friendly st.error alerts
6. Presentation       -> Injected CSS (glassmorphism, animations, hover effects)
7. Internationalisation -> English (default) / Thai text dictionary, language saved in st.session_state

Run with:  streamlit run app.py
(Optional but recommended: keep `.streamlit/config.toml` next to this file so that
native Streamlit widgets/popups also use the dark theme.)
"""

import inspect
import re
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
TEXT_COLOR, MUTED, GRID = "#e2e8f0", "#94a3b8", "rgba(148,163,184,0.12)"   # colour tokens (plain CSS colour strings)
FONT_SANS = "Plus Jakarta Sans, Inter, Noto Sans Thai, sans-serif"      # headings, labels, body (Thai glyph fallback)
FONT_MONO = "JetBrains Mono, Noto Sans Thai, ui-monospace, Menlo, monospace"  # numbers, prices, currency
PALETTE = [
    "#34d399", "#22d3ee", "#818cf8", "#f472b6", "#fbbf24", "#fb7185",
    "#a78bfa", "#2dd4bf", "#60a5fa", "#f97316", "#a3e635", "#e879f9",
]
PORTFOLIO_STYLE = {
    # Dict keys ("Max Sharpe", "Min Variance") are internal ids. Visible names come from TRANSLATIONS (pf_*_main/sub/short/desc)
    # so they follow the selected language.
    "Max Sharpe": dict(ident="max", color=EMERALD, accent="emerald", badge="ok", icon="⭐"),
    "Min Variance": dict(ident="min", color=CYAN, accent="cyan", badge="cyan", icon="🛡️"),
}


# --------------------------------------------------------------------------- #
# Chart & currency settings
# --------------------------------------------------------------------------- #
# Passed to every st.plotly_chart(): wheel-zoom at the cursor, always-visible mode bar,
# responsive sizing. Touch devices get drag-to-pan and pinch-to-zoom from Plotly itself.
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "responsive": True,
    "displaylogo": False,
}
DEFAULT_CAPITAL = 1_000_000.0   # in THB by default
DEFAULT_FX = 35.0               # THB per 1 USD (editable in the sidebar)


# --------------------------------------------------------------------------- #
# Ticker universe (searchable suggestions for the multiselect)
# --------------------------------------------------------------------------- #
MAX_TICKERS = 30
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"]
TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")   # sane Yahoo-style symbol
TICKER_ALIASES = {"TSMC": "TSM", "BRK.B": "BRK-B", "BRK.A": "BRK-A", "GOOGLE": "GOOGL"}

# (symbol, display name), listed in APPROXIMATE market-cap / prominence order
# (ETFs by assets under management, crypto by market cap). The order is only used
# to rank suggestions, so the largest names surface first when you type.
_TICKER_UNIVERSE = [
    ("NVDA", "NVIDIA"), ("MSFT", "Microsoft"), ("AAPL", "Apple"),
    ("GOOGL", "Alphabet (A)"), ("GOOG", "Alphabet (C)"), ("AMZN", "Amazon"),
    ("BTC-USD", "Bitcoin"), ("META", "Meta Platforms"), ("AVGO", "Broadcom"),
    ("TSLA", "Tesla"), ("BRK-B", "Berkshire Hathaway B"), ("TSM", "TSMC"),
    ("SPY", "SPDR S&P 500 ETF"), ("VOO", "Vanguard S&P 500 ETF"),
    ("IVV", "iShares Core S&P 500"), ("VTI", "Vanguard Total Market"),
    ("QQQ", "Invesco QQQ (Nasdaq-100)"), ("LLY", "Eli Lilly"), ("WMT", "Walmart"),
    ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("ORCL", "Oracle"), ("MA", "Mastercard"),
    ("XOM", "Exxon Mobil"), ("NFLX", "Netflix"), ("COST", "Costco"),
    ("JNJ", "Johnson & Johnson"), ("PLTR", "Palantir"), ("ETH-USD", "Ethereum"),
    ("HD", "Home Depot"), ("BAC", "Bank of America"), ("ABBV", "AbbVie"),
    ("PG", "Procter & Gamble"), ("UNH", "UnitedHealth"), ("KO", "Coca-Cola"),
    ("CVX", "Chevron"), ("GE", "GE Aerospace"), ("AMD", "AMD"), ("CSCO", "Cisco"),
    ("TMUS", "T-Mobile US"), ("WFC", "Wells Fargo"), ("MS", "Morgan Stanley"),
    ("NOW", "ServiceNow"), ("IBM", "IBM"), ("CRM", "Salesforce"), ("ABT", "Abbott Labs"),
    ("AXP", "American Express"), ("MCD", "McDonald's"), ("LIN", "Linde"), ("MRK", "Merck"),
    ("DIS", "Disney"), ("PEP", "PepsiCo"), ("UBER", "Uber"), ("ISRG", "Intuitive Surgical"),
    ("INTU", "Intuit"), ("T", "AT&T"), ("GS", "Goldman Sachs"), ("VZ", "Verizon"),
    ("RTX", "RTX"), ("BKNG", "Booking Holdings"), ("QCOM", "Qualcomm"), ("ADBE", "Adobe"),
    ("TXN", "Texas Instruments"), ("NEE", "NextEra Energy"), ("PFE", "Pfizer"),
    ("CAT", "Caterpillar"), ("AMGN", "Amgen"), ("SPGI", "S&P Global"), ("BLK", "BlackRock"),
    ("PANW", "Palo Alto Networks"), ("NKE", "Nike"), ("LOW", "Lowe's"), ("HON", "Honeywell"),
    ("UNP", "Union Pacific"), ("AMAT", "Applied Materials"), ("MU", "Micron"),
    ("INTC", "Intel"), ("SBUX", "Starbucks"), ("BA", "Boeing"), ("DE", "Deere"),
    ("LMT", "Lockheed Martin"), ("CRWD", "CrowdStrike"), ("COIN", "Coinbase"),
    ("SHOP", "Shopify"), ("PYPL", "PayPal"), ("ARM", "Arm Holdings"), ("MRVL", "Marvell"),
    ("NEM", "Newmont"), ("NOC", "Northrop Grumman"),
    # Global / ADRs
    ("ASML", "ASML"), ("SAP", "SAP"), ("NVO", "Novo Nordisk"), ("NVS", "Novartis"),
    ("AZN", "AstraZeneca"), ("SHEL", "Shell"), ("TM", "Toyota"), ("HSBC", "HSBC"),
    ("SONY", "Sony"), ("BABA", "Alibaba"), ("TCEHY", "Tencent (US-listed shares)"), ("PDD", "PDD Holdings"),
    ("SE", "Sea Ltd"), ("MELI", "MercadoLibre"), ("SPOT", "Spotify"), ("BIDU", "Baidu"),
    ("NVR", "NVR Inc."), ("NVST", "Envista Holdings"),
    # ETFs
    ("VUG", "Vanguard Growth ETF"), ("VEA", "Vanguard Developed Mkts"),
    ("VWO", "Vanguard Emerging Mkts"), ("GLD", "SPDR Gold Shares"), ("IAU", "iShares Gold Trust"),
    ("AGG", "iShares US Agg Bond"), ("BND", "Vanguard Total Bond"), ("IWM", "iShares Russell 2000"),
    ("DIA", "SPDR Dow Jones ETF"), ("SCHD", "Schwab US Dividend"), ("TLT", "iShares 20+Y Treasury"),
    ("XLK", "Technology Select SPDR"), ("XLF", "Financial Select SPDR"),
    ("XLE", "Energy Select SPDR"), ("XLV", "Health Care Select SPDR"),
    ("SMH", "VanEck Semiconductor"), ("ARKK", "ARK Innovation"), ("VNQ", "Vanguard Real Estate"),
    ("EFA", "iShares MSCI EAFE"), ("EEM", "iShares MSCI Emerging"), ("SLV", "iShares Silver Trust"),
    # Crypto
    ("XRP-USD", "XRP"), ("BNB-USD", "BNB"), ("SOL-USD", "Solana"),
    ("DOGE-USD", "Dogecoin"), ("ADA-USD", "Cardano"), ("AVAX-USD", "Avalanche"),
]
TICKER_UNIVERSE = dict(_TICKER_UNIVERSE)          # symbol -> name (insertion order = rank)
PRESET_TICKERS = list(TICKER_UNIVERSE)            # options shown in the dropdown, ranked


def _supports_new_options() -> bool:
    """`st.multiselect(accept_new_options=...)` exists only in newer Streamlit versions."""
    try:
        return "accept_new_options" in inspect.signature(st.multiselect).parameters
    except (TypeError, ValueError):
        return False


SUPPORTS_NEW_OPTIONS = _supports_new_options()


# --------------------------------------------------------------------------- #
# Custom exception for predictable, user-facing problems
# --------------------------------------------------------------------------- #
class TranslatableError(Exception):
    """User-facing error: stores a message key and is translated when displayed, not when raised."""

    def __init__(self, key: str, **values):
        super().__init__(key)
        self.key, self.values = key, values

    def __str__(self) -> str:
        return t(self.key, **self.values)


class InputError(TranslatableError):
    """Raised when the user's input (tickers, dates, data) is unusable."""


class OptimError(TranslatableError):
    """Raised when the optimiser cannot find a solution."""


# =========================================================================== #
#                      INTERNATIONALISATION  (English / ไทย)
# =========================================================================== #
# Every user-facing string lives in TRANSLATIONS as (English, Thai). `t(key, **values)` returns the text
# for the language chosen in the sidebar (stored in st.session_state["lang"]).
# Thai wording is deliberately plain and conversational so a first-time investor can follow it.
LANG_EN, LANG_TH = "English", "ไทย"
LANG_OPTIONS = [LANG_EN, LANG_TH]

TRANSLATIONS = {
    # ----------------------------------------------------------------- page / hero
    "page_title": ("Portfolio Optimizer", "ตัวช่วยจัดพอร์ตหุ้น"),
    "hero_title": ("Stock Portfolio Allocation Optimizer", "ตัวช่วยจัดสัดส่วนพอร์ตหุ้นที่เหมาะสม"),
    "hero_sub": (
        "Find the stock mix that earns the most for the risk you take, or the one that keeps risk lowest, "
        "using Harry Markowitz's Mean-Variance model.",
        "หาสัดส่วนหุ้นที่ให้ผลตอบแทนคุ้มที่สุดเมื่อเทียบกับความเสี่ยงที่ต้องรับ หรือสัดส่วนที่เสี่ยงต่ำที่สุด "
        "ด้วยแนวคิดของ Harry Markowitz ผู้คิดค้นทฤษฎีพอร์ตการลงทุนสมัยใหม่",
    ),
    "badge_live": ("Live market data", "ข้อมูลตลาดล่าสุด"),
    "badge_model": ("Markowitz Mean-Variance model", "แบบจำลองของ Markowitz"),
    "badge_model_tip": (
        "Solved with SLSQP (Sequential Least Squares Programming)",
        "คำนวณด้วยวิธี SLSQP (Sequential Least Squares Programming)",
    ),
    "badge_long": ("Long-only (no short-selling)", "ซื้ออย่างเดียว (ไม่มีการขายชอร์ต)"),
    "badge_long_tip": (
        "Each weight is between 0% and 100% and all weights add up to 100%",
        "สัดส่วนของแต่ละตัวอยู่ระหว่าง 0–100% และรวมกันได้ 100%",
    ),
    # ----------------------------------------------------------------- landing
    "how_title": ("How it works", "ใช้งานอย่างไร"),
    "card1_t": ("1 · Choose your assets", "1 · เลือกหุ้นที่สนใจ"),
    "card1_p": (
        "Search and pick two or more stocks in the sidebar (e.g. AAPL, MSFT, TSLA).",
        "ค้นหาและเลือกหุ้นอย่างน้อย 2 ตัวจากแถบด้านซ้าย (เช่น AAPL, MSFT, TSLA)",
    ),
    "card2_t": ("2 · Set the parameters", "2 · ตั้งค่าเงื่อนไข"),
    "card2_p": (
        "Choose the date range and the risk-free rate (the return you could earn with no risk).",
        "เลือกช่วงเวลาของราคาย้อนหลัง และอัตราดอกเบี้ยอ้างอิงไร้ความเสี่ยง (ผลตอบแทนที่ได้โดยแทบไม่มีความเสี่ยง)",
    ),
    "card3_t": ("3 · Run the optimization", "3 · กดคำนวณ"),
    "card3_p": (
        "Get the Max Sharpe Ratio (Best Risk-Adjusted Return) and Min Variance (Lowest Risk) portfolios.",
        "ดูพอร์ตที่คุ้มค่าที่สุดเมื่อเทียบกับความเสี่ยง และพอร์ตที่เสี่ยงต่ำที่สุด พร้อมสัดส่วนที่ควรลงทุนในแต่ละตัว",
    ),
    "footer": (
        "Built with Streamlit · yfinance · SciPy · Plotly  |  For educational purposes only — not financial "
        "advice. Past performance does not guarantee future results.",
        "สร้างด้วย Streamlit · yfinance · SciPy · Plotly  |  เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำในการลงทุน "
        "ผลตอบแทนในอดีตไม่ได้รับประกันผลตอบแทนในอนาคต",
    ),
    # ----------------------------------------------------------------- sidebar
    "brand_sub": ("Markowitz Optimizer", "ตัวช่วยจัดพอร์ต Markowitz"),
    "sec_assets": ("Assets", "หุ้น / สินทรัพย์"),
    "sec_time": ("Time window", "ช่วงเวลา"),
    "sec_risk": ("Risk assumption", "ค่าอ้างอิงความเสี่ยง"),
    "sec_capital": ("Investment capital", "เงินลงทุน"),
    "pick_label": ("Select assets", "เลือกหุ้นหรือสินทรัพย์"),
    "pick_placeholder": ("Search ticker or company…", "พิมพ์ค้นหาชื่อย่อหุ้นหรือชื่อบริษัท…"),
    "pick_help": (
        "Type to filter the list (largest companies first). Click × on a tag to remove it. "
        "ETF = Exchange-Traded Fund, a basket of assets that trades like a stock. "
        "Symbols ending in -USD are cryptocurrencies priced in US Dollars.",
        "พิมพ์เพื่อค้นหา (บริษัทใหญ่จะอยู่ก่อน) กด × ที่ป้ายเพื่อเอาออก "
        "ETF คือกองทุนรวมที่ซื้อขายได้เหมือนหุ้น ส่วนชื่อที่ลงท้ายด้วย -USD คือเหรียญคริปโตที่ตีราคาเป็นดอลลาร์สหรัฐ",
    ),
    "pick_hint": (
        "Type to search by <b>ticker or company</b>. Not listed? Type the exact symbol "
        "(e.g. <b>PTT.BK</b>) and press <b>Enter</b> to add it.",
        "พิมพ์ค้นหาด้วย <b>ชื่อย่อหุ้นหรือชื่อบริษัท</b> ไม่เจอในรายการ? "
        "พิมพ์ชื่อย่อให้ตรง (เช่น <b>PTT.BK</b>) แล้วกด <b>Enter</b> เพื่อเพิ่ม",
    ),
    "custom_label": ("Add a custom ticker", "เพิ่มชื่อย่อหุ้นเอง"),
    "custom_placeholder": ("e.g. PTT.BK  (press Enter)", "เช่น PTT.BK (แล้วกด Enter)"),
    "notice_invalid": ("Ignored invalid symbol(s): {items}", "ข้ามชื่อย่อที่ไม่ถูกต้อง: {items}"),
    "count_ok": ("{n} assets selected", "เลือกแล้ว {n} ตัว"),
    "count_low": ("{n} selected · pick at least 2", "เลือกแล้ว {n} ตัว · ต้องเลือกอย่างน้อย 2 ตัว"),
    "btn_clear": ("Clear all", "ล้างทั้งหมด"),
    "btn_example": ("Load example", "ใช้ตัวอย่าง"),
    "start_label": ("Start date", "วันที่เริ่มต้น"),
    "start_help": (
        "First day of historical prices used in the analysis.",
        "วันแรกของราคาย้อนหลังที่ใช้คำนวณ",
    ),
    "end_label": ("End date", "วันที่สิ้นสุด"),
    "end_help": (
        "Last day of historical prices (today at the latest).",
        "วันสุดท้ายของราคาย้อนหลัง (ไม่เกินวันนี้)",
    ),
    "rf_label": ("Risk-free rate (annual, %)", "อัตราดอกเบี้ยอ้างอิงไร้ความเสี่ยง (ต่อปี, %)"),
    "rf_help": (
        "The return you could earn with no risk, e.g. a US Treasury yield of about 4%. "
        "It is used to calculate the Sharpe Ratio.",
        "ผลตอบแทนที่ได้โดยแทบไม่มีความเสี่ยง เช่น พันธบัตรรัฐบาล (ดอกเบี้ยพันธบัตรสหรัฐฯ ประมาณ 4%) "
        "ใช้เป็นเกณฑ์เทียบเพื่อคำนวณดัชนีวัดความคุ้มค่าเทียบความเสี่ยง",
    ),
    "ccy_label": ("Capital currency", "สกุลเงินของเงินลงทุน"),
    "ccy_help": (
        "Choose the currency you will enter the amount in. Switching converts the amount at the FX rate below.",
        "เลือกสกุลเงินที่จะกรอกจำนวนเงิน เมื่อสลับสกุลเงิน ระบบจะแปลงจำนวนเงินตามอัตราแลกเปลี่ยนด้านล่างให้อัตโนมัติ",
    ),
    "ccy_THB": ("Thai Baht (THB ฿)", "บาทไทย (THB ฿)"),
    "ccy_USD": ("US Dollars (USD $)", "ดอลลาร์สหรัฐ (USD $)"),
    "amount_label": ("Investment capital amount", "เงินลงทุนเริ่มต้นของคุณ"),
    "amount_help": (
        "How much you plan to invest, in the currency selected above.",
        "จำนวนเงินที่คุณวางแผนจะลงทุน ในสกุลเงินที่เลือกด้านบน",
    ),
    "fx_label": (
        "FX (Foreign Exchange Rate) · Baht per 1 US Dollar",
        "อัตราแลกเปลี่ยนเงินตรา (FX) · กี่บาทต่อ 1 ดอลลาร์สหรัฐ",
    ),
    "fx_help": (
        "Thai Baht (฿) per 1 US Dollar. It only affects the money amounts, not the optimization.",
        "จำนวนบาทต่อ 1 ดอลลาร์สหรัฐ ใช้แปลงจำนวนเงินเท่านั้น ไม่มีผลต่อการคำนวณสัดส่วนพอร์ต",
    ),
    "btn_run": ("Run Optimization", "คำนวณพอร์ตที่ดีที่สุด"),
    "btn_guide": ("📖 How to Use", "📖 วิธีใช้งาน"),
    # ----------------------------------------------------------------- progress / errors
    "status_running": ("Running optimization…", "กำลังคำนวณพอร์ต…"),
    "step_validate": ("🔎 Validating inputs…", "🔎 ตรวจสอบข้อมูลที่กรอก…"),
    "step_download": (
        "📡 Downloading market data from Yahoo Finance…",
        "📡 ดาวน์โหลดข้อมูลราคาจาก Yahoo Finance…",
    ),
    "step_returns": (
        "🧮 Computing log returns and the covariance matrix…",
        "🧮 คำนวณผลตอบแทนรายวันและความสัมพันธ์ของการขึ้นลงระหว่างหุ้น…",
    ),
    "step_solve": (
        "⚙️ Solving the Max Sharpe Ratio and Min Variance portfolios with SLSQP "
        "(Sequential Least Squares Programming)…",
        "⚙️ กำลังหาพอร์ตที่คุ้มค่าที่สุดและพอร์ตที่เสี่ยงต่ำที่สุด (ด้วยวิธี SLSQP)…",
    ),
    "status_done": ("Optimization complete ✔", "คำนวณเสร็จแล้ว ✔"),
    "status_input_err": ("Input problem", "ข้อมูลที่กรอกยังไม่ถูกต้อง"),
    "status_optim_err": ("Optimization failed", "คำนวณไม่สำเร็จ"),
    "status_unexpected": ("Unexpected error", "เกิดข้อผิดพลาดที่ไม่คาดคิด"),
    "err_optim": ("Optimization error: {msg}", "คำนวณไม่สำเร็จ: {msg}"),
    "err_generic": (
        "Something went wrong while processing your request. Please verify your tickers, dates, "
        "and internet connection, then try again.",
        "เกิดข้อผิดพลาดระหว่างประมวลผล กรุณาตรวจสอบชื่อย่อหุ้น วันที่ และการเชื่อมต่ออินเทอร์เน็ต แล้วลองใหม่อีกครั้ง",
    ),
    "tech_details": ("Technical details", "รายละเอียดทางเทคนิค"),
    "e_no_data": (
        "No price data was returned. Check the ticker symbols and the date range.",
        "ไม่พบข้อมูลราคา กรุณาตรวจสอบชื่อย่อหุ้นและช่วงวันที่",
    ),
    "e_bad_tickers": (
        "No data found for: {items}. Please check that these ticker symbols are valid for the selected dates.",
        "ไม่พบข้อมูลของ: {items} กรุณาตรวจสอบว่าชื่อย่อหุ้นถูกต้องและมีการซื้อขายในช่วงวันที่ที่เลือก",
    ),
    "e_no_overlap": (
        "The selected tickers have no overlapping trading days in this date range.",
        "หุ้นที่เลือกไม่มีวันซื้อขายที่ตรงกันในช่วงวันที่นี้",
    ),
    "e_min2": (
        "Please enter at least two distinct ticker symbols.",
        "กรุณาเลือกหุ้นอย่างน้อย 2 ตัวที่ไม่ซ้ำกัน",
    ),
    "e_max": ("Please limit the portfolio to {n} tickers or fewer.", "กรุณาเลือกไม่เกิน {n} ตัว"),
    "e_start_end": (
        "The start date must be earlier than the end date.",
        "วันที่เริ่มต้นต้องอยู่ก่อนวันที่สิ้นสุด",
    ),
    "e_future": ("The end date cannot be in the future.", "วันที่สิ้นสุดต้องไม่เป็นวันในอนาคต"),
    "e_range": (
        "Please select a date range of at least two months.",
        "กรุณาเลือกช่วงวันที่อย่างน้อย 2 เดือน",
    ),
    "e_few_obs": (
        "Only {n} trading days of overlapping data were found; at least {m} are required. "
        "Try a longer date range.",
        "พบข้อมูลวันซื้อขายที่ตรงกันเพียง {n} วัน แต่ต้องมีอย่างน้อย {m} วัน ลองเลือกช่วงวันที่ให้ยาวขึ้น",
    ),
    "e_maxsharpe": (
        "Max Sharpe Ratio optimisation failed: {msg}",
        "คำนวณพอร์ตที่คุ้มค่าที่สุดไม่สำเร็จ: {msg}",
    ),
    "e_minvar": (
        "Minimum variance optimisation failed: {msg}",
        "คำนวณพอร์ตที่เสี่ยงต่ำที่สุดไม่สำเร็จ: {msg}",
    ),
    # ----------------------------------------------------------------- results header & tabs
    "b_done": ("Optimization complete", "คำนวณเสร็จแล้ว"),
    "b_assets": ("{n} assets", "{n} สินทรัพย์"),
    "b_days": ("{n} trading days", "{n} วันทำการซื้อขาย"),
    "b_rf": ("Risk-Free Rate {v}", "ดอกเบี้ยอ้างอิงไร้ความเสี่ยง {v}"),
    "tab_summary": ("📊 Executive Summary", "📊 สรุปภาพรวม"),
    "tab_alloc": ("💼 Asset Allocation Weights", "💼 สัดส่วนการลงทุน"),
    "tab_data": ("📋 Raw Historical Data", "📋 ข้อมูลราคาย้อนหลัง"),
    # ----------------------------------------------------------------- portfolio names
    "pf_max_main": ("Max Sharpe Ratio", "อัตราผลตอบแทนเทียบความเสี่ยงสูงสุด"),
    "pf_max_sub": ("Best Risk-Adjusted Return", "เลือกตัวที่คุ้มค่าที่สุด"),
    "pf_max_short": ("Max Sharpe", "คุ้มค่าที่สุด"),
    "pf_max_desc": (
        "Highest return for each unit of risk taken",
        "ได้ผลตอบแทนสูงสุดเมื่อเทียบกับความเสี่ยงที่ต้องรับ",
    ),
    "pf_min_main": ("Min Variance", "พอร์ตความเสี่ยงต่ำที่สุด"),
    "pf_min_sub": ("Lowest Risk", "เน้นปลอดภัยเงินต้นไม่ค่อยแกว่ง"),
    "pf_min_short": ("Min Variance", "เสี่ยงต่ำสุด"),
    "pf_min_desc": (
        "Lowest possible overall volatility (price swings)",
        "ราคาขึ้นลงน้อยที่สุดเท่าที่เป็นไปได้",
    ),
    # ----------------------------------------------------------------- KPI cards
    "kpi_ret": ("Expected Return", "กำไรคาดหวังต่อปี (เฉลี่ย)"),
    "kpi_ret_sub": ("Annualised average gain", "ค่าเฉลี่ยต่อปี"),
    "kpi_ret_tip": (
        "Average yearly gain estimated from historical log returns.",
        "กำไรเฉลี่ยต่อปีที่ประมาณจากราคาย้อนหลัง ไม่ใช่การรับประกันผลตอบแทนในอนาคต",
    ),
    "kpi_vol": ("Volatility (Risk)", "ความเสี่ยง / ความผันผวนต่อปี"),
    "kpi_vol_sub": ("Annualised price swings", "ราคาขึ้นลงมากน้อยแค่ไหน"),
    "kpi_vol_tip": (
        "Annualised standard deviation of returns: how much the value typically moves up and down.",
        "ยิ่งตัวเลขสูง มูลค่าพอร์ตยิ่งขึ้นลงแรง (คำนวณจากส่วนเบี่ยงเบนมาตรฐานของผลตอบแทนต่อปี)",
    ),
    "kpi_sharpe": ("Sharpe Ratio", "ดัชนีวัดความคุ้มค่าเทียบความเสี่ยง"),
    "kpi_sharpe_sub": ("Extra return per unit of risk", "กำไรส่วนเกินต่อความเสี่ยง 1 หน่วย"),
    "kpi_sharpe_tip": (
        "(Expected return − risk-free rate) ÷ volatility. Risk-free rate used: {rf}.",
        "(กำไรคาดหวัง − ดอกเบี้ยอ้างอิงไร้ความเสี่ยง) ÷ ความผันผวน ดอกเบี้ยอ้างอิงที่ใช้: {rf} ยิ่งสูงยิ่งคุ้มค่า",
    ),
    "sec_summary": ("Portfolio Summary Metrics", "สรุปตัวเลขของแต่ละพอร์ต"),
    "th_portfolio": ("Portfolio", "พอร์ต"),
    "th_exp_ret": ("Expected Annual Return", "กำไรคาดหวังต่อปี (เฉลี่ย)"),
    "th_vol": ("Annual Volatility (Risk)", "ความเสี่ยง / ความผันผวนต่อปี"),
    "th_sharpe": ("Sharpe Ratio", "ดัชนีวัดความคุ้มค่าเทียบความเสี่ยง"),
    # ----------------------------------------------------------------- allocation tab
    "sec_alloc": (
        "Asset Allocation & Capital Breakdown",
        "สัดส่วนการลงทุนและจำนวนเงินที่ควรลงในแต่ละตัว",
    ),
    "cap_badge": (
        "💰 Investment Capital: {usd} (US Dollars) ≈ {thb} (Thai Baht)",
        "💰 เงินลงทุนเริ่มต้นของคุณ: {usd} (ดอลลาร์สหรัฐ) ≈ {thb} (บาทไทย)",
    ),
    "fx_badge": ("FX (Foreign Exchange Rate): {v}", "อัตราแลกเปลี่ยนเงินตรา (FX): {v}"),
    "alloc_caption": (
        "Amount = weight × investment capital, converted at the FX (Foreign Exchange Rate) set in the sidebar. "
        "Weights use each asset's own trading currency, with no currency adjustment.",
        "จำนวนเงิน = สัดส่วน × เงินลงทุน แปลงตามอัตราแลกเปลี่ยนเงินตรา (FX) ที่ตั้งไว้ในแถบด้านซ้าย "
        "การคำนวณสัดส่วนใช้ผลตอบแทนในสกุลเงินของหุ้นแต่ละตัว โดยไม่ปรับอัตราแลกเปลี่ยน",
    ),
    "th_asset": ("Asset", "สินทรัพย์"),
    "th_weight": ("Weight (%)", "สัดส่วน (%)"),
    "th_usd": ("US Dollars (USD $)", "ดอลลาร์สหรัฐ (USD $)"),
    "th_thb": ("Thai Baht (THB ฿)", "บาทไทย (THB ฿)"),
    "th_delta": ("Δ Difference ({a} − {b})", "Δ ส่วนต่าง ({a} − {b})"),
    "th_delta_sub": ("in pp (Percentage Points)", "หน่วย: pp (จุดต่างเปอร์เซ็นต์)"),
    "pp_tip": ("Percentage Points", "จุดต่างเปอร์เซ็นต์"),
    "total": ("Total", "รวม"),
    "donut_assets": ("assets", "ตัว"),
    "donut_hidden": (
        "{n} other asset(s) at 0.5% or less not shown",
        "ไม่แสดงอีก {n} ตัวที่สัดส่วน 0.5% หรือน้อยกว่า",
    ),
    # ----------------------------------------------------------------- raw data tab
    "perf_title": (
        "Price Performance · Rebased to 100 at the Start Date",
        "การเติบโตของราคา · ปรับให้เริ่มที่ 100 ในวันแรก",
    ),
    "perf_y": ("Rebased value (start date = 100)", "มูลค่าเทียบวันแรก (วันแรก = 100)"),
    "perf_value": ("Value", "มูลค่า"),
    "perf_caption": (
        "Every line starts at 100, so growth is easy to compare. Scroll to zoom, drag to pan, double-click to reset.",
        "ทุกเส้นเริ่มที่ 100 ในวันแรก จึงเปรียบเทียบการเติบโตได้ง่าย "
        "เลื่อนเมาส์เพื่อซูม ลากเพื่อเลื่อน ดับเบิลคลิกเพื่อรีเซ็ต",
    ),
    "corr_title": ("Correlation Matrix", "ตารางความสัมพันธ์ของหุ้นแต่ละตัว"),
    "corr_hover": ("Correlation", "ความสัมพันธ์"),
    "corr_caption": (
        "Do the stocks move in the same direction? Close to +1 = they move together, close to 0 = unrelated, "
        "close to −1 = they move in opposite directions.",
        "(ดูว่าหุ้นเคลื่อนไหวไปในทิศทางเดียวกันหรือไม่) ค่าใกล้ +1 = ไปทางเดียวกัน, "
        "ใกล้ 0 = ไม่เกี่ยวกัน, ใกล้ −1 = สวนทางกัน",
    ),
    "stats_title": ("Per-Asset Statistics (Annualised)", "สถิติของหุ้นแต่ละตัว (ต่อปี)"),
    "prices_title": (
        "Adjusted Closing Prices",
        "ราคาปิดรายวัน (ปรับปันผลและแตกพาร์แล้ว)",
    ),
    "prices_caption": (
        "Darker blue = higher price within each ticker's own range. Newest dates first.",
        "สีฟ้าเข้ม = ราคาสูงเมื่อเทียบกับช่วงราคาของหุ้นตัวนั้นเอง แสดงวันล่าสุดก่อน",
    ),
    "prices_dl": (
        "⬇️ Download prices (CSV: Comma-Separated Values file)",
        "⬇️ ดาวน์โหลดราคา (ไฟล์ CSV เปิดด้วย Excel ได้)",
    ),
    "col_date": ("Date", "วันที่"),
    # ----------------------------------------------------------------- welcome guide
    "guide_title": (
        "Welcome to Stock Portfolio Allocation Optimizer",
        "ยินดีต้อนรับสู่ตัวช่วยจัดสัดส่วนพอร์ตหุ้น",
    ),
    "guide_intro": (
        "Build a smarter portfolio in four quick steps using Harry Markowitz's Mean-Variance model.",
        "จัดพอร์ตให้ฉลาดขึ้นใน 4 ขั้นตอนง่าย ๆ ด้วยแนวคิดของ Harry Markowitz",
    ),
    "g1_t": ("Select 2 or more stock tickers", "เลือกหุ้นอย่างน้อย 2 ตัว"),
    "g1_b": (
        "Use the <b>Select assets</b> box in the left sidebar. Search by ticker or company name "
        "(e.g. <b>MSFT</b>, <b>NVDA</b>, <b>TSLA</b>, <b>SPY</b>) and click a suggestion to add it. "
        "Click the × on a tag to remove it. Symbol not listed? Type it exactly and press <b>Enter</b>.",
        "ที่แถบด้านซ้าย พิมพ์ค้นหาในช่อง <b>เลือกหุ้นหรือสินทรัพย์</b> ด้วยชื่อย่อหรือชื่อบริษัท "
        "(เช่น <b>MSFT</b>, <b>NVDA</b>, <b>TSLA</b>, <b>SPY</b>) แล้วคลิกที่รายการเพื่อเพิ่ม "
        "กด × ที่ป้ายเพื่อเอาออก ถ้าไม่มีในรายการ ให้พิมพ์ชื่อย่อให้ตรงแล้วกด <b>Enter</b>",
    ),
    "g2_t": ("Set the time window and risk-free rate", "ตั้งช่วงเวลาและดอกเบี้ยอ้างอิง"),
    "g2_b": (
        "Pick a <b>Start date</b> and <b>End date</b> for the historical prices. Then move the "
        "<b>Risk-free rate</b> slider to the benchmark return you could earn with no risk "
        "&mdash; for example about <b>4%</b>, similar to a US Treasury yield.",
        "เลือก <b>วันที่เริ่มต้น</b> และ <b>วันที่สิ้นสุด</b> ของราคาย้อนหลัง จากนั้นเลื่อนแถบ "
        "<b>อัตราดอกเบี้ยอ้างอิงไร้ความเสี่ยง</b> ไปที่ผลตอบแทนที่ได้โดยแทบไม่มีความเสี่ยง "
        "&mdash; เช่น ประมาณ <b>4%</b> ใกล้เคียงดอกเบี้ยพันธบัตรรัฐบาลสหรัฐฯ",
    ),
    "g3_t": ("Click “Run Optimization”", "กดปุ่ม “คำนวณพอร์ตที่ดีที่สุด”"),
    "g3_b": (
        "The app downloads the prices and calculates the best portfolio weights for you.",
        "ระบบจะดึงราคาหุ้นและคำนวณสัดส่วนการลงทุนที่ดีที่สุดให้คุณ",
    ),
    "g4_t": ("Explore the result tabs", "ดูผลลัพธ์ในแต่ละแท็บ"),
    "g4_b": (
        "<ul class='guide-list'>"
        "<li><b>📊 Executive Summary</b> &mdash; compare expected return, volatility (risk) and the "
        "Sharpe Ratio (extra return per unit of risk) of each portfolio.</li>"
        "<li><b>💼 Asset Allocation Weights</b> &mdash; see the exact percentage breakdown for the "
        "<b>Max Sharpe Ratio</b> portfolio (best risk-adjusted return) and the <b>Min Variance</b> "
        "portfolio (lowest risk).</li>"
        "<li><b>📋 Raw Historical Data</b> &mdash; inspect historical growth charts, correlations "
        "and daily closing prices.</li></ul>",
        "<ul class='guide-list'>"
        "<li><b>📊 สรุปภาพรวม</b> &mdash; เปรียบเทียบกำไรคาดหวัง ความเสี่ยง และดัชนีวัดความคุ้มค่าเทียบความเสี่ยง "
        "(ยิ่งสูงยิ่งคุ้ม) ของแต่ละพอร์ต</li>"
        "<li><b>💼 สัดส่วนการลงทุน</b> &mdash; ดูว่าควรลงทุนในหุ้นแต่ละตัวกี่เปอร์เซ็นต์ และเป็นเงินเท่าไร "
        "ทั้งพอร์ต<b>อัตราผลตอบแทนเทียบความเสี่ยงสูงสุด</b> (เลือกตัวที่คุ้มค่าที่สุด) "
        "และพอร์ต<b>ความเสี่ยงต่ำที่สุด</b> (เน้นปลอดภัย)</li>"
        "<li><b>📋 ข้อมูลราคาย้อนหลัง</b> &mdash; ดูกราฟการเติบโต ความสัมพันธ์ระหว่างหุ้น "
        "และราคาปิดรายวัน</li></ul>",
    ),
    "g_note": (
        "Tip: you can reopen this guide anytime with the <b>📖 How to Use</b> button at the bottom of the "
        "sidebar. For educational purposes only &mdash; not financial advice.",
        "เคล็ดลับ: เปิดคู่มือนี้ได้ทุกเมื่อด้วยปุ่ม <b>📖 วิธีใช้งาน</b> ที่ด้านล่างของแถบด้านซ้าย "
        "เพื่อการศึกษาเท่านั้น &mdash; ไม่ใช่คำแนะนำในการลงทุน",
    ),
    "g_start": ("🚀 Get Started", "🚀 เริ่มใช้งาน"),
}


def lang_code() -> str:
    """'th' when Thai is selected in the sidebar, otherwise 'en' (default)."""
    return "th" if st.session_state.get("lang", LANG_EN) == LANG_TH else "en"


def t(key: str, **values) -> str:
    """Translate `key` into the active language and fill in {placeholders}."""
    english, thai = TRANSLATIONS[key]
    text = thai if lang_code() == "th" else english
    return text.format(**values) if values else text


def tx(key: str, **values) -> str:
    """Like t(), but HTML-escaped (for plain-text strings placed inside HTML)."""
    return esc(t(key, **values))


def pf_text(name: str, field: str) -> str:
    """Translated portfolio text. name: 'Max Sharpe' | 'Min Variance'; field: main | sub | short | desc."""
    return t(f"pf_{PORTFOLIO_STYLE[name]['ident']}_{field}")


def pf_full(name: str) -> str:
    """e.g. 'Max Sharpe Ratio (Best Risk-Adjusted Return)'."""
    return f"{pf_text(name, 'main')} ({pf_text(name, 'sub')})"


# =========================================================================== #
#                         PRESENTATION LAYER  (CSS + HTML)
# =========================================================================== #
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

:root {
  --font: 'Plus Jakarta Sans', 'Inter', 'Noto Sans Thai', sans-serif;
  --mono: 'JetBrains Mono', 'Noto Sans Thai', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
  --bg: #0b1120;
  --card: rgba(30, 41, 59, 0.42);
  --emerald: #10b981;
  --emerald-2: #34d399;
  --cyan: #22d3ee;
  --amber: #fbbf24;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --border: rgba(148, 163, 184, 0.14);
}

/* ---------- Keyframes ---------- */
@keyframes fadeUp   { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
@keyframes shimmer  { to { background-position: 200% center; } }
@keyframes growX    { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes pulseDot {
  0%   { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.6); }
  70%  { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
  100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

/* ---------- App shell ---------- */
.stApp {
  background:
    radial-gradient(1100px 560px at 6% -12%, rgba(16, 185, 129, 0.11), transparent 60%),
    radial-gradient(900px 460px at 100% 0%, rgba(34, 211, 238, 0.08), transparent 55%),
    linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
  color: var(--text);
  font-family: var(--font);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3.5rem; max-width: 1360px; }

/* ---------- Typography: Plus Jakarta Sans for text, JetBrains Mono for numbers ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-baseweb], .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
[data-baseweb="popover"], [data-baseweb="popover"] li, [data-baseweb="calendar"], [data-testid="stDialog"] {
  font-family: var(--font);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #f1f5f9; letter-spacing: -0.015em; font-weight: 700; }
.stApp p, .stApp label { color: var(--text); }
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; font-size: 0.8rem; line-height: 1.6; }

.mono, .money, .num, .wnum, .kpi-value,
input[type="number"], [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input,
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
  font-family: var(--mono) !important;
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1;
}
.num { font-size: 0.92em; font-weight: 600; color: #f1f5f9; }
abbr[title] { text-decoration: none; border-bottom: 1px dotted rgba(148, 163, 184, 0.6); cursor: help; }

/* Scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.22); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(52, 211, 153, 0.5); }

/* ---------- Hero ---------- */
.hero {
  padding: 30px 34px; margin-bottom: 22px; border-radius: 22px; border: 1px solid transparent;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(30, 41, 59, 0.55)) padding-box,
    linear-gradient(135deg, rgba(16, 185, 129, 0.5), rgba(34, 211, 238, 0.3), rgba(148, 163, 184, 0.08)) border-box;
  backdrop-filter: blur(14px);
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
  animation: fadeUp 0.7s ease both;
}
.hero h1 {
  margin: 0 0 8px 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15;
  background: linear-gradient(90deg, #34d399, #22d3ee, #34d399); background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 7s linear infinite;
}
.hero p { margin: 0 0 18px 0; color: var(--muted); font-size: 0.98rem; line-height: 1.65; max-width: 720px; }

/* ---------- Badges ---------- */
.badge-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 10px; animation: fadeIn 0.8s ease both; }
.badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
  font-size: 0.76rem; font-weight: 600; color: var(--muted);
  border: 1px solid var(--border); background: rgba(15, 23, 42, 0.55); transition: all 0.25s ease;
}
.badge:hover { transform: translateY(-1px); border-color: rgba(52, 211, 153, 0.45); color: #e2e8f0; }
.badge.ok   { color: #6ee7b7; border-color: rgba(52, 211, 153, 0.32); background: rgba(16, 185, 129, 0.10); }
.badge.cyan { color: #67e8f9; border-color: rgba(34, 211, 238, 0.32); background: rgba(34, 211, 238, 0.08); }
.badge.warn { color: #fcd34d; border-color: rgba(251, 191, 36, 0.32); background: rgba(251, 191, 36, 0.08); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--emerald-2); animation: pulseDot 1.8s infinite; }

/* ---------- Section titles ---------- */
.section-title {
  display: flex; align-items: center; gap: 14px; margin: 34px 0 14px;
  font-size: 1.02rem; font-weight: 700; color: #f1f5f9; animation: fadeUp 0.6s ease both;
}
.section-title::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(148, 163, 184, 0.3), transparent); }

/* ---------- KPI cards (glassmorphism, minimal) ---------- */
.pf-head { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; margin: 26px 0 6px; animation: fadeUp 0.6s ease both; }
.pf-desc { color: var(--muted); font-size: 0.84rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 18px; margin: 12px 0 8px; }
.kpi {
  --accent: #34d399; position: relative; overflow: hidden; padding: 24px 26px; border-radius: 18px;
  border: 1px solid transparent;
  background:
    linear-gradient(160deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.55)) padding-box,
    linear-gradient(135deg, rgba(148, 163, 184, 0.24), rgba(148, 163, 184, 0.05)) border-box;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.22);
  transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
  animation: fadeUp 0.65s ease both;
}
.kpi.emerald { --accent: #34d399; }
.kpi.cyan    { --accent: #22d3ee; }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent), transparent 70%); opacity: 0.8; }
.kpi:hover {
  transform: translateY(-3px);
  background:
    linear-gradient(160deg, rgba(30, 41, 59, 0.82), rgba(15, 23, 42, 0.62)) padding-box,
    linear-gradient(135deg, var(--accent), rgba(148, 163, 184, 0.1)) border-box;
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.32);
}
.kpi-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: var(--muted); }
.kpi-value { margin-top: 12px; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc; line-height: 1.1; }
.kpi-sub   { margin-top: 8px; font-size: 0.8rem; color: var(--muted); }

/* ---------- Glass panels & tables ---------- */
.glass {
  border-radius: 18px; border: 1px solid var(--border); background: var(--card);
  backdrop-filter: blur(12px); box-shadow: 0 8px 26px rgba(0, 0, 0, 0.2);
  transition: border-color 0.3s ease; animation: fadeUp 0.7s ease both;
}
.glass:hover { border-color: rgba(52, 211, 153, 0.3); }
.table-wrap { overflow-x: auto; padding: 8px 10px; }
.dtable { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.9rem; }
.dtable th {
  text-align: left; padding: 16px 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em; line-height: 1.5;
  text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); white-space: nowrap; vertical-align: bottom;
}
.dtable td { padding: 16px 20px; border-bottom: 1px solid rgba(148, 163, 184, 0.06); color: var(--text); white-space: nowrap; }
.dtable tbody tr { transition: background 0.2s ease; }
.dtable tbody tr:hover { background: rgba(52, 211, 153, 0.05); }
.dtable tbody tr:last-child td { border-bottom: none; }
.dtable th.grp { text-align: center; white-space: normal; border-top: 2px solid transparent; }
.dtable th.grp.emerald { color: #6ee7b7; border-top-color: rgba(52, 211, 153, 0.6); background: rgba(16, 185, 129, 0.05); }
.dtable th.grp.cyan    { color: #67e8f9; border-top-color: rgba(34, 211, 238, 0.6); background: rgba(34, 211, 238, 0.05); }
.th-sub { display: block; margin-top: 4px; font-size: 0.7rem; font-weight: 500; letter-spacing: 0; text-transform: none; color: var(--muted); }
.dtable tr.total td { border-top: 1px solid var(--border); background: rgba(148, 163, 184, 0.06); font-weight: 700; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 10px; vertical-align: middle; }
.mono  { font-weight: 600; }
.money { font-weight: 500; font-size: 0.85rem; }
.pos { color: #6ee7b7; } .neg { color: #fda4af; } .flat { color: var(--muted); }

/* Weight cells: number above a slim progress track (compact, so the table fits without scrolling) */
.wcell { min-width: 0; }
.wnum { display: block; margin-bottom: 7px; font-size: 0.85rem; font-weight: 600; }
.wtrack { height: 4px; border-radius: 999px; background: rgba(148, 163, 184, 0.12); overflow: hidden; }
.wfill { height: 100%; border-radius: 999px; transform-origin: left center; animation: growX 1s cubic-bezier(.22, 1, .36, 1) both; }
.f-emerald { background: linear-gradient(90deg, #10b981, #34d399); }
.f-cyan    { background: linear-gradient(90deg, #06b6d4, #22d3ee); }

/* Allocation & capital table: fixed column widths so every column fits standard screens */
.alloc { table-layout: fixed; width: 100%; }
.alloc th { white-space: normal; padding: 14px 10px; font-size: 0.66rem; letter-spacing: 0.05em; line-height: 1.45; }
.alloc th.grp { font-size: 0.68rem; padding: 14px 12px; }
.alloc td { padding: 16px 10px; overflow: hidden; text-overflow: ellipsis; }
.alloc td.asset b { display: inline-block; max-width: calc(100% - 22px); overflow: hidden; text-overflow: ellipsis; vertical-align: middle; }
.alloc .money { font-size: clamp(0.68rem, 0.8vw, 0.82rem); letter-spacing: -0.01em; }
.alloc .mono  { font-size: clamp(0.72rem, 0.85vw, 0.86rem); }

/* Donut cards: centred header, chart, legend chips */
[class*="st-key-donut_"] {
  gap: 0.5rem; padding: 24px 22px 16px; border-radius: 20px; border: 1px solid var(--border);
  background: var(--card); backdrop-filter: blur(12px); box-shadow: 0 8px 26px rgba(0, 0, 0, 0.2);
  transition: border-color 0.3s ease; animation: fadeUp 0.7s ease both;
}
[class*="st-key-donut_"]:hover { border-color: rgba(52, 211, 153, 0.3); }
[class*="st-key-donut_"] [data-testid="stPlotlyChart"] { border: none; background: transparent; box-shadow: none; padding: 0; animation: none; }
.donut-head { text-align: center; }
.donut-title { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; }
.donut-sub { margin-top: 3px; font-size: 0.8rem; color: var(--muted); }
.donut-head::after { content: ""; display: block; width: 36px; height: 3px; margin: 14px auto 0; border-radius: 999px; background: var(--accent); }
.donut-legend { display: flex; flex-wrap: wrap; justify-content: center; align-content: flex-start; gap: 8px 18px; min-height: 56px; font-size: 0.8rem; }
.donut-legend .lg { white-space: nowrap; color: var(--text); }
.donut-note { margin-top: 8px; text-align: center; font-size: 0.74rem; color: var(--muted); }

/* ---------- Landing cards ---------- */
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; margin: 10px 0; }
.info-card { padding: 24px 26px; border-radius: 18px; border: 1px solid var(--border); background: var(--card); backdrop-filter: blur(12px);
  transition: transform 0.25s ease, border-color 0.25s ease; animation: fadeUp 0.7s ease both; }
.info-card:hover { transform: translateY(-3px); border-color: rgba(34, 211, 238, 0.4); }
.info-card .ic { font-size: 1.5rem; }
.info-card h4 { margin: 12px 0 8px; font-size: 1rem; font-weight: 700; }
.info-card p  { margin: 0; font-size: 0.88rem; color: var(--muted); line-height: 1.65; }

/* ---------- Sidebar: clearly separated sections ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(11, 17, 32, 0.98));
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.4rem 1.3rem 2.6rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 1.05rem; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #cbd5e1; }
.brand { display: flex; align-items: center; gap: 12px; margin: 2px 0 6px; animation: fadeIn 0.8s ease both; }
.logo { width: 42px; height: 42px; display: grid; place-items: center; font-size: 1.3rem; border-radius: 12px;
  background: linear-gradient(135deg, #10b981, #06b6d4); box-shadow: 0 8px 20px rgba(16, 185, 129, 0.28); }
.brand-name { font-weight: 800; font-size: 1.05rem; color: #f8fafc; line-height: 1.1; }
.brand-sub  { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px; }
.side-section {
  margin: 30px 0 2px; padding-top: 24px; border-top: 1px solid var(--border);
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
}
.side-section.first { margin-top: 18px; padding-top: 0; border-top: none; }
.side-divider { height: 1px; margin: 14px 0 4px; background: var(--border); }
.hint { margin: 2px 0 4px; font-size: 0.76rem; line-height: 1.6; color: var(--muted); }
.hint b { color: #a7f3d0; font-weight: 600; }

/* ---------- Welcome guide dialog ---------- */
div[data-testid="stDialog"] div[role="dialog"] {
  background: linear-gradient(160deg, rgba(15, 23, 42, 0.98), rgba(17, 26, 46, 0.98)) !important;
  border: 1px solid rgba(52, 211, 153, 0.26); border-radius: 22px !important;
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.6);
  animation: fadeUp 0.5s ease both;
}
.guide-intro { margin: 0 0 8px; color: var(--muted); font-size: 0.95rem; line-height: 1.65; }
.guide-step {
  display: flex; gap: 16px; align-items: flex-start; padding: 16px 18px; margin: 12px 0; border-radius: 16px;
  border: 1px solid var(--border); background: rgba(30, 41, 59, 0.42);
  transition: transform 0.25s ease, border-color 0.25s ease; animation: fadeUp 0.5s ease both;
}
.guide-step:hover { transform: translateY(-2px); border-color: rgba(52, 211, 153, 0.4); }
.guide-num {
  flex: 0 0 34px; width: 34px; height: 34px; display: grid; place-items: center; border-radius: 50%;
  font-weight: 800; color: #04221c; background: linear-gradient(135deg, #10b981, #06b6d4);
}
.guide-title { font-weight: 700; color: #f1f5f9; margin-bottom: 4px; }
.guide-text { font-size: 0.9rem; color: var(--muted); line-height: 1.65; }
.guide-text b { color: #a7f3d0; font-weight: 600; }
.guide-list { margin: 8px 0 0; padding-left: 18px; }
.guide-list li { margin: 8px 0; }
.guide-note { margin-top: 14px; font-size: 0.78rem; color: var(--muted); line-height: 1.6; }

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], .stTextInput input, .stDateInput input {
  background-color: rgba(15, 23, 42, 0.7) !important; color: #f1f5f9 !important;
}
[data-baseweb="input"] { border-radius: 10px !important; border: 1px solid var(--border) !important; transition: border-color 0.2s ease, box-shadow 0.2s ease; }
[data-baseweb="input"]:hover { border-color: rgba(52, 211, 153, 0.4) !important; }
[data-baseweb="input"]:focus-within { border-color: var(--emerald-2) !important; box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.16) !important; }
.stSlider [role="slider"] { background: var(--emerald-2) !important; box-shadow: 0 0 0 5px rgba(52, 211, 153, 0.16); transition: box-shadow 0.2s ease; }
.stSlider [role="slider"]:hover { box-shadow: 0 0 0 9px rgba(52, 211, 153, 0.24); }
[data-testid="stSliderThumbValue"] { color: var(--emerald-2) !important; font-weight: 700; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  width: 100%; padding: 0.7rem 1rem; border-radius: 12px; border: 1px solid rgba(167, 243, 208, 0.4);
  background: linear-gradient(135deg, #10b981, #06b6d4); color: #04221c; font-weight: 700; letter-spacing: 0.01em;
  box-shadow: 0 6px 20px rgba(16, 185, 129, 0.22); transition: all 0.25s ease;
}
.stButton > button p, .stDownloadButton > button p { color: inherit; font-weight: 700; }
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px); border-color: #a7f3d0; color: #022c22; box-shadow: 0 12px 28px rgba(34, 211, 238, 0.28);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(0.99); }
.stDownloadButton > button { background: rgba(30, 41, 59, 0.7); color: #a7f3d0; box-shadow: none; }
.stDownloadButton > button:hover { background: linear-gradient(135deg, #10b981, #06b6d4); }
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
  padding: 0.5rem 0.7rem; font-size: 0.82rem; background: rgba(30, 41, 59, 0.6); color: #cbd5e1;
  border: 1px solid var(--border); box-shadow: none;
}
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {
  background: rgba(16, 185, 129, 0.12); color: #a7f3d0; border-color: rgba(52, 211, 153, 0.5); box-shadow: none;
}

/* ---------- Ticker multiselect ---------- */
div[data-baseweb="select"] > div {
  background-color: rgba(15, 23, 42, 0.7) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-baseweb="select"] > div:hover { border-color: rgba(52, 211, 153, 0.4) !important; }
div[data-baseweb="select"] > div:focus-within { border-color: var(--emerald-2) !important; box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.16) !important; }
div[data-baseweb="select"] input { color: #f1f5f9 !important; }
span[data-baseweb="tag"] {
  background-color: rgba(16, 185, 129, 0.14) !important; border: 1px solid rgba(52, 211, 153, 0.34);
  border-radius: 8px !important; transition: background-color 0.2s ease, border-color 0.2s ease; animation: fadeIn 0.35s ease both;
}
span[data-baseweb="tag"] span { color: #a7f3d0 !important; font-weight: 600; }
span[data-baseweb="tag"]:hover { background-color: rgba(16, 185, 129, 0.26) !important; border-color: #6ee7b7; }
div[data-baseweb="popover"] > div { border-radius: 12px; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul[role="listbox"] {
  background: #0f172a !important; border: 1px solid rgba(52, 211, 153, 0.22); border-radius: 12px; box-shadow: 0 18px 40px rgba(0, 0, 0, 0.5);
}
[data-baseweb="popover"] li[role="option"] { color: #e2e8f0 !important; transition: background-color 0.15s ease; }
[data-baseweb="popover"] li[role="option"]:hover, [data-baseweb="popover"] li[role="option"][aria-selected="true"] { background-color: rgba(52, 211, 153, 0.12) !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; padding: 6px; border-radius: 16px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.55); backdrop-filter: blur(10px); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { height: auto; padding: 10px 20px; border-radius: 11px; background: transparent; transition: all 0.25s ease; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; font-size: 0.9rem; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(52, 211, 153, 0.08); }
.stTabs [data-baseweb="tab"]:hover p { color: #e2e8f0; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(16, 185, 129, 0.9), rgba(6, 182, 212, 0.85)) !important; }
.stTabs [aria-selected="true"] p { color: #04221c !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.3rem; animation: fadeUp 0.55s ease both; }

/* ---------- Native elements ---------- */
[data-testid="stPlotlyChart"] {
  border-radius: 18px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.4); padding: 10px;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.22); overflow: hidden; animation: fadeIn 0.9s ease both; transition: border-color 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover { border-color: rgba(52, 211, 153, 0.3); }
[data-testid="stExpander"], [data-testid="stStatus"] { border: 1px solid var(--border) !important; border-radius: 14px !important; background: rgba(30, 41, 59, 0.35); }
[data-testid="stAlert"] { border-radius: 14px; backdrop-filter: blur(8px); animation: fadeUp 0.5s ease both; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }

.footer { margin-top: 48px; padding-top: 18px; text-align: center; font-size: 0.78rem; line-height: 1.7; color: var(--muted); border-top: 1px solid var(--border); }

/* ---------- Accessibility: respect reduced-motion preference ---------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
/* ---------- Small screens ---------- */
@media (max-width: 640px) {
  .hero { padding: 22px; } .hero h1 { font-size: 1.6rem; } .kpi-value { font-size: 1.55rem; }
}
</style>
"""


# =========================================================================== #
#                         PRESENTATION HELPERS (HTML)
# =========================================================================== #
# Thai needs no letter-spacing / forced upper-casing and a slightly taller line height.
THAI_CSS = """
<style>
.kpi-label, .dtable th, .side-section, .brand-sub, .th-sub, .badge, .section-title, .guide-title,
.donut-title, .donut-sub, .alloc th { letter-spacing: 0 !important; text-transform: none !important; }
.dtable th { font-size: 0.76rem !important; line-height: 1.55 !important; }
.alloc th, .alloc th.grp { font-size: 0.72rem !important; }
.hero h1 { line-height: 1.4; }
.hero p, .kpi-sub, .info-card p, .guide-text, .hint, .guide-intro, .guide-note, .donut-sub { line-height: 1.8; }
</style>
"""


def esc(text) -> str:
    """HTML-escape user-influenced text (also neutralises '$' so Markdown won't treat it as LaTeX)."""
    return _escape(str(text)).replace("$", "&#36;")


def num(value) -> str:
    """Wrap a number in the monospaced style (JetBrains Mono, tabular figures)."""
    return f'<span class="num">{esc(value)}</span>'


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
    if lang_code() == "th":
        st.markdown(THAI_CSS, unsafe_allow_html=True)


def card_container(key: str):
    """Keyed container styled as a glass card via CSS (falls back to a plain container on old Streamlit)."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def side_section(title: str, first: bool = False) -> None:
    """Sidebar section heading with a subtle divider above it (none for the first section)."""
    render_html(f'<div class="side-section{" first" if first else ""}">{esc(title)}</div>')


def section_title(text: str) -> None:
    render_html(f'<div class="section-title">{esc(text)}</div>')


def render_hero() -> None:
    render_html(
        f"""
        <div class="hero">
          <h1>{tx('hero_title')}</h1>
          <p>{tx('hero_sub')}</p>
          <div class="badge-row">
            <span class="badge ok"><span class="dot"></span>{tx('badge_live')}</span>
            <span class="badge cyan" title="{tx('badge_model_tip')}">{tx('badge_model')}</span>
            <span class="badge" title="{tx('badge_long_tip')}">{tx('badge_long')}</span>
          </div>
        </div>
        """
    )


def render_landing() -> None:
    """Empty-state shown before the first optimisation run."""
    section_title(t("how_title"))
    render_html(
        f"""
        <div class="card-grid">
          <div class="info-card" style="animation-delay:.05s"><span class="ic">🧺</span>
            <h4>{tx('card1_t')}</h4><p>{tx('card1_p')}</p></div>
          <div class="info-card" style="animation-delay:.15s"><span class="ic">🎛️</span>
            <h4>{tx('card2_t')}</h4><p>{tx('card2_p')}</p></div>
          <div class="info-card" style="animation-delay:.25s"><span class="ic">🚀</span>
            <h4>{tx('card3_t')}</h4><p>{tx('card3_p')}</p></div>
        </div>
        """
    )


def render_footer() -> None:
    render_html(f'<div class="footer">{tx("footer")}</div>')


# =========================================================================== #
#                       TICKER SELECTION + APP STATE
# =========================================================================== #
def normalize_tickers(raw_items) -> tuple[list[str], list[str]]:
    """
    Clean user-entered symbols: split on commas/spaces, upper-case, apply aliases
    (e.g. TSMC -> TSM), drop invalid symbols and duplicates while keeping order.
    Returns (clean_symbols, rejected_tokens).
    """
    clean, rejected = [], []
    for item in raw_items:
        for token in re.split(r"[,;\s]+", str(item).strip().upper()):
            if not token:
                continue
            token = TICKER_ALIASES.get(token, token)
            if not TICKER_RE.match(token):
                rejected.append(token)
            elif token not in clean:
                clean.append(token)
    return clean[:MAX_TICKERS], rejected


# `st.session_state["tickers"]` is the single source of truth for the selection. Every way of
# changing it (picking, removing, typing a custom symbol, Clear all, Load example) goes through a
# callback that rewrites that list, so tickers accumulate instead of being overwritten.
#
# The multiselect's own key carries a version number. Whenever the canonical list has to be pushed
# back into the widget (clear, example, normalised custom input) the version is bumped, which
# re-creates the widget from `default=` without assigning to a live widget's state. This also makes
# the widget immune to the label/placeholder changing when the language is switched.
def ticker_label(symbol: str) -> str:
    """Dropdown / tag label, e.g. 'NVDA - NVIDIA'. Custom symbols are shown as typed."""
    name = TICKER_UNIVERSE.get(symbol)
    return f"{symbol} - {name}" if name else symbol


def init_app_state() -> None:
    """Defaults for every persisted setting (language, tickers, dates, rate, capital, FX)."""
    ss = st.session_state
    ss.setdefault("lang", LANG_EN)                      # default language: English
    ss.setdefault("tickers", list(DEFAULT_TICKERS))
    ss.setdefault("picker_v", 0)
    ss.setdefault("ticker_rejected", [])
    ss.setdefault("start_date", date.today() - timedelta(days=5 * 365))
    ss.setdefault("end_date", date.today())
    ss.setdefault("risk_free_pct", 4.0)
    ss.setdefault("capital_ccy", "THB")
    ss.setdefault("capital_amount", DEFAULT_CAPITAL)
    ss.setdefault("fx_rate", DEFAULT_FX)


# Widgets whose label / help text change with the language get a new internal identity in
# Streamlit. Re-assigning their current value at the top of every run keeps what the user entered.
PINNED_KEYS = ["start_date", "end_date", "risk_free_pct", "capital_ccy", "capital_amount", "fx_rate"]


def pin_widget_state() -> None:
    for key in PINNED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _picker_key() -> str:
    return f"ticker_picker_{st.session_state.get('picker_v', 0)}"


def _set_tickers(tickers: list[str], rejected=None) -> None:
    """Replace the canonical list and re-create the multiselect from it."""
    st.session_state["tickers"] = tickers
    st.session_state["ticker_rejected"] = rejected or []
    st.session_state["picker_v"] += 1


def on_picker_change() -> None:
    """Multiselect callback: runs on every add / remove / custom entry."""
    raw = list(st.session_state.get(_picker_key(), []))
    clean, rejected = normalize_tickers(raw)
    st.session_state["tickers"] = clean
    st.session_state["ticker_rejected"] = rejected
    if clean != raw:                      # e.g. 'ptt.bk' -> 'PTT.BK', or an invalid entry was dropped
        st.session_state["picker_v"] += 1


def on_clear() -> None:
    _set_tickers([])


def on_example() -> None:
    _set_tickers(list(DEFAULT_TICKERS))


def on_custom_add() -> None:
    """Fallback for older Streamlit versions without `accept_new_options`."""
    raw = st.session_state.get("custom_ticker_input", "")
    st.session_state["custom_ticker_input"] = ""
    clean, rejected = normalize_tickers(list(st.session_state["tickers"]) + [raw])
    _set_tickers(clean, rejected)


def render_ticker_picker() -> None:
    """Searchable, state-backed ticker multiselect with removable tags."""
    tickers = list(st.session_state["tickers"])
    # Ranked presets first; any custom symbols already selected are appended so they stay valid options.
    options = PRESET_TICKERS + [x for x in tickers if x not in TICKER_UNIVERSE]

    kwargs = dict(
        options=options,
        default=tickers,
        format_func=ticker_label,
        key=_picker_key(),
        on_change=on_picker_change,
        placeholder=t("pick_placeholder"),
        max_selections=MAX_TICKERS,
        help=t("pick_help"),
    )
    if SUPPORTS_NEW_OPTIONS:
        kwargs["accept_new_options"] = True    # lets users type any symbol, e.g. PTT.BK, and press Enter
    st.multiselect(t("pick_label"), **kwargs)

    if SUPPORTS_NEW_OPTIONS:
        render_html(f'<div class="hint">{t("pick_hint")}</div>')
    else:  # older Streamlit: separate input for custom symbols
        st.text_input(
            t("custom_label"), key="custom_ticker_input", on_change=on_custom_add,
            placeholder=t("custom_placeholder"),
        )

    rejected = st.session_state.get("ticker_rejected", [])
    if rejected:
        render_html(
            f'<div class="badge-row"><span class="badge warn">⚠ {tx("notice_invalid", items=", ".join(rejected))}</span></div>'
        )

    n = len(tickers)
    if n >= 2:
        render_html(
            f'<div class="badge-row"><span class="badge ok"><span class="dot"></span>{t("count_ok", n=num(n))}</span></div>'
        )
    else:
        render_html(f'<div class="badge-row"><span class="badge warn">{t("count_low", n=num(n))}</span></div>')

    col_a, col_b = st.columns(2)
    col_a.button(t("btn_clear"), key="btn_clear", on_click=on_clear)
    col_b.button(t("btn_example"), key="btn_example", on_click=on_example)


# --------------------------------------------------------------------------- #
# Currency helpers (capital is entered in THB or USD; FX = THB per 1 USD)
# --------------------------------------------------------------------------- #
def capital_in_both(amount: float, currency: str, fx: float) -> tuple[float, float]:
    """Return the capital as (USD, THB) given the entered amount, its currency code and THB-per-USD rate."""
    if currency == "USD":
        return amount, amount * fx
    return (amount / fx if fx > 0 else 0.0), amount


def on_capital_ccy_change() -> None:
    """When the capital currency is switched, convert the entered amount at the current rate."""
    new = st.session_state.get("capital_ccy", "THB")
    old = st.session_state.get("_capital_ccy_prev", "THB")
    fx = st.session_state.get("fx_rate", DEFAULT_FX)
    amount = st.session_state.get("capital_amount", 0.0)
    if new != old and fx > 0:
        st.session_state["capital_amount"] = round(amount / fx, 2) if new == "USD" else round(amount * fx, 2)
    st.session_state["_capital_ccy_prev"] = new


# --------------------------------------------------------------------------- #
# Data Acquisition Layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Returns a DataFrame (rows = dates, columns = tickers) with NaNs removed.
    Raises InputError (translated when displayed) for invalid tickers / empty results.
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
        raise InputError("e_no_data")

    # `Close` is a DataFrame for multiple tickers and (in some versions) also for one.
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Identify tickers that returned no data at all (invalid / delisted symbols).
    empty_cols = [c for c in prices.columns if prices[c].isna().all()]
    if empty_cols:
        raise InputError("e_bad_tickers", items=", ".join(empty_cols))

    # Clean: drop rows with any missing value so all assets share the same dates.
    prices = prices.dropna()
    if prices.empty:
        raise InputError("e_no_overlap")

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
        raise OptimError("e_maxsharpe", msg=result.message)
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
        raise OptimError("e_minvar", msg=result.message)
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
            {"type": "eq", "fun": lambda w, tgt=target: np.dot(w, mean_returns) - tgt},
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
        style = PORTFOLIO_STYLE[name]
        rows.append([
            f'<span class="swatch" style="background:{style["color"]}"></span><b>{esc(pf_full(name))}</b>',
            f'<span class="mono">{r["Expected Annual Return"]:.2%}</span>',
            f'<span class="mono">{r["Annual Volatility"]:.2%}</span>',
            f'<span class="mono" style="color:{style["color"]}">{r["Sharpe Ratio"]:.3f}</span>',
        ])
    return table_html([t("th_portfolio"), t("th_exp_ret"), t("th_vol"), t("th_sharpe")], rows)


def weight_bar_html(weight: float, css_class: str) -> str:
    """Compact weight cell: percentage (monospaced) above a slim progress track."""
    width = max(0.0, min(weight, 1.0)) * 100
    return (
        f'<div class="wcell"><span class="wnum">{weight:.2%}</span>'
        f'<div class="wtrack"><div class="wfill {css_class}" style="width:{width:.2f}%"></div></div></div>'
    )


def allocation_table_html(allocation: pd.DataFrame, color_map: dict, capital_usd: float, fx: float) -> str:
    """Weights (%) plus the exact amount per stock in US Dollars and Thai Baht for each portfolio."""
    names = list(allocation.columns)
    capital_thb = capital_usd * fx

    # Fixed column widths (sum = 100%) so the whole table fits the page without horizontal scrolling.
    cols = '<col style="width:10%">' + '<col style="width:12%"><col style="width:12.5%"><col style="width:14%">' * len(names)
    cols += '<col style="width:13%">'

    head1 = f'<tr><th rowspan="2">{tx("th_asset")}</th>'
    head2 = "<tr>"
    for n in names:
        style = PORTFOLIO_STYLE[n]
        head1 += f'<th class="grp {style["accent"]}" colspan="3">{style["icon"]} {esc(pf_full(n))}</th>'
        head2 += f'<th>{tx("th_weight")}</th><th>{tx("th_usd")}</th><th>{tx("th_thb")}</th>'
    short_a, short_b = pf_text(names[0], "short"), pf_text(names[1], "short")
    head1 += (
        f'<th rowspan="2">{tx("th_delta", a=short_a, b=short_b)}'
        f'<span class="th-sub">{tx("th_delta_sub")}</span></th></tr>'
    )
    head2 += "</tr>"

    body = ""
    for ticker, r in allocation.iterrows():
        body += (
            f'<tr><td class="asset"><span class="swatch" style="background:{color_map[ticker]}"></span>'
            f"<b>{esc(ticker)}</b></td>"
        )
        for n in names:
            w = float(r[n])
            body += (
                f"<td>{weight_bar_html(w, 'f-' + PORTFOLIO_STYLE[n]['accent'])}</td>"
                f'<td><span class="money">&#36;{w * capital_usd:,.2f}</span></td>'
                f'<td><span class="money">฿{w * capital_thb:,.2f}</span></td>'
            )
        diff = float(r[names[0]] - r[names[1]]) * 100
        cls = "pos" if diff > 0.05 else "neg" if diff < -0.05 else "flat"
        body += (
            f'<td><span class="mono {cls}">{diff:+.1f} '
            f'<abbr title="{tx("pp_tip")}">pp</abbr></span></td></tr>'
        )

    body += f'<tr class="total"><td>{tx("total")}</td>'
    for n in names:
        tot = float(allocation[n].sum())
        body += (
            f'<td><span class="mono">{tot:.2%}</span></td>'
            f'<td><span class="money">&#36;{tot * capital_usd:,.2f}</span></td>'
            f'<td><span class="money">฿{tot * capital_thb:,.2f}</span></td>'
        )
    body += "<td></td></tr>"

    return (
        '<div class="glass table-wrap"><table class="dtable alloc">'
        f"<colgroup>{cols}</colgroup><thead>{head1}{head2}</thead><tbody>{body}</tbody></table></div>"
    )


def kpi_card(label: str, value: str, sub: str, accent: str, delay: float, tip: str = "") -> str:
    tip_attr = f' title="{esc(tip)}"' if tip else ""
    return (
        f'<div class="kpi {accent}" style="animation-delay:{delay:.2f}s"{tip_attr}>'
        f'<div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
    )


def render_portfolio_kpis(name, weights, mean_returns, cov_matrix, rf, base_delay=0.0) -> None:
    """Three glass KPI cards (return / volatility / Sharpe Ratio) for one portfolio."""
    style = PORTFOLIO_STYLE[name]
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - rf) / vol if vol > 1e-12 else float("nan")
    cards = "".join([
        kpi_card(t("kpi_ret"), f"{ret:.2%}", t("kpi_ret_sub"), style["accent"], base_delay, t("kpi_ret_tip")),
        kpi_card(t("kpi_vol"), f"{vol:.2%}", t("kpi_vol_sub"), style["accent"], base_delay + 0.08, t("kpi_vol_tip")),
        kpi_card(t("kpi_sharpe"), f"{sharpe:.2f}", t("kpi_sharpe_sub"), style["accent"], base_delay + 0.16,
                 t("kpi_sharpe_tip", rf=f"{rf:.2%}")),
    ])
    render_html(
        f'<div><div class="pf-head"><span class="badge {style["badge"]}">{style["icon"]} {esc(pf_full(name))}</span>'
        f'<span class="pf-desc">{tx("pf_" + style["ident"] + "_desc")}</span></div>'
        f'<div class="kpi-grid">{cards}</div></div>'
    )


# --------------------------------------------------------------------------- #
# Output helpers – Plotly figures (dark, translucent)
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
        font=dict(family=FONT_SANS, color=TEXT_COLOR, size=13),
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.96)", bordercolor=EMERALD,
            font=dict(family=FONT_SANS, color="#f1f5f9", size=13),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        dragmode="pan",                       # click-and-drag (or one-finger drag) pans; wheel / pinch zooms
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=EMERALD),
    )
    layout.update(kwargs)
    fig.update_layout(**layout)
    # Numbers on axes use the monospaced font so they align and scan easily.
    fig.update_xaxes(tickfont=dict(family=FONT_MONO, size=12))
    fig.update_yaxes(tickfont=dict(family=FONT_MONO, size=12))
    return fig


def donut_slices(weights) -> list[int]:
    """Indices of the slices drawn in a donut (weight above 0.5%), largest first."""
    return sorted((i for i, w in enumerate(weights) if w > 0.005), key=lambda i: -float(weights[i]))


def build_donut(name: str, weights, tickers, color_map, capital_usd: float = 0.0, fx: float = DEFAULT_FX) -> go.Figure:
    """
    Symmetrical donut: equal 30 px margins on every side, the ring centred in a full-size domain,
    no in-chart title or legend (both are rendered as HTML in the card so nothing can overflow).
    """
    keep = donut_slices(weights)
    labels = [tickers[i] for i in keep]
    values = [float(weights[i]) for i in keep]
    amounts = [[v * capital_usd, v * capital_usd * fx] for v in values]  # [USD, THB] per slice
    fig = go.Figure(go.Pie(
        labels=labels, values=values, customdata=amounts, hole=0.62, sort=False,
        direction="clockwise", rotation=0, domain=dict(x=[0, 1], y=[0, 1]), showlegend=False,
        marker=dict(colors=[color_map[x] for x in labels], line=dict(color="#0b1120", width=3)),
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        textfont=dict(family=FONT_MONO, color="#0b1120", size=13),
        hovertemplate=("<b>%{label}</b><br>%{percent}<br>"
                       "$%{customdata[0]:,.2f}  ·  ฿%{customdata[1]:,.2f}<extra></extra>"),
    ))
    style = PORTFOLIO_STYLE[name]
    _dark_layout(
        fig, height=340,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
        uniformtext=dict(minsize=11, mode="hide"),          # hide percentages that do not fit a thin slice
        annotations=[dict(
            text=f"<b>{len(labels)}</b><br><span style='font-size:11px;color:{MUTED}'>{t('donut_assets')}</span>",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(family=FONT_MONO, size=24, color=style["color"]),
        )],
    )
    return fig


def donut_header_html(name: str) -> str:
    style = PORTFOLIO_STYLE[name]
    return (
        f'<div class="donut-head" style="--accent:{style["color"]}">'
        f'<div class="donut-title">{style["icon"]} {esc(pf_text(name, "main"))}</div>'
        f'<div class="donut-sub">({esc(pf_text(name, "sub"))})</div></div>'
    )


def donut_legend_html(weights, tickers, color_map) -> str:
    keep = donut_slices(weights)
    chips = "".join(
        f'<span class="lg"><span class="swatch" style="background:{color_map[tickers[i]]}"></span>'
        f'{esc(tickers[i])} {num(f"{float(weights[i]):.1%}")}</span>'
        for i in keep
    )
    hidden = len(weights) - len(keep)
    note = f'<div class="donut-note">{tx("donut_hidden", n=hidden)}</div>' if hidden else ""
    return f'<div><div class="donut-legend">{chips}</div>{note}</div>'


def build_correlation_heatmap(log_returns: pd.DataFrame) -> go.Figure:
    corr = log_returns.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=[[0, "#f472b6"], [0.5, "#1e293b"], [1, EMERALD]],
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(family=FONT_MONO, color="#f1f5f9"),
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(family=FONT_MONO, color=MUTED)),
        hovertemplate=f"%{{y}} vs %{{x}}<br>{t('corr_hover')}: %{{z:.2f}}<extra></extra>",
        xgap=3, ygap=3,
    ))
    _dark_layout(
        fig, height=460, margin=dict(l=20, r=20, t=55, b=20),
        title=dict(text=t("corr_title"), x=0, font=dict(size=16, color="#f1f5f9")),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_performance_chart(prices: pd.DataFrame, color_map) -> go.Figure:
    """Price history rebased to 100 at the start date."""
    rebased = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for tk in rebased.columns:
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased[tk], mode="lines", name=tk,
            line=dict(color=color_map[tk], width=2),
            hovertemplate=f"<b>{esc(tk)}</b><br>%{{x|%Y-%m-%d}}<br>{esc(t('perf_value'))}: %{{y:.1f}}<extra></extra>",
        ))
    _dark_layout(
        fig, height=440, margin=dict(l=20, r=20, t=55, b=20), hovermode="x unified",
        title=dict(text=t("perf_title"), x=0, font=dict(size=16, color="#f1f5f9")),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, title=t("perf_y")),
    )
    return fig


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(tickers: list[str], start: date, end: date) -> None:
    """Raise InputError (translated when displayed) for anything we can detect before hitting the network."""
    if len(tickers) < 2:
        raise InputError("e_min2")
    if len(tickers) > MAX_TICKERS:
        raise InputError("e_max", n=MAX_TICKERS)
    if start >= end:
        raise InputError("e_start_end")
    if end > date.today():
        raise InputError("e_future")
    if (end - start).days < 60:
        raise InputError("e_range")


# --------------------------------------------------------------------------- #
# Results dashboard (tabs)
# --------------------------------------------------------------------------- #
def render_results(res: dict, capital_usd: float, fx: float) -> None:
    tickers, prices, log_returns = res["tickers"], res["prices"], res["log_returns"]
    mean_returns, cov_matrix = res["mean_returns"], res["cov_matrix"]
    portfolios, rf = res["portfolios"], res["rf"]
    color_map = {x: PALETTE[i % len(PALETTE)] for i, x in enumerate(tickers)}

    # Status badges
    first_day, last_day = prices.index[0].date(), prices.index[-1].date()
    render_html(
        f"""
        <div class="badge-row">
          <span class="badge ok"><span class="dot"></span>{tx('b_done')}</span>
          <span class="badge cyan">{t('b_assets', n=num(len(tickers)))}</span>
          <span class="badge">{t('b_days', n=num(f'{len(log_returns):,}'))}</span>
          <span class="badge">{num(first_day)} → {num(last_day)}</span>
          <span class="badge warn">{t('b_rf', v=num(f'{rf:.2%}'))}</span>
        </div>
        """
    )

    tab_summary, tab_alloc, tab_data = st.tabs([t("tab_summary"), t("tab_alloc"), t("tab_data")])

    summary = build_summary_table(portfolios, mean_returns, cov_matrix, rf)
    allocation = build_allocation_table(portfolios, tickers)

    # ---------------------------- Executive Summary --------------------------- #
    with tab_summary:
        render_portfolio_kpis("Max Sharpe", portfolios["Max Sharpe"], mean_returns, cov_matrix, rf, 0.0)
        render_portfolio_kpis("Min Variance", portfolios["Min Variance"], mean_returns, cov_matrix, rf, 0.25)
        section_title(t("sec_summary"))
        render_html(summary_table_html(summary))

    # ------------------------------ Allocation -------------------------------- #
    with tab_alloc:
        col1, col2 = st.columns(2, gap="large")
        for col, key in zip((col1, col2), ("Max Sharpe", "Min Variance")):
            with col:
                with card_container(f"donut_{PORTFOLIO_STYLE[key]['accent']}"):
                    render_html(donut_header_html(key))
                    st.plotly_chart(
                        build_donut(key, portfolios[key], tickers, color_map, capital_usd, fx),
                        config=PLOTLY_CONFIG,
                    )
                    render_html(donut_legend_html(portfolios[key], tickers, color_map))
        section_title(t("sec_alloc"))
        render_html(
            f"""
            <div class="badge-row">
              <span class="badge ok">{t('cap_badge', usd=num(f'${capital_usd:,.2f}'), thb=num(f'฿{capital_usd * fx:,.2f}'))}</span>
              <span class="badge cyan">{t('fx_badge', v=num(f'$1 = ฿{fx:,.2f}'))}</span>
            </div>
            """
        )
        render_html(allocation_table_html(allocation, color_map, capital_usd, fx))
        st.caption(t("alloc_caption"))

    # ------------------------------- Raw data --------------------------------- #
    with tab_data:
        st.plotly_chart(build_performance_chart(prices, color_map), config=PLOTLY_CONFIG)
        st.caption(t("perf_caption"))
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(build_correlation_heatmap(log_returns), config=PLOTLY_CONFIG)
            st.caption(t("corr_caption"))
        with c2:
            section_title(t("stats_title"))
            vols = np.sqrt(np.diag(cov_matrix))
            rows = [[
                f'<span class="swatch" style="background:{color_map[tk]}"></span><b>{esc(tk)}</b>',
                f'<span class="mono">{mean_returns[i]:.2%}</span>',
                f'<span class="mono">{vols[i]:.2%}</span>',
                f'<span class="mono">{(mean_returns[i] - rf) / vols[i]:.2f}</span>',
            ] for i, tk in enumerate(tickers)]
            render_html(table_html([t("th_asset"), t("kpi_ret"), t("kpi_vol"), t("kpi_sharpe")], rows))
        section_title(t("prices_title"))
        price_view = prices.round(2).iloc[::-1].copy()            # newest trading day first
        price_view.index = price_view.index.strftime("%Y-%m-%d")
        price_view.index.name = t("col_date")
        try:
            # Per-column blue gradient (axis=0): each stock is scaled to its own price range.
            # `low` lifts the minimum off pure white so it still reads well on the dark UI.
            styled_prices = (
                price_view.style
                .background_gradient(cmap="Blues", axis=0, low=0.3, high=0.0)
                .format("{:,.2f}")
            )
            st.dataframe(styled_prices, height=420)
        except ImportError:                                       # matplotlib not installed
            st.dataframe(price_view, height=420)
        st.caption(t("prices_caption"))
        st.download_button(
            t("prices_dl"),
            data=prices.to_csv().encode("utf-8"),
            file_name="historical_prices.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Welcome guide (popup)
# --------------------------------------------------------------------------- #
def open_guide() -> None:
    """Beginner-friendly how-to popup. The dialog is built at call time so its title follows the language."""

    @st.dialog(t("guide_title"), width="large")
    def _guide() -> None:
        render_html(f'<div class="guide-intro">{tx("guide_intro")}</div>')

        def step(number: int, title: str, body: str) -> str:
            return (
                f'<div class="guide-step" style="animation-delay:{number * 0.07:.2f}s">'
                f'<div class="guide-num">{number}</div><div>'
                f'<div class="guide-title">{esc(title)}</div><div class="guide-text">{body}</div></div></div>'
            )

        render_html(
            "<div>"
            + step(1, t("g1_t"), t("g1_b"))
            + step(2, t("g2_t"), t("g2_b"))
            + step(3, t("g3_t"), t("g3_b"))
            + step(4, t("g4_t"), t("g4_b"))
            + f'<div class="guide-note">{t("g_note")}</div></div>'
        )
        st.write("")
        if st.button(t("g_start"), type="primary", key="guide_close"):
            st.session_state.has_seen_guide = True
            st.rerun()

    _guide()


# --------------------------------------------------------------------------- #
# Streamlit application
# --------------------------------------------------------------------------- #
def main() -> None:
    st.session_state.setdefault("lang", LANG_EN)          # default language: English
    st.set_page_config(page_title=t("page_title"), page_icon="📈", layout="wide")
    inject_css()
    init_app_state()
    pin_widget_state()

    # ------------------------- Input Layer (Sidebar) ------------------------ #
    with st.sidebar:
        render_html(
            f"""
            <div class="brand"><div class="logo">📈</div>
              <div><div class="brand-name">PortfolioLab</div>
              <div class="brand-sub">{tx('brand_sub')}</div></div></div>
            """
        )
        # The selection lives in st.session_state["lang"], so it persists across every interaction.
        st.radio("🌐 Language / ภาษา", LANG_OPTIONS, key="lang", horizontal=True)

        side_section(t("sec_assets"))
        render_ticker_picker()

        side_section(t("sec_time"))
        start_date = st.date_input(t("start_label"), key="start_date", help=t("start_help"))
        end_date = st.date_input(t("end_label"), key="end_date", help=t("end_help"))

        side_section(t("sec_risk"))
        risk_free_pct = st.slider(
            t("rf_label"), min_value=0.0, max_value=10.0, step=0.1, key="risk_free_pct", help=t("rf_help"),
        )

        side_section(t("sec_capital"))
        capital_ccy = st.radio(
            t("ccy_label"), ["THB", "USD"], key="capital_ccy", format_func=lambda c: t(f"ccy_{c}"),
            on_change=on_capital_ccy_change, help=t("ccy_help"),
        )
        capital_amount = st.number_input(
            t("amount_label"), min_value=0.0, step=1000.0, format="%.2f", key="capital_amount",
            help=t("amount_help"),
        )
        fx_rate = st.number_input(
            t("fx_label"), min_value=1.0, max_value=200.0, step=0.05, format="%.2f", key="fx_rate",
            help=t("fx_help"),
        )

        render_html('<div class="side-divider"></div>')
        run = st.button(t("btn_run"), type="primary")
        open_guide_clicked = st.button(t("btn_guide"), key="btn_guide")

    render_hero()
    risk_free_rate = risk_free_pct / 100.0
    capital_usd, _capital_thb = capital_in_both(capital_amount, capital_ccy, fx_rate)

    # Welcome guide: pops up automatically ONCE per session (flag is set when it opens, so
    # dismissing it with the X also counts), and re-opens whenever the sidebar button is clicked.
    if "has_seen_guide" not in st.session_state:
        st.session_state.has_seen_guide = True
        open_guide()
    elif open_guide_clicked:
        open_guide()

    # ----------------------- Core logic (error-guarded) --------------------- #
    if run:
        st.session_state.pop("result", None)   # clear any previous result
        error = None                           # (kind, message) if something fails

        with st.status(t("status_running"), expanded=True) as status:
            try:
                st.write(t("step_validate"))
                tickers = list(st.session_state.get("tickers", []))
                validate_inputs(tickers, start_date, end_date)

                st.write(t("step_download"))
                prices = fetch_prices(tuple(tickers), start_date, end_date)

                st.write(t("step_returns"))
                log_returns = compute_log_returns(prices)
                if len(log_returns) < MIN_OBSERVATIONS:
                    raise InputError("e_few_obs", n=len(log_returns), m=MIN_OBSERVATIONS)
                mean_returns, cov_matrix = annualised_stats(log_returns)

                st.write(t("step_solve"))
                w_sharpe = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate)
                w_minvar = min_variance_portfolio(mean_returns, cov_matrix)
                portfolios = {"Max Sharpe": w_sharpe, "Min Variance": w_minvar}

                # Keep everything in session state so the dashboard survives reruns
                # (e.g. clicking a download button, switching tabs or changing the language).
                st.session_state["result"] = dict(
                    tickers=tickers, prices=prices, log_returns=log_returns,
                    mean_returns=mean_returns, cov_matrix=cov_matrix,
                    portfolios=portfolios, rf=risk_free_rate,
                )
                status.update(label=t("status_done"), state="complete", expanded=False)

            # Known, user-correctable problems -> friendly message
            except InputError as exc:
                error = ("input", exc)
                status.update(label=t("status_input_err"), state="error", expanded=False)
            # Optimiser failures (e.g. singular covariance matrix)
            except OptimError as exc:
                error = ("optim", exc)
                status.update(label=t("status_optim_err"), state="error", expanded=False)
            # Anything unexpected (network down, yfinance changes, etc.)
            except Exception as exc:  # noqa: BLE001 – deliberate catch-all so the app never crashes
                error = ("other", f"{type(exc).__name__}: {exc}")
                status.update(label=t("status_unexpected"), state="error", expanded=False)

        if error:
            kind, detail = error
            if kind == "input":
                st.error(f"⚠️ {detail}")                         # __str__ translates in the active language
            elif kind == "optim":
                st.error(f"⚠️ {t('err_optim', msg=detail)}")
            else:
                st.error(f"⚠️ {t('err_generic')}")
                with st.expander(t("tech_details")):
                    st.code(detail)
            render_footer()
            return

    # --------------------------- Output Layer --------------------------- #
    result = st.session_state.get("result")
    if result is None:
        render_landing()
    else:
        render_results(result, capital_usd, fx_rate)
    render_footer()


if __name__ == "__main__":
    main()
