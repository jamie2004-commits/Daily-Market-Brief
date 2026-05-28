"""
Daily market brief generator — Gemini free-tier edition.

Prices: yfinance (free).
News + analysis: Gemini 2.5 Flash with Google Search grounding (free tier).
PDF: ReportLab.
Delivery: Gmail SMTP.

Runs on a schedule via GitHub Actions. Reads config from environment.
"""

import json
import os
import re
import smtplib
import ssl
import sys
import time
import urllib.parse
from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
import requests
from google import genai
from google.genai import types
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

# ---------- Watchlist ----------
# (yfinance_symbol, display_name, kind)
TICKERS = [
    # US indices
    ("^GSPC", "S&P 500", "index"),
    ("^IXIC", "Nasdaq", "index"),
    ("^DJI", "Dow Jones", "index"),
    ("^RUT", "Russell 2000", "index"),
    # Asian indices
    ("^N225", "Nikkei 225", "index"),
    ("^KS11", "KOSPI", "index"),
    ("^HSI", "Hang Seng", "index"),
    ("^STI", "Straits Times", "index"),
    # Vol / rates / FX
    ("^VIX", "VIX", "vol"),
    ("^TNX", "US 10Y yield", "rate"),
    ("DX-Y.NYB", "Dollar (DXY)", "fx"),
    # Commodities
    ("GC=F", "Gold", "commodity"),
    ("CL=F", "WTI crude", "commodity"),
]

# ---------- Colors ----------
COL_TEXT = colors.HexColor("#1a1a1a")
COL_MUTED = colors.HexColor("#666666")
COL_DIM = colors.HexColor("#888888")
COL_BORDER = colors.HexColor("#cccccc")
COL_BORDER_LIGHT = colors.HexColor("#ececec")
COL_POS = colors.HexColor("#0f7a3a")
COL_NEG = colors.HexColor("#b8243c")
# Section-header accent palette
COL_HEADER_BG = colors.HexColor("#1f2c4a")   # dark navy bar
COL_HEADER_SUB = colors.HexColor("#9aa6b8")  # subtitle on navy
COL_ACCENT = colors.HexColor("#a01d2e")      # rich red accent
COL_CELL_BG = colors.HexColor("#fafbfc")     # light cell background
COL_TODAY_BG = colors.HexColor("#fff8e1")    # today column tint (warm)
COL_CLOSED_BG = colors.HexColor("#fdf2f4")   # closure-day column tint (faint red)


# ---------- Localization ----------
# We render three editions per run: English, Simplified Chinese, Traditional
# Chinese. Each gets its own PDF. Chinese editions use an embedded TrueType CJK
# font (WenQuanYi Zen Hei) so glyphs render in any PDF viewer (Gmail, mobile).

LANGS = ["en", "sc", "tc"]

# Candidate paths for a TrueType (glyf-outline) CJK font. ReportLab cannot use
# Noto Sans CJK (PostScript/CFF outlines), so we rely on WenQuanYi Zen Hei,
# installed via the workflow (apt: fonts-wqy-zenhei).
_CJK_FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
]
_CJK_FONT_NAME = "CJK"
_cjk_state = {"checked": False, "available": False}


def _ensure_cjk_font():
    """Register the embedded CJK font once. Returns True if available.
    Bold/italic are aliased to the same face so <b>/<i> on Chinese text don't
    fall back to Helvetica (which would render Chinese as blank boxes)."""
    if _cjk_state["checked"]:
        return _cjk_state["available"]
    _cjk_state["checked"] = True
    path = next((p for p in _CJK_FONT_PATHS if os.path.exists(p)), None)
    if not path:
        print("  WARN: no CJK font found on system; Chinese editions will be "
              "skipped. Ensure the workflow installs fonts-wqy-zenhei.")
        _cjk_state["available"] = False
        return False
    try:
        pdfmetrics.registerFont(TTFont(_CJK_FONT_NAME, path, subfontIndex=0))
        registerFontFamily(
            _CJK_FONT_NAME, normal=_CJK_FONT_NAME, bold=_CJK_FONT_NAME,
            italic=_CJK_FONT_NAME, boldItalic=_CJK_FONT_NAME,
        )
        _cjk_state["available"] = True
        print(f"  CJK font registered from {path}")
    except Exception as e:
        print(f"  WARN: failed to register CJK font ({type(e).__name__}: {e}); "
              "Chinese editions will be skipped.")
        _cjk_state["available"] = False
    return _cjk_state["available"]


def _prose_font(lang):
    """Font for prose / Chinese text. English uses Helvetica; CJK uses the
    embedded font when available, else Helvetica (latin fallback)."""
    if lang in ("sc", "tc") and _cjk_state.get("available"):
        return _CJK_FONT_NAME
    return "Helvetica"


def _prose_font_bold(lang):
    # CJK has no separate bold; reuse the same face. English keeps Helvetica-Bold.
    if lang in ("sc", "tc") and _cjk_state.get("available"):
        return _CJK_FONT_NAME
    return "Helvetica-Bold"


def _sanitize_cjk(text, lang):
    """The CJK font lacks U+2212 (typographic minus). Swap it for an ASCII
    hyphen-minus in Chinese editions so negative figures don't vanish."""
    if lang in ("sc", "tc") and text:
        return text.replace("\u2212", "-")
    return text


# Per-language UI strings (everything that isn't model-generated content).
LABELS = {
    "en": {
        "eyebrow": "DAILY MARKET BRIEF",
        "snapshot": "Market snapshot",
        "snapshot_sub": "Closes as of {date}",
        "drivers": "Top market drivers",
        "drivers_sub": "What moved markets and which names felt it",
        "calendar": "Forward calendar",
        "calendar_sub": "This week · holidays, Fed events, data releases & earnings",
        "legend": "indicates market closure",
        "today": "TODAY",
        "beyond": "Beyond this week:",
        "impacted": "Impacted:",
        "read_article": "Read the full article",
        "find_news": "Find articles on Google News \u2192",
        "developments": "Developments to watch into the next session",
        "developments_sub": "Notable scheduled events beyond the forward calendar",
        "col_asset": "Asset", "col_close": "Close", "col_pct": "Day %",
        "footer": "Generated automatically · prices from Yahoo Finance · news via Gemini + Google Search",
        "page": "Page {n}",
        "email_body": ("Your daily market brief for {date} is attached in three "
                       "editions: English, Simplified Chinese, and Traditional "
                       "Chinese.\n\nSent automatically by your market-brief GitHub Action."),
    },
    "sc": {
        "eyebrow": "每日市场简报",
        "snapshot": "市场快照",
        "snapshot_sub": "截至 {date} 收盘",
        "drivers": "市场主要驱动因素",
        "drivers_sub": "推动市场的事件及受影响标的",
        "calendar": "前瞻日历",
        "calendar_sub": "本周 · 假期、美联储事件、数据发布与财报",
        "legend": "表示休市",
        "today": "今日",
        "beyond": "本周以外：",
        "impacted": "受影响标的：",
        "read_article": "阅读全文",
        "find_news": "在 Google 新闻查找相关报道 \u2192",
        "developments": "下一交易时段值得关注的进展",
        "developments_sub": "前瞻日历之外的重要既定事件",
        "col_asset": "资产", "col_close": "收盘", "col_pct": "当日 %",
        "footer": "自动生成 · 价格来自 Yahoo Finance · 新闻经由 Gemini + Google 搜索",
        "page": "第 {n} 页",
        "email_body": ("您 {date} 的每日市场简报已附上，共三个版本："
                       "英文、简体中文、繁体中文。\n\n由您的 market-brief GitHub Action 自动发送。"),
    },
    "tc": {
        "eyebrow": "每日市場簡報",
        "snapshot": "市場快照",
        "snapshot_sub": "截至 {date} 收盤",
        "drivers": "市場主要驅動因素",
        "drivers_sub": "推動市場的事件及受影響標的",
        "calendar": "前瞻日曆",
        "calendar_sub": "本週 · 假期、聯準會事件、數據發布與財報",
        "legend": "表示休市",
        "today": "今日",
        "beyond": "本週以外：",
        "impacted": "受影響標的：",
        "read_article": "閱讀全文",
        "find_news": "在 Google 新聞查找相關報導 \u2192",
        "developments": "下一交易時段值得關注的發展",
        "developments_sub": "前瞻日曆之外的重要既定事件",
        "col_asset": "資產", "col_close": "收盤", "col_pct": "當日 %",
        "footer": "自動生成 · 價格來自 Yahoo Finance · 新聞經由 Gemini + Google 搜尋",
        "page": "第 {n} 頁",
        "email_body": ("您 {date} 的每日市場簡報已附上，共三個版本："
                       "英文、簡體中文、繁體中文。\n\n由您的 market-brief GitHub Action 自動發送。"),
    },
}

# Snapshot asset display names per language. Follows the SC/TC variant rules
# (e.g. KOSPI = 韩综 / 韓綜). Tickers like VIX stay in Latin.
DISPLAY_NAMES = {
    "en": {s: d for s, d, _ in TICKERS},
    "sc": {
        "^GSPC": "标普500", "^IXIC": "纳斯达克", "^DJI": "道琼斯",
        "^RUT": "罗素2000", "^N225": "日经225", "^KS11": "韩综",
        "^HSI": "恒生", "^STI": "海峡时报", "^VIX": "VIX",
        "^TNX": "美国10年期国债收益率", "DX-Y.NYB": "美元指数 (DXY)",
        "GC=F": "黄金", "CL=F": "WTI原油",
    },
    "tc": {
        "^GSPC": "標普500", "^IXIC": "納斯達克", "^DJI": "道瓊斯",
        "^RUT": "羅素2000", "^N225": "日經225", "^KS11": "韓綜",
        "^HSI": "恒生", "^STI": "海峽時報", "^VIX": "VIX",
        "^TNX": "美國10年期公債殖利率", "DX-Y.NYB": "美元指數 (DXY)",
        "GC=F": "黃金", "CL=F": "WTI原油",
    },
}

# Weekday names for calendar day-column headers.
_WEEKDAYS = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "sc": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
    "tc": ["週一", "週二", "週三", "週四", "週五", "週六", "週日"],
}
# Full weekday names for the title date line.
_WEEKDAYS_FULL = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "sc": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
    "tc": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
}


def _fmt_title_date(d, lang):
    """Format the big title date per language."""
    if lang == "en":
        return d.strftime("%A, %B %d, %Y")
    wd = _WEEKDAYS_FULL[lang][d.weekday()]
    return f"{d.year}年{d.month}月{d.day}日 {wd}"


def _fmt_day_header(d, lang):
    """Format a calendar day-column header, e.g. 'Mon 25 May' / '周一 5月25日'."""
    wd = _WEEKDAYS[lang][d.weekday()]
    if lang == "en":
        return f'{wd} <font color="#c0c8d4">{d.strftime("%-d %b")}</font>'
    return f'{wd} <font color="#c0c8d4">{d.month}月{d.day}日</font>'


def _fmt_day_header_today(d, lang):
    """Today's calendar header, with a highlighted weekday + TODAY tag."""
    today_label = LABELS[lang]["today"]
    if lang == "en":
        wd = _WEEKDAYS[lang][d.weekday()]
        return (f'<font color="#ffd166">{wd}</font> {d.strftime("%-d %b")}  '
                f'<font color="#ffd166" size="7">· {today_label}</font>')
    wd = _WEEKDAYS[lang][d.weekday()]
    return (f'<font color="#ffd166">{wd}</font> {d.month}月{d.day}日  '
            f'<font color="#ffd166" size="7">· {today_label}</font>')


def _fmt_short_date(d, lang):
    """Short date used in the 'beyond this week' overflow line."""
    if lang == "en":
        return d.strftime("%a %-d %b")
    wd = _WEEKDAYS[lang][d.weekday()]
    return f"{wd} {d.month}月{d.day}日"


# ---------- Prices ----------

def fetch_prices():
    out = []
    today_sgt = datetime.now(ZoneInfo("Asia/Singapore")).date()
    for symbol, display, kind in TICKERS:
        try:
            hist = yf.Ticker(symbol).history(period="10d")
            if len(hist) < 2:
                print(f"  WARN: only {len(hist)} bars for {symbol}, skipping")
                continue
            # Filter out today's bar — for Asian indices, yfinance may
            # return an intraday bar dated today while the market is still
            # open. We want only fully completed daily closes.
            completed = hist[hist.index.date < today_sgt]
            if len(completed) < 2:
                print(f"  WARN: only {len(completed)} completed bars for {symbol}, skipping")
                continue
            close = float(completed["Close"].iloc[-1])
            prev = float(completed["Close"].iloc[-2])
            pct = (close - prev) / prev * 100.0
            close_date = completed.index[-1].date()
            out.append({
                "symbol": symbol,
                "display": display,
                "kind": kind,
                "close": close,
                "pct": pct,
                "date": close_date,
            })
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")
    return out


def get_ticker_pct(symbol):
    try:
        today_sgt = datetime.now(ZoneInfo("Asia/Singapore")).date()
        hist = yf.Ticker(symbol).history(period="10d")
        completed = hist[hist.index.date < today_sgt]
        if len(completed) < 2:
            return None
        close = float(completed["Close"].iloc[-1])
        prev = float(completed["Close"].iloc[-2])
        return (close - prev) / prev * 100.0
    except Exception:
        return None


# ---------- News fetching (Marketaux) ----------

def fetch_marketaux_news(api_token, total=15):
    """Fetch financial news articles from Marketaux's free tier.
    Returns up to `total` deduplicated articles across a few targeted queries."""
    if not api_token:
        return []

    base = "https://api.marketaux.com/v1/news/all"
    # Each query returns up to 3 articles on the free tier
    queries = [
        {"countries": "us", "limit": 3},
        {"countries": "us", "search": "stocks earnings", "limit": 3},
        {"search": "Federal Reserve interest rates inflation", "limit": 3},
        {"search": "oil crude commodities", "limit": 3},
        {"search": "Asia markets Nikkei Hang Seng KOSPI", "limit": 3},
    ]

    articles = []
    seen = set()

    for q in queries:
        if len(articles) >= total:
            break
        params = {
            "api_token": api_token,
            "language": "en",
            "filter_entities": "true",
            **q,
        }
        try:
            r = requests.get(base, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            for art in data.get("data", []):
                url = art.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                articles.append({
                    "title": art.get("title", ""),
                    "snippet": (art.get("snippet") or art.get("description") or "")[:400],
                    "url": url,
                    "source": art.get("source", ""),
                    "published_at": art.get("published_at", ""),
                    "entities": [
                        e.get("symbol", "")
                        for e in art.get("entities", [])
                        if e.get("symbol")
                    ][:5],
                })
        except Exception as e:
            print(f"  Marketaux query failed ({q}): {type(e).__name__}: {e}")

    return articles


# ---------- Stale market detection ----------

def _expected_gap_days(today_sgt):
    """How many calendar days back the latest close should be on a normal day,
    accounting for weekends. Returns 1 on Tue-Sat, 3 on Mon, 2 on Sun."""
    weekday = today_sgt.weekday()  # 0=Mon, 6=Sun
    if weekday == 0:
        return 3  # Monday → expect Friday's close
    if weekday == 6:
        return 2  # Sunday → expect Friday's close
    return 1      # Tue-Sat → expect yesterday's close


def detect_stale_markets(prices, today_sgt):
    """Identify tickers whose latest close is older than normal trading
    would explain. These are candidates for holiday closures."""
    expected_gap = _expected_gap_days(today_sgt)
    stale = []
    for p in prices:
        actual_gap = (today_sgt - p["date"]).days
        if actual_gap > expected_gap:
            stale.append({
                "display": p["display"],
                "symbol": p["symbol"],
                "latest_close": p["date"].strftime("%a, %b %d"),
                "extra_days": actual_gap - expected_gap,
            })
    return stale


# ---------- LLM synthesis ----------

PROMPT_TEMPLATE = """You are writing a US equity market summary for a reader in Singapore.

Context:
- The reader is opening this brief on {sgt_date_str} (Singapore time).
- The most recent completed US trading session was {us_close_str}.
- The current trading week runs from {week_mon_str} through {week_fri_str}.
- If those two dates are several days apart — for example, it is Monday morning SGT and the last US close was the previous Friday, or there was a US holiday in between — then a weekend or holiday gap exists. In that case, your news section MUST cover relevant developments from across that gap, not just the trading day itself.

Latest close per asset (from Yahoo Finance):
{prices_block}

{stale_block}

{articles_block}

In addition to the past closures already evident from the stale data above, use Google Search to find any OTHER full-day equity market closures occurring during this week ({week_mon_str} to {week_fri_str}, Monday through Friday inclusive). Check the official holiday calendars for major exchanges in: United States (NYSE/Nasdaq), United Kingdom (LSE), Eurozone (Xetra/Euronext), Japan (TSE), Korea (KRX), Hong Kong (HKEX), China (SSE/SZSE), Singapore (SGX), Australia (ASX). Include both PAST closures (already happened this week) AND UPCOMING closures (later this week).

For your three drivers, primarily use the curated Marketaux articles above (they are real, dated, and have direct URLs). Supplement with Google Search if a clearly bigger market-moving story isn't covered by the curated set. Either way, find the day's top market-moving news.

Then output ONLY a JSON object (no markdown fences, no preamble, no commentary) with this exact structure:

{{
  "drivers": [
    {{
      "headline": "Short punchy headline under 12 words",
      "body": "4-6 sentences explaining what happened, why it mattered, and how the market reacted. Specific and factual.",
      "impacted_tickers": ["AFFECTED_STOCK_OR_SECTOR_ETF", "ANOTHER"],
      "source_url": "https://reuters.com/article-xyz (or empty string if no direct URL)",
      "source_title": "Original article title or empty string"
    }}
  ],
  "catalysts_ahead": [
    {{"date": "Tue, May 26", "event": "Specific event description"}}
  ],
  "market_closures": [
    {{"date": "Mon, May 25", "markets": "US", "reason": "Memorial Day"}},
    {{"date": "Mon, May 25", "markets": "Korea, Hong Kong", "reason": "Buddha's Birthday"}},
    {{"date": "Wed, May 27", "markets": "Singapore", "reason": "Hari Raya Haji"}}
  ],
  "developments_to_watch": [
    {{
      "headline": "Name a specific upcoming event + its date, under 14 words. Use **markdown** to emphasize the date/name.",
      "body": "1 brief sentence on why it matters or what to watch.",
      "categories": ["TECH", "PRODUCT"]
    }}
  ]
}}

Rules:
- Provide 2 to 4 drivers, ordered by market impact (most impactful first). Select GLOBALLY — across US, Asia, Europe, commodities, rates, FX and geopolitics — whatever genuinely moved markets most, not just US equities. A story being merely newsworthy is NOT enough; if markets didn't react, do not include it. Each driver MUST satisfy at least ONE of the following inclusion thresholds:
  - Index move > 0.75%, or a sector ETF move > 1.5%
  - Single-stock move > 5% in a Magnificent 7 name, an S&P 100 constituent, or a major Asia heavyweight
  - Commodity move > 3%
  - Yield move > 5 bp, or a 30Y key-level breach
  - A central-bank decision, or an on-cycle macro release (CPI, PCE, NFP, GDP; jobless claims only if a genuine shock)
  - A geopolitical event with clear energy / safe-haven / risk-asset transmission (e.g. Strait of Hormuz, Taiwan Strait, Korean peninsula, US-China tariffs, Russia-Ukraine)
  - A major corporate event in a heavyweight name (earnings beat/miss WITH a stock move, M&A, or a guidance change)
- QUALITY BAR / NO PADDING: Return only as many drivers as genuinely clear the bar above. Two strong drivers are far better than three padded ones — if only two stories qualify, return exactly two. NEVER manufacture a driver to reach a count, and NEVER include filler such as: the VIX or any volatility index simply rising/falling; "markets edged higher/lower"; generic "investor sentiment / confidence / risk appetite" with no concrete catalyst; an index closing only fractionally (<0.75%) up or down; or routine, low-impact commentary. A driver about volatility "easing" or "fear gauge" movement is BANNED unless it is itself a >2 std-dev event with a named cause. Only if a session is genuinely quiet and fewer than two stories clear the bar may you include the single next-most-impactful item — but even then, never a volatility/sentiment throwaway.
- For driver headlines AND bodies, you may use **double asterisks** sparingly to highlight the 1-3 most important numbers/moves/levels (e.g., "Brent **−7%**", "**30Y 5.20% breach**"). These render as bold red in the PDF. Use only for genuinely market-moving figures, not decorative emphasis.
- impacted_tickers (drivers): 2-5 real US-listed symbols that ACTUALLY MOVED on this specific story. Use the affected individual stocks (e.g. NVDA, AAPL, JPM), sector ETFs (XLE, SMH, XLF, XLK), or bond ETFs (TLT, HYG, LQD). Do NOT use broad index-proxy ETFs — specifically NEVER SPY, QQQ, DIA, IWM, VOO or IVV — because the S&P 500, Nasdaq, Dow and Russell are already shown in the snapshot table above, so listing their ETFs is redundant and (on flat days) contradicts the snapshot. Only include names with a genuine move (roughly > 0.3%); never pad with names that barely moved. Pick tickers whose move is directionally consistent with the story. Standard symbols only; never invent tickers.
- Prioritize stories from the most recent 24-48 hours when impact is comparable.
- source_url and source_title: ALWAYS fill these from the curated Marketaux articles above whenever the driver is supported by one of them. If no curated article covers the story, leave both as empty strings (the PDF will fall back to a Google News search link). NEVER use a Vertex/grounding redirect URL — only real direct article URLs.
- Make headlines specific enough that someone searching for them on Google News will find the actual articles you're referencing.
- catalysts_ahead: items scheduled in the next 3 calendar days from today ({sgt_date_str}). Could be 1-6 items depending on what's on the calendar. Include FOMC / Fed meetings and speeches, US government and policy announcements, major economic data releases (CPI, PCE, GDP, NFP, jobless claims, retail sales, etc.), and major upcoming earnings (use real ticker symbols). Use real dates. Order chronologically.
- market_closures: one entry per (date, holiday) pair. "markets" must be country names only (US, UK, Japan, Korea, Hong Kong, Singapore, China, Eurozone, Australia) — NEVER ticker symbols or index names. Group multiple countries observing the same holiday on the same date into one entry. Cover the FULL week {week_mon_str} to {week_fri_str}: include past closures already in the stale list AND any upcoming closures you find via search. Order chronologically. If no closures, return [].
- developments_to_watch: 0 to 2 SPECIFIC upcoming events worth flagging — each must be a concrete, named, scheduled-or-rumored event with a real date or approximate timeframe. Good types of events to surface here: major company IPOs (an upcoming or newly-priced listing, an S-1 filing, a confirmed listing date, or a credibly-reported IPO in the pipeline — e.g. Stripe, SpaceX, Klarna, Databricks, Discord, or any sizable name), product launches and keynotes (Apple WWDC, Nvidia GTC, etc.), central-bank decision dates beyond the next 3 days, OPEC meetings, antitrust rulings, big M&A milestones, and notable scheduled earnings further out. Concrete examples: "Stripe expected to file S-1 in coming weeks", "Nvidia GTC keynote Jun 11", "Apple WWDC opens Jun 9", "OPEC+ ministerial Jun 1", "EU antitrust ruling on [company] expected mid-June". These are events that sit BEYOND the next-3-day calendar window and are NOT already shown anywhere else in this brief. Actively use Google Search to check for any major IPOs that are upcoming, recently filed, or rumored — these are especially worth flagging when present.
  HARD RULES for this field:
  1. EVENTS ONLY. Each item must name a specific, identifiable event (an IPO/listing, a keynote, a filing, a meeting, a ruling, a launch, a scheduled earnings date, a central-bank decision date, etc.). Do NOT write generic thematic commentary, market-outlook musings, or analysis. BANNED: vague headlines like "Federal Reserve's inflation stance", "AI sector momentum", "investors will monitor upcoming data", "watch for further earnings". If it isn't a nameable event with a when, it does NOT belong here.
  2. NO OVERLAP. Do NOT include anything already listed in catalysts_ahead or market_closures, and do NOT repeat any of the 3 drivers. The forward calendar already covers the next 3 days of Fed events, data releases, earnings, and holidays — developments_to_watch is strictly for events FURTHER OUT than that window.
  3. NO PADDING. If there are no specific notable events beyond the calendar window, return an empty array []. Do NOT invent filler to reach a count. Quality over quantity — zero good items is better than one vague one.
  - headline: under 14 words, naming the event and its date/timeframe. Use **markdown** for 1 key emphasis (e.g., the date or name) if helpful.
  - body: 1 brief sentence on why it matters / what to watch. Concise, not analytical.
  - categories: 2-3 short uppercase tags as an array. Pick from common buckets like: RATES, FED, FOMC, INFLATION, MACRO, GDP, CPI, PCE, EARNINGS, AI, TECH, SEMIS, CONSUMER, ENERGY, OIL, OPEC, COMMODITIES, GOLD, M&A, IPO, ANTITRUST, REGULATION, ASIA, CHINA, JAPAN, KOREA, GEOPOLITICS, ELECTION, TARIFFS, FX, PRODUCT. You may invent additional short tags if needed.
- Output ONLY the JSON object. Nothing before or after."""


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


def synthesize(prices, gemini_key, us_close_str, sgt_date_str,
               stale_markets, week_mon_str, week_fri_str, marketaux_articles):
    client = genai.Client(api_key=gemini_key)
    prices_block = "\n".join(
        f"- {p['display']}: {p['close']:.2f} ({p['pct']:+.2f}%) — latest close {p['date'].strftime('%a %b %d')}"
        for p in prices
    )

    if stale_markets:
        stale_lines = "\n".join(
            f"- {s['display']}: latest close {s['latest_close']} (older than expected by {s['extra_days']} day(s))"
            for s in stale_markets
        )
        stale_block = (
            "STALE MARKETS — the following assets have a latest-close date "
            "older than normal trading would explain, meaning they were closed "
            "for a holiday or unusual event. You MUST identify the SPECIFIC "
            "public holiday in each market's home country:\n\n"
            + stale_lines
        )
    else:
        stale_block = (
            "All assets have current latest-close dates consistent with "
            "normal trading; no past closures detected from data."
        )

    if marketaux_articles:
        articles_lines = []
        for i, a in enumerate(marketaux_articles, start=1):
            tickers = ", ".join(a["entities"]) if a["entities"] else "—"
            articles_lines.append(
                f"[{i}] ({a['source']}, {a['published_at']}) {a['title']}\n"
                f"    Tickers: {tickers}\n"
                f"    Snippet: {a['snippet']}\n"
                f"    URL: {a['url']}"
            )
        articles_block = (
            "Curated financial news articles (from Marketaux) — these are real, "
            "published articles you may use as primary sources for your drivers. "
            "Prefer these over generic web search hits when the same story is "
            "covered. For each driver you build from one of these articles, "
            "include the article's URL in `source_url` and title in `source_title`:\n\n"
            + "\n\n".join(articles_lines)
        )
    else:
        articles_block = (
            "No curated articles provided. Use Google Search to find market-moving news."
        )

    prompt = PROMPT_TEMPLATE.format(
        us_close_str=us_close_str,
        sgt_date_str=sgt_date_str,
        prices_block=prices_block,
        stale_block=stale_block,
        week_mon_str=week_mon_str,
        week_fri_str=week_fri_str,
        articles_block=articles_block,
    )
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.2,
    )

    # Retry on transient 503 / 429 / UNAVAILABLE errors with progressive backoff.
    # If gemini-2.5-flash stays unavailable, fall back to gemini-2.5-flash-lite.
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    backoff_seconds = [20, 45, 90]
    last_error = None

    for model_idx, model_name in enumerate(models_to_try):
        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                if model_idx > 0:
                    print(f"  (succeeded on fallback model: {model_name})")
                return _extract_json(response.text)
            except Exception as e:
                last_error = e
                msg = str(e)
                transient = any(
                    token in msg
                    for token in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "INTERNAL")
                )
                if not transient or attempt >= len(backoff_seconds):
                    # Either non-transient, or we've exhausted backoffs for this model
                    print(f"  Model {model_name} failed: {type(e).__name__}: {msg[:200]}")
                    break
                wait = backoff_seconds[attempt]
                print(f"  Gemini transient error on {model_name} "
                      f"(attempt {attempt + 1}/{len(backoff_seconds)}). "
                      f"Retrying in {wait}s...")
                time.sleep(wait)

    # All models and retries exhausted
    raise last_error


# Broad index-proxy ETFs that merely duplicate the snapshot indices. Listing
# them as 'impacted' is redundant and, on near-flat days, can show a move that
# contradicts the index in the snapshot (e.g. SPY -0.02% vs S&P +0.02%). We drop
# them outright and map them to the index we'd reconcile against.
_INDEX_PROXY_ETFS = {
    "SPY": "^GSPC", "VOO": "^GSPC", "IVV": "^GSPC", "SPLG": "^GSPC",
    "QQQ": "^IXIC", "QQQM": "^IXIC",
    "DIA": "^DJI",
    "IWM": "^RUT", "VTWO": "^RUT",
}

# Minimum absolute move (%) for a name to count as genuinely "impacted".
# Below this, a ticker isn't meaningfully part of the story and (near zero) is
# where sign-flips happen, so we drop it.
_MIN_IMPACT_MOVE = 0.25


def verify_drivers(drivers, prices=None):
    """Verify and sanity-check impacted_tickers against yfinance.

    Guards applied to every impacted ticker:
      - Drop broad index-proxy ETFs (SPY/QQQ/DIA/IWM/...) — redundant with the
        snapshot and a frequent source of sign-flip contradictions.
      - Drop tickers we can't price.
      - Drop negligible moves (|pct| < _MIN_IMPACT_MOVE): an "impacted" name
        should actually have moved, and near-zero is where the sign-flip the
        reader noticed comes from.
      - For any ticker that maps to a snapshot index, reconcile its sign with
        the snapshot: if they disagree, trust the snapshot value (same source,
        same session) rather than a separately-fetched figure.

    Accepts raw strings or already-verified {symbol, pct} dicts (idempotent)."""
    snapshot_pct = {p["symbol"]: p["pct"] for p in (prices or [])}
    for d in drivers:
        verified = []
        for sym in d.get("impacted_tickers", []):
            if isinstance(sym, dict) and "symbol" in sym and "pct" in sym:
                symbol, pct = sym.get("symbol"), sym.get("pct")
            else:
                symbol = sym.get("symbol") if isinstance(sym, dict) else sym
                pct = None
            if not symbol:
                continue
            symbol = str(symbol).upper().strip()
            # Drop redundant index-proxy ETFs
            if symbol in _INDEX_PROXY_ETFS:
                continue
            if pct is None:
                pct = get_ticker_pct(symbol)
            if pct is None:
                continue
            # Reconcile against the snapshot when we track the same instrument
            if symbol in snapshot_pct:
                pct = snapshot_pct[symbol]
            # Drop negligible / near-flat moves
            if abs(pct) < _MIN_IMPACT_MOVE:
                continue
            verified.append({"symbol": symbol, "pct": round(pct, 2)})
        d["impacted_tickers"] = verified
    return drivers


# ---------- Translation (EN -> SC / TC) ----------

_STYLE_GUIDE = """You are a financial-press translator. Rewrite the English market-brief content into {lang_name}, following these rules EXACTLY.

TRANSLATION DISCIPLINE:
- REWRITE, don't translate literally. Take the facts and write fresh {lang_name} sentences in financial-press style. Never lift English sentence structure.
- Topic-prominent: lead with the subject of interest, then what happened.
- Use press action verbs ({verbs_hint}) instead of plain 上涨/下跌 (上漲/下跌).
- Integrate attribution inline (e.g. 「据 X 报道」/「據 X 報導」), not parenthetically.
- Drop English-style inline markers (no "HERO ·", "Read-through:", "Base case:").
- Tickers stay in Latin (NVDA, TLT, D05.SI, 0992.HK, 005930.KS).
- Numbers stay in Western digits + currency symbols ($96.31, never spelled out).
- Acronyms stay in Latin: IPO, AI, GDP, CPI, PCE, NFP, FOMC, OPEC, WTI, VIX, ETF, DXY, M&A.

{variant_block}

COMPLIANCE (rewrite to hedged, non-advisory language):
- "may possibly affect" -> 「可能影响」/「可能影響」or「市场关注」/「市場關注」(NEVER「将影响」/「將影響」or「必定」).
- "investors should" -> REMOVE. Rewrite as 「市场参与者可能关注」/「市場參與者可能關注」.
- "recommended" -> NEVER use 推荐/推薦. Use 「市场解读」/「市場解讀」or「市场观察」/「市場觀察」.
- "fully priced in" -> 「市场已大致计入」/「市場已大致計入」(hedged, NOT「完全计入」/「完全計入」).
- NEVER output: 投资者应/投資者應, 建议/建議+买卖, 推荐/推薦, 近乎确定/近乎確定, 完全计入/完全計入, 必涨/必漲, 必跌, 必定.

You will receive a JSON object. Translate ONLY these text fields into {lang_name}: every "headline", every "body", every "event", every "markets", every "reason", every "categories" entry, every "source_title". 
DO NOT translate or alter: any "date" field (keep the exact English string like "Tue, May 28" — it is parsed by code), any "source_url", any "symbol"/ticker, any "pct" or numeric value, the JSON keys themselves, or any **double-asterisk** markers (keep them around the same emphasized figures).
For "categories": translate sector words (RATES->利率/利率, FED->美联储/聯準會, EARNINGS->财报/財報, ENERGY->能源, etc.) but keep acronyms (IPO, AI, GDP, CPI, PCE, FOMC, OPEC) in Latin.

Output ONLY the translated JSON object with the identical structure and keys. No markdown fences, no commentary."""

_VARIANT_SC = """SIMPLIFIED CHINESE (Mainland/SG) variant rules — use these forms, NEVER the Traditional ones:
收益率 (NOT 殖利率), 概率 (NOT 機率), 软件 (NOT 軟體), 数据 (NOT 資料), 数据中心 (NOT 資料中心), 英伟达 (NOT 輝達), 信息 (NOT 資訊), 互联网 (NOT 網際網路), 环比/同比 (NOT 月增/年增), 内存 (NOT 記憶體), 服务器 (NOT 伺服器), 霍尔木兹海峡 (NOT 荷姆茲海峽), 芯片/半导体 (NOT 晶片/半導體), 期货 (NOT 期貨), 特朗普 (NOT 川普), 鲍威尔 (NOT 鮑爾), 贝森特 (NOT 貝森特), 韩综 (NOT 韓綜). Use Simplified characters throughout."""

_VARIANT_TC = """TRADITIONAL CHINESE (Taiwan default) variant rules — use these forms, NEVER the Simplified ones:
殖利率 (NOT 收益率), 機率 (NOT 概率), 軟體 (NOT 软件), 資料 (NOT 数据), 資料中心 (NOT 数据中心), 輝達 (NOT 英伟达), 資訊 (NOT 信息), 網際網路 (NOT 互联网), 月增/年增 (NOT 环比/同比), 記憶體 (NOT 内存), 伺服器 (NOT 服务器), 荷姆茲海峽 (NOT 霍尔木兹海峡), 晶片/半導體 (NOT 芯片/半导体), 期貨 (NOT 期货), 川普 (NOT 特朗普), 鮑爾 (NOT 鲍威尔), 貝森特 (NOT 贝森特), 韓綜 (NOT 韩综). Use Traditional characters throughout."""


# High-confidence terminology normalization. Maps any non-target spelling (in
# either script) to the correct variant for the target language. Only includes
# distinctive financial/tech terms and proper nouns where the 1:1 mapping is
# unambiguous, so we never corrupt ordinary words.
_VARIANT_FIX = {
    "tc": {  # force Taiwan-standard terminology
        "輝達": ["英伟达", "英偉達", "辉达"],
        "川普": ["特朗普"],
        "鮑爾": ["鲍威尔", "鮑威爾"],
        "貝森特": ["贝森特"],
        "殖利率": ["收益率"],
        "機率": ["概率"],
        "軟體": ["软件", "軟件"],
        "記憶體": ["内存", "內存"],
        "伺服器": ["服务器", "服務器"],
        "晶片": ["芯片"],
        "半導體": ["半导体"],
        "資訊": ["信息"],
        "網際網路": ["互联网", "互聯網"],
        "資料中心": ["数据中心", "數據中心"],
        "荷姆茲": ["霍尔木兹", "霍爾木茲"],
        "期貨": ["期货"],
        "月增": ["环比", "環比"],
        "年增": ["同比"],
    },
    "sc": {  # force Mainland/SG-standard terminology
        "英伟达": ["輝達", "辉达", "英偉達"],
        "特朗普": ["川普"],
        "鲍威尔": ["鮑爾", "鮑威爾"],
        "贝森特": ["貝森特"],
        "收益率": ["殖利率"],
        "概率": ["機率"],
        "软件": ["軟體", "軟件"],
        "内存": ["記憶體", "內存"],
        "服务器": ["伺服器", "服務器"],
        "芯片": ["晶片"],
        "半导体": ["半導體"],
        "信息": ["資訊"],
        "互联网": ["網際網路", "互聯網"],
        "数据中心": ["資料中心", "數據中心"],
        "霍尔木兹": ["荷姆茲", "霍爾木茲"],
        "期货": ["期貨"],
        "环比": ["月增"],
        "同比": ["年增"],
    },
}

# Compliance: collapse over-confident / advisory phrasing to hedged forms.
_COMPLIANCE_FIX = {
    "sc": {"完全计入": "大致计入", "推荐": "市场观察", "近乎确定": "市场预期"},
    "tc": {"完全計入": "大致計入", "推薦": "市場觀察", "近乎確定": "市場預期"},
}
# Phrases that should never appear — logged as warnings for manual review.
_COMPLIANCE_WARN = {
    "sc": ["投资者应", "必涨", "必跌", "必定"],
    "tc": ["投資者應", "必漲", "必跌", "必定"],
}


def _enforce_variants(text, lang):
    """Apply high-confidence variant + compliance normalization to one string."""
    if lang not in ("sc", "tc") or not text:
        return text
    for target, sources in _VARIANT_FIX[lang].items():
        for s in sources:
            if s in text:
                text = text.replace(s, target)
    for bad, good in _COMPLIANCE_FIX[lang].items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def _scrub_translation(brief_data, lang):
    """Walk the translated brief and normalize every text field in place.
    Returns (brief_data, warnings)."""
    warnings = []
    fix = lambda s: _enforce_variants(s, lang)

    for d in brief_data.get("drivers", []):
        for k in ("headline", "body", "source_title"):
            if d.get(k):
                d[k] = fix(d[k])
    for c in brief_data.get("catalysts_ahead", []):
        if c.get("event"):
            c["event"] = fix(c["event"])
    for c in brief_data.get("market_closures", []):
        for k in ("markets", "reason"):
            if c.get(k):
                c[k] = fix(c[k])
    for dev in brief_data.get("developments_to_watch", []):
        for k in ("headline", "body"):
            if dev.get(k):
                dev[k] = fix(dev[k])
        if dev.get("categories"):
            dev["categories"] = [fix(x) for x in dev["categories"]]

    blob = json.dumps(brief_data, ensure_ascii=False)
    for bad in _COMPLIANCE_WARN.get(lang, []):
        if bad in blob:
            warnings.append(bad)
    return brief_data, warnings


def translate_brief(brief_data, lang, gemini_key):
    """Translate the model-generated text fields of brief_data into SC or TC,
    following the financial-press style guide. Returns a new translated dict.
    Date fields, tickers, URLs and numbers are preserved verbatim."""
    if lang == "en":
        return brief_data

    lang_name = "Simplified Chinese (简体中文)" if lang == "sc" else "Traditional Chinese (繁體中文)"
    verbs_hint = ("飙升/走强/收高/应声下挫/续创新高/失守/收复/回吐/走疲" if lang == "sc"
                  else "飆升/走強/收高/應聲下挫/續創新高/失守/收復/回吐/走疲")
    variant_block = _VARIANT_SC if lang == "sc" else _VARIANT_TC

    system = _STYLE_GUIDE.format(
        lang_name=lang_name, verbs_hint=verbs_hint, variant_block=variant_block,
    )
    # Only send the translatable subset to keep the model focused
    payload = {
        "drivers": brief_data.get("drivers", []),
        "catalysts_ahead": brief_data.get("catalysts_ahead", []),
        "market_closures": brief_data.get("market_closures", []),
        "developments_to_watch": brief_data.get("developments_to_watch", []),
    }
    prompt = system + "\n\nJSON to translate:\n" + json.dumps(payload, ensure_ascii=False)

    client = genai.Client(api_key=gemini_key)
    config = types.GenerateContentConfig(temperature=0.4)
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    backoff_seconds = [15, 30]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config,
                )
                translated = _extract_json(response.text)
                # Merge back: keep verified impacted_tickers from the English
                # drivers (they hold {symbol, pct} dicts we don't want re-translated).
                out = dict(brief_data)
                for key in ("drivers", "catalysts_ahead", "market_closures",
                            "developments_to_watch"):
                    if key in translated:
                        out[key] = translated[key]
                # Restore impacted_tickers (verified) onto drivers by position
                en_drivers = brief_data.get("drivers", [])
                for i, d in enumerate(out.get("drivers", [])):
                    if i < len(en_drivers):
                        d["impacted_tickers"] = en_drivers[i].get("impacted_tickers", [])
                        # Preserve source_url verbatim (URL must not be translated)
                        d["source_url"] = en_drivers[i].get("source_url", "")
                # Enforce SC/TC variant discipline + compliance scrub
                out, warns = _scrub_translation(out, lang)
                if warns:
                    print(f"  Translation [{lang}] compliance warnings: {warns}")
                return out
            except Exception as e:
                last_error = e
                msg = str(e)
                transient = any(t in msg for t in
                                ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "INTERNAL"))
                if not transient or attempt >= len(backoff_seconds):
                    print(f"  Translation [{lang}] failed on {model_name}: "
                          f"{type(e).__name__}: {msg[:160]}")
                    break
                wait = backoff_seconds[attempt]
                print(f"  Translation [{lang}] transient error; retry in {wait}s...")
                time.sleep(wait)

    raise last_error


# ---------- PDF ----------

def fmt_close(value, kind):
    if kind == "crypto": return f"${value:,.0f}"
    if kind == "commodity": return f"${value:,.2f}"
    if kind == "rate": return f"{value:.2f}%"
    if kind == "fx": return f"{value:.2f}"
    return f"{value:,.2f}"


def fmt_pct(pct):
    sign = "+" if pct >= 0 else "−"
    return f"{sign}{abs(pct):.2f}%"


def _section_header(title, subtitle=None, total_width_cm=17.0, lang="en"):
    """Navy bar with a red left-accent stripe and a white title.
    The optional subtitle renders in muted grey, separated by a bullet."""
    title_style = ParagraphStyle(
        "sechdr_title", fontName=_prose_font_bold(lang), fontSize=11,
        textColor=colors.white, leading=15,
    )
    title = _sanitize_cjk(title, lang)
    subtitle = _sanitize_cjk(subtitle, lang)
    if subtitle:
        text = (f"{title}  "
                f"<font color='#9aa6b8' size='9'>·  {subtitle}</font>")
    else:
        text = title
    para = Paragraph(text, title_style)
    accent_w = 0.18
    tbl = Table(
        [["", para]],
        colWidths=[accent_w*cm, (total_width_cm - accent_w)*cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), COL_ACCENT),
        ("BACKGROUND", (1, 0), (1, 0), COL_HEADER_BG),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), 10),
        ("RIGHTPADDING", (1, 0), (1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _parse_event_date(s, sgt_date):
    """Parse 'Tue, May 26' style strings into a date.
    Year is inferred from sgt_date, with rollover for cross-year boundaries."""
    if not s:
        return None
    s = s.strip().rstrip(",").strip()
    formats_no_year = [
        "%a, %b %d", "%a %b %d", "%A, %B %d", "%A %B %d",
        "%a, %B %d", "%A, %b %d", "%b %d", "%B %d",
    ]
    for fmt in formats_no_year:
        try:
            parsed = datetime.strptime(s, fmt).date().replace(year=sgt_date.year)
            # Resolve year-rollover when sgt_date is near Jan/Dec
            if (parsed - sgt_date).days < -200:
                parsed = parsed.replace(year=sgt_date.year + 1)
            elif (parsed - sgt_date).days > 200:
                parsed = parsed.replace(year=sgt_date.year - 1)
            return parsed
        except ValueError:
            continue
    formats_with_year = ["%a, %b %d, %Y", "%A, %B %d, %Y",
                         "%b %d, %Y", "%B %d, %Y"]
    for fmt in formats_with_year:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _build_forward_calendar(sgt_date, catalysts, closures, lang="en"):
    """5-column horizontal calendar combining catalysts and market closures.

    Layout:
      Row 0: day labels (Mon 25 May / 周一 5月25日) on dark navy bg, white text.
      Row 1: per-day content cell — closures rendered in red at top,
             events in normal text below. Empty days show '—'.

    Returns (table, overflow_events, overflow_closures) where overflow lists
    contain anything outside the visible Mon-Fri window."""
    weekday = sgt_date.weekday()  # 0=Mon, 6=Sun
    if weekday < 5:
        week_start = sgt_date - timedelta(days=weekday)
    else:
        # Weekend SGT brief: pivot to next week's Monday
        week_start = sgt_date + timedelta(days=(7 - weekday))
    days = [week_start + timedelta(days=i) for i in range(5)]

    # Closures use an en-dash separator (CJK font lacks it? it has U+2013) — use
    # a localized separator that the CJK font supports.
    sep = " — " if lang == "en" else "·"

    day_content = {d: {"closures": [], "events": []} for d in days}
    overflow_events = []
    overflow_closures = []

    for c in (closures or []):
        d = _parse_event_date(c.get("date", ""), sgt_date)
        markets = _sanitize_cjk((c.get("markets") or "").strip(), lang)
        reason = _sanitize_cjk((c.get("reason") or "").strip(), lang)
        if not (markets or reason):
            continue
        line = f"{markets}{sep}{reason}" if (markets and reason) else (markets or reason)
        if d in day_content:
            day_content[d]["closures"].append(line)
        else:
            overflow_closures.append((d, line))

    for c in (catalysts or []):
        d = _parse_event_date(c.get("date", ""), sgt_date)
        event = _sanitize_cjk((c.get("event") or "").strip(), lang)
        if not event:
            continue
        if d in day_content:
            day_content[d]["events"].append(event)
        else:
            overflow_events.append((d, event))

    pfont = _prose_font(lang)
    pfont_b = _prose_font_bold(lang)
    # Cell paragraph styles
    closure_style = ParagraphStyle(
        "cal_closure", fontName=pfont_b, fontSize=7.5,
        textColor=COL_ACCENT, leading=11, spaceAfter=2,
    )
    event_style = ParagraphStyle(
        "cal_event", fontName=pfont, fontSize=8,
        textColor=COL_TEXT, leading=11, spaceAfter=2,
    )
    empty_style = ParagraphStyle(
        "cal_empty", fontName=pfont, fontSize=8,
        textColor=colors.HexColor("#bbbbbb"), leading=10, alignment=1,
    )
    hdr_style = ParagraphStyle(
        "cal_hdr", fontName=pfont_b, fontSize=8.5,
        textColor=colors.white, leading=12,
    )

    # Header row — highlight today with a yellow accent
    header_cells = []
    today_idx = None
    for i, d in enumerate(days):
        if d == sgt_date:
            today_idx = i
            label = _fmt_day_header_today(d, lang)
        else:
            label = _fmt_day_header(d, lang)
        header_cells.append(Paragraph(label, hdr_style))

    # Content row — list of Paragraphs per cell (ReportLab handles vertical stack)
    content_cells = []
    for d in days:
        parts = []
        for cl in day_content[d]["closures"]:
            parts.append(Paragraph(f"● {cl}", closure_style))
        for ev in day_content[d]["events"]:
            parts.append(Paragraph(ev, event_style))
        if not parts:
            parts.append(Paragraph("—", empty_style))
        content_cells.append(parts)

    data = [header_cells, content_cells]
    col_w = 17.0 / 5.0
    col_widths = [col_w*cm] * 5

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), COL_HEADER_BG),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("VALIGN", (0, 1), (-1, 1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 1), (-1, 1), COL_CELL_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("LINEAFTER", (0, 0), (-2, 0), 0.5, colors.HexColor("#3a4868")),
    ]
    # Tint closure-day columns and the today column
    for i, d in enumerate(days):
        if day_content[d]["closures"]:
            style.append(("BACKGROUND", (i, 1), (i, 1), COL_CLOSED_BG))
    if today_idx is not None:
        style.append(("BACKGROUND", (today_idx, 1), (today_idx, 1), COL_TODAY_BG))

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style))
    return tbl, overflow_events, overflow_closures


def _highlight_emphasis(text, color="#a01d2e"):
    """Convert **markdown** emphasis to bold red HTML for ReportLab Paragraphs.
    Safe no-op when no asterisks are present."""
    if not text:
        return text
    return re.sub(
        r"\*\*([^*]+?)\*\*",
        rf'<font color="{color}"><b>\1</b></font>',
        text,
    )


def _build_development_item(num, dev, lang="en"):
    """Render one 'developments to watch' item with red left bar, a numbered
    headline, category tags right-aligned at top, and body. Visual style mirrors
    the 'Developments to Watch' sections in finance briefs."""
    pfont = _prose_font(lang)
    pfont_b = _prose_font_bold(lang)
    headline_style = ParagraphStyle(
        "dev_headline", fontName=pfont_b, fontSize=10.5,
        textColor=COL_TEXT, leading=15,
    )
    cat_style = ParagraphStyle(
        "dev_cat", fontName=pfont_b, fontSize=8.5,
        textColor=COL_ACCENT, leading=13, alignment=2,  # right-aligned
    )
    body_style = ParagraphStyle(
        "dev_body", fontName=pfont, fontSize=9.5,
        textColor=COL_TEXT, leading=15,
    )

    headline_txt = _sanitize_cjk(dev.get("headline", ""), lang)
    body_txt = _sanitize_cjk(dev.get("body", ""), lang)
    headline_html = _highlight_emphasis(f'<b>{num}.</b>&nbsp;&nbsp;{headline_txt}')
    body_html = _highlight_emphasis(body_txt)

    headline_para = Paragraph(headline_html, headline_style)

    cats = dev.get("categories") or []
    if cats:
        # Latin tags get upper-cased; CJK tags pass through unchanged
        def _fmt_cat(c):
            c = _sanitize_cjk(str(c).strip(), lang)
            return c.upper() if c.isascii() else c
        cat_text = "  ·  ".join(_fmt_cat(c) for c in cats if str(c).strip())
        cat_para = Paragraph(cat_text, cat_style)
    else:
        cat_para = Paragraph("", cat_style)

    # Title row: headline on the left, categories right-aligned on the right
    title_row = Table(
        [[headline_para, cat_para]],
        colWidths=[10.8*cm, 4.7*cm],
    )
    title_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    body_para = Paragraph(body_html, body_style)

    cell_contents = [title_row, Spacer(1, 4), body_para]

    outer = Table(
        [["", cell_contents]],
        colWidths=[0.18*cm, 16.82*cm],
    )
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), COL_ACCENT),
        ("BACKGROUND", (1, 0), (1, 0), COL_CELL_BG),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), 12),
        ("RIGHTPADDING", (1, 0), (1, 0), 12),
    ]))
    return outer


def build_pdf(out_path, sgt_date_str, us_close_str, prices, drivers, catalysts,
              market_closures=None, developments_to_watch=None, sgt_date=None,
              lang="en"):
    # Resolve sgt_date for the Forward calendar window. Prefer the explicit
    # kwarg; otherwise parse sgt_date_str (format produced by main()).
    if sgt_date is None:
        try:
            sgt_date = datetime.strptime(sgt_date_str, "%A, %B %d, %Y").date()
        except ValueError:
            sgt_date = datetime.now(ZoneInfo("Asia/Singapore")).date()

    L = LABELS[lang]
    pfont = _prose_font(lang)
    pfont_b = _prose_font_bold(lang)
    # Localized date strings
    title_date = _fmt_title_date(sgt_date, lang)
    if lang == "en":
        us_close_disp = us_close_str
    else:
        # us_close_str is an English date like "Wednesday, May 27"; reformat
        try:
            ucd = datetime.strptime(f"{us_close_str}, {sgt_date.year}",
                                    "%A, %B %d, %Y").date()
            us_close_disp = _fmt_short_date(ucd, lang)
        except ValueError:
            us_close_disp = us_close_str

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Market brief — {title_date}",
        author="Daily market brief",
    )
    eyebrow = ParagraphStyle("eyebrow", fontName=pfont, fontSize=8.5,
                             textColor=COL_DIM, leading=11, spaceAfter=4)
    h1 = ParagraphStyle("h1", fontName=pfont, fontSize=18,
                        textColor=COL_TEXT, leading=24)
    h3 = ParagraphStyle("h3", fontName=pfont_b, fontSize=10.5,
                        textColor=COL_TEXT, leading=15, spaceAfter=3)
    body = ParagraphStyle("body", fontName=pfont, fontSize=9.5,
                          textColor=COL_TEXT, leading=15)
    # Numeric/Latin styles always use Helvetica for crisp digits + proper signs
    body_latin = ParagraphStyle("body_latin", fontName="Helvetica", fontSize=9.5,
                                textColor=COL_TEXT, leading=14)
    src = ParagraphStyle("src", fontName=pfont, fontSize=7.5,
                         textColor=COL_DIM, leading=11, spaceBefore=4)

    story = []
    story.append(Paragraph(L["eyebrow"], eyebrow))
    story.append(Paragraph(title_date, h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_BORDER,
                            spaceBefore=4, spaceAfter=14))

    # Snapshot
    story.append(_section_header(L["snapshot"],
                                 L["snapshot_sub"].format(date=us_close_disp),
                                 lang=lang))
    story.append(Spacer(1, 8))
    names = DISPLAY_NAMES.get(lang, DISPLAY_NAMES["en"])
    data = [[L["col_asset"], L["col_close"], L["col_pct"]]]
    st = [
        # Header row uses the prose font so localized labels render (CJK-safe)
        ("FONT", (0,0), (-1,0), pfont_b, 8.5),
        ("TEXTCOLOR", (0,0), (-1,0), COL_MUTED),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        # Asset-name column in prose font; numeric columns in Helvetica
        ("FONT", (0,1), (0,-1), pfont, 10),
        ("FONT", (1,1), (-1,-1), "Helvetica", 10),
        ("TEXTCOLOR", (0,1), (-1,-1), COL_TEXT),
        ("LINEBELOW", (0,0), (-1,0), 0.5, COL_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]
    for i, p in enumerate(prices, start=1):
        disp = names.get(p["symbol"], p["display"])
        data.append([disp, fmt_close(p["close"], p["kind"]), fmt_pct(p["pct"])])
        c = COL_POS if p["pct"] >= 0 else COL_NEG
        st.append(("TEXTCOLOR", (2,i), (2,i), c))
        st.append(("FONT", (2,i), (2,i), "Helvetica-Bold", 10))
        if i < len(prices):
            st.append(("LINEBELOW", (0,i), (-1,i), 0.25, COL_BORDER_LIGHT))
    snap = Table(data, colWidths=[8*cm, 4*cm, 4*cm])
    snap.setStyle(TableStyle(st))
    story.append(snap)

    story.append(Spacer(1, 18))

    # Drivers
    story.append(_section_header(L["drivers"], L["drivers_sub"], lang=lang))
    story.append(Spacer(1, 8))
    for i, d in enumerate(drivers, start=1):
        headline_txt = _sanitize_cjk(d.get("headline", ""), lang)
        body_txt = _sanitize_cjk(d.get("body", ""), lang)
        headline_html = _highlight_emphasis(f'<b>{i}.</b>&nbsp;&nbsp;{headline_txt}')
        body_html = _highlight_emphasis(body_txt)
        block = [Paragraph(headline_html, h3), Paragraph(body_html, body)]
        if d.get("impacted_tickers"):
            parts = []
            for t in d["impacted_tickers"]:
                hx = "#0f7a3a" if t["pct"] >= 0 else "#b8243c"
                sign = "+" if t["pct"] >= 0 else "−"
                parts.append(
                    f'<b>{t["symbol"]}</b> '
                    f'<font color="{hx}"><b>{sign}{abs(t["pct"]):.2f}%</b></font>'
                )
            block.append(Spacer(1, 4))
            # Impacted label in prose font; tickers/pcts in Helvetica (Latin)
            impacted_html = (
                f'<font name="{pfont}"><i>{L["impacted"]}</i></font>&nbsp;&nbsp;'
                + "  ·  ".join(parts)
            )
            block.append(Paragraph(impacted_html, body_latin))
        # Prefer a real article URL from Marketaux when Gemini attached one;
        # fall back to a Google News search link otherwise.
        source_url = (d.get("source_url") or "").strip()
        source_title = _sanitize_cjk((d.get("source_title") or "").strip(), lang)
        is_real_url = (
            source_url.startswith("http")
            and "grounding-api-redirect" not in source_url
            and "vertexaisearch.cloud.google.com" not in source_url
        )
        if is_real_url:
            label = source_title if source_title else L["read_article"]
            link_html = (
                f'<a href="{source_url}"><font color="#1a5fb4"><u>{label}</u></font></a>'
            )
        else:
            # Strip **markdown** so the search query is clean
            clean_headline = re.sub(r"\*\*([^*]+?)\*\*", r"\1",
                                    d.get("headline", ""))
            q = urllib.parse.quote_plus(clean_headline)
            news_url = f"https://news.google.com/search?q={q}"
            link_html = (
                f'<a href="{news_url}"><font color="#1a5fb4">'
                f'<u>{L["find_news"]}</u></font></a>'
            )
        block.append(Paragraph(link_html, src))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 12))

    # Forward calendar — combined market closures + catalysts (Fed events,
    # data releases, earnings) laid out as a horizontal week view.
    story.append(Spacer(1, 8))
    story.append(_section_header(L["calendar"], L["calendar_sub"], lang=lang))
    # Small legend right under the header bar so the red bullets are unambiguous.
    legend_style = ParagraphStyle(
        "cal_legend", fontName=pfont, fontSize=8,
        textColor=COL_MUTED, leading=10, alignment=2,  # right-aligned
        spaceBefore=4, spaceAfter=4,
    )
    story.append(Paragraph(
        f'<font color="#a01d2e"><b>●</b></font>&nbsp;&nbsp;{L["legend"]}',
        legend_style,
    ))

    cal_tbl, overflow_events, overflow_closures = _build_forward_calendar(
        sgt_date, catalysts or [], market_closures or [], lang=lang,
    )
    story.append(cal_tbl)

    # If any items fall outside the visible Mon-Fri window, list them below
    if overflow_events or overflow_closures:
        sep = " · "
        more_lines = []
        for d, line in overflow_closures:
            ds = _fmt_short_date(d, lang) if d else "?"
            more_lines.append(
                f'<font color="#a01d2e"><b>●</b></font> <b>{ds}</b>{sep}{line}'
            )
        for d, line in overflow_events:
            ds = _fmt_short_date(d, lang) if d else "?"
            more_lines.append(f'<b>{ds}</b>{sep}{line}')
        if more_lines:
            extra_style = ParagraphStyle(
                "extra_cal", fontName=pfont, fontSize=8.5,
                textColor=COL_MUTED, leading=13,
            )
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                f'<b>{L["beyond"]}</b>  ' + "  ·  ".join(more_lines),
                extra_style,
            ))

    # Developments to watch — forward-looking events, numbered continuing
    # from drivers so the brief reads as one continuous list.
    if developments_to_watch:
        story.append(Spacer(1, 18))
        story.append(_section_header(L["developments"], L["developments_sub"],
                                     lang=lang))
        story.append(Spacer(1, 8))
        start_num = (len(drivers) if drivers else 3) + 1
        for offset, dev in enumerate(developments_to_watch):
            story.append(_build_development_item(start_num + offset, dev, lang=lang))
            story.append(Spacer(1, 10))

    footer_text = L["footer"]
    footer_font = pfont if lang in ("sc", "tc") and _cjk_state.get("available") else "Helvetica"
    page_label = L["page"]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(footer_font, 7.5)
        canvas.setFillColor(COL_DIM)
        canvas.drawRightString(A4[0] - 2*cm, 1*cm, page_label.format(n=doc.page))
        canvas.drawString(2*cm, 1*cm, footer_text)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------- Email ----------

def send_email(pdf_paths, recipient, gmail_user, gmail_pw, subject, sgt_date_str):
    """Attach one or more PDFs to a single email. pdf_paths may be a single
    path (str) or a list of paths."""
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]
    # Support comma-separated list of recipients in the RECIPIENT_EMAIL secret
    recipients = [r.strip() for r in recipient.split(",") if r.strip()]
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    text = (
        f"Your daily market brief for {sgt_date_str} is attached in three "
        "editions: English (-en), Simplified Chinese (-sc), and Traditional "
        "Chinese (-tc).\n\nSent automatically by your market-brief GitHub Action."
    )
    msg.attach(MIMEText(text, "plain"))
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header("Content-Disposition", "attachment",
                              filename=Path(pdf_path).name)
            msg.attach(attach)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(gmail_user, gmail_pw)
        server.send_message(msg)


# ---------- Main ----------

def main():
    try:
        gemini_key = os.environ["GEMINI_API_KEY"]
        gmail_user = os.environ["GMAIL_USER"]
        gmail_pw = os.environ["GMAIL_APP_PASSWORD"]
        recipient = os.environ["RECIPIENT_EMAIL"]
    except KeyError as e:
        print(f"FATAL: missing required env var: {e}")
        sys.exit(1)

    # Optional: Marketaux for curated article URLs
    marketaux_key = os.environ.get("MARKETAUX_API_KEY", "").strip()

    # SGT date = the day the user reads the brief (cron runs 8am SGT)
    sgt_now = datetime.now(ZoneInfo("Asia/Singapore"))
    sgt_date = sgt_now.date()
    sgt_date_str = sgt_date.strftime("%A, %B %d, %Y")
    print(f"SGT date (label): {sgt_date_str}")

    print("Fetching prices...")
    prices = fetch_prices()
    if len(prices) < 3:
        print("FATAL: fewer than 3 tickers fetched; aborting")
        sys.exit(1)

    # US close date = most recent close from the US EQUITY indexes only.
    # We deliberately exclude VIX/TNX/DXY here because rates and FX can
    # carry a different "latest date" than equities (FX trades 24h), which
    # would otherwise mislabel the brief's subtitle.
    us_equity_symbols = {"^GSPC", "^IXIC", "^DJI", "^RUT"}
    us_ticker_dates = [p["date"] for p in prices if p["symbol"] in us_equity_symbols]
    us_close_date = max(us_ticker_dates) if us_ticker_dates else prices[0]["date"]
    us_close_str = us_close_date.strftime("%A, %B %d")
    print(f"US close covered: {us_close_str}, {len(prices)} tickers")

    print("Detecting market closures...")
    stale_markets = detect_stale_markets(prices, sgt_date)
    if stale_markets:
        print(f"  {len(stale_markets)} stale markets (likely holiday closures):")
        for s in stale_markets:
            print(f"    - {s['display']}: latest {s['latest_close']} (+{s['extra_days']}d)")
    else:
        print("  No stale markets detected.")

    # Current trading-week Monday and Friday in SGT
    week_mon = sgt_date - timedelta(days=sgt_date.weekday())
    week_fri = week_mon + timedelta(days=4)
    week_mon_str = week_mon.strftime("%a, %b %d")
    week_fri_str = week_fri.strftime("%a, %b %d")
    print(f"  Week scope for closure detection: {week_mon_str} – {week_fri_str}")

    if marketaux_key:
        print("Fetching news from Marketaux...")
        marketaux_articles = fetch_marketaux_news(marketaux_key, total=15)
        print(f"  Got {len(marketaux_articles)} articles")
    else:
        print("MARKETAUX_API_KEY not set; using Google Search only.")
        marketaux_articles = []

    print("Calling Gemini (synthesis + Google Search grounding)...")
    brief_data = synthesize(
        prices, gemini_key, us_close_str, sgt_date_str,
        stale_markets, week_mon_str, week_fri_str, marketaux_articles,
    )

    print("Verifying & sanity-checking impacted tickers against yfinance...")
    drivers = verify_drivers(brief_data["drivers"], prices)
    brief_data["drivers"] = drivers
    developments = brief_data.get("developments_to_watch") or []

    # Register the embedded CJK font (needed for SC/TC editions).
    cjk_ok = _ensure_cjk_font()

    # Build the per-language brief data: English as-is, then translations.
    editions = {"en": brief_data}
    for lang in ("sc", "tc"):
        if not cjk_ok:
            print(f"Skipping {lang} edition (no CJK font available).")
            continue
        try:
            print(f"Translating brief -> {lang}...")
            editions[lang] = translate_brief(brief_data, lang, gemini_key)
        except Exception as e:
            print(f"  WARN: {lang} translation failed ({type(e).__name__}); "
                  "skipping this edition.")

    print("Building PDFs...")
    pdf_paths = []
    for lang in LANGS:
        if lang not in editions:
            continue
        bd = editions[lang]
        out_path = f"/tmp/market-brief-{sgt_date.isoformat()}-{lang}.pdf"
        build_pdf(
            out_path, sgt_date_str, us_close_str, prices,
            bd.get("drivers", drivers), bd.get("catalysts_ahead", []),
            market_closures=bd.get("market_closures") or [],
            developments_to_watch=bd.get("developments_to_watch") or [],
            sgt_date=sgt_date, lang=lang,
        )
        pdf_paths.append(out_path)
        print(f"  built {Path(out_path).name}")

    print(f"Emailing {len(pdf_paths)} edition(s) to {recipient}...")
    subject = f"Market brief — {sgt_date.strftime('%a %b %d, %Y')}"
    send_email(pdf_paths, recipient, gmail_user, gmail_pw, subject,
               sgt_date_str)
    print("Done.")


if __name__ == "__main__":
    main()
