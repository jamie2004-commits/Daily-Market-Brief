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
      "impacted_tickers": ["TICKER1", "TICKER2", "TICKER3", "TICKER4"],
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
- Exactly 3 drivers, ordered by market impact. A story being merely newsworthy is NOT enough; if markets didn't react, do not promote it to a driver. Each driver MUST satisfy at least ONE of the following inclusion criteria:
  - Index move > 0.75%, or a sector ETF move > 1.5%
  - Single-stock move > 5% in a Magnificent 7 name, an S&P 100 constituent, or a major Asia heavyweight
  - Commodity move > 3%
  - Yield move > 5 bp, or a 30Y key-level breach
  - A central-bank decision, or an on-cycle macro release (CPI, PCE, NFP, GDP; jobless claims only if a genuine shock)
  - A geopolitical event with clear energy / safe-haven / risk-asset transmission (e.g. Strait of Hormuz, Taiwan Strait, Korean peninsula, US-China tariffs, Russia-Ukraine)
  - A major corporate event in a heavyweight name (earnings beat/miss WITH a stock move, M&A, or a guidance change)
  If you genuinely cannot find three stories meeting any of the above, choose the next-most-impactful even if it falls slightly short, but never include throwaway commentary or routine policy meetings with negligible market reaction. Prioritize stories from the most recent 24-48 hours when impact is comparable.
- For driver headlines AND bodies, you may use **double asterisks** sparingly to highlight the 1-3 most important numbers/moves/levels (e.g., "Brent **−7%**", "**30Y 5.20% breach**"). These render as bold red in the PDF. Use only for genuinely market-moving figures, not decorative emphasis.
- impacted_tickers (drivers): 3-5 real US-listed ticker symbols per driver. Standard symbols only (AAPL, NVDA, JPM, XLE, etc.). Never invent tickers.
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
        temperature=0.3,
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


def verify_drivers(drivers):
    """Verify impacted_tickers against yfinance. Accepts items where
    impacted_tickers is a list of strings (raw from Gemini) or already-verified
    dicts (idempotent). Returns drivers with dicts of {symbol, pct}."""
    for d in drivers:
        verified = []
        for sym in d.get("impacted_tickers", []):
            if isinstance(sym, dict) and "symbol" in sym and "pct" in sym:
                verified.append(sym)  # already verified; pass through
                continue
            symbol = sym.get("symbol") if isinstance(sym, dict) else sym
            if not symbol:
                continue
            pct = get_ticker_pct(symbol)
            if pct is not None:
                verified.append({"symbol": symbol, "pct": pct})
        d["impacted_tickers"] = verified
    return drivers


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


def _section_header(title, subtitle=None, total_width_cm=17.0):
    """Navy bar with a red left-accent stripe and a white title.
    The optional subtitle renders in muted grey, separated by a bullet."""
    title_style = ParagraphStyle(
        "sechdr_title", fontName="Helvetica-Bold", fontSize=11,
        textColor=colors.white, leading=14,
    )
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


def _build_forward_calendar(sgt_date, catalysts, closures):
    """5-column horizontal calendar combining catalysts and market closures.

    Layout:
      Row 0: day labels (Mon 25 May, Tue 26 May, ...) on dark navy bg, white text.
      Row 1: per-day content cell — closures rendered in red bold at top,
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

    day_content = {d: {"closures": [], "events": []} for d in days}
    overflow_events = []
    overflow_closures = []

    for c in (closures or []):
        d = _parse_event_date(c.get("date", ""), sgt_date)
        markets = (c.get("markets") or "").strip()
        reason = (c.get("reason") or "").strip()
        if not (markets or reason):
            continue
        line = f"{markets} — {reason}" if (markets and reason) else (markets or reason)
        if d in day_content:
            day_content[d]["closures"].append(line)
        else:
            overflow_closures.append((d, line))

    for c in (catalysts or []):
        d = _parse_event_date(c.get("date", ""), sgt_date)
        event = (c.get("event") or "").strip()
        if not event:
            continue
        if d in day_content:
            day_content[d]["events"].append(event)
        else:
            overflow_events.append((d, event))

    # Cell paragraph styles
    closure_style = ParagraphStyle(
        "cal_closure", fontName="Helvetica-Bold", fontSize=7.5,
        textColor=COL_ACCENT, leading=10, spaceAfter=2,
    )
    event_style = ParagraphStyle(
        "cal_event", fontName="Helvetica", fontSize=8,
        textColor=COL_TEXT, leading=10.5, spaceAfter=2,
    )
    empty_style = ParagraphStyle(
        "cal_empty", fontName="Helvetica", fontSize=8,
        textColor=colors.HexColor("#bbbbbb"), leading=10, alignment=1,
    )
    hdr_style = ParagraphStyle(
        "cal_hdr", fontName="Helvetica-Bold", fontSize=8.5,
        textColor=colors.white, leading=11,
    )

    # Header row — highlight today with a yellow accent
    header_cells = []
    today_idx = None
    for i, d in enumerate(days):
        if d == sgt_date:
            today_idx = i
            label = (f'<font color="#ffd166">{d.strftime("%a")}</font> '
                     f'{d.strftime("%-d %b")}  '
                     f'<font color="#ffd166" size="7">· TODAY</font>')
        else:
            label = (f'{d.strftime("%a")} '
                     f'<font color="#c0c8d4">{d.strftime("%-d %b")}</font>')
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


def _build_development_item(num, dev):
    """Render one 'developments to watch' item with red left bar, a numbered
    headline, category tags right-aligned at top, body, and impacted tickers
    at the bottom. Visual style mirrors the 'Developments to Watch' sections
    in finance briefs."""
    headline_style = ParagraphStyle(
        "dev_headline", fontName="Helvetica-Bold", fontSize=10.5,
        textColor=COL_TEXT, leading=14,
    )
    cat_style = ParagraphStyle(
        "dev_cat", fontName="Helvetica-Bold", fontSize=8.5,
        textColor=COL_ACCENT, leading=13, alignment=2,  # right-aligned
    )
    body_style = ParagraphStyle(
        "dev_body", fontName="Helvetica", fontSize=9.5,
        textColor=COL_TEXT, leading=14,
    )

    headline_html = _highlight_emphasis(
        f'<b>{num}.</b>&nbsp;&nbsp;{dev.get("headline", "")}'
    )
    body_html = _highlight_emphasis(dev.get("body", ""))

    headline_para = Paragraph(headline_html, headline_style)

    cats = dev.get("categories") or []
    if cats:
        cat_text = "  ·  ".join(str(c).strip().upper() for c in cats if str(c).strip())
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
              market_closures=None, developments_to_watch=None, sgt_date=None):
    # Resolve sgt_date for the Forward calendar window. Prefer the explicit
    # kwarg; otherwise parse sgt_date_str (format produced by main()).
    if sgt_date is None:
        try:
            sgt_date = datetime.strptime(sgt_date_str, "%A, %B %d, %Y").date()
        except ValueError:
            sgt_date = datetime.now(ZoneInfo("Asia/Singapore")).date()
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Market brief — {sgt_date_str}",
        author="Daily market brief",
    )
    eyebrow = ParagraphStyle("eyebrow", fontName="Helvetica", fontSize=8.5,
                             textColor=COL_DIM, leading=11, spaceAfter=4)
    h1 = ParagraphStyle("h1", fontName="Helvetica", fontSize=18,
                        textColor=COL_TEXT, leading=22)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13,
                        textColor=COL_TEXT, leading=16, spaceAfter=2)
    h3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5,
                        textColor=COL_TEXT, leading=14, spaceAfter=3)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                          textColor=COL_TEXT, leading=14)
    muted = ParagraphStyle("muted", fontName="Helvetica", fontSize=9,
                           textColor=COL_MUTED, leading=12, spaceAfter=6)
    src = ParagraphStyle("src", fontName="Helvetica", fontSize=7.5,
                         textColor=COL_DIM, leading=10, spaceBefore=4)

    story = []
    story.append(Paragraph("DAILY MARKET BRIEF", eyebrow))
    story.append(Paragraph(sgt_date_str, h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_BORDER,
                            spaceBefore=4, spaceAfter=14))

    # Snapshot
    story.append(_section_header("Market snapshot",
                                 f"Closes as of {us_close_str}"))
    story.append(Spacer(1, 8))
    data = [["Asset", "Close", "Day %"]]
    st = [
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 8.5),
        ("TEXTCOLOR", (0,0), (-1,0), COL_MUTED),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        ("FONT", (0,1), (-1,-1), "Helvetica", 10),
        ("TEXTCOLOR", (0,1), (-1,-1), COL_TEXT),
        ("LINEBELOW", (0,0), (-1,0), 0.5, COL_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]
    for i, p in enumerate(prices, start=1):
        data.append([p["display"], fmt_close(p["close"], p["kind"]),
                     fmt_pct(p["pct"])])
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
    story.append(_section_header("Top market drivers",
                                 "What moved markets and which names felt it"))
    story.append(Spacer(1, 8))
    for i, d in enumerate(drivers, start=1):
        headline_html = _highlight_emphasis(
            f'<b>{i}.</b>&nbsp;&nbsp;{d.get("headline", "")}'
        )
        body_html = _highlight_emphasis(d.get("body", ""))
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
            block.append(Paragraph(
                f'<i>Impacted:</i>&nbsp;&nbsp;{"  ·  ".join(parts)}', body))
        # Prefer a real article URL from Marketaux when Gemini attached one;
        # fall back to a Google News search link otherwise.
        source_url = (d.get("source_url") or "").strip()
        source_title = (d.get("source_title") or "").strip()
        is_real_url = (
            source_url.startswith("http")
            and "grounding-api-redirect" not in source_url
            and "vertexaisearch.cloud.google.com" not in source_url
        )
        if is_real_url:
            label = source_title if source_title else "Read the full article"
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
                f'<u>Find articles on Google News &rarr;</u></font></a>'
            )
        block.append(Paragraph(link_html, src))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 12))

    # Forward calendar — combined market closures + catalysts (Fed events,
    # data releases, earnings) laid out as a horizontal week view.
    story.append(Spacer(1, 8))
    story.append(_section_header(
        "Forward calendar",
        "This week · holidays, Fed events, data releases & earnings",
    ))
    # Small legend right under the header bar so the red bullets are unambiguous.
    legend_style = ParagraphStyle(
        "cal_legend", fontName="Helvetica", fontSize=8,
        textColor=COL_MUTED, leading=10, alignment=2,  # right-aligned
        spaceBefore=4, spaceAfter=4,
    )
    story.append(Paragraph(
        '<font color="#a01d2e"><b>●</b></font>&nbsp;&nbsp;indicates market closure',
        legend_style,
    ))

    cal_tbl, overflow_events, overflow_closures = _build_forward_calendar(
        sgt_date, catalysts or [], market_closures or [],
    )
    story.append(cal_tbl)

    # If any items fall outside the visible Mon-Fri window, list them below
    if overflow_events or overflow_closures:
        more_lines = []
        for d, line in overflow_closures:
            ds = d.strftime("%a %-d %b") if d else "Date?"
            more_lines.append(
                f'<font color="#a01d2e"><b>●</b></font> <b>{ds}</b> · {line}'
            )
        for d, line in overflow_events:
            ds = d.strftime("%a %-d %b") if d else "Date?"
            more_lines.append(f'<b>{ds}</b> · {line}')
        if more_lines:
            extra_style = ParagraphStyle(
                "extra_cal", fontName="Helvetica", fontSize=8.5,
                textColor=COL_MUTED, leading=12,
            )
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "<b>Beyond this week:</b>  " + "  ·  ".join(more_lines),
                extra_style,
            ))

    # Developments to watch into the next session — forward-looking setup
    # items styled with a red left bar, numbered continuing from drivers
    # (so the brief reads as one continuous 1-6 narrative).
    if developments_to_watch:
        story.append(Spacer(1, 18))
        story.append(_section_header(
            "Developments to watch into the next session",
            "Notable scheduled events beyond the forward calendar",
        ))
        story.append(Spacer(1, 8))
        # Items numbered starting at len(drivers) + 1 so drivers (1-3) and
        # developments (4-6) form a continuous list.
        start_num = (len(drivers) if drivers else 3) + 1
        for offset, dev in enumerate(developments_to_watch):
            story.append(_build_development_item(start_num + offset, dev))
            story.append(Spacer(1, 10))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(COL_DIM)
        canvas.drawRightString(A4[0] - 2*cm, 1*cm, f"Page {doc.page}")
        canvas.drawString(2*cm, 1*cm,
            "Generated automatically · prices from Yahoo Finance · news via Gemini + Google Search")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------- Email ----------

def send_email(pdf_path, recipient, gmail_user, gmail_pw, subject, sgt_date_str):
    # Support comma-separated list of recipients in the RECIPIENT_EMAIL secret
    recipients = [r.strip() for r in recipient.split(",") if r.strip()]
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    text = (
        f"Your daily market brief for {sgt_date_str} is attached.\n\n"
        "Sent automatically by your market-brief GitHub Action."
    )
    msg.attach(MIMEText(text, "plain"))
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

    print("Verifying impacted tickers against yfinance...")
    drivers = verify_drivers(brief_data["drivers"])
    developments = brief_data.get("developments_to_watch") or []

    print("Building PDF...")
    pdf_path = f"/tmp/market-brief-{sgt_date.isoformat()}.pdf"
    build_pdf(
        pdf_path, sgt_date_str, us_close_str, prices, drivers,
        brief_data["catalysts_ahead"],
        market_closures=brief_data.get("market_closures") or [],
        developments_to_watch=developments,
        sgt_date=sgt_date,
    )

    print(f"Emailing to {recipient}...")
    subject = f"Market brief — {sgt_date.strftime('%a %b %d, %Y')}"
    send_email(pdf_path, recipient, gmail_user, gmail_pw, subject,
               sgt_date_str)
    print("Done.")


if __name__ == "__main__":
    main()
