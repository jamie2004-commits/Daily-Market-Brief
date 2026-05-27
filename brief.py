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

In addition to the past closures already evident from the stale data above, use Google Search to find any OTHER full-day equity market closures occurring during this week ({week_mon_str} to {week_fri_str}, Monday through Friday inclusive). Check the official holiday calendars for major exchanges in: United States (NYSE/Nasdaq), United Kingdom (LSE), Eurozone (Xetra/Euronext), Japan (TSE), Korea (KRX), Hong Kong (HKEX), China (SSE/SZSE), Singapore (SGX), Australia (ASX). Include both PAST closures (already happened this week) AND UPCOMING closures (later this week).

Use Google Search to find the day's top market-moving news. Cover both the US trading session of {us_close_str} AND any market-relevant news that happened between that close and {sgt_date_str} — including weekends, holidays, after-hours moves, and any major Asian session developments since.

Then output ONLY a JSON object (no markdown fences, no preamble, no commentary) with this exact structure:

{{
  "drivers": [
    {{
      "headline": "Short punchy headline under 12 words",
      "body": "4-6 sentences explaining what happened, why it mattered, and how the market reacted. Specific and factual.",
      "impacted_tickers": ["TICKER1", "TICKER2", "TICKER3", "TICKER4"]
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
  "on_the_horizon": "A short prose paragraph (3-5 sentences) covering 2-4 MAJOR upcoming events that sit BEYOND the next 3 days but have significant market implications."
}}

Rules:
- Exactly 3 drivers, ordered by market impact. Each driver MUST be tied to meaningful price action — typically a 1%+ move in major names, a notable sector move, or a clear inflection in a major index, yield, or commodity. A story being merely newsworthy is NOT enough; if markets didn't react, do not promote it to a driver. If you can't find three high-impact stories, choose the next-most-impactful even if smaller, but never include throwaway commentary or routine policy meetings with negligible market reaction. Prioritize stories from the most recent 24-48 hours when impact is comparable.
- impacted_tickers: 3-5 real US-listed ticker symbols per driver. Standard symbols only (AAPL, NVDA, JPM, XLE, etc.). Never invent tickers.
- Make headlines specific enough that someone searching for them on Google News will find the actual articles you're referencing.
- catalysts_ahead: items scheduled in the next 3 calendar days from today ({sgt_date_str}). Could be 1-6 items depending on what's on the calendar. Include FOMC / Fed meetings and speeches, US government and policy announcements, major economic data releases (CPI, PCE, GDP, NFP, jobless claims, retail sales, etc.), and major upcoming earnings (use real ticker symbols). Use real dates. Order chronologically.
- market_closures: one entry per (date, holiday) pair. "markets" must be country names only (US, UK, Japan, Korea, Hong Kong, Singapore, China, Eurozone, Australia) — NEVER ticker symbols or index names. Group multiple countries observing the same holiday on the same date into one entry. Cover the FULL week {week_mon_str} to {week_fri_str}: include past closures already in the stale list AND any upcoming closures you find via search. Order chronologically. If no closures, return [].
- on_the_horizon: a prose paragraph (3-5 sentences, no bullet points, no headers) summarizing 2-4 MAJOR strategic events expected beyond the next 3 days but within roughly the next 2 months. These are forward-looking developments distinct from the daily-calendar catalysts above. Examples of what belongs here: rumored or scheduled IPOs (e.g., SpaceX, Stripe, Klarna), big-deal M&A milestones, antitrust rulings, major product launches (Apple events, Nvidia GTC, etc.), election milestones, central bank decisions further out, OPEC meetings, big sector inflection points. Use Google Search to identify what's actually upcoming. Be specific with names and approximate dates. If genuinely nothing notable is coming up, return a single short sentence acknowledging that — but this should be rare.
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
               stale_markets, week_mon_str, week_fri_str):
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

    prompt = PROMPT_TEMPLATE.format(
        us_close_str=us_close_str,
        sgt_date_str=sgt_date_str,
        prices_block=prices_block,
        stale_block=stale_block,
        week_mon_str=week_mon_str,
        week_fri_str=week_fri_str,
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
    for d in drivers:
        verified = []
        for sym in d.get("impacted_tickers", []):
            pct = get_ticker_pct(sym)
            if pct is not None:
                verified.append({"symbol": sym, "pct": pct})
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


def build_pdf(out_path, sgt_date_str, us_close_str, prices, drivers, catalysts,
              market_closures=None, on_the_horizon=None):
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
    story.append(Paragraph(f"Post-close summary · US session of {us_close_str}", muted))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_BORDER,
                            spaceBefore=4, spaceAfter=14))

    # Snapshot
    story.append(Paragraph("Market snapshot", h2))
    story.append(Spacer(1, 4))
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

    # Market holidays this week (grouped by date)
    if market_closures:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Market holidays this week", h3))
        story.append(Spacer(1, 4))

        # Group entries by date so date column merges naturally
        from itertools import groupby
        closures_grouped = []
        for date_str, group in groupby(market_closures, key=lambda x: x.get("date", "")):
            entries = []
            for c in group:
                markets = c.get("markets", "")
                reason = c.get("reason", "")
                if markets and reason:
                    entries.append((markets, reason))
            if entries:
                closures_grouped.append((date_str, entries))

        if closures_grouped:
            cl_data = []
            cl_spans = []
            cl_last_row_of_day = []
            row_idx = 0
            for date_str, entries in closures_grouped:
                start_row = row_idx
                for j, (markets, reason) in enumerate(entries):
                    cl_data.append([
                        date_str if j == 0 else "",
                        markets,
                        reason,
                    ])
                    row_idx += 1
                if len(entries) > 1:
                    cl_spans.append(("SPAN", (0, start_row), (0, row_idx - 1)))
                cl_last_row_of_day.append(row_idx - 1)

            cl_table = Table(cl_data, colWidths=[3.2*cm, 5.5*cm, 7.8*cm])
            cl_style = [
                ("FONT", (0,0), (-1,-1), "Helvetica", 9),
                ("TEXTCOLOR", (0,0), (0,-1), COL_MUTED),
                ("TEXTCOLOR", (1,0), (1,-1), COL_TEXT),
                ("TEXTCOLOR", (2,0), (2,-1), COL_MUTED),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("TOPPADDING", (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ] + cl_spans
            for end_row in cl_last_row_of_day[:-1]:
                cl_style.append(("LINEBELOW", (0, end_row), (-1, end_row), 0.25, COL_BORDER_LIGHT))
            cl_table.setStyle(TableStyle(cl_style))
            story.append(cl_table)
    story.append(Spacer(1, 18))

    # Drivers
    story.append(Paragraph("Top market drivers", h2))
    story.append(Paragraph("What moved markets and which names felt it", muted))
    for d in drivers:
        block = [Paragraph(d["headline"], h3), Paragraph(d["body"], body)]
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
        # Generate a Google News search link based on the headline
        # This avoids broken grounding-redirect URLs from Gemini
        q = urllib.parse.quote_plus(d["headline"])
        news_url = f"https://news.google.com/search?q={q}"
        block.append(Paragraph(
            f'<a href="{news_url}"><font color="#1a5fb4"><u>Find articles on Google News &rarr;</u></font></a>',
            src))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 12))

    # Catalysts ahead
    story.append(Spacer(1, 4))
    story.append(Paragraph("Catalysts ahead", h2))
    story.append(Paragraph("Next 3 days · Fed events, data releases, earnings", muted))

    # Group catalysts by day so the date appears once per group
    from itertools import groupby
    grouped = []
    for date, group in groupby(catalysts, key=lambda x: x["date"]):
        grouped.append((date, [g["event"] for g in group]))

    cd = []
    spans = []
    last_row_of_day = []  # for separator lines between day groups
    row_idx = 0
    for date, events in grouped:
        start_row = row_idx
        for j, event in enumerate(events):
            cd.append([date if j == 0 else "", event])
            row_idx += 1
        if len(events) > 1:
            spans.append(("SPAN", (0, start_row), (0, row_idx - 1)))
        last_row_of_day.append(row_idx - 1)

    ct = Table(cd, colWidths=[3.5*cm, 13*cm])
    cst = [
        ("FONT", (0,0), (-1,-1), "Helvetica", 9.5),
        ("TEXTCOLOR", (0,0), (0,-1), COL_MUTED),
        ("TEXTCOLOR", (1,0), (1,-1), COL_TEXT),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ] + spans
    # Add separator lines between day groups (not within a day's events)
    for end_row in last_row_of_day[:-1]:
        cst.append(("LINEBELOW", (0, end_row), (-1, end_row), 0.25, COL_BORDER_LIGHT))
    ct.setStyle(TableStyle(cst))
    story.append(ct)

    # On the horizon — strategic forward-looking prose paragraph
    if on_the_horizon and on_the_horizon.strip():
        story.append(Spacer(1, 16))
        story.append(Paragraph("On the horizon", h2))
        story.append(Paragraph("Bigger events expected over the coming weeks", muted))
        horizon_style = ParagraphStyle(
            "horizon", fontName="Helvetica", fontSize=9.5,
            textColor=COL_TEXT, leading=14, spaceBefore=2, spaceAfter=2,
        )
        story.append(Paragraph(on_the_horizon.strip(), horizon_style))

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

    print("Calling Gemini (synthesis + Google Search grounding)...")
    brief_data = synthesize(
        prices, gemini_key, us_close_str, sgt_date_str,
        stale_markets, week_mon_str, week_fri_str,
    )

    print("Verifying impacted tickers against yfinance...")
    drivers = verify_drivers(brief_data["drivers"])

    print("Building PDF...")
    pdf_path = f"/tmp/market-brief-{sgt_date.isoformat()}.pdf"
    build_pdf(
        pdf_path, sgt_date_str, us_close_str, prices, drivers,
        brief_data["catalysts_ahead"],
        market_closures=brief_data.get("market_closures") or [],
        on_the_horizon=brief_data.get("on_the_horizon") or "",
    )

    print(f"Emailing to {recipient}...")
    subject = f"Market brief — {sgt_date.strftime('%a %b %d, %Y')}"
    send_email(pdf_path, recipient, gmail_user, gmail_pw, subject,
               sgt_date_str)
    print("Done.")


if __name__ == "__main__":
    main()
