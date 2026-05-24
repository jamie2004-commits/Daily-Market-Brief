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
from datetime import datetime
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
    ("^NDX", "Nasdaq 100", "index"),
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
    # Crypto
    ("BTC-USD", "Bitcoin", "crypto"),
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
    for symbol, display, kind in TICKERS:
        try:
            hist = yf.Ticker(symbol).history(period="5d")
            if len(hist) < 2:
                print(f"  WARN: only {len(hist)} bars for {symbol}, skipping")
                continue
            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            pct = (close - prev) / prev * 100.0
            out.append({
                "symbol": symbol,
                "display": display,
                "kind": kind,
                "close": close,
                "pct": pct,
                "date": hist.index[-1].date(),
            })
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")
    return out


def get_ticker_pct(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        if len(hist) < 2:
            return None
        close = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        return (close - prev) / prev * 100.0
    except Exception:
        return None


# ---------- LLM synthesis ----------

PROMPT_TEMPLATE = """You are writing a US equity market post-close summary covering the US session of {us_close_str}.

Official close prices from Yahoo Finance:
{prices_block}

Use Google Search to find the day's top market-moving news. Then output ONLY a JSON object (no markdown fences, no preamble, no commentary) with this exact structure:

{{
  "drivers": [
    {{
      "headline": "Short punchy headline under 12 words",
      "body": "4-6 sentences explaining what happened, why it mattered, and how the market reacted. Specific and factual.",
      "impacted_tickers": ["TICKER1", "TICKER2", "TICKER3", "TICKER4"],
      "sources": [
        {{"title": "Article title", "url": "https://..."}}
      ]
    }}
  ],
  "catalysts_ahead": [
    {{"date": "Tue, May 26", "event": "Specific event description"}}
  ]
}}

Rules:
- Exactly 3 drivers, ordered by market impact.
- impacted_tickers: 3-5 real US-listed ticker symbols per driver. Standard symbols only (AAPL, NVDA, JPM, XLE, etc.). Never invent tickers.
- sources: 1-2 real article URLs per driver from your search results.
- catalysts_ahead: items scheduled in the next 3 calendar days from today ({sgt_date_str}). Could be 1-6 items depending on what's on the calendar. Include FOMC / Fed meetings and speeches, US government and policy announcements, major economic data releases (CPI, PCE, GDP, NFP, jobless claims, retail sales, etc.), and major upcoming earnings (use real ticker symbols). Use real dates. Order chronologically.
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


def synthesize(prices, gemini_key, us_close_str, sgt_date_str):
    client = genai.Client(api_key=gemini_key)
    prices_block = "\n".join(
        f"- {p['display']}: {p['close']:.2f} ({p['pct']:+.2f}%)" for p in prices
    )
    prompt = PROMPT_TEMPLATE.format(
        us_close_str=us_close_str,
        sgt_date_str=sgt_date_str,
        prices_block=prices_block,
    )
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.3,
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )
    return _extract_json(response.text)


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


def build_pdf(out_path, sgt_date_str, us_close_str, prices, drivers, catalysts):
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
        if d.get("sources"):
            links = [
                f'<a href="{s["url"]}"><font color="#1a5fb4"><u>{s.get("title", s["url"])}</u></font></a>'
                for s in d["sources"][:2] if s.get("url")
            ]
            if links:
                block.append(Paragraph(
                    f"Read more: {' &nbsp;·&nbsp; '.join(links)}", src))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 12))

    # Catalysts ahead
    story.append(Spacer(1, 4))
    story.append(Paragraph("Catalysts ahead", h2))
    story.append(Paragraph("Major announcements, Fed events, economic data, earnings", muted))
    cd = [[c["date"], c["event"]] for c in catalysts]
    ct = Table(cd, colWidths=[3.5*cm, 13*cm])
    cst = [
        ("FONT", (0,0), (-1,-1), "Helvetica", 9.5),
        ("TEXTCOLOR", (0,0), (0,-1), COL_MUTED),
        ("TEXTCOLOR", (1,0), (1,-1), COL_TEXT),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]
    for i in range(len(cd) - 1):
        cst.append(("LINEBELOW", (0,i), (-1,i), 0.25, COL_BORDER_LIGHT))
    ct.setStyle(TableStyle(cst))
    story.append(ct)

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

    # US close date = most recent close that yfinance returned (yesterday in US terms)
    # We pick this from a US-only ticker to avoid Asian holidays misleading us.
    us_ticker_dates = [p["date"] for p in prices if p["symbol"].startswith("^G") or p["symbol"] == "^NDX" or p["symbol"] == "^DJI"]
    us_close_date = max(us_ticker_dates) if us_ticker_dates else prices[0]["date"]
    us_close_str = us_close_date.strftime("%A, %B %d")
    print(f"US close covered: {us_close_str}, {len(prices)} tickers")

    print("Calling Gemini (synthesis + Google Search grounding)...")
    brief_data = synthesize(prices, gemini_key, us_close_str, sgt_date_str)

    print("Verifying impacted tickers against yfinance...")
    drivers = verify_drivers(brief_data["drivers"])

    print("Building PDF...")
    pdf_path = f"/tmp/market-brief-{sgt_date.isoformat()}.pdf"
    build_pdf(pdf_path, sgt_date_str, us_close_str, prices, drivers,
              brief_data["catalysts_ahead"])

    print(f"Emailing to {recipient}...")
    subject = f"Market brief — {sgt_date.strftime('%a %b %d, %Y')}"
    send_email(pdf_path, recipient, gmail_user, gmail_pw, subject,
               sgt_date_str)
    print("Done.")


if __name__ == "__main__":
    main()
