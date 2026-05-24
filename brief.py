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
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

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
TICKERS = [
    ("^GSPC", "S&P 500", "index"),
    ("^NDX", "Nasdaq 100", "index"),
    ("^DJI", "Dow Jones", "index"),
    ("^RUT", "Russell 2000", "index"),
    ("^VIX", "VIX", "vol"),
    ("^TNX", "US 10Y yield", "rate"),
    ("DX-Y.NYB", "Dollar (DXY)", "fx"),
    ("GC=F", "Gold", "commodity"),
    ("CL=F", "WTI crude", "commodity"),
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

PROMPT_TEMPLATE = """You are writing a US equity market post-close summary for {date_str}.

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
  "watch_next_week": [
    {{"date": "Mon, May 25", "event": "Specific event"}}
  ]
}}

Rules:
- Exactly 3 drivers, ordered by market impact.
- impacted_tickers: 3-5 real US-listed ticker symbols per driver. Standard symbols only (AAPL, NVDA, JPM, XLE, etc.). Never invent tickers.
- sources: 1-2 real article URLs per driver from your search results.
- watch_next_week: 4-6 catalysts for the next 5 business days. Real tickers, real dates.
- Output ONLY the JSON object. Nothing before or after."""


def _extract_json(text):
    """Pull the first valid JSON object out of model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        text = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find outermost { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


def synthesize(prices, gemini_key, brief_date_str):
    client = genai.Client(api_key=gemini_key)
    prices_block = "\n".join(
        f"- {p['display']}: {p['close']:.2f} ({p['pct']:+.2f}%)" for p in prices
    )
    prompt = PROMPT_TEMPLATE.format(
        date_str=brief_date_str, prices_block=prices_block,
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
    """Re-fetch every model-mentioned ticker against yfinance.
    Drop tickers that can't be verified."""
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


def build_pdf(out_path, brief_date_str, prices, drivers, watch):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Market brief — {brief_date_str}",
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
    story.append(Paragraph(f"{brief_date_str} — US close", h1))
    story.append(Paragraph("Post-close summary", muted))
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
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
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
                f'<a href="{s["url"]}" color="#888888">{s.get("title", s["url"])}</a>'
                for s in d["sources"][:2] if s.get("url")
            ]
            if links:
                block.append(Paragraph(f"Sources: {' · '.join(links)}", src))
        story.append(KeepTogether(block))
        story.append(Spacer(1, 12))

    # What to watch
    story.append(Spacer(1, 4))
    story.append(Paragraph("What to watch next week", h2))
    story.append(Paragraph("Catalysts on the radar", muted))
    wd = [[w["date"], w["event"]] for w in watch]
    wt = Table(wd, colWidths=[3.5*cm, 13*cm])
    wst = [
        ("FONT", (0,0), (-1,-1), "Helvetica", 9.5),
        ("TEXTCOLOR", (0,0), (0,-1), COL_MUTED),
        ("TEXTCOLOR", (1,0), (1,-1), COL_TEXT),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]
    for i in range(len(wd) - 1):
        wst.append(("LINEBELOW", (0,i), (-1,i), 0.25, COL_BORDER_LIGHT))
    wt.setStyle(TableStyle(wst))
    story.append(wt)

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

def send_email(pdf_path, recipient, gmail_user, gmail_pw, subject, brief_date_str):
    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = recipient
    msg["Subject"] = subject
    text = (
        f"Your daily market brief for {brief_date_str} is attached.\n\n"
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

    print("Fetching prices...")
    prices = fetch_prices()
    if len(prices) < 3:
        print("FATAL: fewer than 3 tickers fetched; aborting")
        sys.exit(1)

    brief_date = prices[0]["date"]
    brief_date_str = brief_date.strftime("%A, %B %d, %Y")
    print(f"  Brief date: {brief_date_str}, {len(prices)} tickers")

    print("Calling Gemini (synthesis + Google Search grounding)...")
    brief_data = synthesize(prices, gemini_key, brief_date_str)

    print("Verifying impacted tickers against yfinance...")
    drivers = verify_drivers(brief_data["drivers"])

    print("Building PDF...")
    pdf_path = f"/tmp/market-brief-{brief_date.isoformat()}.pdf"
    build_pdf(pdf_path, brief_date_str, prices, drivers,
              brief_data["watch_next_week"])

    print(f"Emailing to {recipient}...")
    subject = f"Market brief — {brief_date.strftime('%a %b %d, %Y')}"
    send_email(pdf_path, recipient, gmail_user, gmail_pw, subject,
               brief_date_str)
    print("Done.")


if __name__ == "__main__":
    main()
