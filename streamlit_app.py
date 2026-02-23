import streamlit as st
import yfinance as yf
import json
from groq import Groq

# --- API KEY SETUP ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

# --- CUSTOM CSS: Bloomberg-style dark theme ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }

/* ── App background ── */
.stApp {
    background-color: #0a0a0a !important;
    background-image:
        linear-gradient(rgba(0, 255, 136, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 136, 0.02) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Header bar ── */
.stocksense-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 0 20px 0;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 24px;
}
.stocksense-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #00ff88;
    letter-spacing: -1px;
}
.stocksense-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 3px 8px;
    border: 1px solid #222;
    border-radius: 2px;
}
.live-dot {
    width: 7px;
    height: 7px;
    background: #00ff88;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,136,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(0,255,136,0); }
}

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    background-color: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 4px !important;
    padding: 14px 18px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    color: #d0d0d0 !important;
}
/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    border-left: 3px solid #00ff88 !important;
    background-color: #0d1a13 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e0e0e0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #00ff88 !important;
    box-shadow: 0 0 0 1px rgba(0,255,136,0.2) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #444 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #1a1a1a !important;
}
[data-testid="stSidebar"] * {
    color: #c0c0c0 !important;
}

/* ── Metric cards ── */
.metric-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #555 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: #00ff88 !important;
}
.metric-sub {
    font-size: 0.7rem;
    color: #444 !important;
    margin-top: 2px;
}

/* ── Ticker strip ── */
.ticker-strip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    border-top: 1px solid #1a1a1a;
    padding-top: 8px;
    margin-top: 8px;
    letter-spacing: 0.5px;
}
.ticker-up { color: #00ff88 !important; }
.ticker-down { color: #ff4466 !important; }

/* ── Sidebar section headers ── */
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #333 !important;
    margin: 16px 0 8px 0;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 4px;
}

/* ── Suggestion chips ── */
.chip {
    display: inline-block;
    font-size: 0.72rem;
    color: #666 !important;
    border: 1px solid #222;
    border-radius: 2px;
    padding: 3px 8px;
    margin: 3px 2px;
    cursor: pointer;
    font-family: 'IBM Plex Mono', monospace;
    transition: all 0.15s;
}

/* ── Markdown tables ── */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
th {
    background: #161616 !important;
    color: #00ff88 !important;
    border: 1px solid #222 !important;
    padding: 8px 12px !important;
    text-align: left !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
}
td {
    border: 1px solid #1e1e1e !important;
    padding: 7px 12px !important;
    color: #c0c0c0 !important;
}
tr:hover td { background: #141414 !important; }

/* ── Spinners & status ── */
.stSpinner > div { border-top-color: #00ff88 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #333; }
</style>
"""

# --- YFINANCE TOOL IMPLEMENTATIONS ---

def get_stock_price(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker.upper()).info
        price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap:,.0f}"
        volume = info.get("volume", "N/A")
        volume_str = f"{volume:,}" if isinstance(volume, int) else str(volume)
        return (
            f"Stock data for {info.get('longName', ticker)} ({ticker.upper()}):\n"
            f"Current Price: ${price}\n"
            f"Previous Close: ${info.get('previousClose', 'N/A')}\n"
            f"Day High: ${info.get('dayHigh', 'N/A')}\n"
            f"Day Low: ${info.get('dayLow', 'N/A')}\n"
            f"52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"Market Cap: {market_cap}\n"
            f"Volume: {volume_str}\n"
        )
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {e}"


def get_company_overview(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker.upper()).info
        description = info.get("longBusinessSummary", "N/A")
        if description and len(description) > 400:
            description = description[:400] + "..."
        return (
            f"Company overview for {info.get('longName', ticker)} ({ticker.upper()}):\n"
            f"Sector: {info.get('sector', 'N/A')}\n"
            f"Industry: {info.get('industry', 'N/A')}\n"
            f"PE Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"EPS: {info.get('trailingEps', 'N/A')}\n"
            f"EBITDA: {info.get('ebitda', 'N/A')}\n"
            f"Total Revenue: {info.get('totalRevenue', 'N/A')}\n"
            f"Profit Margin: {info.get('profitMargins', 'N/A')}\n"
            f"Return on Equity: {info.get('returnOnEquity', 'N/A')}\n"
            f"Debt to Equity: {info.get('debtToEquity', 'N/A')}\n"
            f"Description: {description}\n"
        )
    except Exception as e:
        return f"Error fetching company overview for {ticker}: {e}"


def get_analyst_recommendations(ticker: str) -> str:
    try:
        recs = yf.Ticker(ticker.upper()).recommendations
        if recs is None or recs.empty:
            return f"No analyst recommendations found for {ticker.upper()}."
        return f"Recent analyst recommendations for {ticker.upper()}:\n{recs.tail(5).to_string()}"
    except Exception as e:
        return f"Error fetching recommendations for {ticker}: {e}"


def get_market_snapshot() -> dict:
    """Fetch quick stats for the sidebar dashboard."""
    tickers = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "VIX": "^VIX"}
    data = {}
    for name, symbol in tickers.items():
        try:
            info = yf.Ticker(symbol).info
            price = info.get("regularMarketPrice") or info.get("previousClose", 0)
            prev = info.get("previousClose", price)
            change_pct = ((price - prev) / prev * 100) if prev else 0
            data[name] = {"price": price, "change": change_pct}
        except:
            data[name] = {"price": "N/A", "change": 0}
    return data


TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_company_overview": get_company_overview,
    "get_analyst_recommendations": get_analyst_recommendations,
}

STOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price and trading stats for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": "Get fundamental data: sector, PE ratio, EBITDA, revenue, margins.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_recommendations",
            "description": "Get analyst buy/sell/hold recommendations for a stock.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
]

STOCK_SYSTEM_PROMPT = """You are a financial analyst with tools to fetch live stock data.

RULES:
- You MUST call tools to get data. Never write answers from memory.
- Call tools ONE AT A TIME. Do not batch multiple tickers in one call.
- For a comparison (e.g. GOOGL vs TSLA), call get_stock_price(GOOGL), then get_stock_price(TSLA), then get_company_overview(GOOGL), then get_company_overview(TSLA), one by one.
- Only write your final answer AFTER you have received all tool results.
- Present data in markdown tables. End with a brief summary.
"""

def run_stock_agent(client: Groq, query: str, history: list) -> str:
    messages = [{"role": "system", "content": STOCK_SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    for i in range(10):
        tool_choice = "required" if i == 0 else "auto"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=STOCK_TOOLS,
            tool_choice=tool_choice,
            max_tokens=2048,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content or "No response generated."
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        })
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            result = TOOL_MAP.get(fn_name, lambda **k: "Unknown tool.")(**fn_args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Unable to complete analysis after multiple attempts. Please try again."


WEB_SYSTEM_PROMPT = """You are a world-class financial analyst with real-time web search.
Always search for current information before answering.
Present findings clearly in bullet points or tables with a brief summary at the end.
"""

def run_web_agent(client: Groq, query: str, history: list) -> str:
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="groq/compound",
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content


CLASSIFIER_PROMPT = """Classify the user's financial question into ONE category:

"stock" — questions about specific stock prices, PE ratios, revenue, EBITDA, analyst ratings, 
or comparisons between specific tickers (e.g. "Apple stock price", "Compare GOOGL and TSLA", "NVDA fundamentals")

"web" — questions about news, recent events, earnings announcements, financial concepts, 
macroeconomics, or general market trends (e.g. "Latest news on Nvidia", "What is a PE ratio?", "How does inflation affect stocks?")

Reply with ONLY one word: stock or web"""

def classify_query(client: Groq, query: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": query}
        ],
        max_tokens=5,
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()
    return "stock" if "stock" in result else "web"


def run_agent(query: str, history: list) -> str:
    client = Groq(api_key=groq_api_key)
    query_type = classify_query(client, query)
    if query_type == "stock":
        return run_stock_agent(client, query, history)
    else:
        return run_web_agent(client, query, history)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY is not configured. Go to Settings → Secrets and add your key.")
    st.stop()

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600; color:#00ff88; letter-spacing:-0.5px;">
            ◈ STOCKSENSE
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#333; letter-spacing:2px; margin-top:2px;">
            ALL-IN-ONE FINANCE ADVISOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Market snapshot
    st.markdown('<div class="sidebar-section">Market Snapshot</div>', unsafe_allow_html=True)

    with st.spinner(""):
        snapshot = get_market_snapshot()

    for name, vals in snapshot.items():
        price = vals["price"]
        change = vals["change"]
        if isinstance(price, float):
            price_str = f"{price:,.2f}"
        else:
            price_str = str(price)
        color = "#00ff88" if change >= 0 else "#ff4466"
        arrow = "▲" if change >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-value">{price_str}</div>
            <div class="metric-sub" style="color:{color} !important;">{arrow} {abs(change):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Try Asking</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stSidebar"] .stButton > button {
        background: #111 !important;
        border: 1px solid #222 !important;
        border-radius: 2px !important;
        color: #555 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.68rem !important;
        padding: 4px 10px !important;
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 2px !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #00ff88 !important;
        color: #00ff88 !important;
        background: #0d1a13 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    suggestions = [
        "Latest news on Nvidia",
        "Compare GOOGL and TSLA",
        "What is a PE ratio?",
        "Apple fundamentals",
        "How does inflation affect stocks?",
        "Analyst rating for Amazon",
    ]
    for s in suggestions:
        if st.button(f"› {s}", key=f"chip_{s}"):
            st.session_state["prefill"] = s
            st.rerun()

    st.markdown('<div class="sidebar-section">Powered By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#333; line-height:2;">
        GROQ · LLAMA 3.3<br>
        GROQ COMPOUND<br>
        YFINANCE
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ──
st.markdown("""
<div class="stocksense-header">
    <div>
        <span class="live-dot"></span>
    </div>
    <div class="stocksense-logo">StockSense</div>
    <div class="stocksense-tag">Live Market Intelligence</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = None

# Show example cards when chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div style="margin: 8px 0 28px 0;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#333; letter-spacing:2px; margin-bottom:16px;">
            WHAT CAN I HELP YOU WITH?
        </div>
    </div>
    """, unsafe_allow_html=True)

    examples = [
        ("📰", "Market News", "Latest news on Nvidia", "Real-time headlines & analysis"),
        ("📊", "Stock Data", "Compare GOOGL and TSLA", "Live prices & fundamentals"),
        ("🎓", "Finance 101", "What is a PE ratio?", "Concepts explained simply"),
        ("📈", "Analyst Ratings", "Analyst rating for Amazon", "Buy / Hold / Sell signals"),
        ("🏢", "Company Deep Dive", "Apple fundamentals", "Revenue, margins, EBITDA"),
        ("🌍", "Macro Markets", "How does inflation affect stocks?", "Big picture market forces"),
    ]

    cols = st.columns(3)
    for i, (icon, title, query, desc) in enumerate(examples):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background:#111;
                border:1px solid #1e1e1e;
                border-radius:4px;
                padding:14px 16px;
                margin-bottom:4px;
            ">
                <div style="font-size:1.1rem; margin-bottom:6px;">{icon}</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#00ff88; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">{title}</div>
                <div style="font-size:0.8rem; color:#aaa; margin-bottom:6px;">"{query}"</div>
                <div style="font-size:0.7rem; color:#444;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try →", key=f"ex_{i}"):
                st.session_state["prefill"] = query
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle prefilled prompt from sidebar or example cards
prefilled = st.session_state.pop("prefill", None)
user_input = st.chat_input("Enter ticker, ask about markets, or any finance question...")
prompt = prefilled or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing markets..."):
            try:
                response = run_agent(prompt, st.session_state.messages[:-1])
            except Exception as e:
                response = f"⚠️ Error: `{str(e)}`\n\nPlease try again in a moment."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
