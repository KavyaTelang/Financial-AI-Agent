import streamlit as st
import yfinance as yf
import json
from groq import Groq

# --- API KEY SETUP ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

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


# --- STOCK AGENT ---
# Uses llama-3.3-70b-versatile which is more reliable with tool_choice=required
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
        # First call always requires a tool. Subsequent calls use auto.
        tool_choice = "required" if i == 0 else "auto"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=STOCK_TOOLS,
            tool_choice=tool_choice,
            max_tokens=2048,
        )
        msg = response.choices[0].message

        # No tool calls = final answer ready
        if not msg.tool_calls:
            return msg.content or "No response generated."

        # Append assistant tool call message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            result = TOOL_MAP.get(fn_name, lambda **k: "Unknown tool.")(** fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    return "Unable to complete analysis after multiple attempts. Please try again."


# --- WEB AGENT (Groq Compound — server-side web search, no libraries needed) ---
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


# --- QUERY CLASSIFIER ---
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


# --- STREAMLIT UI ---
st.set_page_config(page_title="StockSense", page_icon="📈")
st.title("Financial AI Agent")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY is not configured.")
    st.markdown("""
    **How to add your Groq API key on Streamlit Cloud:**
    1. Go to your app on [share.streamlit.io](https://share.streamlit.io)
    2. Click **Settings → Secrets**
    3. Add:
    ```toml
    GROQ_API_KEY = "your_groq_api_key_here"
    ```
    """)
    st.stop()

st.sidebar.markdown("### StockSense")
st.sidebar.markdown("Ask me anything — stock prices, fundamentals, news, or finance concepts.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Try asking:**")
st.sidebar.markdown("- Latest news on Nvidia")
st.sidebar.markdown("- Compare GOOGL and TSLA")
st.sidebar.markdown("- What is a PE ratio?")
st.sidebar.markdown("- How does inflation affect stocks?")
st.sidebar.markdown("- What do analysts think about Apple?")
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:** Groq · LLaMA 3.3 · Groq Compound · yFinance")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Financial AI Agent, StockSense! Ask me about stock prices, company fundamentals, analyst recommendations, market news, or any finance concept."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about stocks, markets, or finance concepts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing..."):
            try:
                response = run_agent(prompt, st.session_state.messages[:-1])
            except Exception as e:
                response = f"⚠️ An error occurred: `{str(e)}`\n\nPlease try again in a moment."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
