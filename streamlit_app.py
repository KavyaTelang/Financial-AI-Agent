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


# --- TOOL REGISTRY (for stock data agent) ---
STOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price and trading stats for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL, TSLA, GOOGL, NVDA"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": "Get fundamental company data: sector, PE ratio, EBITDA, revenue, margins.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL, TSLA, GOOGL, NVDA"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_recommendations",
            "description": "Get the latest analyst buy/sell/hold recommendations for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL, TSLA, GOOGL, NVDA"}
                },
                "required": ["ticker"]
            }
        }
    },
]

TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_company_overview": get_company_overview,
    "get_analyst_recommendations": get_analyst_recommendations,
}

# --- QUERY CLASSIFIER ---
# Decide whether the question needs live stock data or web search / general knowledge
CLASSIFIER_PROMPT = """You are a query router for a financial assistant. 
Classify the user's question into ONE of two categories:

- "stock": questions about specific stock prices, company fundamentals, analyst ratings, 
  stock comparisons (e.g. "What is Apple's stock price?", "Compare GOOGL and TSLA", "NVDA PE ratio")
  
- "web": questions about news, recent events, financial concepts, macroeconomics, 
  market trends, earnings reports, or anything general 
  (e.g. "Latest news on Nvidia", "What is a PE ratio?", "How does inflation affect stocks?")

Reply with ONLY the single word: stock or web"""


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


# --- STOCK DATA AGENT ---
STOCK_SYSTEM_PROMPT = """You are a world-class financial analyst. 
You have tools to fetch live stock prices, company fundamentals, and analyst recommendations.
Always call the appropriate tools to get live data — never answer from memory.
For comparisons, call tools for each ticker one at a time.
Present results in clean markdown tables or bullet points, then give a brief analytical summary.
"""

def run_stock_agent(client: Groq, query: str, history: list) -> str:
    messages = [{"role": "system", "content": STOCK_SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    for _ in range(8):
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            tools=STOCK_TOOLS,
            tool_choice="required",
            max_tokens=2048,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

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
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_MAP[fn_name](**fn_args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # After tools are executed, get the final answer with auto mode
        follow_up = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            tools=STOCK_TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )
        follow_msg = follow_up.choices[0].message
        if not follow_msg.tool_calls:
            return follow_msg.content

        # If it wants more tools, add them and loop
        messages.append({
            "role": "assistant",
            "content": follow_msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in follow_msg.tool_calls
            ]
        })
        for tc in follow_msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_MAP[fn_name](**fn_args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Unable to complete analysis. Please try again."


# --- WEB SEARCH AGENT (Groq Compound — web search is built-in, no libraries needed) ---
WEB_SYSTEM_PROMPT = """You are a world-class financial analyst assistant with access to real-time web search.
Always search the web to find current, accurate information before answering.
Present findings clearly with bullet points or tables.
Always end with a concise analytical summary.
"""

def run_web_agent(client: Groq, query: str, history: list) -> str:
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    # groq/compound has web search built in server-side — no external libraries needed
    response = client.chat.completions.create(
        model="groq/compound",
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# --- MAIN AGENT ROUTER ---
def run_agent(query: str, history: list) -> str:
    client = Groq(api_key=groq_api_key)
    query_type = classify_query(client, query)

    if query_type == "stock":
        return run_stock_agent(client, query, history)
    else:
        return run_web_agent(client, query, history)


# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")

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

st.sidebar.markdown("### 📊 Financial AI Agent")
st.sidebar.markdown("Ask me anything — stock prices, fundamentals, news, or finance concepts.")
st.sidebar.markdown("**Powered by:** Groq · Llama 4 Scout · Groq Compound · yFinance")
st.sidebar.markdown("---")
st.sidebar.markdown("**Try asking:**")
st.sidebar.markdown("- Latest news on Nvidia")
st.sidebar.markdown("- Compare GOOGL and TSLA")
st.sidebar.markdown("- What is a PE ratio?")
st.sidebar.markdown("- How does inflation affect stocks?")
st.sidebar.markdown("- What do analysts think about Apple?")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Financial AI Agent 📈 Ask me about stock prices, company fundamentals, analyst recommendations, market news, or any finance concept."}
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
