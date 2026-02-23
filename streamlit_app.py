import streamlit as st
import yfinance as yf
import json
from groq import Groq
from duckduckgo_search import DDGS

# --- API KEY SETUP ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

# --- TOOL IMPLEMENTATIONS ---

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


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title', 'N/A')}\nURL: {r.get('href', 'N/A')}\nSummary: {r.get('body', 'N/A')}\n")
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Error performing web search: {e}"


# --- TOOL REGISTRY ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price and trading stats (day high/low, 52-week range, market cap, volume) for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol e.g. AAPL, TSLA, GOOGL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": "Get fundamental company data: sector, industry, PE ratio, EPS, EBITDA, revenue, profit margin, ROE, debt/equity, and business description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol e.g. AAPL, TSLA, GOOGL"}
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
                    "ticker": {"type": "string", "description": "Stock ticker symbol e.g. AAPL, TSLA, GOOGL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for any information. Use this for: general finance knowledge, "
                "recent news about a company or market, explanations of financial concepts "
                "(e.g. what is a PE ratio, how does inflation affect stocks), macroeconomic topics, "
                "earnings reports, or anything not covered by the stock data tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to look up"}
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_company_overview": get_company_overview,
    "get_analyst_recommendations": get_analyst_recommendations,
    "web_search": web_search,
}

SYSTEM_PROMPT = """You are a world-class financial analyst assistant with access to real-time stock data and web search.

Tool usage guidelines:
- For stock prices, trading stats → use get_stock_price
- For fundamentals (PE ratio, revenue, margins) → use get_company_overview
- For analyst sentiment → use get_analyst_recommendations
- For recent news, earnings, market events, or any knowledge/concept questions → use web_search
- For comparisons between stocks, call tools for each ticker one at a time
- You can use multiple tools in sequence to build a complete answer

Response guidelines:
- Always use tools to get current data before answering — never rely on your training knowledge for stock prices or recent events
- Present financial data in clean markdown tables or bullet points
- For knowledge questions (e.g. "what is EBITDA"), use web_search to give an accurate, sourced answer
- Always finish with a concise summary
"""


def run_agent(user_query: str, history: list) -> str:
    client = Groq(api_key=groq_api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    for _ in range(8):  # max 8 tool call iterations
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in msg.tool_calls
            ]
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_MAP[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "I was unable to complete the analysis after multiple attempts. Please try rephrasing your question."


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
st.sidebar.markdown("**Powered by:** Groq · Llama 4 Scout · yFinance · DuckDuckGo")
st.sidebar.markdown("---")
st.sidebar.markdown("**Try asking:**")
st.sidebar.markdown("- Compare GOOGL and TSLA")
st.sidebar.markdown("- What is a PE ratio?")
st.sidebar.markdown("- Latest news on Nvidia")
st.sidebar.markdown("- How does inflation affect the stock market?")
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
        with st.spinner("Analyzing..."):
            try:
                response = run_agent(prompt, st.session_state.messages[:-1])
            except Exception as e:
                response = f"⚠️ An error occurred: `{str(e)}`\n\nPlease try again in a moment."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
