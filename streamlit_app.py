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
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
                f"Summary: {r.get('body', 'N/A')}"
            )
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Error performing web search: {e}"


# --- TOOL REGISTRY ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price and trading stats for a ticker symbol.",
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
            "description": "Get fundamental company data: sector, PE ratio, EBITDA, revenue, margins, and business description.",
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
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for real-time information. "
                "ALWAYS use this tool for: latest news about any company or stock, "
                "recent earnings results, market events, financial concept explanations, "
                "macroeconomic topics, IPOs, mergers, acquisitions, or anything knowledge-based. "
                "NEVER answer news or knowledge questions from memory — always search first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
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

SYSTEM_PROMPT = """You are a world-class financial analyst assistant with access to real-time tools.

CRITICAL RULES — you must follow these without exception:
1. NEVER answer from your own training knowledge. ALWAYS call a tool first.
2. For ANY question about news, recent events, or what's happening with a company → call web_search immediately.
3. For stock prices or trading data → call get_stock_price.
4. For fundamentals (PE ratio, revenue, EBITDA etc.) → call get_company_overview.
5. For analyst ratings → call get_analyst_recommendations.
6. For comparisons → call tools for each ticker one at a time.
7. You may chain multiple tool calls to build a complete answer.

FORBIDDEN responses (never say these):
- "You can find this information online..."
- "I recommend checking a news source..."
- "My knowledge only goes up to..."
- Any response that tells the user to search elsewhere.

You have the web_search tool — USE IT. The user expects you to find the answer, not redirect them.

After getting tool results, present findings in clean markdown with bullet points or tables, then give a brief analytical summary.
"""


def run_agent(user_query: str, history: list) -> str:
    client = Groq(api_key=groq_api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    for _ in range(8):
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            tools=TOOLS,
            tool_choice="required",  # Force the model to always call a tool on first pass
            max_tokens=2048,
        )

        msg = response.choices[0].message

        # If no tool calls returned, we have the final answer
        if not msg.tool_calls:
            return msg.content

        # Add assistant message with tool calls to history
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

        # Execute each tool and feed results back
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            result = TOOL_MAP[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # After first tool call round, switch to auto so it can finish naturally
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

        # If it wants more tool calls, add them and loop again
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

    return "I was unable to complete the analysis. Please try rephrasing your question."


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
