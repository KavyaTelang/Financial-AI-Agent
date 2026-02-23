import streamlit as st
import yfinance as yf

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.run.response import RunResponse

# --- API KEY SETUP ---
groq_api_key = st.secrets.get("GROQ_API_KEY")


# --- STANDALONE TOOL FUNCTIONS (more reliable with phidata 2.7.7) ---

def get_stock_price(ticker: str) -> str:
    """Gets the current stock price and basic trading stats for a given ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap:,.0f}"
        volume = info.get("volume", "N/A")
        volume_str = f"{volume:,}" if isinstance(volume, int) else str(volume)
        return (
            f"**{info.get('longName', ticker)} ({ticker.upper()})**\n"
            f"- Current Price: ${price}\n"
            f"- Previous Close: ${info.get('previousClose', 'N/A')}\n"
            f"- Day High: ${info.get('dayHigh', 'N/A')}\n"
            f"- Day Low: ${info.get('dayLow', 'N/A')}\n"
            f"- 52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"- 52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"- Market Cap: {market_cap}\n"
            f"- Volume: {volume_str}\n"
        )
    except Exception as e:
        return f"Error getting stock price for {ticker}: {e}"


def get_company_overview(ticker: str) -> str:
    """Gets fundamental company data including sector, PE ratio, EBITDA, revenue, and a business description."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        description = info.get("longBusinessSummary", "N/A")
        if description and len(description) > 300:
            description = description[:300] + "..."
        return (
            f"**Company Overview: {info.get('longName', ticker)} ({ticker.upper()})**\n"
            f"- Sector: {info.get('sector', 'N/A')}\n"
            f"- Industry: {info.get('industry', 'N/A')}\n"
            f"- PE Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"- EPS: {info.get('trailingEps', 'N/A')}\n"
            f"- EBITDA: {info.get('ebitda', 'N/A')}\n"
            f"- Total Revenue: {info.get('totalRevenue', 'N/A')}\n"
            f"- Profit Margin: {info.get('profitMargins', 'N/A')}\n"
            f"- Return on Equity: {info.get('returnOnEquity', 'N/A')}\n"
            f"- Debt to Equity: {info.get('debtToEquity', 'N/A')}\n"
            f"- Description: {description}\n"
        )
    except Exception as e:
        return f"Error getting company overview for {ticker}: {e}"


def get_analyst_recommendations(ticker: str) -> str:
    """Gets the latest analyst buy/sell/hold recommendations for a stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        recs = stock.recommendations
        if recs is None or recs.empty:
            return f"No analyst recommendations found for {ticker}."
        recent = recs.tail(5).to_string()
        return f"Recent Analyst Recommendations for {ticker.upper()}:\n{recent}"
    except Exception as e:
        return f"Error getting recommendations for {ticker}: {e}"


# --- AGENT CREATION ---
@st.cache_resource
def get_financial_agent():
    return Agent(
        name="Financial Analyst",
        role="You are a world-class financial analyst. You help users understand stocks, company fundamentals, and market news.",
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
        tools=[get_stock_price, get_company_overview, get_analyst_recommendations, DuckDuckGo()],
        instructions=[
            "When comparing two stocks, call get_stock_price and get_company_overview for EACH ticker separately.",
            "When asked about a single stock, call get_stock_price and get_company_overview together for a full picture.",
            "Use get_analyst_recommendations when the user asks about analyst sentiment or buy/sell ratings.",
            "Use DuckDuckGo to find recent news, earnings updates, or context not available via yfinance tools.",
            "Always present data clearly using markdown tables or bullet points.",
            "If unsure of a ticker symbol, search for it using DuckDuckGo before calling other tools.",
            "Always end with a brief summary of your findings.",
        ],
        markdown=True,
    )


# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY is not configured.")
    st.markdown("""
    **How to add your secret on Streamlit Cloud:**
    1. Go to your app on [share.streamlit.io](https://share.streamlit.io)
    2. Click **Settings → Secrets**
    3. Add:
    ```toml
    GROQ_API_KEY = "your_groq_api_key_here"
    ```
    """)
    st.stop()

st.sidebar.markdown("### 📊 Financial AI Agent")
st.sidebar.markdown("Ask me about any stock — prices, fundamentals, analyst ratings, or news.")
st.sidebar.markdown("**Powered by:** Groq (LLaMA 3.3), yFinance, DuckDuckGo")
st.sidebar.markdown("---")
st.sidebar.markdown("**Example questions:**")
st.sidebar.markdown("- What is Apple's current stock price?")
st.sidebar.markdown("- Compare GOOGL and TSLA")
st.sidebar.markdown("- Give me Tesla's fundamentals")
st.sidebar.markdown("- What do analysts think about NVDA?")
st.sidebar.markdown("- Any recent news on Microsoft?")

financial_agent = get_financial_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Financial AI Agent 📈 Ask me about any stock — prices, company fundamentals, analyst recommendations, or comparisons."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a stock, company, or market trend..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            final_output = ""
            for chunk in financial_agent.run(prompt, stream=True):
                if isinstance(chunk, str):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, RunResponse):
                    if chunk.output:
                        final_output = chunk.output

            if final_output:
                full_response = final_output

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ An error occurred: `{str(e)}`\n\nIf this is a rate limit, please wait a moment and try again."
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
