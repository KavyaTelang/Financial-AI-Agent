import streamlit as st
import yfinance as yf

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.toolkit import Toolkit
from phi.run.response import RunResponse

# --- API KEY SETUP ---
groq_api_key = st.secrets.get("GROQ_API_KEY")

# --- YFINANCE TOOL (no API key needed, completely free) ---
class YFinanceTools(Toolkit):
    def __init__(self):
        super().__init__(name="yfinance_tools")
        self.register(self.get_stock_price)
        self.register(self.get_company_overview)
        self.register(self.get_analyst_recommendations)

    def get_stock_price(self, ticker: str) -> str:
        """Gets the current stock price and basic stats for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
            return (
                f"**{info.get('longName', ticker)} ({ticker})**\n"
                f"- Current Price: ${price}\n"
                f"- Previous Close: ${info.get('previousClose', 'N/A')}\n"
                f"- Day High: ${info.get('dayHigh', 'N/A')}\n"
                f"- Day Low: ${info.get('dayLow', 'N/A')}\n"
                f"- 52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
                f"- 52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
                f"- Market Cap: ${info.get('marketCap', 'N/A'):,}\n" if isinstance(info.get('marketCap'), int) else
                f"- Market Cap: {info.get('marketCap', 'N/A')}\n"
            )
        except Exception as e:
            return f"Error getting stock price for {ticker}: {e}"

    def get_company_overview(self, ticker: str) -> str:
        """Gets fundamental company data including PE ratio, EBITDA, and description."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return (
                f"**Company Overview: {info.get('longName', ticker)}**\n"
                f"- Sector: {info.get('sector', 'N/A')}\n"
                f"- Industry: {info.get('industry', 'N/A')}\n"
                f"- PE Ratio: {info.get('trailingPE', 'N/A')}\n"
                f"- EPS: {info.get('trailingEps', 'N/A')}\n"
                f"- EBITDA: {info.get('ebitda', 'N/A')}\n"
                f"- Revenue: {info.get('totalRevenue', 'N/A')}\n"
                f"- Profit Margin: {info.get('profitMargins', 'N/A')}\n"
                f"- Description: {info.get('longBusinessSummary', 'N/A')[:300]}...\n"
            )
        except Exception as e:
            return f"Error getting company overview for {ticker}: {e}"

    def get_analyst_recommendations(self, ticker: str) -> str:
        """Gets the latest analyst recommendations for a stock."""
        try:
            stock = yf.Ticker(ticker)
            recs = stock.recommendations
            if recs is None or recs.empty:
                return f"No analyst recommendations found for {ticker}."
            recent = recs.tail(5).to_string()
            return f"Recent Analyst Recommendations for {ticker}:\n{recent}"
        except Exception as e:
            return f"Error getting recommendations for {ticker}: {e}"


# --- AGENT CREATION ---
@st.cache_resource
def get_financial_agent():
    return Agent(
        name="Financial Analyst",
        role="You are a world-class financial analyst. You help users understand stocks, company fundamentals, and market news.",
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
        tools=[YFinanceTools(), DuckDuckGo()],
        instructions=[
            "When a user asks about a stock or company, use get_stock_price and get_company_overview together to give a full picture.",
            "Use get_analyst_recommendations when the user asks about analyst sentiment or buy/sell ratings.",
            "Use DuckDuckGo to find recent news, earnings updates, or any context not available via yfinance.",
            "Always present data in a clean, readable format using markdown tables or bullet points.",
            "If you are unsure of a ticker symbol, search for it using DuckDuckGo before calling yfinance tools.",
            "Always include a brief summary of your findings at the end.",
        ],
        markdown=True,
    )


# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY is not configured. Please add it in your Streamlit Cloud secrets.")
    st.markdown("""
    **How to add your secret:**
    1. Go to your app on [share.streamlit.io](https://share.streamlit.io)
    2. Click **Settings → Secrets**
    3. Add the following:
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
st.sidebar.markdown("- Give me an overview of Tesla's fundamentals")
st.sidebar.markdown("- What do analysts think about NVDA?")
st.sidebar.markdown("- Any recent news on Microsoft?")

financial_agent = get_financial_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Financial AI Agent 📈 Ask me about any stock — prices, company fundamentals, analyst recommendations, or the latest news."}
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
            final_output_from_run_response = ""
            for chunk in financial_agent.run(prompt, stream=True):
                if isinstance(chunk, str):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, RunResponse):
                    if chunk.output:
                        final_output_from_run_response = chunk.output

            if final_output_from_run_response:
                full_response = final_output_from_run_response

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ An error occurred: `{str(e)}`\n\nThis may be a Groq rate limit. Please wait a moment and try again."
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
