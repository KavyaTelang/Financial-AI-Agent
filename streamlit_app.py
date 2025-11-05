# streamlit_app.py

import os
import streamlit as st
from dotenv import load_dotenv
import traceback

# Phidata imports
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.toolkit import Toolkit

# Alpha Vantage library import
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.newsandsentiment import NewsAndSentiment

# Load environment variables
load_dotenv()

# --- THIS IS THE CUSTOM TOOL DEFINITION ---
# We are defining the tool ourselves to fix the ModuleNotFoundError

class AlphaVantageTools(Toolkit):
    def __init__(
        self,
        api_key: str | None = None,
        stock_news: bool = True,
        company_overview: bool = True,
    ):
        super().__init__(name="alpha_vantage_tools")
        
        # Get the API key from the environment if not provided
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        
        if stock_news:
            self.register(self.get_stock_news)
        if company_overview:
            self.register(self.get_company_overview)

    def get_stock_news(self, ticker: str) -> str:
        """
        Get the latest news for a stock ticker from Alpha Vantage.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'NVDA').

        Returns:
            str: A formatted string of the latest news articles or an error message.
        """
        try:
            ns = NewsAndSentiment(key=self.api_key)
            news_data, _ = ns.get_news_sentiment(tickers=ticker, limit=5)
            if not news_data or "feed" not in news_data or not news_data["feed"]:
                return f"No news found for {ticker}."
            
            articles = []
            for article in news_data["feed"]:
                articles.append(
                    f"- **{article['title']}**\n  - Source: {article['source']}\n  - Summary: {article['summary']}\n  - URL: {article['url']}"
                )
            return "Latest News:\n" + "\n".join(articles)
        except Exception as e:
            return f"Error getting news for {ticker}: {e}"

    def get_company_overview(self, ticker: str) -> str:
        """
        Get the company overview and fundamentals for a stock ticker.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'NVDA').

        Returns:
            str: A formatted string of the company overview or an error message.
        """
        try:
            fd = FundamentalData(key=self.api_key)
            overview, _ = fd.get_company_overview(symbol=ticker)
            if not overview:
                return f"No company overview found for {ticker}."
            
            # Format the overview into a readable string
            details = [f"**{key}**: {value}" for key, value in overview.items()]
            return f"Company Overview for {ticker}:\n" + "\n".join(details)
        except Exception as e:
            return f"Error getting company overview for {ticker}: {e}"


@st.cache_resource
def get_multi_ai_agent():
    """This function creates and returns the multi-agent assistant."""
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for information.",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[DuckDuckGo()],
        markdown=True,
    )
    
    finance_agent = Agent(
        name="Finance AI Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        role="You are a world-class financial analyst.",
        tools=[AlphaVantageTools()], # Use our custom-defined tool
        instructions=["Use tables to display data"],
        markdown=True,
    )
    
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "Always include sources", 
            "Use tables to display data",
            "When delegating tasks, provide clear details in the 'additional_information' field for the specialist agent."
        ],
        markdown=True,
    )
    return multi_ai_agent

# --- STREAMLIT UI (No changes from here down) ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")
st.sidebar.markdown("### Built by Kavya Telang")
st.sidebar.markdown("This multi-agent assistant can search the web and access real-time financial data.")

multi_ai_agent = get_multi_ai_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your financial research today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about stocks, news, and more..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in multi_ai_agent.run(prompt, stream=True):
                if isinstance(chunk, dict) and "content" in chunk and chunk["content"] is not None:
                    full_response += chunk["content"]
                    placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, str):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, dict) and "tool_name" in chunk:
                    tool_name = chunk["tool_name"]
                    if tool_name == "transfer_task_to_finance_ai_agent":
                        placeholder.markdown("🔍 Accessing financial data...")
                    elif tool_name == "transfer_task_to_web_search_agent":
                        placeholder.markdown("🌐 Searching the web...")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = "Sorry, an error occurred. This can happen if the external data source is slow or unavailable. Please try your request again in a moment."
            placeholder.markdown(full_response)
            traceback.print_exc()
    st.session_state.messages.append({"role": "assistant", "content": full_response})