# streamlit_app.py (FINAL OPTIMIZED VERSION)

import os
import streamlit as st
import traceback
import requests

# Phidata imports
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.toolkit import Toolkit

# Alpha Vantage library import
from alpha_vantage.fundamentaldata import FundamentalData

# --- API KEY SETUP ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
alpha_vantage_api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY")

# --- OPTIMIZED TOOL DEFINITION ---
class AlphaVantageTools(Toolkit):
    def __init__(self, api_key: str | None = None):
        super().__init__(name="alpha_vantage_tools")
        self.api_key = api_key or alpha_vantage_api_key
        if not self.api_key: raise ValueError("Alpha Vantage API key not found.")
        self.register(self.get_stock_news)
        self.register(self.get_company_overview)

    def get_stock_news(self, ticker: str) -> str:
        """Gets the 3 latest news article summaries for a stock ticker."""
        try:
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=3&apikey={self.api_key}'
            r = requests.get(url, timeout=20); r.raise_for_status(); data = r.json()
            if "feed" not in data or not data["feed"]: return f"No news found for {ticker}."
            articles = [f"- {a['title']}: {a['summary']}" for a in data["feed"]]
            return "Latest News:\n" + "\n".join(articles)
        except Exception as e: return f"Error getting news: {e}"

    def get_company_overview(self, ticker: str) -> str:
        """Gets key financial metrics for a company from its overview."""
        try:
            fd = FundamentalData(key=self.api_key, output_format='pandas', treat_info_as_error=True, timeout=20)
            overview, _ = fd.get_company_overview(symbol=ticker)
            if overview.empty: return f"No company overview found for {ticker}."
            
            # --- THE FIX: Select only the most important fields to save tokens ---
            key_metrics = {
                "Name": overview.get("Name", ["N/A"])[0],
                "Description": overview.get("Description", ["N/A"])[0][:200] + "...", # Truncate description
                "MarketCapitalization": overview.get("MarketCapitalization", ["N/A"])[0],
                "EBITDA": overview.get("EBITDA", ["N/A"])[0],
                "PERatio": overview.get("PERatio", ["N/A"])[0],
                "52WeekHigh": overview.get("52WeekHigh", ["N/A"])[0],
                "52WeekLow": overview.get("52WeekLow", ["N/A"])[0],
            }
            details = [f"**{key}**: {value}" for key, value in key_metrics.items()]
            return f"Key Metrics for {ticker}:\n" + "\n".join(details)
        except Exception as e: return f"Error getting company overview: {e}"

# --- OPTIMIZED AGENT CREATION ---
@st.cache_resource
def get_financial_agent():
    """Creates a single, token-efficient financial agent."""
    return Agent(
        name="Financial Analyst",
        role="You are a helpful financial analyst. You have access to stock news, company overviews, and web search.",
        model=Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key),
        tools=[AlphaVantageTools(), DuckDuckGo()],
        instructions=[
            "If the user asks a vague question (like just a ticker), ask for clarification.",
            "Choose the best tool for the user's specific question.",
            "Present the final answer clearly and concisely."
        ],
        markdown=True,
    )

# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")

if not groq_api_key or not alpha_vantage_api_key:
    st.error("API keys are not configured. Please add GROQ_API_KEY and ALPHA_VANTAGE_API_KEY to your Streamlit secrets.")
else:
    st.sidebar.markdown("### Built by Kavya Telang")
    st.sidebar.markdown("This assistant can access real-time financial data and search the web.")
    
    financial_agent = get_financial_agent()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask about stock news, company metrics, etc..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in financial_agent.run(prompt, stream=True):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = "Sorry, an error occurred. The daily token limit may have been reached or the external data source is unavailable. Please try again in a few minutes."
                full_response += f"\n\n**Debug Info:**\n```\n{traceback.format_exc()}\n```"
                placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})