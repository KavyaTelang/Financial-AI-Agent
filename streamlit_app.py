# streamlit_app.py (FINAL OPTIMIZED VERSION)

import os
import streamlit as st
from dotenv import load_dotenv
import traceback
import requests

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.toolkit import Toolkit
from alpha_vantage.fundamentaldata import FundamentalData

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY") or st.secrets.get("ALPHA_VANTAGE_API_KEY")

class AlphaVantageTools(Toolkit):
    def __init__(self, api_key: str | None = None, stock_news: bool = True, company_overview: bool = True):
        super().__init__(name="alpha_vantage_tools")
        self.api_key = api_key or alpha_vantage_api_key
        if not self.api_key: raise ValueError("Alpha Vantage API key not found.")
        if stock_news: self.register(self.get_stock_news_and_sentiment)
        if company_overview: self.register(self.get_company_overview)
    def get_stock_news_and_sentiment(self, ticker: str) -> str:
        try:
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=5&apikey={self.api_key}'
            r = requests.get(url, timeout=30); r.raise_for_status(); data = r.json()
            if "feed" not in data or not data["feed"]: return f"No news found for {ticker}."
            articles = [f"- **{a['title']}**\n  - Summary: {a['summary']}" for a in data["feed"]]
            full_output = "Latest News:\n" + "\n".join(articles)
            return full_output[:3500] + "..." if len(full_output) > 3500 else full_output
        except Exception as e: return f"Error getting news for {ticker}: {e}"
    def get_company_overview(self, ticker: str) -> str:
        try:
            fd = FundamentalData(key=self.api_key, output_format='pandas', treat_info_as_error=True, timeout=30)
            overview, _ = fd.get_company_overview(symbol=ticker)
            if overview.empty: return f"No company overview found for {ticker}."
            details = [f"**{key}**: {value[0]}" for key, value in overview.items()]
            full_output = f"Company Overview for {ticker}:\n" + "\n".join(details)
            return full_output[:3500] + "..." if len(full_output) > 3500 else full_output
        except Exception as e: return f"Error getting company overview for {ticker}: {e}"

@st.cache_resource
def get_financial_agent():
    # --- THIS IS THE FINAL CONFIGURATION ---
    # The optimal balance of power and token efficiency
    model_id = "qwen/qwen3-32b" 
    
    financial_agent = Agent(
        name="Financial AI Agent",
        role="You are a world-class financial analyst. Your goal is to help users by accessing financial data and web search.",
        model=Groq(id=model_id, api_key=groq_api_key),
        tools=[AlphaVantageTools(), DuckDuckGo()],
        instructions=[
            "First, understand the user's query.",
            "Then, choose the best tool to answer their question (either financial data or web search).",
            "Present the answer clearly to the user, using tables and formatting where appropriate.",
            "If the user's query is unclear, ask for clarification."
        ],
        markdown=True,
    )
    return financial_agent

# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")
if not groq_api_key or not alpha_vantage_api_key:
    st.error("API keys are not configured. Please add GROQ_API_KEY and ALPHA_VANTAGE_API_KEY to your Streamlit secrets.")
else:
    st.sidebar.markdown("### Built by Kavya Telang")
    st.sidebar.markdown("This assistant can search the web and access real-time financial data.")
    financial_agent = get_financial_agent()
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
                for chunk in financial_agent.run(prompt, stream=True):
                    if isinstance(chunk, dict) and "content" in chunk and chunk["content"] is not None:
                        full_response += chunk["content"]
                        placeholder.markdown(full_response + "▌")
                    elif isinstance(chunk, str):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = "Sorry, an error occurred. This can happen if an external data source is slow or if the daily token limit has been reached. Please try again in a few minutes."
                placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})