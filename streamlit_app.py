# streamlit_app.py (FINAL POLISHED VERSION)

import os
import streamlit as st
from dotenv import load_dotenv
import traceback
import requests

# Phidata imports
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.toolkit import Toolkit

# Alpha Vantage library import
from alpha_vantage.fundamentaldata import FundamentalData

# Load environment variables
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
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
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
def get_multi_ai_agent():
    model_id = "qwen/qwen3-32b" # Stick with this optimal model
    
    web_search_agent = Agent(name="Web Search Agent", role="Search the web for information.", model=Groq(id=model_id, api_key=groq_api_key), tools=[DuckDuckGo()], markdown=True)
    
    finance_agent = Agent(name="Finance AI Agent", model=Groq(id=model_id, api_key=groq_api_key), role="You are a world-class financial analyst.", tools=[AlphaVantageTools()], instructions=["Use tables to display data"], markdown=True)
    
    # --- THIS IS THE AGENT WITH THE FINAL FIX ---
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        model=Groq(id=model_id, api_key=groq_api_key),
        instructions=[
            "Always include sources", 
            "Use tables to display data",
            "When delegating tasks, provide clear details in the 'additional_information' field for the specialist agent.",
            # --- THIS NEW LINE IS THE FIX ---
            "If the user's query is unclear, ambiguous, or too short, ask for more details to clarify their intent."
        ],
        markdown=True,
    )
    return multi_ai_agent

# --- STREAMLIT UI (No changes needed from here down) ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")
if not groq_api_key or not alpha_vantage_api_key:
    st.error("API keys are not configured. Please add GROQ_API_KEY and ALPHA_VANTAGE_API_KEY to your Streamlit secrets.")
else:
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
                    if isinstance(chunk, dict) and "content" in chunk and "content" is not None:
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
        st.session_state.messages.append({"role": "assistant", "content": full_response})