# streamlit_app.py (Final Version with Enhanced Error Reporting)

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
    # ... (The AlphaVantageTools class is exactly the same as before, no changes needed)
    def __init__(self, api_key: str | None = None, stock_news: bool = True, company_overview: bool = True):
        super().__init__(name="alpha_vantage_tools")
        self.api_key = api_key or alpha_vantage_api_key
        if not self.api_key: raise ValueError("Alpha Vantage API key not found.")
        if stock_news: self.register(self.get_stock_news_and_sentiment)
        if company_overview: self.register(self.get_company_overview)
    def get_stock_news_and_sentiment(self, ticker: str) -> str:
        try:
            url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=5&apikey={self.api_key}'
            r = requests.get(url, timeout=30) # Added a 30-second timeout
            r.raise_for_status() # This will raise an error for bad responses (4xx or 5xx)
            data = r.json()
            if "feed" not in data or not data["feed"]: return f"No news found for {ticker}."
            articles = [f"- **{a['title']}**\n  - Source: {a['source']}\n  - Summary: {a['summary']}\n  - URL: {a['url']}" for a in data["feed"]]
            return "Latest News:\n" + "\n".join(articles)
        except Exception as e: return f"Error getting news for {ticker}: {e}"
    def get_company_overview(self, ticker: str) -> str:
        try:
            fd = FundamentalData(key=self.api_key, output_format='pandas', treat_info_as_error=True, timeout=30)
            overview, _ = fd.get_company_overview(symbol=ticker)
            if overview.empty: return f"No company overview found for {ticker}."
            details = [f"**{key}**: {value[0]}" for key, value in overview.items()]
            return f"Company Overview for {ticker}:\n" + "\n".join(details)
        except Exception as e: return f"Error getting company overview for {ticker}: {e}"

# In streamlit_app.py

@st.cache_resource
def get_multi_ai_agent():
    """This function creates and returns the multi-agent assistant."""
    # Using a smaller, more token-efficient model to stay in the free tier
    model_id = "llama3-8b-8192"

    web_search_agent = Agent(
        name="Web Search Agent",
        role="Search the web for information.",
        model=Groq(id=model_id, api_key=groq_api_key),
        tools=[DuckDuckGo()],
        markdown=True,
    )
    
    finance_agent = Agent(
        name="Finance AI Agent",
        model=Groq(id=model_id, api_key=groq_api_key),
        role="You are a world-class financial analyst.",
        tools=[AlphaVantageTools()],
        instructions=["Use tables to display data"],
        markdown=True,
    )
    
    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        model=Groq(id=model_id, api_key=groq_api_key),
        instructions=[
            "Always include sources", 
            "Use tables to display data",
            "When delegating tasks, provide clear details in the 'additional_information' field for the specialist agent."
        ],
        markdown=True,
    )
    return multi_ai_agent

# --- STREAMLIT UI ---
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
                # ... (The stream-handling loop is the same)
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
            # --- THIS IS THE ONLY PART THAT CHANGED ---
            except Exception as e:
                # Keep the user-friendly message
                full_response = "Sorry, an error occurred. This can happen if the external data source is slow or unavailable. Please try your request again in a moment."
                # Add the specific technical error for debugging purposes
                full_response += f"\n\n**Debug Info:**\n```\n{traceback.format_exc()}\n```"
                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})