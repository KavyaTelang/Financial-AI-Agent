# streamlit_app.py (Final Hardened Version)

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
from alpha_vantage.fundamentaldata import FundamentalData

# Load environment variables
load_dotenv()

# --- NEW: Sanity check to ensure API keys are loaded ---
# This helps us debug if the secrets are the problem.
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY") or st.secrets.get("ALPHA_VANTAGE_API_KEY")

# --- MODIFIED: The custom tool now has a longer timeout ---
class AlphaVantageTools(Toolkit):
    def __init__(
        self,
        api_key: str | None = None,
        company_overview: bool = True,
    ):
        super().__init__(name="alpha_vantage_tools")
        self.api_key = api_key or alpha_vantage_api_key
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found.")
        if company_overview:
            self.register(self.get_company_overview)

    def get_company_overview(self, ticker: str) -> str:
        """
        Get the company overview and fundamentals for a stock ticker.
        """
        try:
            # --- MODIFIED: Increased timeout from default (10s) to 30s ---
            fd = FundamentalData(key=self.api_key, output_format='pandas', treat_info_as_error=True, timeout=30)
            overview, _ = fd.get_company_overview(symbol=ticker)
            if overview.empty:
                return f"No company overview found for {ticker}."
            
            # Convert pandas DataFrame to a readable string
            details = [f"**{key}**: {value[0]}" for key, value in overview.items()]
            return f"Company Overview for {ticker}:\n" + "\n".join(details)
        except Exception as e:
            return f"Error getting company overview for {ticker}: {e}"


@st.cache_resource
def get_multi_ai_agent():
    """This function creates and returns the multi-agent assistant."""
    # (The rest of the agent definitions are the same)
    web_search_agent = Agent(name="Web Search Agent", role="Search the web for information.", model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key), tools=[DuckDuckGo()], markdown=True)
    finance_agent = Agent(name="Finance AI Agent", model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key), role="You are a world-class financial analyst.", tools=[AlphaVantageTools(api_key=alpha_vantage_api_key)], instructions=["Use tables to display data"], markdown=True)
    multi_ai_agent = Agent(team=[web_search_agent, finance_agent], model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key), instructions=["Always include sources", "Use tables to display data", "When delegating tasks, provide clear details in the 'additional_information' field for the specialist agent."], markdown=True)
    return multi_ai_agent

# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")

# --- NEW: Display a warning if API keys are missing ---
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
                traceback.print_exc()
        st.session_state.messages.append({"role": "assistant", "content": full_response})