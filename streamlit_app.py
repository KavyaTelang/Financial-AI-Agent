# streamlit_app.py

import os
import streamlit as st
from dotenv import load_dotenv
import traceback
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
# --- New Import ---
from phi.tools.alpha_vantage import AlphaVantageTools

# Load environment variables
load_dotenv()

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
    
    # --- UPGRADED FINANCE AGENT ---
    finance_agent = Agent(
        name="Finance AI Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        role="You are a world-class financial analyst.",
        # Replace the old tool with the new, more reliable one
        tools=[AlphaVantageTools(stock_news=True, company_overview=True)],
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

# --- STREAMLIT UI (No changes needed from here down) ---
# ... (the rest of your Streamlit UI code is perfect as is)
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