import os
import streamlit as st
from dotenv import load_dotenv

# Important: You must import your agent and tools here
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()

# --- AGENT DEFINITION (The same as before) ---
# This runs once and is cached by Streamlit

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
        tools=[
            YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
        ],
        instructions=["Use tables to display data"],
        markdown=True,
    )

    multi_ai_agent = Agent(
        team=[web_search_agent, finance_agent],
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=["Always include sources", "Use tables to display data"],
        markdown=True,
    )
    return multi_ai_agent

# --- STREAMLIT UI ---

st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")
st.sidebar.markdown("### Built by Kavya Telang")
st.sidebar.markdown("This multi-agent assistant can search the web and access real-time financial data to answer your questions.")

# Get the agent
multi_ai_agent = get_multi_ai_agent()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your financial research today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me about stocks, news, and more..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the assistant's response
    with st.chat_message("assistant"):
        # The magic function for streaming responses
        response = st.write_stream(multi_ai_agent.run(prompt, stream=True))
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})