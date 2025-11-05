# streamlit_app.py (FINAL MANUAL CONTROL VERSION)

import os
import streamlit as st
import traceback
import requests
import json

# Phidata imports (we only need the model wrapper now)
from phi.model.groq import Groq

# Alpha Vantage library import
from alpha_vantage.fundamentaldata import FundamentalData

# --- TOOL DEFINITION ---
# These are now just regular Python functions, not part of a Toolkit.

def get_stock_news_and_sentiment(ticker: str) -> str:
    """Gets the latest news for a stock ticker."""
    api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY")
    if not api_key: return "Error: Alpha Vantage API key not found."
    try:
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=5&apikey={api_key}'
        r = requests.get(url, timeout=20); r.raise_for_status(); data = r.json()
        if "feed" not in data or not data["feed"]: return f"No news found for {ticker}."
        articles = [f"- **{a['title']}**: {a['summary']}" for a in data["feed"]]
        full_output = "Latest News:\n" + "\n".join(articles)
        return full_output[:3500]
    except Exception as e: return f"Error getting news: {e}"

def get_company_overview(ticker: str) -> str:
    """Gets the company overview for a stock ticker."""
    api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY")
    if not api_key: return "Error: Alpha Vantage API key not found."
    try:
        fd = FundamentalData(key=api_key, output_format='pandas', treat_info_as_error=True, timeout=20)
        overview, _ = fd.get_company_overview(symbol=ticker)
        if overview.empty: return f"No company overview found for {ticker}."
        details = [f"**{key}**: {value[0]}" for key, value in overview.items()]
        full_output = f"Company Overview for {ticker}:\n" + "\n".join(details)
        return full_output[:3500]
    except Exception as e: return f"Error getting company overview: {e}"

# --- CORE LOGIC: MANUALLY HANDLING TOOLS ---

# Define the tools the AI can use
tools = {
    "get_stock_news_and_sentiment": get_stock_news_and_sentiment,
    "get_company_overview": get_company_overview,
}

# Define the JSON structure the AI must return
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_news_and_sentiment",
            "description": "Get the latest news and sentiment for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "The stock ticker symbol, e.g., 'NVDA'"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_overview",
            "description": "Get the company overview and fundamentals for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "The stock ticker symbol, e.g., 'TSLA'"}},
                "required": ["ticker"],
            },
        },
    },
]

# In streamlit_app.py

def get_llm_response(prompt: str):
    """This function orchestrates the entire AI interaction manually."""
    
    llm = Groq(model="llama-3.3-70b-versatile", api_key=st.secrets.get("GROQ_API_KEY"))

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Given a user query, decide if you need to call a function to get information. If so, respond with the function call in JSON format."},
        {"role": "user", "content": prompt},
    ]
    
    # --- THIS IS THE FIX ---
    # Change .create() to .invoke()
    first_response = llm.invoke(messages=messages, tools=tool_definitions, tool_choice="auto")
    
    if first_response.choices[0].message.tool_calls:
        tool_call = first_response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name in tools:
            yield "🔍 Accessing financial data...\n\n"
            function_to_call = tools[function_name]
            tool_output = function_to_call(**function_args)
            
            messages.append(first_response.choices[0].message)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": tool_output})
            
            second_response_stream = llm.response_stream(messages=messages)
            for chunk in second_response_stream:
                yield chunk
        else:
            yield "Error: AI chose a tool that doesn't exist."
    else:
        response_stream = llm.response_stream(messages=messages)
        for chunk in response_stream:
            yield chunk

# --- STREAMLIT UI ---
st.set_page_config(page_title="Financial AI Agent", page_icon="📈")
st.title("📈 Financial AI Agent")

if not st.secrets.get("GROQ_API_KEY") or not st.secrets.get("ALPHA_VANTAGE_API_KEY"):
    st.error("API keys are not configured. Please add GROQ_API_KEY and ALPHA_VANTAGE_API_KEY to your Streamlit secrets.")
else:
    st.sidebar.markdown("### Built by Kavya Telang")
    st.sidebar.markdown("This assistant can access real-time financial data.")
    
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
                # Call our new, manual function
                for chunk in get_llm_response(prompt):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = "Sorry, an error occurred. This can happen if an external data source is slow or if the daily token limit has been reached. Please try again in a few minutes."
                full_response += f"\n\n**Debug Info:**\n```\n{traceback.format_exc()}\n```"
                placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})