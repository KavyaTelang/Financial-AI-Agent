import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    role="You are a world-class financial analyst. Your primary skill is providing stock market data using your tools.",
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=["Use tables to display data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tools_calls=True,
    markdown=True,
)

# --- API DEFINITION ---

# Create the FastAPI application
app = FastAPI()

# Define the structure of the request body using Pydantic
class QueryRequest(BaseModel):
    query: str

# Define the API endpoint
@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    This endpoint receives a query and streams the agent's response.
    """
    # Use the agent's .run() method, which returns a generator for streaming
    # We do not use .print_response() because we need to return the data, not print it
    response_generator = multi_ai_agent.run(request.query, stream=True)
    
    return StreamingResponse(response_generator, media_type="text/plain")

# Optional: Add a root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok"}