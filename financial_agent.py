from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# web search agent
websearch_agent = Agent(
    name = "web search agent",
    role = "Search the web for the information",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

## Financial agent
finance_agent= Agent(
    name="finance_agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    show_tool_calls=True,
    instructions=["Use tables to display the data"],
    markdown=True,
)

multi_ai_agent= Agent(
    team=[websearch_agent,finance_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for the nvidia", stream=True)




