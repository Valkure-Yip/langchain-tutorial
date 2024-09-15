# https://python.langchain.com/v0.2/docs/tutorials/agents/

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["DASHSCOPE_API_KEY"] = "sk-f2fff14ce1c3436dad4855e94c3d3c54"
# os.environ["ANTHROPIC_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

search = TavilySearchResults(max_results=2)
# search_results = search.invoke("what is the weather in SF")
# print(search_results)
# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]
memory = SqliteSaver.from_conn_string(":memory:")
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

# Use the agent
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")