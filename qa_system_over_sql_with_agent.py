# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/#agents

import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import ast
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

# llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", model="anthropic/claude-3.5-sonnet")


db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# use SQL agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

# create a retriever tool that gets the most similar proper nouns
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")
vector_db = FAISS.from_texts(artists + albums, hf)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

# create agent_executor
system_message = SystemMessage(content=SQL_PREFIX)
tools.append(retriever_tool) # add the self-defined retriever tool
agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

# 使用agent_executor来执行agent：向db查询数据
# Meta Llama 7B: 以下步骤中生成的sql是错误的，无法执行，LLM会自己瞎编结果，严重hallucination
# Claude 3.5 sonnet: 能够正确执行
# for s in agent_executor.stream(
#     {"messages": [HumanMessage(content="Which country's customers spent the most?")]}
# ):
#     print(s)
#     print("----")

# for s in agent_executor.stream(
#     {"messages": [HumanMessage(content="Describe the playlisttrack table")]}
# ):
#     print(s)
#     print("----")

for s in agent_executor.stream(
    {"messages": [HumanMessage(content="How many albums does alis in chain have?")]}
):
    print(s)
    print("----")
