# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/

import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import re
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
# llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", model="anthropic/claude-3.5-sonnet")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM Artist LIMIT 10;"))

# NOTE 需自定义一个lambda表达式来提取LLM回答中的sql，使用的llamma 3 7B 模型直接把sql写在一大段md里了
write_query = (create_sql_query_chain(llm, db)|(lambda markdown: re.search(r'```sql\n(.*?)\n```', markdown, re.DOTALL).group(1).strip() if re.search(r'```sql\n(.*?)\n```', markdown, re.DOTALL) else None))
execute_query = QuerySQLDataBaseTool(db=db) # query execution
# chain = write_query | execute_query
# response = chain.invoke({"question": "How many employees are there"})
# print(response)
# print(db.run(response))
# print(chain.get_prompts()[0].pretty_print())

# get final answer with SQL query result
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": "How many employees are there"}))