# https://python.langchain.com/v0.2/docs/tutorials/query_analysis/
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
import datetime

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", model="anthropic/claude-3.5-sonnet")

# load transcripts of a few LangChain videos
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# BUG youtubeLoader api not working
docs = []
for url in urls:
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())
# Add some additional metadata: what year the video was published
for doc in docs:
    doc.metadata["publish_year"] = int(
        datetime.datetime.strptime(
            doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y")
    )

print([doc.metadata["title"] for doc in docs])