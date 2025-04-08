from fastapi import FastAPI
from langserve import add_routes
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()

class QAInput(BaseModel):
    question: str

class QAOutput(BaseModel):
    answer: str

@app.on_event("startup")
def init_qa():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables!")

    loader = TextLoader("data/promtior_info.txt")
    docs = loader.load()

    embeddings = OpenAIEmbeddings(api_key=openai_key)
    db = Chroma.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    qa_runnable = RunnableLambda(
        lambda input: {
            "answer": qa_chain.invoke({"query": input["question"]})["result"]
        }
    )

    add_routes(
        app,
        qa_runnable,
        path="/chat",
        input_type=QAInput,
        output_type=QAOutput
    )
