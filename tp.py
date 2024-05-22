from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
    ConversationalRetrievalChain,
)
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from langchain_groq import ChatGroq
from langchain_together.embeddings import TogetherEmbeddings

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from os import system, environ
from dotenv import dotenv_values
from typing import Dict, List

import uvicorn
import json

vector_store_directory = "vector_store/"
chat_history_file = "chat_history.json"

config: Dict = {
    **dotenv_values(dotenv_path=".env"),  
    **environ,  
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


### VECTOR TOOLS

def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
) -> VectorStore:
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=vector_store_directory,
    )


def get_vector_store(
    embeddings: Embeddings,
) -> VectorStore:
    return Chroma(
        persist_directory=vector_store_directory,
        embedding_function=embeddings,
    )


def clear_vector_store():
    system("rm -rf " + vector_store_directory)


### DOCUMENT TOOLS


def split_documents(
    documents: List[Document],
) -> List[Document]:
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter()
    return text_splitter.split_documents(documents)


def pdf_to_documents(
    pdf_path: str,
) -> List[Document]:
    loader: PyPDFLoader = PyPDFLoader(file_path=pdf_path)
    documents: List[Document] = loader.load()
    return documents

### CHAT HISTORY

def load_chat_history() -> List[tuple[str]]:
    return list(
        map(lambda x: (x[0], x[1]), json.load(open(chat_history_file)))
    )
    
def add_to_chat_history(question:str, answer:str):
    chat_history: List[tuple[str]] = load_chat_history()
    chat_history.append((question, answer))
    with open(chat_history_file, 'w') as f:
        json.dump(chat_history, f)
        
def clear_chat_history():
    with open(chat_history_file, 'w') as f:
        json.dump([], f)


### CHAT TOOLS


def load_files(
    pdf_paths: List[str],
    embeddings: Embeddings,
):
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        documents.extend(split_documents(pdf_to_documents(pdf_path=pdf_path)))
    create_vector_store(
        documents=documents,
        embeddings=embeddings,
    )


def ask(
    question: str,
    chat_model: BaseChatModel,
    embeddings: Embeddings,
) -> str:
    chat_history: List[tuple[str]] = load_chat_history()
    vector_store: Chroma = get_vector_store(embeddings=embeddings)
    retriever: VectorStoreRetriever = vector_store.as_retriever()
    prompt_template = """Answer the question at the end using this context :

    {context}

    Only use information in the context or say "I do not know".

    Question: {question}
    """
    promptTemplate: PromptTemplate = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    conversational_chain: BaseConversationalRetrievalChain = (
        ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": promptTemplate},
        )
    )
    result = conversational_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )
    add_to_chat_history(question=question, answer=result["answer"])
    return result["answer"]


### DEMO

chat_groq: ChatGroq = ChatGroq(temperature=0, groq_api_key=config["GROQ_API_KEY"], model_name="mixtral-8x7b-32768")
embeddings_together_ai: TogetherEmbeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=config["TOGETHER_AI_API_KEY"])

@app.get("/ask/{question}")
async def ask_api(question):
    return {"answer": ask(question=question, embeddings=embeddings_together_ai, chat_model=chat_groq,)}

@app.get("/load_pdf/{pdf}")
async def load_files_api(pdf):
    load_files(pdf_paths=["pdf/"+pdf], embeddings=embeddings_together_ai,)
    return {"status": "done"}

@app.get("/clear_pdfs")
async def clear_vector_store_api():
    clear_vector_store()
    return {"status": "done"}

@app.get("/clear_chat_history")
async def clear_chat_history_api():
    clear_chat_history()
    return {"status": "done"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

