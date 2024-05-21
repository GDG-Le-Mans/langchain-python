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

from os import system
from typing import List

from tools.ollama_model import embeddings_mistral, llm_txt

### VECTOR TOOLS

vector_store_directory = "vector_store/"


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


### CHAT TOOLS


def load_files(
    pdf_paths: List[str],
    embeddings: Embeddings,
):
    clear_vector_store()
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        documents.extend(split_documents(pdf_to_documents(pdf_path=pdf_path)))
    create_vector_store(
        documents=documents,
        embeddings=embeddings,
    )


def ask(
    question: str,
    chat_history: List[tuple[str]],
    chat_model: BaseChatModel,
    embeddings: Embeddings,
) -> str:
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
    return result["answer"]


### DEMO

my_pdfs: List[str] = ["example_docs/pdf/dogs.pdf", "example_docs/pdf/cats2.pdf"]
load_files(
    pdf_paths=my_pdfs,
    embeddings=embeddings_mistral,
)

my_chat_history: List[tuple[str]] = [
    (
        "What is a bird?",
        " A bird, according to the given context, is not mentioned or discussed. I do not know.",
    ),
    (
        "What is a lizard?",
        " A lizard, according to the given context, is not mentioned or discussed. I do not know.",
    )
]
my_question: str = "tell me more"
answer: str = ask(
    question=my_question,
    chat_history=my_chat_history,
    embeddings=embeddings_mistral,
    chat_model=llm_txt,
)
print(answer)
