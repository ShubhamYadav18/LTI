import redis
import faiss
import numpy as np
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Initialize the embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loader = CSVLoader("security_incidents.csv")
docs = loader.load()

# Split Documents
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# FAISS for vector search
vector_store = FAISS.from_documents(chunks, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# Hybrid Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# LLM (MISTRAL)
llm = ChatOllama(model="mistral", temperature=0.7)

@tool
def kb_lookup(query: str) -> str:
    """Lookup internal knowledge base for fixes."""
    fixes = {
        "outlook crash": "Restart Outlook → Repair Office → Delete corrupt OST.",
        "printer issue": "Reinstall drivers and restart print spooler.",
    }
    for key in fixes:
        if key in query.lower():
            return fixes[key]
    return "No KB entry found."

# Bind tools to LLM
llm_tools = llm.bind_tools([kb_lookup])

# Prompt Template
prompt_template = """
You are an intelligent SOC Analyst Assistant.

Your response MUST ALWAYS follow this exact format:

From your past: <summarize relevant history or say 'No previous issues found'>.
Suggested: <the fix/solution in one short sentence>.

Use retrieved context + chat memory + tools when needed.
If unclear, infer from patterns.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

# Conversational Chain
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_tools,
    retriever=hybrid_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Multi-user session memory
user_memory_store = {}

def get_user_memory(user_id: str):
    if user_id not in user_memory_store:
        user_memory_store[user_id] = InMemoryChatMessageHistory()
    return user_memory_store[user_id]

rag_chain = RunnableWithMessageHistory(
    base_rag_chain,
    get_user_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Check if query exists in Redis
def check_cache(query: str):
    cached_response = redis_client.get(query)
    if cached_response:
        return cached_response
    return None

# Save response to Redis cache
def cache_response(query: str, response: str):
    redis_client.set(query, response)

# Function to process the query
def process_query(user_id: str, query: str):
    """Process the user's query with RAG chain and cache the response."""
    # Check if response is in cache
    cached_response = check_cache(query)
    if cached_response:
        return cached_response
    
    # Otherwise, invoke RAG chain
    user_memory = get_user_memory(user_id)
    response = base_rag_chain.invoke(
        {
            "question": query,
            "chat_history": user_memory.messages
        }
    )
    
    # Add the user query and response to memory
    user_memory.add_user_message(query)
    user_memory.add_ai_message(response["answer"])

    # Cache the response
    cache_response(query, response["answer"])

    return response["answer"]

