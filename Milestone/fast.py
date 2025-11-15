from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain

# Tool support
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Multi-user session memory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Load CSV
loader = CSVLoader("tickets.csv")
docs = loader.load()

# 2. Split Documents
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# 3. Embeddings (MiniLM)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Vector Store (FAISS)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_store")

faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 5. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# 6. Hybrid Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# 7. LLM (MISTRAL)
llm = ChatOllama(model="mistral", temperature=0.7)

# Tool for internal knowledge base lookup
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

# Bind tools to the LLM
llm_tools = llm.bind_tools([kb_lookup])

# 8. Prompt Template
prompt_template = """
You are an intelligent IT support assistant.

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

# 9. Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

# Initialize the Conversational Retrieval Chain
base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_tools,
    retriever=hybrid_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# User memory store for multi-user session memory
user_memory_store = {}

def get_user_memory(user_id: str):
    """Retrieve or create user memory."""
    if user_id not in user_memory_store:
        user_memory_store[user_id] = InMemoryChatMessageHistory()
    return user_memory_store[user_id]

# Define a function to invoke the RAG chain with a user query and session memory
def process_query(user_id: str, query: str):
    """Process the user's query with the RAG chain."""
    user_memory = get_user_memory(user_id)

    # Invoke the RAG chain with user memory and query
    response = base_rag_chain.invoke(
        {
            "question": query,
            "chat_history": user_memory.messages
        }
    )
    
    # Add the user query and response to memory
    user_memory.add_user_message(query)
    user_memory.add_ai_message(response["answer"])

    return response["answer"]

