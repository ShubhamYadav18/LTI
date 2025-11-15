# =====================================================================
# HYBRID RAG (FAISS + BM25) + TOOL SUPPORT + MULTI-USER MEMORY (MISTRAL)
# =====================================================================
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





# 1. LOAD CSV

loader = CSVLoader("security_incidents.csv")
docs = loader.load()


#2. SPLIT DOCUMENTS
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)


#3. EMBEDDINGS (MiniLM)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#4. VECTOR STORE (FAISS)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_store")

faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})


#5. BM25 RETRIEVER
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4


#6. HYBRID RETRIEVER
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)


#7. LLM (MISTRAL)
llm = ChatOllama(model="mistral", temperature=0.7)



#TOOL
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

llm_tools = llm.bind_tools([kb_lookup])


#8. PROMPT
prompt_template = """
You are an intelligent SOC Analyst Assistant .

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


#9. Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)



base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_tools,
    retriever=hybrid_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)



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



print("Multi-User RAG Chat — enter user_id & question\n")

while True:
    user_id = input("\nEnter user_id: ").strip()
    if user_id.lower() in ["exit", "quit", "stop"]:
        break

    query = input("You: ")

    response = rag_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": user_id}}
    )

    print(f"AI ({user_id}): {response['answer']}")