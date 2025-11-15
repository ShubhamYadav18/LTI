import re
import json
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

# -------------------------------------------------------
# ENTITY EXTRACTION FUNCTION
# -------------------------------------------------------
def extract_entities(text):
    entities = {}

    ip_pattern = r"\b(\d{1,3}(?:\.\d{1,3}){3})\b"
    os_pattern = r"(Windows|Ubuntu|CentOS|Debian|RedHat|Linux|macOS)[\s\d\.]*"
    hostname_pattern = r"Host=([\w-]+)"
    mitre_pattern = r"(T\d{4})"
    severity_pattern = r"\b(High|Medium|Critical|Low|None)\b"

    ip_match = re.findall(ip_pattern, text)
    os_match = re.findall(os_pattern, text)
    hostname_match = re.findall(hostname_pattern, text)
    mitre_match = re.findall(mitre_pattern, text)
    severity_match = re.findall(severity_pattern, text)

    if ip_match:
        entities["IP"] = ip_match[0]
    if os_match:
        entities["OS"] = os_match[0]
    if hostname_match:
        entities["Hostname"] = hostname_match[0]
    if mitre_match:
        entities["MITRE"] = mitre_match[0]
    if severity_match:
        entities["Severity"] = severity_match[0]

    return entities

# -------------------------------------------------------
# TOOL: KB LOOKUP (FIXED WITH DOCSTRING)
# -------------------------------------------------------
@tool
def kb_lookup(query: str) -> dict:
    """
    Lookup internal SOC knowledge base and return issue, severity, and fix.
    The match is based on the content of the user query or retrieved context.
    """
    kb = {
        "Suspicious PowerShell encoded command detected": {
            "issue": "Suspicious PowerShell command detected.",
            "severity": "HIGH ALERT",
            "fix": "Terminated process --> Disabled PowerShell v2 --> Quarantined artifacts"
        },
        "Malicious Python script executed": {
            "issue": "Malicious python code executed.",
            "severity": "MEDIUM",
            "fix": "Stopped script --> Performed malware analysis --> Cleaned environment."
        },
        "SSH brute-force authentication attempts": {
            "issue": "SSH Breach",
            "severity": "CRITICAL",
            "fix": "Blocked IP; Enabled fail2ban; Reset credentials"
        },
        "Unusual sudo usage": {
            "issue": "Unusual sudo usage",
            "severity": "LOW",
            "fix": "Reviewed sudoers file; Revoked access."
        },
    }

    q = query.lower()
    for key in kb:
        if key.lower() in q:
            return kb[key]

    return {
        "issue": "Unknown",
        "severity": "NONE",
        "fix": "No KB entry found."
    }

# -------------------------------------------------------
# 1. Load CSV
# -------------------------------------------------------
loader = CSVLoader("incidents.csv")
docs = loader.load()

# -------------------------------------------------------
# 2. Split documents
# -------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# -------------------------------------------------------
# 3. Embeddings (MiniLM)
# -------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------------------------------
# 4. Vector Store (FAISS)
# -------------------------------------------------------
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_store")

faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------------
# 5. BM25 Retriever
# -------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# -------------------------------------------------------
# 6. Hybrid Retriever
# -------------------------------------------------------
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# -------------------------------------------------------
# 7. LLM (MISTRAL)
# -------------------------------------------------------
llm = ChatOllama(model="mistral", temperature=0.7)
llm_tools = llm.bind_tools([kb_lookup])

# -------------------------------------------------------
# 8. Prompt Template
# -------------------------------------------------------
prompt_template = """
You are an SOC(Security Operations Center) Analyst Assistant.

Your response MUST ALWAYS follow this exact format:


summarize relevant history or say 'No previous issues found' in one sentence>.
Suggest the fix/solution in one short sentence>. 

Also include If severity is HIGH ALERT or CRITICAL.


Use retrieved context + chat memory + tools when needed.
If unclear, infer from patterns.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

# -------------------------------------------------------
# 9. Conversational Retrieval Chain
# -------------------------------------------------------
base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_tools,
    retriever=hybrid_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# -------------------------------------------------------
# 10. Multi-user chat memory
# -------------------------------------------------------
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

# -------------------------------------------------------
# 11. Chat loop with JSON output
# -------------------------------------------------------
print("Multi-User RAG Chat — enter user_id & question\n")

while True:
    user_id = input("\nEnter user_id: ").strip()
    if user_id.lower() in ["exit", "quit", "stop"]:
        break

    query = input("You: ")

    print(f"AI ({user_id}): ", end="", flush=True)

    stream = rag_chain.stream(
        {"question": query},
        config={"configurable": {"session_id": user_id}}
    )

    full_response = ""
    extracted_entities = {}

    for chunk in stream:
        if "answer" in chunk:
            token = chunk["answer"]
            full_response += token
            print(token, end="", flush=True)

            extracted_entities.update(extract_entities(token))

    structured_output = {
        "response": full_response,
        "extracted_entities": extracted_entities
    }

    print("\nStructured Output: ", json.dumps(structured_output, indent=2))
    print("\n")
