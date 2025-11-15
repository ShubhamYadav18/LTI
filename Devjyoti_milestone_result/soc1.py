# ================================================================
# SOC RAG Assistant - Console Version with JSON Output
# ================================================================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import re, json

# ===========
# STEP 1: LOAD DATA
# ===========
print("\n=== STEP 1: LOAD DATA ===")
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded {len(docs)} incidents")
if docs:
    print("Sample record:\n", docs[0].page_content[:200])

# ===========
# STEP 2: CHUNK DATA
# ===========
print("\n=== STEP 2: CHUNKING ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
splits = splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

# ===========
# STEP 3: EMBEDDINGS + FAISS
# ===========
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("FAISS vector retriever ready (k=4)")

# ===========
# BONUS: HYBRID RETRIEVAL
# ===========
print("\n=== BONUS: HYBRID RETRIEVAL ===")
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4
hybrid = EnsembleRetriever(
    retrievers=[retriever, bm25],
    weights=[0.7, 0.3],
)
print("Hybrid retriever ready.\n")

# ===========
# STEP 4: PROMPT + RAG CHAIN
# ===========
print("\n=== STEP 4: RAG CHAIN ===")
llm = ChatOllama(model="mistral", temperature=0.3)

def _format_docs(docs_list):
    if not docs_list:
        return "No relevant past incidents retrieved."
    return "\n---\n".join([d.page_content for d in docs_list])

format_docs = RunnableLambda(lambda x: _format_docs(x))

prompt = ChatPromptTemplate.from_template("""
You are a SOC Analyst Assistant.

Analyst History:
{history}

Query:
{question}

Entities:
{entities}

Retrieved Context:
{context}

Answer with resolution and enriched analysis.
""")

parser = StrOutputParser()

context_chain = RunnableLambda(lambda x: x["question"]) | hybrid | format_docs
entities_passthrough = RunnableLambda(lambda x: x.get("entities", {}))

chain = (
    {
        "context": context_chain,
        "question": RunnablePassthrough(),
        "entities": entities_passthrough,
        "history": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)
print("RAG chain constructed.\n")

# ===========
# STEP 5: MEMORY
# ===========
print("\n=== STEP 5: MEMORY ===")
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
print("Memory wrapper ready.\n")

# ===========
# STEP 6: ENTITY EXTRACTION + THREAT SCORE
# ===========
def extract_entities(query: str, context: str):
    text = f"{query}\n{context}"
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    mitre = re.findall(r"T\d{4}", text)
    severity = re.findall(r"\b(Critical|High|Medium|Low)\b", text, flags=re.IGNORECASE)
    return {
        "ips": ips,
        "mitre": mitre,
        "severity": severity[0].title() if severity else None
    }

def compute_threat_score(entities):
    score = 0
    sev = (entities.get("severity") or "").lower()
    if sev == "critical": score += 40
    elif sev == "high": score += 30
    elif sev == "medium": score += 20
    elif sev == "low": score += 10

    mitre = set(entities.get("mitre", []))
    if any(code.startswith("T1059") for code in mitre): score += 15   # PowerShell
    if "T1110" in mitre: score += 15                                  # Brute force

    ips = entities.get("ips", [])
    if any(ip.endswith(".13") or ip.startswith(("45.", "185.")) for ip in ips):
        score += 20

    return min(score, 100)

def to_structured_output(answer, entities, context, threat_score):
    return {
        "answer": answer,
        "entities": entities,
        "threat_score": threat_score,
        "retrieved_context": context,
    }

# ===========
# STEP 7: INTERACTIVE CONSOLE LOOP
# ===========
print("\n=== STEP 7: CONSOLE LOOP ===")
while True:
    raw = input("Enter: <analyst_id> <query> (or q): ")
    if raw.strip().lower() == "q":
        break

    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Format: analyst42 suspicious ssh activity")
        continue

    analyst_id, query = parts
    retrieved_docs = retriever._get_relevant_documents(query, k=4, run_manager=None)
    context_str = _format_docs(retrieved_docs)
    entities = extract_entities(query, context_str)

    response = memory_chain.invoke(
        {"question": query, "entities": entities, "history": ""},
        config={"configurable": {"session_id": analyst_id}}
    )

    threat_score = compute_threat_score(entities)
    structured = to_structured_output(response, entities, context_str, threat_score)

    print(json.dumps(structured, indent=2))
    print("\n")