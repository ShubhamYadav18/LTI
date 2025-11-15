# ================================================================
# SOC RAG Assistant - Assessment Template (Students: Fill TODOs)
# ================================================================

# ===========
# STEP 0: IMPORTS (DO NOT REMOVE — Add only if needed)
# ===========
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import re
import json

# Helper: format documents to plain text for the prompt
def format_docs(docs):
    if not docs:
        return "(No relevant context retrieved)"
    return "\n\n".join([f"[Chunk {i+1}]\n" + d.page_content for i, d in enumerate(docs)])

# Simple validation for IPv4 octets
def _valid_ip(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except ValueError:
        return False

# LLM Setup (Students keep as-is)
llm = ChatOllama(model="mistral", temperature=0.3)

# ===========
# STEP 1: LOAD DATA
# ===========
print("\n=== STEP 1: LOAD DATA ===")
# TODO: Load security_incidents.txt using TextLoader
# TODO: Print doc count + sample record
loader = TextLoader("security_incidents.txt")
# STUDENT IMPLEMENTATION:
docs = loader.load()
print(f"Loaded {len(docs)} document(s).")
if len(docs) > 0:
    sample_preview = docs[0].page_content[:500]
    print("Sample record preview (first 500 chars):\n", sample_preview)
else:
    print("No documents loaded—ensure security_incidents.txt exists in the working directory.")

# ===========
# STEP 2: CHUNK DATA
# ===========
print("\n=== STEP 2: CHUNKING ===")
# TODO: Implement RecursiveCharacterTextSplitter
# TODO: Produce splits
# STUDENT IMPLEMENTATION:
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
splits = splitter.split_documents(docs)
print(f"Generated {len(splits)} chunk(s).")
if len(splits) > 0:
    print("Chunk[0] preview:\n", splits[0].page_content[:300])

# ===========
# STEP 3: EMBEDDINGS + FAISS
# ===========
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
# TODO: Create embeddings using MiniLM-L6-v2
# TODO: Create FAISS vectorstore
# TODO: Create vector retriever (k=4)
# STUDENT IMPLEMENTATION:
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Vector retriever ready.")

# ===========
# BONUS: HYBRID RETRIEVAL
# ===========
print("\n=== BONUS: HYBRID RETRIEVAL ===")
# TODO: Create BM25 retriever
# TODO: Create EnsembleRetriever with weights [0.7, 0.3]
# STUDENT IMPLEMENTATION:
bm25 = BM25Retriever.from_documents(splits)
hybrid = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.7, 0.3])
print("Hybrid retriever ready.\n")

# ===========
# STEP 4: PROMPT + LCEL RAG CHAIN
print("\n=== STEP 4: RAG CHAIN ===")

# Build prompt template (must include placeholders for context, question, entities, history)
# NOTE: all literal JSON braces are doubled to avoid being interpreted as template vars.
prompt = ChatPromptTemplate.from_template(
    """
You are a SOC Analyst Assistant helping triage and resolve security alerts.
Use the retrieved incident context, extracted entities, and prior analyst conversation history to provide precise guidance.

[Retrieved Context]
{context}

[Entity Memory]
{entities}

[Analyst History]
{history}

[Analyst Query]
{question}

Instructions:
- Recommend resolutions and playbooks drawing from similar incidents in the context.
- Call tools if helpful (e.g., threat_enrich for IPs) to enrich the analysis.
- Reference MITRE ATT&CK techniques when present.
- Be explicit about risks, likely root cause, and next actions.
- Return a concise, structured JSON-like response with keys:
  {{
    "recommendations": [...],
    "resolution_steps": [...],
    "similar_incidents_summary": "...",
    "entities_extracted": {{ ... }},
    "mitre_mapping": [...],
    "threat_score": "use provided score or estimate",
    "analysis": "clear reasoning"
  }}
"""
)
parser = StrOutputParser()


# Build helpers as runnables
context_runnable = itemgetter("question") | hybrid | RunnableLambda(format_docs)
entities_runnable = RunnableLambda(lambda x: (
    {
        # Extract based on the input dict (expects 'question')
        **extract_entities(x.get("question", ""), "")
    }
))
history_runnable = itemgetter("history")

# TODO: Build LCEL runnable chain
# STUDENT IMPLEMENTATION:
chain = (
    {
        "context": context_runnable,
        "question": itemgetter("question"),
        "entities": entities_runnable,
        "history": history_runnable,
    }
    | prompt
    | llm  # will be replaced with llm_with_tools after binding tools in STEP BONUS
    | parser
)
print("RAG chain constructed.\n")

# ===========
# BONUS: TOOL
# ===========
print("\n=== BONUS: TOOL ===")
@tool
def threat_enrich(ip: str):
    """Students: Return mock threat intel for IP"""
    # TODO: Implement enrichment logic
    # Private ranges and RFC special-use for demo purposes
    intel = {
        "ip": ip,
        "is_valid": _valid_ip(ip),
        "category": "unknown",
        "confidence": 50,
        "geo": "N/A",
        "asn": "AS-UNKNOWN",
        "last_seen": "N/A",
        "notes": []
    }
    if not intel["is_valid"]:
        intel["category"] = "invalid-ip"
        intel["confidence"] = 0
        intel["notes"].append("Not a valid IPv4 address")
        return json.dumps(intel)

    # RFC 1918 private
    private_blocks = [
        ("10.", "private"),
        ("192.168.", "private"),
        ("172.16.", "private"), ("172.17.", "private"), ("172.18.", "private"), ("172.19.", "private"),
        ("172.20.", "private"), ("172.21.", "private"), ("172.22.", "private"), ("172.23.", "private"),
        ("172.24.", "private"), ("172.25.", "private"), ("172.26.", "private"), ("172.27.", "private"),
        ("172.28.", "private"), ("172.29.", "private"), ("172.30.", "private"), ("172.31.", "private"),
    ]
    for prefix, cat in private_blocks:
        if ip.startswith(prefix):
            intel["category"] = cat
            intel["confidence"] = 90
            intel["notes"].append("RFC1918 private address")
            return json.dumps(intel)

    # TEST-NET ranges used in documentation (treat as suspicious demo)
    if ip.startswith("203.0.113.") or ip.startswith("198.51.100.") or ip.startswith("192.0.2."):
        intel["category"] = "known-bad-demo"
        intel["confidence"] = 80
        intel["geo"] = "Global"
        intel["asn"] = "AS-TESTNET"
        intel["last_seen"] = "recent"
        intel["notes"].append("Matched TEST-NET reserved block; treat as suspicious for demo")
    else:
        intel["category"] = "internet"
        intel["confidence"] = 60
        intel["geo"] = "Unknown"
        intel["asn"] = "AS-UNKNOWN"
        intel["notes"].append("No specific intel; treat as internet host")

    return json.dumps(intel)

# TODO: Bind tool to LLM
llm_with_tools = llm.bind_tools([threat_enrich])

# Replace LLM in chain with tool-enabled one
chain = (
    {
        "context": context_runnable,
        "question": itemgetter("question"),
        "entities": entities_runnable,
        "history": history_runnable,
    }
    | prompt
    | llm_with_tools
    | parser
)

# ===========
# STEP 5: MEMORY (RunnableWithMessageHistory)
# ===========
print("\n=== STEP 5: MEMORY ===")
store = {}
# DO NOT CHANGE

def get_session_history(session_id: str):
    # TODO: Create InMemoryChatMessageHistory if missing
    # TODO: Return the history object
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# TODO: Wrap chain with RunnableWithMessageHistory
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
print("Memory wrapper ready.\n")

# ===========
# STEP 6: ENTITY EXTRACTION (BONUS)
# ===========
def extract_entities(query: str, context: str):
    """Students: Extract IPs, hostname, OS, MITRE tags, severity"""
    # TODO: Your custom extraction logic
    text = f"{query}\n{context}" if context else query

    # IP addresses
    ip_candidates = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    ips = [ip for ip in ip_candidates if _valid_ip(ip)]

    # OS detection
    os_list = []
    os_keywords = ["windows", "linux", "ubuntu", "debian", "rhel", "centos", "macos", "osx"]
    for kw in os_keywords:
        if re.search(rf"\b{kw}\b", text, flags=re.IGNORECASE):
            os_list.append(kw.capitalize())
    os_list = sorted(list(set(os_list)))

    # Hostname heuristic (alphanumeric with dashes/underscores and digits)
    host_candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text)
    # filter obvious generic words
    blacklist = {"User", "Hostname", "Host", "Alert", "Summary", "Steps", "MITRE", "Severity"}
    hostnames = [h for h in host_candidates if h not in blacklist and (any(c.isdigit() for c in h) or ("-" in h) or ("_" in h))]
    hostnames = sorted(list(set(hostnames)))

    # MITRE ATT&CK techniques
    mitre = sorted(list(set(re.findall(r"\bT\d{4}\b", text))))

    # Severity
    sev_match = re.findall(r"\b(Low|Medium|High|Critical)\b", text, flags=re.IGNORECASE)
    severity = sev_match[-1].capitalize() if sev_match else "Unknown"

    # Indicators/flags
    flags = []
    def flag_if(pattern, label):
        if re.search(pattern, text, flags=re.IGNORECASE):
            flags.append(label)
    flag_if(r"encoded\s*powershell|powershell\s*-enc|\-enc\s*", "encoded_powershell")
    flag_if(r"brute\s*force|failed\s*ssh|ssh\s*login\s*fail", "brute_force_ssh")
    flag_if(r"ransomware|crypto\s*locker", "ransomware")
    flag_if(r"credential\s*dump|mimikatz|lsass", "credential_dumping")
    flag_if(r"unauthorized\s*rdp|rdp\s*access", "unauthorized_rdp")
    flag_if(r"phishing|malicious\s*email|attachment", "phishing")
    flag_if(r"vpn\s*misuse|improper\s*vpn", "vpn_misuse")
    flag_if(r"sudo\s*misuse|irregular\s*sudo", "irregular_sudo")
    flag_if(r"suspicious\s*outbound|data\s*exfil|c2\s*traffic", "suspicious_outbound")

    # Simple malicious IP heuristic (TEST-NET ranges as demo)
    malicious_ip_present = any(ip.startswith(prefix) for ip in ips for prefix in ["203.0.113.", "198.51.100.", "192.0.2."])

    # Threat score computation
    base = {"Low": 10, "Medium": 30, "High": 60, "Critical": 85}
    score = base.get(severity, 20)
    if mitre:
        score += 15
    if malicious_ip_present:
        score += 20
    # Add per-flag risk
    score += min(30, 5 * len(flags))
    # Specific boosts
    if "brute_force_ssh" in flags:
        score += 10
    if "credential_dumping" in flags:
        score += 10
    score = max(0, min(100, score))

    return {
        "ips": ips,
        "os": os_list,
        "hostnames": hostnames,
        "mitre": mitre,
        "severity": severity,
        "flags": flags,
        "malicious_ip_present": malicious_ip_present,
        "threat_score": score,
    }

# ===========
# STEP 7: INTERACTIVE CONSOLE LOOP
# ===========
print("\n=== STEP 7: CONSOLE LOOP ===")
while True:
    raw = input("Enter: <analyst_id> <query> (or q): ")
    if raw == "q":
        break
    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Format: analyst42 suspicious ssh activity")
        continue
    analyst_id, query = parts

    # TODO: Retrieve entities from query or context
    # Retrieve hybrid context explicitly for display
    try:
        retrieved_docs = hybrid.get_relevant_documents(query)
    except Exception:
        retrieved_docs = []
    context_text = format_docs(retrieved_docs)
    entities = extract_entities(query, context_text)

    # TODO: Invoke the memory-wrapped RAG chain
    response = memory_chain.invoke({
        "question": query,
        # Note: entities/context are computed inside the chain for the prompt;
        # we still display our richer extraction below.
    }, config={"configurable": {"session_id": analyst_id}})

    # Optional: tool-based resolution using threat_enrich for each IP found
    intel_results = []
    for ip in entities.get("ips", []):
        try:
            intel = threat_enrich.invoke(ip)
        except Exception:
            # fallback to direct call
            intel = threat_enrich.func(ip) if hasattr(threat_enrich, "func") else json.dumps({"ip": ip, "error": "tool call failed"})
        intel_results.append(json.loads(intel) if isinstance(intel, str) else intel)

    # TODO: Print the response
    print("\n--- Retrieved Context (RAG) ---\n", context_text)

    # Injected user memory (show last few messages)
    history = get_session_history(analyst_id)
    msgs_preview = []
    for m in history.messages[-6:]:  # last 6 messages
        role = "user" if isinstance(m, HumanMessage) else ("assistant" if isinstance(m, AIMessage) else "other")
        msgs_preview.append({"role": role, "content": m.content})
    print("\n--- Injected User Memory (last turns) ---\n", json.dumps(msgs_preview, indent=2))

    # Injected entity memory
    print("\n--- Injected Entity Memory ---\n", json.dumps(entities, indent=2))

    # Final LLM answer
    print("\n--- Final LLM Answer ---\n", response)

    # Optional tool-based enrichment
    if intel_results:
        print("\n--- Threat Enrichment (tool) ---\n", json.dumps(intel_results, indent=2))

    # Structured Output (JSON)
    structured = {
        "analyst_id": analyst_id,
        "query": query,
        "retrieved_context": context_text,
        "user_memory": msgs_preview,
        "entity_memory": entities,
        "threat_enrichment": intel_results,
        "final_answer": response,
    }
    print("\n=== Structured Output ===\n" + json.dumps(structured, indent=2) + "\n")