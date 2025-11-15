"""
SOC Analyst Assistant - Incident RAG Investigation System
Run: python soc_rag_app.py
Dataset: security_incidents.txt
"""

# Step 1: Import required libraries with updated imports
import os
import re
import json
from typing import List, Dict, Any

# Updated LangChain imports for latest version
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✓ Using langchain-huggingface")
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("⚠️  Using deprecated embeddings. Install: pip install langchain-huggingface")
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama
    print("✓ Using langchain-ollama")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        print("⚠️  Using deprecated ChatOllama. Install: pip install langchain-ollama")
    except ImportError:
        from langchain.chat_models import ChatOllama

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

try:
    from langchain_core.runnables.history import RunnableWithMessageHistory
except ImportError:
    from langchain.runnables.history import RunnableWithMessageHistory

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory
except ImportError:
    try:
        from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
    except ImportError:
        from langchain.memory import ChatMessageHistory as InMemoryChatMessageHistory

# BM25 imports
try:
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

print("="*80)
print(" 🔐 SOC ANALYST ASSISTANT - INCIDENT RAG INVESTIGATION SYSTEM")
print("="*80)

# Step 2: Load the Dataset
print("\n[STEP 1] Loading security_incidents.txt...")
try:
    with open("security_incidents.txt", "r", encoding="utf-8") as f:
        incident_lines = f.readlines()
    
    print(f"✓ Loaded {len(incident_lines)} incident records")
    if incident_lines:
        sample = incident_lines[0][:150] + "..." if len(incident_lines[0]) > 150 else incident_lines[0]
        print(f"\nSample incident:\n{sample}")
except FileNotFoundError:
    print("❌ ERROR: security_incidents.txt not found!")
    print("Please create the file in the same directory as this script.")
    exit(1)

# Convert to Document format
documents = [Document(page_content=line.strip()) for line in incident_lines if line.strip()]
print(f"✓ Converted to {len(documents)} Document objects")

# Step 3: Chunking
print("\n[STEP 2] Chunking incidents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n", " | ", ". ", " "]
)

# Split documents
split_docs = text_splitter.split_documents(documents)
print(f"✓ Created {len(split_docs)} chunks (chunk_size=400, overlap=50)")

# Step 4: Embeddings + FAISS Vector Store
print("\n[STEP 3] Building FAISS vector store with all-MiniLM-L6-v2...")
print("(First run will download the model, please wait...)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✓ FAISS index created (k=4)")

# Step 5: BONUS - Hybrid Retrieval (FAISS + BM25)
if HYBRID_AVAILABLE:
    print("\n[STEP 4 - BONUS] Creating Hybrid Retriever (Vector 0.7 + BM25 0.3)...")
    try:
        bm25_retriever = BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = 4
        
        # Ensemble retriever with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        print("✓ Hybrid retriever ready (improves exact term matching for IPs/hosts)")
        USE_HYBRID = True
    except Exception as e:
        print(f"⚠️  Hybrid retriever failed: {e}")
        print("Using vector retriever only")
        ensemble_retriever = vector_retriever
        USE_HYBRID = False
else:
    print("\n[STEP 4] Using vector retriever only")
    print("  (Install 'rank-bm25' for bonus hybrid retrieval: pip install rank-bm25)")
    ensemble_retriever = vector_retriever
    USE_HYBRID = False

# Step 6: Initialize LLM (Ollama/Mistral)
print("\n[STEP 5] Initializing Ollama/Mistral LLM...")
try:
    llm = ChatOllama(model="mistral", temperature=0.3)
    print("✓ LLM ready")
except Exception as e:
    print(f"❌ ERROR: Could not connect to Ollama: {e}")
    print("Please ensure Ollama is running and Mistral is installed:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Pull Mistral: ollama pull mistral")
    exit(1)

# Step 7: Memory Setup
print("\n[STEP 6] Setting up user-specific memory...")
store = {}  # Chat history per analyst
entity_store = {}  # Entity memory per analyst

def get_session_history(analyst_id: str):
    """Retrieve or create chat history for analyst"""
    if analyst_id not in store:
        store[analyst_id] = InMemoryChatMessageHistory()
        entity_store[analyst_id] = {}
    return store[analyst_id]

# Step 8: BONUS - Entity Memory Extraction
def extract_entities(analyst_id: str, query: str, context: str, response: str) -> Dict[str, Any]:
    """Extract entities from query, context, and response (BONUS FEATURE)"""
    combined_text = f"{query} {context} {response}"
    entities = {}
    
    # Extract IP addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = re.findall(ip_pattern, combined_text)
    if ips:
        entities["IPs"] = list(set(ips))[:3]
    
    # Extract OS
    os_pattern = r'(Windows \d+|Windows Server \d+|Ubuntu \d+|CentOS \d+|RedHat \d+|Mac OS)'
    os_matches = re.findall(os_pattern, combined_text, re.IGNORECASE)
    if os_matches:
        entities["OS"] = list(set(os_matches))[:2]
    
    # Extract Hostname
    host_pattern = r'Host=([A-Za-z0-9\-]+)'
    hosts = re.findall(host_pattern, combined_text)
    if hosts:
        entities["Hostname"] = list(set(hosts))[:3]
    
    # Extract MITRE technique codes
    mitre_pattern = r'(T\d{4}(?:\.\d{3})?)'
    mitre_codes = re.findall(mitre_pattern, combined_text)
    if mitre_codes:
        entities["MITRE"] = list(set(mitre_codes))
    
    # Extract Severity
    severity_pattern = r'Severity=(Critical|High|Medium|Low)'
    severities = re.findall(severity_pattern, combined_text, re.IGNORECASE)
    if severities:
        entities["Severity"] = list(set(severities))
    
    entity_store[analyst_id] = entities
    return entities

def format_entities(analyst_id: str) -> str:
    """Format entity memory for prompt injection"""
    if analyst_id in entity_store and entity_store[analyst_id]:
        formatted = []
        for key, values in entity_store[analyst_id].items():
            if isinstance(values, list):
                formatted.append(f"{key}: {', '.join(map(str, values))}")
            else:
                formatted.append(f"{key}: {values}")
        return "\n🔍 Entity Memory:\n" + "\n".join(formatted)
    return ""

# Step 9: BONUS - Threat Enrichment Tool
def threat_enrichment_tool(query: str, entities: Dict[str, Any]) -> str:
    """Enrich threat intelligence based on query and entities (BONUS FEATURE)"""
    enrichment = ["🛡️ Threat Enrichment:"]
    
    # MITRE technique mapping
    if "MITRE" in entities:
        mitre_map = {
            "T1110": "Brute Force - Common credential attack, implement rate limiting & MFA",
            "T1059": "Command/Script Interpreter - Monitor PowerShell/bash execution",
            "T1486": "Data Encrypted for Impact (Ransomware) - CRITICAL: Isolate immediately",
            "T1568": "Dynamic Resolution - DNS tunneling possible, check egress traffic",
            "T1068": "Exploitation for Privilege Escalation - Patch systems urgently",
            "T1003": "OS Credential Dumping - Domain compromise likely, rotate credentials",
            "T1190": "Exploit Public-Facing Application - Web app vulnerability, apply WAF rules"
        }
        for mitre in entities["MITRE"]:
            base_code = mitre.split(".")[0]
            if base_code in mitre_map:
                enrichment.append(f"  • {mitre}: {mitre_map[base_code]}")
    
    # IP reputation check
    if "IPs" in entities:
        for ip in entities["IPs"]:
            if ip.startswith("10.") or ip.startswith("192.168."):
                enrichment.append(f"  • IP {ip}: Internal network (check for lateral movement)")
            else:
                enrichment.append(f"  • IP {ip}: External source (recommend block at perimeter)")
    
    # OS-specific recommendations
    if "OS" in entities:
        for os in entities["OS"]:
            if "Windows" in os:
                enrichment.append(f"  • {os}: Enable Windows Defender ATP, check Event IDs 4624/4625")
            elif "Ubuntu" in os or "CentOS" in os:
                enrichment.append(f"  • {os}: Review /var/log/auth.log, enable auditd")
    
    return "\n".join(enrichment) if len(enrichment) > 1 else "  No specific enrichment available"

# Step 10: BONUS - Threat Score Calculator
def calculate_threat_score(entities: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Calculate threat score based on multiple factors (BONUS FEATURE)"""
    score = 0
    factors = []
    
    # Severity scoring
    if "Severity" in entities:
        severity_scores = {"Critical": 40, "High": 30, "Medium": 15, "Low": 5}
        for sev in entities["Severity"]:
            score += severity_scores.get(sev, 0)
            factors.append(f"Severity={sev} (+{severity_scores.get(sev, 0)})")
    
    # MITRE technique scoring
    high_risk_mitre = ["T1486", "T1003", "T1068", "T1190", "T1021", "T1558"]
    if "MITRE" in entities:
        for mitre in entities["MITRE"]:
            if any(hr in mitre for hr in high_risk_mitre):
                score += 25
                factors.append(f"High-risk MITRE {mitre} (+25)")
            else:
                score += 10
                factors.append(f"MITRE {mitre} (+10)")
    
    # Malicious indicators
    if "PowerShell" in context or "encoded command" in context.lower():
        score += 15
        factors.append("PowerShell execution detected (+15)")
    
    if "brute-force" in context.lower() or "failed login" in context.lower():
        score += 10
        factors.append("Brute-force indicators (+10)")
    
    if "ransomware" in context.lower() or "encryption" in context.lower():
        score += 30
        factors.append("Ransomware behavior (+30)")
    
    # External IPs
    if "IPs" in entities:
        external_ips = [ip for ip in entities["IPs"] if not (ip.startswith("10.") or ip.startswith("192.168."))]
        if external_ips:
            score += 5 * len(external_ips)
            factors.append(f"{len(external_ips)} external IP(s) (+{5*len(external_ips)})")
    
    # Determine risk level
    if score >= 70:
        risk_level = "CRITICAL"
    elif score >= 50:
        risk_level = "HIGH"
    elif score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "threat_score": min(score, 100),
        "risk_level": risk_level,
        "factors": factors
    }

# Step 11: BONUS - Structured Output Parser
def parse_to_json(response: str, entities: Dict, threat_score: Dict) -> Dict[str, Any]:
    """Convert response to structured JSON format (BONUS FEATURE)"""
    return {
        "analyst_response": response,
        "extracted_entities": entities,
        "threat_assessment": threat_score,
        "timestamp": "2025-11-15T10:30:00Z"
    }

# Step 12: Build LCEL RAG Chain
print("\n[STEP 7] Building LCEL RAG chain...")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a SOC analyst assistant helping investigate security incidents.

📋 Retrieved Context from Past Incidents:
{context}

{entity_memory}

Provide actionable recommendations based on retrieved incidents. Reference MITRE techniques and specific resolution steps."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])

# LCEL Chain with FIXED retriever method
def retrieve_context(inputs):
    """Retrieve context using invoke method (updated API)"""
    # Use invoke() instead of get_relevant_documents()
    docs = ensemble_retriever.invoke(inputs["query"])
    return "\n\n".join([doc.page_content for doc in docs])

chain = (
    {
        "context": lambda x: retrieve_context(x),
        "query": lambda x: x["query"],
        "entity_memory": lambda x: format_entities(x["analyst_id"]),
        "history": lambda x: x.get("history", [])
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Wrap with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

print("✓ LCEL chain ready (Retrieve → Prompt → LLM → Parse)")

# Step 13: Console Interaction Loop
print("\n" + "="*80)
print(" 🚀 SOC RAG ASSISTANT READY")
print("="*80)
print("Commands: 'exit' to quit | 'clear <analyst_id>' to reset session")
print("="*80 + "\n")

while True:
    try:
        # Get analyst input
        analyst_id = input("👤 Analyst ID: ").strip()
        if analyst_id.lower() == "exit":
            print("\n✓ Exiting SOC RAG Assistant...")
            break
        
        if analyst_id.lower().startswith("clear"):
            clear_id = analyst_id.split()[1] if len(analyst_id.split()) > 1 else ""
            if clear_id in store:
                store.pop(clear_id)
                entity_store.pop(clear_id, None)
                print(f"✓ Session cleared for {clear_id}\n")
            continue
        
        query = input(f"🔍 Query: ").strip()
        if not query:
            continue
        
        print("\n" + "="*80)
        if USE_HYBRID:
            print("🔄 Processing with Hybrid RAG (Vector 0.7 + BM25 0.3)...")
        else:
            print("🔄 Processing with Vector RAG...")
        
        # Get context for entity extraction - FIXED METHOD
        context_docs = ensemble_retriever.invoke(query)
        context_text = "\n".join([doc.page_content for doc in context_docs])
        
        # Invoke RAG chain
        response = chain_with_history.invoke(
            {"query": query, "analyst_id": analyst_id},
            config={"configurable": {"session_id": analyst_id}}
        )
        
        # Extract entities (BONUS)
        entities = extract_entities(analyst_id, query, context_text, response)
        
        # Display RAG response
        print(f"\n🤖 SOC ANALYST RESPONSE:")
        print(response)
        
        # Show entity memory (BONUS)
        if entities:
            print("\n" + format_entities(analyst_id))
        
        # Threat enrichment (BONUS)
        if entities:
            enrichment = threat_enrichment_tool(query, entities)
            print(f"\n{enrichment}")
        
        # Threat score (BONUS)
        threat_score = calculate_threat_score(entities, context_text)
        print(f"\n⚠️ Threat Score: {threat_score['threat_score']}/100 (Risk Level: {threat_score['risk_level']})")
        if threat_score['factors']:
            print(f"Factors: {', '.join(threat_score['factors'][:3])}")
        
        # Structured output (BONUS)
        structured = parse_to_json(response, entities, threat_score)
        print(f"\n📊 Structured Output Available (JSON): {len(json.dumps(structured))} chars")
        
        # Show session info
        history = get_session_history(analyst_id)
        print(f"\n💬 Session: {len(history.messages)//2} exchanges for {analyst_id}")
        
    except KeyboardInterrupt:
        print("\n\n✓ Exiting SOC RAG Assistant...")
        break
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("="*80 + "\n")

print("\n✓ Stay vigilant! 🔐\n")