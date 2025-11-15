# backend/app.py
"""
FastAPI backend for SOC RAG Assistant.
Place `security_incidents.txt` in same folder or set DATA_PATH variable.
Run:
  pip install -r requirements.txt
  uvicorn backend.app:app --reload --port 8000
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Replace these imports with exact packages in your environment ---
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
    from langchain_ollama import ChatOllama
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
except Exception as e:
    # If your environment uses different import paths, update them as needed.
    raise

# -----------------------
# Config
# -----------------------
DATA_PATH = "security_incidents.txt"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
HF_MODEL = "all-MiniLM-L6-v2"
K = 4

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="SOC RAG Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Simple request/response models
# -----------------------
class QueryRequest(BaseModel):
    analyst_id: str
    query: str

class ResolveKBRequest(BaseModel):
    analyst_id: str
    query: str

# -----------------------
# Utilities & core logic (adapted)
# -----------------------
def normalize_session_id(session_id: str) -> str:
    return session_id.strip().lower()

IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
MITRE_REGEX = re.compile(r"\bT\d{4}\b", re.IGNORECASE)
SEVERITY_REGEX = re.compile(r"\b(low|medium|high|critical)\b", re.IGNORECASE)
HOST_REGEX = re.compile(r"\b([a-zA-Z0-9_\-]{3,}\-[A-Za-z0-9_\-]{1,})\b")
OS_KEYWORDS = ["windows", "ubuntu", "centos", "debian", "redhat", "macos", "osx"]

def extract_entities_from_text(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    ips = list(dict.fromkeys(IP_REGEX.findall(text)))
    mitres = list(dict.fromkeys([m.upper() for m in MITRE_REGEX.findall(text)]))
    severities = list(dict.fromkeys([s.capitalize() for s in SEVERITY_REGEX.findall(text)]))
    hosts = []
    for m in HOST_REGEX.findall(text):
        if len(m) >= 3 and not re.fullmatch(r"\d{2,}", m):
            hosts.append(m)
    hosts = list(dict.fromkeys(hosts))
    os_found = []
    lower = text.lower()
    for os_kw in OS_KEYWORDS:
        if os_kw in lower:
            os_found.append(os_kw.capitalize())
    return {
        "ip": ips[0] if ips else None,
        "ips": ips,
        "host": hosts[0] if hosts else None,
        "hosts": hosts,
        "os": os_found[0] if os_found else None,
        "oses": os_found,
        "mitre": mitres[0] if mitres else None,
        "mitres": mitres,
        "severity": severities[0] if severities else None,
        "severities": severities,
    }

def threat_enrich(ip: str) -> str:
    if not ip:
        return "No IP provided."
    if ip.startswith("10.") or ip.startswith("192.168."):
        return f"IP {ip} appears to be internal/private. No malicious history in mock DB."
    if ip.startswith("8.8.8.8"):
        return f"IP {ip} is a public DNS resolver (Google). Likely benign."
    return f"IP {ip} flagged in mock-threat-db: previous brute-force attempts. Recommend block."

def compute_threat_score(entities: Dict[str, Any], query_text: str, retrieved_context: str) -> int:
    score = 0
    sev = (entities.get("severity") or "").lower() if entities.get("severity") else ""
    if sev == "critical":
        score += 40
    elif sev == "high":
        score += 25
    elif sev == "medium":
        score += 10
    if entities.get("mitres"):
        score += 10
    if entities.get("ips"):
        for ip in entities.get("ips", []):
            if not (ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172.")):
                score += 15
                break
    txt = (query_text or "") + " " + (retrieved_context or "")
    if re.search(r"powershell|encoded|base64", txt, re.IGNORECASE):
        score += 15
    if re.search(r"brute|failed login|ssh|multiple attempts", txt, re.IGNORECASE):
        score += 15
    if score > 100:
        score = 100
    return int(score)

# -----------------------
# RAG system state (loaded once)
# -----------------------
_docs = []
_splits = []
_retrievers = {}
_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_or_create_history(session_id: str) -> InMemoryChatMessageHistory:
    sid = normalize_session_id(session_id)
    if sid not in _store:
        _store[sid] = InMemoryChatMessageHistory()
    return _store[sid]

def get_or_load_dataset(path=DATA_PATH):
    global _docs, _splits, _retrievers
    if _docs:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    loader = TextLoader(path)
    _docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    _splits = splitter.split_documents(_docs)
    # embeddings & stores
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL)
    vectorstore = FAISS.from_documents(_splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(k=K)
    bm25_retriever = BM25Retriever.from_documents(_splits)
    hybrid_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.7, 0.3])
    _retrievers = {
        "vectorstore": vectorstore,
        "faiss_retriever": faiss_retriever,
        "bm25_retriever": bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
    }

def build_context_from_docs(docs, max_docs=5, max_chars_per_doc=1500):
    doc_texts = []
    for d in docs[:max_docs]:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + " ...[truncated]"
        doc_texts.append(text)
    return "\n\n".join(doc_texts) if doc_texts else "No relevant context found."

def fetch_relevant_documents(retriever, query):
    for method_name in ("get_relevant_documents", "get_documents", "retrieve", "get_relevant_documents_for_query"):
        fn = getattr(retriever, method_name, None)
        if callable(fn):
            try:
                return fn(query)
            except Exception:
                continue
    try:
        return retriever(query)
    except Exception:
        return []

# -----------------------
# LLM (wrap)
# -----------------------
_llm = ChatOllama(model="mistral", temperature=0.0)

def call_llm(prompt_text: str) -> str:
    # basic wrapper — tune based on your environment
    try:
        out = _llm.invoke(prompt_text)
        return str(out)
    except Exception:
        try:
            return str(_llm(prompt_text))
        except Exception as e:
            # final fallback: raise
            raise

# -----------------------
# KB (mock)
# -----------------------
def kb_lookup(query: str) -> str:
    q = query.lower()
    if "ssh" in q or "brute" in q:
        return "Steps: Check auth logs, block IP, force password reset, enable fail2ban."
    if "powershell" in q or "encoded" in q:
        return "Steps: Quarantine host, collect PowerShell command, inspect for base64/diff, revert malicious changes."
    if "ransomware" in q:
        return "Steps: isolate host, preserve logs, contact IR, restore from backups after containment."
    return ""

# -----------------------
# API startup endpoint (load dataset)
# -----------------------
@app.on_event("startup")
def startup_event():
    get_or_load_dataset(DATA_PATH)

@app.get("/status")
def status():
    loaded = len(_docs)
    chunks = len(_splits)
    return {"status": "ok", "documents_loaded": loaded, "chunks": chunks}

@app.get("/sample")
def sample():
    if not _docs:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    sample_text = _docs[0].page_content[:800]
    return {"sample": sample_text, "count": len(_docs)}

@app.get("/sessions")
def list_sessions():
    return {"sessions": list(_store.keys())}

@app.get("/history/{analyst_id}")
def get_history(analyst_id: str):
    sid = normalize_session_id(analyst_id)
    hist = _store.get(sid)
    if not hist:
        return {"history": []}
    msgs = getattr(hist, "messages", None) or getattr(hist, "get_messages", lambda: [])()
    # convert to serializable
    out = []
    for m in msgs:
        content = getattr(m, "content", getattr(m, "text", str(m)))
        role = getattr(m, "type", getattr(m, "role", None))
        out.append({"role": role, "content": content})
    return {"history": out}

@app.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.analyst_id or not req.query:
        raise HTTPException(status_code=400, detail="analyst_id and query required.")
    sid = normalize_session_id(req.analyst_id)
    hist_obj = get_or_create_history(sid)

    # 1) retrieve
    docs = fetch_relevant_documents(_retrievers["hybrid_retriever"], req.query) or []
    context = build_context_from_docs(docs, max_docs=4)

    # 2) entity extraction
    ents_q = extract_entities_from_text(req.query)
    ents_ctx = extract_entities_from_text(context)
    merged = {}
    for key in ("ip", "ips", "host", "hosts", "os", "oses", "mitre", "mitres", "severity", "severities"):
        vq = ents_q.get(key)
        vc = ents_ctx.get(key)
        if isinstance(vq, list):
            merged[key] = (vq or []) + ([x for x in (vc or []) if x not in (vq or [])])
        else:
            merged[key] = vq or vc

    # 3) KB suggestions
    kb_steps = kb_lookup(req.query) or "No KB suggestions."

    # 4) threat enrichment
    ip_for_tool = merged.get("ip") or (merged.get("ips") and merged.get("ips")[0]) or None
    tool_report = threat_enrich(ip_for_tool) if ip_for_tool else "No IP to enrich."

    # 5) threat score
    tscore = compute_threat_score(merged, req.query, context)

    # 6) form prompt for LLM
    prompt_text = (
        "You are a SOC incident assistant.\n\n"
        f"History: (not injected in prompt for brevity)\n"
        f"Entities: {json.dumps(merged)}\n\n"
        f"KB Suggestions: {kb_steps}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Analyst Query:\n{req.query}\n\n"
        "Produce a JSON-like dict containing: answer, resolution_steps (if any), entities, threat_score.\n"
    )

    try:
        llm_out = call_llm(prompt_text)
    except Exception as e:
        llm_out = f"LLM call failed: {str(e)}"

    # try to parse JSON from LLM output
    out_struct = None
    try:
        start = llm_out.find("{")
        end = llm_out.rfind("}") + 1
        if start != -1 and end != -1 and end > start:
            candidate = llm_out[start:end]
            out_struct = json.loads(candidate)
    except Exception:
        out_struct = None

    if not out_struct:
        out_struct = {
            "answer": llm_out,
            "resolution_steps": kb_steps if kb_steps != "No KB suggestions." else "",
            "retrieved_context": context,
            "entities": merged,
            "threat_score": tscore,
            "tool_enrichment": tool_report,
        }

    # 7) append to history
    try:
        if hasattr(hist_obj, "add_user_message"):
            hist_obj.add_user_message(req.query)
            hist_obj.add_ai_message(json.dumps(out_struct))
        else:
            msgs = getattr(hist_obj, "messages", [])
            msgs.append(HumanMessage(content=req.query))
            msgs.append(AIMessage(content=json.dumps(out_struct)))
            try:
                hist_obj.messages = msgs
            except Exception:
                pass
    except Exception:
        pass

    return out_struct

@app.post("/resolve_kb")
def resolve_kb(req: ResolveKBRequest):
    kb = kb_lookup(req.query)
    if not kb:
        return {"resolved": False, "message": "No KB suggestions."}
    sid = normalize_session_id(req.analyst_id)
    hist = get_or_create_history(sid)
    # mark resolved
    resolution_note = f"Resolved using KB steps:\n{kb}"
    try:
        if hasattr(hist, "add_user_message"):
            hist.add_user_message(req.query)
            hist.add_ai_message(resolution_note)
        else:
            msgs = getattr(hist, "messages", [])
            msgs.append(HumanMessage(content=req.query))
            msgs.append(AIMessage(content=resolution_note))
            try:
                hist.messages = msgs
            except Exception:
                pass
    except Exception:
        pass
    return {"resolved": True, "resolution_note": resolution_note}
