import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="SOC RAG UI", layout="wide")
st.title("SOC Analyst Assistant – RAG UI")

menu = st.sidebar.selectbox("Menu", ["Query", "Sessions", "Inspect History"])

def call_backend(analyst_id: str, query: str):
    payload = {"analyst_id": analyst_id, "query": query}
    try:
        r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=150)
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


# -----------------------------
#  SINGLE QUERY
# -----------------------------
if menu == "Query":
    st.header("Single Query")
    analyst_id = st.text_input("Analyst ID", value="analyst1")
    query = st.text_input("Query", value="suspicious powershell on ubuntu")

    if st.button("Send"):
        with st.spinner("Processing..."):
            res = call_backend(analyst_id, query)

        if res:
            st.subheader("Structured Output")
            st.json(res)

            if "retrieved_context" in res:
                st.subheader("Retrieved Context")
                st.code(res["retrieved_context"])

            if "entities" in res:
                st.subheader("Extracted Entities")
                st.json(res["entities"])

            if "threat_score" in res:
                st.metric("Threat Score", res["threat_score"])

            if "tool_enrichment" in res:
                st.subheader("Threat Intelligence (Tool)")
                st.code(res["tool_enrichment"])


# -----------------------------
#  SESSIONS
# -----------------------------
elif menu == "Sessions":
    st.header("Active Sessions")
    try:
        r = requests.get(f"{BACKEND_URL}/sessions")
        data = r.json()
        st.json(data)
    except Exception as e:
        st.error(f"Could not fetch sessions: {e}")


# -----------------------------
#  INSPECT HISTORY
# -----------------------------
elif menu == "Inspect History":
    st.header("Inspect Session History")

    analyst_id = st.text_input("Analyst ID", value="analyst1")

    if st.button("Fetch History"):
        try:
            r = requests.get(f"{BACKEND_URL}/history/{analyst_id}")
            data = r.json()
            st.subheader(f"History for {analyst_id}")
            st.json(data)
        except Exception as e:
            st.error(f"Could not fetch history: {e}")
