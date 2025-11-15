import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/ask/"

# Streamlit App UI
st.title("SOC Analyst Assistant  Chatbot")

st.write("""
    Welcome to the SOC Analyst Assistant  chatbot. 
    Please enter your user ID and ask your SOC Analyst Assistant question below.
""")

# Input fields for user ID and query
user_id = st.text_input("Enter your User ID")
query = st.text_area("Enter your Question")

# Button to send the query
if st.button("Ask"):
    if user_id and query:
        # Send a POST request to the FastAPI backend with the user ID and query
        try:
            # Prepare the data
            payload = {"user_id": user_id, "query": query}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                # Parse and display the answer from the response
                result = response.json()
                answer = result.get("answer")
                st.write(f"**Answer:** {answer}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter both User ID and a Question.")
