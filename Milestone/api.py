from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fast import process_query, get_user_memory  # Importing the functions from dhiraj.py

# Initialize FastAPI app
app = FastAPI()

# Define Pydantic model for input validation
class QueryRequest(BaseModel):
    user_id: str
    query: str

class HistoryRequest(BaseModel):
    user_id: str

# FastAPI endpoint for answering queries
@app.post("/ask/")
async def ask_query(request: QueryRequest):
    try:
        user_id = request.user_id
        query = request.query

        # Invoke the RAG chain with user memory based on user_id
        response = process_query(user_id, query)

        # Return the response in the desired format
        return JSONResponse(content={"answer": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint to get chat history based on user_id
@app.post("/history/")
async def get_history(request: HistoryRequest):
    try:
        user_id = request.user_id

        # Retrieve user memory and return the chat history
        user_memory = get_user_memory(user_id)

        # Return the chat history as a list of messages
        # We need to access the 'content' of the HumanMessage or AIMessage objects
        history = [{"role": msg.__class__.__name__, "message": msg.content} for msg in user_memory.messages]
        return JSONResponse(content={"history": history})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the IT Support API. Use /ask/ endpoint to ask questions, and /history/ to retrieve chat history."}

