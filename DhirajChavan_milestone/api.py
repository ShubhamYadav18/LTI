from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dhiraj import process_query, get_user_memory

app = FastAPI()

class QueryRequest(BaseModel):
    user_id: str
    query: str

class HistoryRequest(BaseModel):
    user_id: str

@app.post("/ask/")
async def ask_query(request: QueryRequest):
    try:
        user_id = request.user_id
        query = request.query

        # Process the query using the cached responses and RAG chain
        response = process_query(user_id, query)

        return JSONResponse(content={"answer": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/history/")
async def get_history(request: HistoryRequest):
    try:
        user_id = request.user_id
        user_memory = get_user_memory(user_id)

        # Return the chat history
        history = [{"role": msg.__class__.__name__, "message": msg.content} for msg in user_memory.messages]
        return JSONResponse(content={"history": history})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the SOC Analyst Assistant API. Use /ask/ endpoint to ask questions."}
