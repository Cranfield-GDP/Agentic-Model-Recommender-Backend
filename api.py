from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from orchestrator import run_agent, memory
import logging


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = FastAPI()


class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(user_message: UserMessage):
    log.info(f"Received message: {user_message.message}")
    try:
        response = run_agent(user_message.message)
        return {"response": response, "chat_history": memory.load_memory_variables({})}
    except Exception as e:
        log.error(f"Error in chat_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Telecom Chatbot Backend is running."}
