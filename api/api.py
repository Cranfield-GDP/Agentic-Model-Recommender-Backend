from fastapi import FastAPI, HTTPException
import logging

from agent.nodes import deployment_confirmation_agent_node
from api.api_schema import ModelResponse, SelectedModel, UserInput
from agent.schema.schema import GraphState

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Telecom Chatbot Backend is running."}


@app.post("/model")
async def select_model(selected_model: SelectedModel):
    initial_state = GraphState(user_id=selected_model.userId, user_chat=selected_model.message)
    return deployment_confirmation_agent_node() 

@app.post("/chat", 
          response_model_exclude_none=True)
async def chat(user_message: UserInput):
    log.info(f"Received request from user: {user_message.userId}, message: {user_message.message} and the model: {user_message.model}")
    try:
        initial_state = GraphState(user_id=user_message.userId, user_chat=user_message.message)
        final_state = app.invoke(initial_state)
        if "deployment_confirmation_agent_result" in final_state:
            print("Final Response from Agent3:", final_state['deployment_confirmation_agent_result'])
        elif "hugging_face_models" in final_state:
            print("Recommended Hugging face model:", final_state["hugging_face_models"])
        elif "requirement_clarification_agent_result" in final_state:
            print("Additional Questions from Agent2:", final_state["requirement_clarification_agent_result"])
        return final_state
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=500, detail=str(e))

