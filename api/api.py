from fastapi import FastAPI, HTTPException
import logging

from agent.nodes import deployment_confirmation_agent_node, save_model
from api.api_schema import SelectedModel, UserInput
from agent.schema.schema import GraphState
from agent.agents import app as agents

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Telecom Chatbot Backend is running."}


@app.post("/model")
async def select_model(selected_model: SelectedModel):
    save_model(selected_model.model, selected_model.user_id)
    initial_state = GraphState(user_id=selected_model.user_id, user_chat=selected_model.model)
    response = deployment_confirmation_agent_node(initial_state) 
    return {"message": response.deployment_confirmation_agent_result.description}

@app.post("/chat", 
          response_model_exclude_none=True)
async def chat(user_message: UserInput):
    log.info(f"Received request from user: {user_message.userId}, message: {user_message.message}")
    try:
        initial_state = GraphState(user_id=user_message.userId, user_chat=user_message.message)
        final_state = agents.invoke(initial_state)
        if "deployment_confirmation_agent_result" in final_state:
            return {"message":final_state['deployment_confirmation_agent_result'].description}
        elif "hugging_face_models" in final_state and  isinstance(final_state["hugging_face_models"], list):
            return {"models": final_state["hugging_face_models"]}
        elif "hugging_face_models" in final_state:
            return {"message": final_state["hugging_face_models"]}
        elif "requirement_clarification_agent_result" in final_state:
            return final_state["requirement_clarification_agent_result"]
        elif "latency_agent_result" in final_state and final_state["latency_agent_result"].latency is None:
            return final_state["latency_agent_result"]
        return final_state
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=500, detail=str(e))

