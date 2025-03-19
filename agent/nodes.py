import asyncio
from typing import List
from langgraph.store.memory import InMemoryStore
from huggingface_hub import list_models
import logging
from agent.memory import (delete_user_conversation, get_saved_variables, 
                          memory_store, 
                          chat_namespace, 
                          save_variable)
from agent.schema.schema import (GraphState, HuggingFaceModel, MoreInfoResponse, UserConfirmationReviewer, VariableStore, 
                    requirement_analyser_agent_format_instructions, 
                    requirement_analysis_agent_parser, 
                    requirement_clarification_agent_parser, 
                    requirement_clarification_agent_format_instruction, 
                    deployment_confirmation_agent_format_instructions, 
                    deployment_confirmation_agent_parser, 
                    getAgentResponse,
                    hugging_face_agent_format_instructions,
                    hugging_face_agent_parser,
                    Agents)


from llm.llm_provider import llm
from agent.template import (requirement_analysis_agent_template, 
                      requirement_clarification_agent_template, 
                      confirmation_agent_template,
                      hugging_face_model_search_agent_template,
                      user_confirmation_reviewer_template)

from agent.hugging_face_tasks import get_huggingface_categories
from utils.deployer import deploy_model_in_cloud
import os

cloud_cost = os.getenv("COST_CLOUD", "3.99£/hr")
edge_cost = os.getenv("COST_EDGE", "6.00£/hr")


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_llm_response(prompt, parser):
    """Formats prompt, calls LLM, and parses the structured output."""
    response = llm(prompt)
    return parser.parse(response.content)

def get_past_messages(memory_store:InMemoryStore, namespace: tuple) -> List[str]:
    """Retrieve conversation history from memory store"""
    messages = memory_store.search(namespace)
    return [m.value for m in messages] 

def requirement_analyis_agent_node(state: GraphState) -> GraphState:
    saved_variables = get_saved_variables(state.user_id, memory_store)
    if saved_variables[VariableStore.IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION.name]:
        print("requirement is already confirmed")
        return state
    namespace = chat_namespace(state.user_id)
    past_messages = get_past_messages(memory_store, namespace)

    result: MoreInfoResponse = get_llm_response(
        requirement_analysis_agent_template.format_messages(
            user_chat=state.user_chat,
            format_instructions=requirement_analyser_agent_format_instructions,
            history=past_messages if len(past_messages) > 0 else "No Past conversation History"
        ),
        requirement_analysis_agent_parser
    )
    past_messages.append({"user": state.user_chat, Agents.RequirementAnalysisAgent.value: getAgentResponse(result)})
    memory_store.put(namespace, "latest", past_messages)
    save_variable(VariableStore.IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION, result.isInfoEnoughToMakeDecision, state.user_id, memory_store)
    if result.isInfoEnoughToMakeDecision:
        save_variable(VariableStore.DEPLOYMENT, result.deployment, state.user_id, memory_store)
        save_variable(VariableStore.NETWORK_SLICE, result.networkSlice, state.user_id, memory_store)
    return state

def requirement_clarification_agent_node(state: GraphState) -> GraphState:
    namespace = chat_namespace(state.user_id)
    past_messages = get_past_messages(memory_store,namespace)
    result = get_llm_response(
        requirement_clarification_agent_template.format_messages(
            user_chat=state.user_chat,
            format_instructions=requirement_clarification_agent_format_instruction,
            history=past_messages 
        ),
        requirement_clarification_agent_parser
    )
    past_messages.append({"user": state.user_chat, Agents.RequirementClarificationAgent.value: getAgentResponse(result)})
    memory_store.put(namespace, "latest", past_messages)
    state.requirement_clarification_agent_result = result
    return state

def deployment_confirmation_agent_node(state: GraphState) -> GraphState:
    """Runs Agent3 if enough information is available."""
    namespace = chat_namespace(state.user_id)
    past_messages = get_past_messages(memory_store,namespace)
    saved_variable = get_saved_variables(state.user_id, memory_store)
    result = get_llm_response(
        user_confirmation_reviewer_template.format_messages(
            user_chat=state.user_chat,
            deployment=saved_variable[VariableStore.DEPLOYMENT.name],
            network_slice=saved_variable[VariableStore.NETWORK_SLICE.name],
            model= saved_variable[VariableStore.SELECTED_MODEL.name],
            cloud_cost= cloud_cost,
            edge_cost= edge_cost,
            format_instructions=deployment_confirmation_agent_format_instructions,
            history=past_messages
        ),
        deployment_confirmation_agent_parser
    )
    past_messages.append({"user": state.user_chat, Agents.DeploymentConfirmationAgent.value: getAgentResponse(result)})
    memory_store.put(namespace, "latest", past_messages)
    state.deployment_confirmation_agent_result = result
    return state

def hugging_face_model_search_agent_node(state: GraphState) -> GraphState:
    """Runs the Hugging Face Model Search agent and stores results in memory."""
    namespace = chat_namespace(state.user_id)
    past_messages = get_past_messages(memory_store,namespace)

    tasks = get_huggingface_categories()

    result: HuggingFaceModel = get_llm_response(hugging_face_model_search_agent_template.format_messages(
            user_chat=state.user_chat,
            format_instructions=hugging_face_agent_format_instructions,
            history=past_messages,
            categories=tasks
        ), hugging_face_agent_parser)
    
    print(result)
    past_messages.append({"user": state.user_chat, Agents.HuggingFaceModelAgent.value: getAgentResponse(result)})
    memory_store.put(namespace, "latest", past_messages)
    if result.is_enough_info_available_for_model_selection:
        models = [{"model": model.id, "modelUrl": f"https://huggingface.co/{model.id}"} for model in list_models(task=result.category, sort="likes", limit=5, direction=-1)]
        state.hugging_face_models = models
        save_variable(VariableStore.IS_REQUIREMENT_CLEAR, result.is_enough_info_available_for_model_selection, state.user_id, memory_store)
        save_variable(VariableStore.MODEL_CATEGORY,result.category, state.user_id, memory_store)
        save_variable(VariableStore.SUGGESTED_MODELS, models, state.user_id, memory_store)
    else:
        state.hugging_face_models = result.clarification
    return state


def user_confirmation_reviewer_node(state: GraphState) -> GraphState:
    """verifies if the user has confirmed the deployment and provide further confirmation to deploy or clarify or even update the user selection"""
    namespace = chat_namespace(state.user_id)
    past_messages = get_past_messages(memory_store, namespace)
    saved_variable = get_saved_variables(state.user_id, memory_store)
    result: UserConfirmationReviewer = get_llm_response(
        confirmation_agent_template.format_messages(
            user_chat=state.user_chat,
            model = saved_variable[VariableStore.SELECTED_MODEL.name],
            deployment=saved_variable[VariableStore.DEPLOYMENT.name],
            network_slice=saved_variable[VariableStore.NETWORK_SLICE.name],
            cloud_cost= cloud_cost,
            edge_cost= edge_cost,
            format_instructions=deployment_confirmation_agent_format_instructions,
            history=past_messages
        ),
        deployment_confirmation_agent_parser
    )
    past_messages.append({"user": state.user_chat, Agents.DeploymentConfirmationAgent.value: getAgentResponse(result)})
    memory_store.put(namespace, "latest", past_messages)
    state.deployment_confirmation_agent_result = result
    if result.isConfirmed:
        print("The user has confirmed the deployment: ", result)
        asyncio.create_task(deploy_model_in_cloud(result.selected_model, 
                              saved_variable[VariableStore.MODEL_CATEGORY.name],
                              result.selected_deployment, result.selected_slice))
        delete_user_conversation(state.user_id)
        print("The variables and the chat_history for the user are deleted successfully")
    return state

def save_model(model: str, userId: str):
    save_variable(VariableStore.SELECTED_MODEL, model, userId, memory_store)