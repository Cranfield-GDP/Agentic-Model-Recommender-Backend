from langgraph.graph import StateGraph, END

from agent.memory import get_saved_variables, memory_store
from agent.nodes import deployment_confirmation_agent_node, hugging_face_model_search_agent_node, requirement_analyis_agent_node, requirement_clarification_agent_node, user_confirmation_reviewer_node
from agent.schema.schema import Agents, GraphState, VariableStore

subgraph = StateGraph(GraphState)

subgraph.add_node(Agents.HuggingFaceModelAgent.value, hugging_face_model_search_agent_node)
subgraph.add_node(Agents.DeploymentConfirmationAgent.value, deployment_confirmation_agent_node)
subgraph.add_node(Agents.UserConfirmationReviewer.value, user_confirmation_reviewer_node)
subgraph.add_node(Agents.RequirementAnalysisAgent.value, requirement_analyis_agent_node)
subgraph.add_node(Agents.RequirementClarificationAgent.value,  requirement_clarification_agent_node)

subgraph.set_entry_point(Agents.RequirementAnalysisAgent.value)

def sub_router(state: GraphState):
    saved_variables = get_saved_variables(
        state.user_id,
        memortStore=memory_store
    )
    if saved_variables[VariableStore.SELECTED_MODEL.name]:
        return Agents.UserConfirmationReviewer.value
    if saved_variables[VariableStore.IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION.name]:
        return Agents.HuggingFaceModelAgent.value
    if not saved_variables[VariableStore.IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION.name]:
        return Agents.RequirementClarificationAgent.value
    return END
    

subgraph.add_conditional_edges(Agents.RequirementAnalysisAgent.value, sub_router, {
    Agents.RequirementClarificationAgent.value,
    
})

clarification_subgraph = subgraph.compile()