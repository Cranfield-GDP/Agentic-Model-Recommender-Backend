from langgraph.graph import StateGraph, END


from agent.nodes import (deployment_confirmation_agent_node, 
                         hugging_face_model_search_agent_node, 
                         requirement_analyis_agent_node, 
                         requirement_clarification_agent_node, user_confirmation_reviewer_node,
                         )

from agent.schema.schema import Agents, GraphState, VariableStore



from agent.memory import (get_saved_variables,
                    memory_store)


graph = StateGraph(GraphState)


graph.add_node(Agents.RequirementAnalysisAgent.value, requirement_analyis_agent_node)
graph.add_node(Agents.RequirementClarificationAgent.value, requirement_clarification_agent_node)
graph.add_node(Agents.DeploymentConfirmationAgent.value, deployment_confirmation_agent_node)
graph.add_node(Agents.HuggingFaceModelAgent.value, hugging_face_model_search_agent_node)
graph.add_node(Agents.UserConfirmationReviewer.value, user_confirmation_reviewer_node)

def router(state: GraphState):
    """Routes between agents based on agent1 decision"""
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
        

graph.set_entry_point(Agents.RequirementAnalysisAgent.value)
graph.add_conditional_edges(Agents.RequirementAnalysisAgent.value, router, {Agents.RequirementClarificationAgent.value, 
                                                                            Agents.DeploymentConfirmationAgent.value,
                                                                            Agents.HuggingFaceModelAgent.value})

app = graph.compile()