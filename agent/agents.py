from langgraph.graph import StateGraph, END
import logging

from agent.nodes import (deployment_confirmation_agent_node, 
                         hugging_face_model_search_agent_node, 
                         latency_gathering_node, 
                        user_confirmation_reviewer_node,
                         )

from agent.schema.schema import Agents, GraphState, VariableStore

from agent.subgraph import clarification_subgraph

from agent.memory import (get_saved_variables,
                    memory_store)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

graph = StateGraph(GraphState)


graph.add_node(Agents.ClarificationSubgraph.value, clarification_subgraph)
graph.add_node(Agents.LatencyAnayserAgent.value, latency_gathering_node)

def router(state: GraphState):
    """Routes between agents based on agent1 decision"""
    saved_variables = get_saved_variables(
        state.user_id,
        memortStore=memory_store
    )
    log.info("The saved varibles are fetched for the user %s", saved_variables)
    if saved_variables[VariableStore.LATENCY.name]:
        return Agents.ClarificationSubgraph.value
    return Agents.ClarificationSubgraph.value
        

graph.set_entry_point(Agents.LatencyAnayserAgent.value)
graph.add_conditional_edges(Agents.LatencyAnayserAgent.value, router, {Agents.ClarificationSubgraph.value})

app = graph.compile()