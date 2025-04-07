from dataclasses import asdict
from langgraph.store.memory import InMemoryStore

from agent.schema.schema import GraphState, VariableStore



memory_store = InMemoryStore()

chat_namespace = lambda user_id : (user_id, "chat")
decision_variable_namespace = lambda user_id : (user_id,"decision_variable")

default_saved_variables = {
    VariableStore.IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION.name : False,
    VariableStore.DEPLOYMENT.name : None,
    VariableStore.NETWORK_SLICE.name : None,
    VariableStore.SUGGESTED_MODELS.name : [],
    VariableStore.SELECTED_MODEL.name : None,
    VariableStore.MODEL_CATEGORY.name : None,
    VariableStore.IS_DEPLOYMENT_CONFIRMED.name : False,
    VariableStore.IS_REQUIREMENT_CLEAR.name : False,
    VariableStore.LATENCY.name: None
}

def save_variable(key: VariableStore, value: str, user_id: str, memoryStore: InMemoryStore):
    "Save a variable to the namespace"
    namespace = decision_variable_namespace(user_id)
    messages = memory_store.search(namespace)
    if len(messages) == 0:
        messages.append({k: v for k, v in default_saved_variables.items()})        
        variables = messages[0]
    else:
        variables = messages[0].value
    variables[key.name] = value
    memory_store.put(namespace, "latest", variables)

def get_saved_variables(user_id: str, memortStore: InMemoryStore):
    "get a variable from the namespace"
    namespace = decision_variable_namespace(user_id)
    messages = memory_store.search(namespace)
    if len(messages) == 0:
        return default_saved_variables
    return messages[0].value  

def save_user_state(user_id: str, state: dict, memory_store: dict):
    """Save the updated state back to memory."""
    namespace = ("memories", user_id)
    memory_store.put(namespace, "user_state", state)

def _delete_namespace(store: InMemoryStore, namespace: tuple):
    items = store.search(namespace)
    for item in items:
        store.delete(namespace, item.key)
    

def delete_user_conversation(user_id: str):
    namespaces = [chat_namespace(user_id), decision_variable_namespace(user_id)]
    for namespace in namespaces:
        _delete_namespace(memory_store, namespace)

