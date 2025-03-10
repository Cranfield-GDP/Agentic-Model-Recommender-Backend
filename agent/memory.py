from langgraph.store.memory import InMemoryStore

from agent.schema.schema import GraphState, VariableStore



memory_store = InMemoryStore()

chat_namespace = lambda user_id : (user_id, "chat")
decision_variable_namespace = lambda user_id : (user_id,"decision_variable")

def save_variable(key: VariableStore, value: str, user_id: str, memoryStore: InMemoryStore):
    "Save a variable to the namespace"
    namespace = decision_variable_namespace(user_id)
    messages = memory_store.search(namespace)
    if len(messages) == 0:
        messages.append({i.name : i.value for i in VariableStore})
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
        return {i.name : i.value for i in VariableStore}
    return messages[0].value  

def load_user_state(user_id: str, memory_store: dict) -> GraphState:
    namespace = ("memories", user_id)
    stored_state = memory_store.get(namespace, "user_state").value if memory_store.get(namespace, "user_state") else GraphState(user_id=user_id, user_chat="")
    return stored_state

def save_user_state(user_id: str, state: dict, memory_store: dict):
    """Save the updated state back to memory."""
    namespace = ("memories", user_id)
    memory_store.put(namespace, "user_state", state)