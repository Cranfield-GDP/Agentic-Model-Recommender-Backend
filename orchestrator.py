import uuid
from langgraph.checkpoint.memory import MemorySaver  # LangGraph's memory backend
from langgraph.graph import StateGraph, START, MessagesState

from langchain.agents import initialize_agent, Tool
from custom_llm import CustomLLM  # Our custom LLM class

from agents.telecom_requirement_agent import run as requirement_run
from agents.telecom_model_suggestion_agent import run as model_suggestion_run
from agents.telecom_qa_agent import run as qa_run
from agents.telecom_deployment_decision_agent import run as deployment_run
from agents.telecom_followup_agent import run as followup_run

from memory_wrapper import MemorySaverWrapper

raw_memory = MemorySaver()
memory = MemorySaverWrapper(raw_memory)

tools = [
    Tool(
        name="RequirementAnalysis",
        func=requirement_run,
        description="Extract and clarify telecom network requirements from the user input."
    ),
    Tool(
        name="ModelSuggestion",
        func=model_suggestion_run,
        description="Suggest telecom models based on the provided requirements."
    ),
    Tool(
        name="QA",
        func=qa_run,
        description="Answer follow-up questions regarding telecom requirements and suggestions."
    ),
    Tool(
        name="DeploymentDecision",
        func=deployment_run,
        description="Advise on deployment options (Cloud or MEC) based on user requirements."
    ),
    Tool(
        name="FollowUp",
        func=lambda _: followup_run(), 
        description="Summarize the conversation and ask if the user has further questions."
    )
]

custom_llm = CustomLLM()

agent = initialize_agent(
    tools,
    custom_llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6
)

def run_agent(user_input: str) -> str:
    return agent.invoke(user_input)

workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("agent_node", lambda state: run_agent(state["input"]))
workflow.add_edge(START, "agent_node")

app = workflow.compile(checkpointer=raw_memory)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}
