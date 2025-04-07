from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class MoreInfoResponse(BaseModel):
    """
    Requirement Analysis agent output
    """
    isInfoEnoughToMakeDecision:bool = Field(description="Boolean indicating if enough info is available")
    deployment: Optional[str] = Field(None, description="Deployment choice (Edge/Cloud), empty if decision cannot be made")
    networkSlice: Optional[str] = Field(None, description="The most suitable network slice between (eMBB, uRLLC, mMTC), empty if decision cannot be made")

requirement_analysis_agent_parser = PydanticOutputParser(pydantic_object=MoreInfoResponse)
requirement_analyser_agent_format_instructions = requirement_analysis_agent_parser.get_format_instructions()

class Question(BaseModel):
    """
    Requirement Clarification agent output
    """
    question: str = Field(description="The question text")
    rationale: str = Field(description="Why this question is critical")

class Questions(BaseModel):
    """
    Requirement Clarification agent output
    """
    questions: List[Question]


requirement_clarification_agent_parser = PydanticOutputParser(pydantic_object=Questions)
requirement_clarification_agent_format_instruction = requirement_clarification_agent_parser.get_format_instructions()

class UserConfirmationReviewer(BaseModel):
    """
    Format to check if the user has confirmed the deployment, model, and network slice.
    """
    isConfirmed: bool = Field(
        description="Indicates whether the user has confirmed the selected model, deployment, and network slice."
    )
    selected_model: str = Field(
        description="The model selected by the user."
    )
    selected_deployment: str = Field(
        description="The selected deployment type, either 'cloud' or 'edge'."
    )
    selected_slice: str = Field(
        description="The selected network slice, which must be one of ['uRLLC', 'mMTC', 'eMBB']."
    )
    description: str = Field(
        description="Provides clarification on the selected deployment and network slice or confirms that deployment is proceeding."
    )

class Message(BaseModel):
    """
    confirmation agent output
    """
    message: str = Field(description="Summary of decision and confirmation request")

deployment_confirmation_agent_parser = PydanticOutputParser(pydantic_object=UserConfirmationReviewer)
deployment_confirmation_agent_format_instructions = deployment_confirmation_agent_parser.get_format_instructions()
 

def getAgentResponse(response: BaseModel):
    schema = response.model_json_schema()["properties"]
    return {
            field: {
                "value": getattr(response, field),  
                "description": schema[field].get("description", "No description available")
            }
            for field in response.model_fields
        }

class HuggingFaceModel(BaseModel):
    """
    HuggingFace agent output template
    """
    category: Optional[str] = Field(description="The most suitable Hugging Face category that best matches the user's requirement, or null if insufficient information")
    rationale: Optional[str] = Field(description="A brief explanation of why the selected category is the best fit, or null if insufficient information")
    is_enough_info_available_for_model_selection: bool = Field(description="True if the input is well defined for selecting a category, false otherwise.")
    clarification: Optional[str] = Field(description="If additional information is needed, specify what further details you require from the user; otherwise, return `None`")

hugging_face_agent_parser = PydanticOutputParser(pydantic_object=HuggingFaceModel)
hugging_face_agent_format_instructions = hugging_face_agent_parser.get_format_instructions()

class LatencyAgentModel(BaseModel):
    latency: Optional[str] = Field(description="The user's provided latency value if valid, otherwise null")
    suggestion: Optional[str] = Field(description="If latency is null, provide a recommendation explaining why a specific latency is important based on 5QI; otherwise, leave this empty")

latency_agent_parser = PydanticOutputParser(pydantic_object=LatencyAgentModel)
latency_agent_format_instruction = latency_agent_parser.get_format_instructions()

class GraphState(BaseModel):
    user_id: str
    user_chat: str
    requirement_analyser_agent_result: Optional[MoreInfoResponse] = None
    requirement_clarification_agent_result: Optional[Questions] = None
    deployment_confirmation_agent_result: Optional[UserConfirmationReviewer] = None
    hugging_face_models: Optional[List[str]] | Optional[str] = None
    latency_agent_result: Optional[LatencyAgentModel] = None


class VariableStore(Enum):
    IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION = "IS_ENOUGH_INFO_AVAILABLE_TO_MAKE_DECISION"
    DEPLOYMENT = "DEPLOYMENT"
    NETWORK_SLICE = "NETWORK_SLICE"
    SUGGESTED_MODELS = "SUGGESTED_MODELS"
    SELECTED_MODEL = "SELECTED_MODEL"
    MODEL_CATEGORY = "MODEL_CATEGORY"
    IS_DEPLOYMENT_CONFIRMED = "IS_DEPLOYMENT_CONFIRMED"
    IS_REQUIREMENT_CLEAR = "IS_REQUIREMENT_CLEAR"
    LATENCY = "LATENCY"

class Agents(Enum):
    RequirementAnalysisAgent = "REQUIREMENT_ANALYSIS_AGENT"
    RequirementClarificationAgent = "REQUIREMENT_CLARIFICATION_AGENT"
    HuggingFaceModelAgent = "HUGGING_FACE_MODEL_AGENT"
    DeploymentConfirmationAgent = "DEPLOYMENT_CONFIRMATION_AGENT"
    DeployerAgent = "DEPLOYER_AGENT"
    UserConfirmationReviewer = "USER_CONFIRMATION_REVIEWER"
    LatencyAnayserAgent = "LATENCY_ANALYSER_AGENT"
    ClarificationSubgraph = "CLARIFICATION_SUBGRAPH"
