from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain.chat_models.base import BaseChatModel
import httpx
from typing import List, Optional
import os

class DeepseekFinetuned(BaseChatModel):
    """
    Custom Chat Model for self-hosted LLM API.
    """
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = os.getenv("LOCAL_ENDPOINT")
        self.api_key = os.getenv("LOCAL_API_KEY", None)

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> AIMessage:
        #headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "messages": [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                         for m in messages],
            "stop": stop,
            **kwargs
        }

        response = httpx.post(self.api_url, 
                              json=payload, 
                              #headers=headers, #ignoring authorization headers for now
                              timeout=30)
        response.raise_for_status()
        result = response.json()

        return AIMessage(content=result.get("response", ""))

    @property
    def _llm_type(self) -> str:
        """
        Returns the LLM type identifier.
        """
        return "deepseek"
