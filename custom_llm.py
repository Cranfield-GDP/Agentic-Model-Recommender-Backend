from langchain.llms.base import LLM
from typing import Optional, List

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        from llm_provider import llm 
        return llm(prompt)

    @property
    def _identifying_params(self) -> dict:
        return {}
