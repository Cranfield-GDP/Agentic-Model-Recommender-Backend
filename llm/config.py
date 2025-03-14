import os
import logging


log = logging.getLogger(__name__)

class Config:
    def __init__(self, llm_provider:str):
        if llm_provider.lower() in ["local", "deepseek", "chatgpt", "gemini", "openai"]:
            self.model = llm_provider
        else:
            log.warning(f"Invalid Argument {llm_provider} : usage - local / chatgpt / gemini ")
            raise Exception(f"Invalid Argument {llm_provider} : usage - local / chatgpt / gemini ")
    
        self.fastApiHost = os.getenv("FASTAPI_HOST", "0.0.0.0")
        self.fastApiPort = int(os.getenv("FASTAPI_PORT", "8000"))