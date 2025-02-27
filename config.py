import os
from dotenv import load_dotenv
import logging


log = logging.getLogger(__name__)

class Config:
    def __init__(self, llm_provider:str):
        if llm_provider.lower() == "local":
            self.endpoint = os.getenv("LOCAL_ENDPOINT")
            self.apiKey = None
        elif llm_provider.lower() == "chatgpt":
            self.endpoint = os.getenv("CHATGPT_ENDPOINT")
            self.apiKey = os.getenv("CHATGPT_API_KEY")
        elif llm_provider.lower() == "gemini":
            self.endpoint = os.getenv("GEMINI_ENDPOINT")
            self.apiKey = os.getenv("GEMINI_API_KEY")
        else:
            log.warning(f"Invalid Argument {llm_provider} : usage - local / chatgpt / gemini ")
            raise Exception(f"Invalid Argument {llm_provider} : usage - local / chatgpt / gemini ")
    
        self.fastApiHost = os.getenv("FASTAPI_HOST", "0.0.0.0")
        self.fastApiPort = int(os.getenv("FASTAPI_PORT", "8000"))
