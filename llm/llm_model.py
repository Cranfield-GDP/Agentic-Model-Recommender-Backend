import logging
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from llm.deepseek_finetuned import DeepseekFinetuned


log = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, model: str):
        if not model:
            log.warning("Model is not present as environmental variable, so not processing")
            raise Exception("Model is not present as environmental variable, so not processing")
        if model.lower() == "openai" or model.lower() == "chatgpt":
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        elif model.lower() == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        elif model.lower() in ["local", "deepseek"] :
            self.llm = DeepseekFinetuned()
        else:
            log.warning(f"Invalid model is provided: {model}")
            raise Exception("Invalid Model")