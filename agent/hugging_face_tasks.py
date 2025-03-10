import os
import requests


CACHED_CATEGORIES = None

HUGGINGFACE_KEY = os.getenv("HUGGINGFACE_KEY")
HUGGINGFACE_ENDPOINT = os.getenv("HUGGINGFACE_ENDPOINT") 

def get_huggingface_categories():
    """
    Retrieve available task categories from Hugging Face and cache the result.
    """
    global CACHED_CATEGORIES
    if CACHED_CATEGORIES is not None:
        return CACHED_CATEGORIES

    url = f"{HUGGINGFACE_ENDPOINT}/tasks"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"} if HUGGINGFACE_KEY else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    categories = list(response.json().keys())
    CACHED_CATEGORIES = categories
    return categories