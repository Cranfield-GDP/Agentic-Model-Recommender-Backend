import os
import json
import requests
from llm_provider import llm

# Global variable to cache categories once loaded.
CACHED_CATEGORIES = None

HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
HUGGING_FACE_URL = os.getenv("HUGGING_FACE_API") 

def get_huggingface_categories():
    """
    Retrieve available task categories from Hugging Face and cache the result.
    """
    global CACHED_CATEGORIES
    if CACHED_CATEGORIES is not None:
        return CACHED_CATEGORIES

    url = f"{HUGGING_FACE_URL}/tasks"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_KEY}"} if HUGGING_FACE_KEY else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    categories = list(response.json().keys())
    CACHED_CATEGORIES = categories
    return categories

def get_prompt_for_requirement(user_requirement, categories):
    """
    Build a prompt that asks the LLM to choose the most relevant category from the available ones.
    """
    prompt = (
        "Help the user choose the most appropriate and relevant category from Hugging Face Tasks "
        "for the following telecom user requirement.\n"
        "User requirement:\n"
        f"'{user_requirement}'\n"
        "Select the best suited category from the list below. Do not include any explanation; "
        "return only the category exactly as listed.\n"
        "Available categories:\n"
        + ", ".join(categories)
    )
    return prompt

def check_and_get_category(response, categories):
    """
    Process the LLM's response to extract a valid category.
    """
    resp = response.lower().replace(" ", "-").strip()
    for category in categories:
        if category.lower() in resp:
            return category
    raise Exception("No matching category found in the LLM response.")

def get_huggingface_models(category):
    """
    Query Hugging Face API for the top 5 models under the provided category.
    """
    url = f"{HUGGING_FACE_URL}/models"
    params = {"filter": category, "sort": "downloads", "limit": 5}
    headers = {"Authorization": f"Bearer {HUGGING_FACE_KEY}"} if HUGGING_FACE_KEY else {}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    models_data = response.json()
    models_list = []
    for model in models_data:
        # 'modelId' may be provided as a key; otherwise, fall back to 'id'
        model_id = model.get("modelId", model.get("id", "unknown"))
        models_list.append({
            "name": model_id,
            "url": f"https://huggingface.co/{model_id}"
        })
    return models_list

def run(user_requirement: str) -> str:
    """
      1. Retrieve available Hugging Face task categories (cached).
      2. Use the LLM to determine the appropriate category for the user requirement.
      3. Query the Hugging Face models API for the top five models in that category.
      4. Return the results as a JSON string.
    """
    try:
        categories = get_huggingface_categories()
        
        prompt = get_prompt_for_requirement(user_requirement, categories)
        
        category_response = llm(prompt)
        category = check_and_get_category(category_response, categories)
        
        models = get_huggingface_models(category)
        
        result = {"buttons": models, "category": category}
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
