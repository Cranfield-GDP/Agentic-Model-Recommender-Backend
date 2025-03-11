import httpx

# API Endpoints
CHAT_URL = "http://0.0.0.0:8000/chat"
MODEL_URL = "http://0.0.0.0:8000/model"

def print_json_response(response, indent=0):
    space = " " * indent  

    if isinstance(response, dict):  
        for key, value in response.items():
            print(f"{space} :: {key}:")
            print_json_response(value, indent + 2)  

    elif isinstance(response, list): 
        for idx, item in enumerate(response):
            print(f"{space}🔹 Item {idx + 1}:")
            print_json_response(item, indent + 2)  

    else:
        print(f"{space}➡ {response}")


def get_user_input(prompt: str):
    return input(prompt).strip()

def send_chat_request(user_id):
    user_message = get_user_input("\nYou: ")

    payload = {
        "message": user_message,
        "userId": user_id
    }

    try:
        response = httpx.post(CHAT_URL, json=payload)
        response.raise_for_status() 
        return response.json()
    except httpx.HTTPStatusError as e:
        print(f"\nServer error: {e.response.text}")
    except Exception as e:
        print(f"\nError: {e}")

    return None 

def validate_and_send_model(user_id, models):
    while True:
        try:
            print("Available models:")
            for idx, model in enumerate(models, start=1):
                print(f"{idx}. {model}")

            selection = int(get_user_input("\nSelect a model (1-5): "))

            if 1 <= selection <= len(models):
                selected_model = models[selection - 1]
                print(f"Sending selected model: {selected_model}")

                payload = {
                    "model": selected_model,  
                    "user_id": user_id        
                }
                response = httpx.post(MODEL_URL, json=payload)
                print_json_response(response.json())
                return
            else:
                print(f"Invalid selection! Please enter a number between 1 and {len(models)}")

        except ValueError:
            print("Invalid input! Please enter a **number** between 1 and", len(models))

def main():
    
    user_id = get_user_input("Enter your User ID: ")
    print("AI: Explain your requirement in detail")
    
    while True:
        chat_response = send_chat_request(user_id)

        if chat_response is None:
            continue  

        if isinstance(chat_response, dict) and "models" in chat_response and isinstance(chat_response["models"], list):
            models = chat_response["models"]

            if not models:
                # print("No models available. Redirecting back to /chat...\n")
                continue 

            validate_and_send_model(user_id, models)
        else:
            print_json_response(chat_response)
            # print("Redirecting back to /chat...\n")

if __name__ == "__main__":
    main()
