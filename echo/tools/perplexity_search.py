import os
import requests
import json
from typing import Dict


# Replace with your actual port if different from 3000
API_URL = "http://localhost:3000/api/search"



def call_api(query: str, history: list = None) -> Dict:
    if history is None:
        history = [
            ["human", "Hi, how are you?"],
            ["assistant", "I am doing well, how can I help you today?"],
        ]

    llm_provider, llm_name = (
        os.getenv("PERPLEXITY_LLM_PROVIDER"),
        os.getenv("PERPLEXITY_LLM"),
    )
    embed_model_provider, embed_model_name_name = (
        os.getenv("PERPLEXITY_EMBEDDING_PROVIDER"),
        os.getenv("PERPLEXITY_EMBEDDING_MODEL"),
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "chatModel": {"provider": llm_provider, "name": llm_name},
        "embeddingModel": {
            "provider": embed_model_provider,
            "name": embed_model_name_name,
        },
        "optimizationMode": "speed",
        "focusMode": "webSearch",
        "query": query,
        "history": history,
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload)).json()
    if response.get("error"):
        raise ValueError(f"API error: {response['error']}")
    return response
