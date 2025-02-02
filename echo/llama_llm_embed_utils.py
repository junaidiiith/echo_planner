from llama_index.llms.fireworks import Fireworks
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from llama_index.core.llms import ChatMessage
from echo.constants import (
    LLM_TYPE,
    OPENAI,
    OPENAI_MODEL,
    OPENAI_API_KEY,
    FIREWORKS,
    FIREWORKS_LLM,
    FIREWORKS_API_KEY,
    EMBED_MODEL_TYPE,
    EMBED_BATCH_SIZE,    
)

def get_llm():
    llm_type = os.getenv(LLM_TYPE)
    if llm_type == OPENAI:
        return OpenAI(model=os.getenv(OPENAI_MODEL), api_key=os.getenv(OPENAI_API_KEY))
    elif llm_type == FIREWORKS:
        return Fireworks(
            model=os.getenv(FIREWORKS_LLM), api_key=os.getenv(FIREWORKS_API_KEY)
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def get_embed_model():
    embed_model_type = os.getenv(EMBED_MODEL_TYPE)
    if embed_model_type == OPENAI:
        return OpenAIEmbedding(
            api_key=os.getenv(OPENAI_API_KEY),
            embed_batch_size=os.getenv(EMBED_BATCH_SIZE, 16),
        )
    elif embed_model_type == FIREWORKS:
        return FireworksEmbedding(
            api_key=os.getenv(FIREWORKS_API_KEY),
            embed_batch_size=os.getenv(EMBED_BATCH_SIZE, 16),
        )
    else:
        raise ValueError(f"Unknown embedding model type: {embed_model_type}")


def get_llm_response(query: str, system_prompt: str = None, llm=None):
    if llm is None:
        llm = get_llm()
    messages = (
        [ChatMessage(role="system", content=system_prompt)] if system_prompt else []
    )
    messages.append(ChatMessage(role="user", content=query))
    response = llm.chat(messages)
    return response.message.blocks[0].text.strip()
