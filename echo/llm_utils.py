import aisuite as ai
from openai import OpenAI
from echo.settings import MAX_CONTEXT_LENGTH
from echo.utils import (
    get_num_tokens, 
    get_text_upto_tokens
)
from typing import List, Union
import os



llm_configs = {
    "fireworks": {
        "model": "fireworks:accounts/fireworks/models/deepseek-v3",
        "base_url": "https://api.fireworks.ai/inference/v1"
    },
    "openai": {
        "model": "openai:gpt-4o"
    },
    "together": {
        "model": "togetherlabs/gpt-3.5-turbo",
        "base_url": "https://api.together.xyz/v1"
    }
}


DATA_SUMMARIZATION_SYS_PROMPT = \
"""
You are an expert in sales such that you can summarize the content extracted from a website of a company that would be relevant to a potential client of that company.
"""

DATA_SUMMARIZATION_PROMPT = \
"""
You are provided with the extracted contents from some webpages.
You need to summarize the extracted content that would be most relevant and persuasive to a potential client.
Focus on key offerings, unique selling points, pricing (if available), case studies, testimonials, competitive advantages, and any value propositions that differentiate this company from its competitors.
Remove any fluff, internal jargon, or non-client-relevant details. The final output should be clear, concise, and sales-oriented.

THE SUMMARY MUST COVER ALL THE KEY POINTS FROM THE EXTRACTED CONTENT.
The whole summary should be between {min_tokens} and {max_tokens} tokens.

Below are the extracted contents from the webpages -
{content}
"""

def get_llm_name():
    llm_type = os.getenv("LLM_TYPE")
    llm_config = llm_configs[llm_type]
    model = llm_config['model']
    return model


def run_openai_query(query: Union[str, List[str]], use_tools: bool = False):
        
    
    client = OpenAI()
    if isinstance(query, str):
        if use_tools:
            response = client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=f"{query}",
            )
        else:
            response = client.responses.create(
                model="gpt-4o",
                input=f"{query}",
            )
    else:
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=query,
        )
        response = chat_response.choices[0].message.content
        
    return response



def get_response(messages: List[str], num_retries = 3):
    
    def get_resp():
        return ai.Client().chat.completions.create(
            model=os.getenv("AISUITE_OPENAI_GPT4o"),
            messages=messages,
        ).choices[0].message.content
    
    tries = 0
    while tries < num_retries:
        try:
            return get_resp()
        except Exception as e:
            raise e
            tries += 1
    return None
    

def summarize_text(text, split_count=5):
    
    def summarize(text):
        print("Summarizing text with total tokens: ", get_num_tokens(text))
        return get_response([
            {"role": "system", "content": DATA_SUMMARIZATION_SYS_PROMPT},
            {"role": "user", "content": DATA_SUMMARIZATION_PROMPT.format(
                    content=text, 
                    min_tokens=min_tokens, 
                    max_tokens=max_tokens
                )
            }
        ])
    
    tokens = get_num_tokens(text)
    split_size = tokens // split_count
    min_tokens = int(0.2*split_size)
    max_tokens = int(0.4*split_size)
    
    if tokens < MAX_CONTEXT_LENGTH:
        return summarize(text)
    else:
        
        print("Splitting text into {} parts".format(split_count))
        print(f"Split size: {split_size}, Min tokens: {min_tokens}, Max tokens: {max_tokens}")
        summaries = [
            summarize(get_text_upto_tokens(text, split_size * i))
            for i in range(1, split_count)
        ]
        return "\n\n".join(summary for summary in summaries if summary)


