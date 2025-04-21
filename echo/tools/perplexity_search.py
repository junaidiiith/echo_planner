import os
import re
from pydantic import BaseModel
import requests
import json
from typing import Dict, List, Optional
from echo.settings import MAX_RETRIES, debug_mode
from echo.utils import get_pydantic_dummy_instance
from echo.tools.web_scraping import extract_data_from_links


# Replace with your actual port if different from 3000
API_URL = os.getenv("PERPLEXICA_BASE_URL", "http://localhost:3000/api/search")


class Metadata(BaseModel):
    title: str
    url: str


class Source(BaseModel):
    pageContent: str
    metadata: Metadata


class SearchResponse(BaseModel):
    message: str
    sources: List[Source]


class ExtractedData(BaseModel):
    link: str
    data: str


class CitedSource(BaseModel):
    citation_id: int
    title: str
    content: str
    url: str


class ExtractedCitedSource(CitedSource):
    data: Optional[str] = None


class SearchResponseWithCitedData(BaseModel):
    message: str
    extracted_sources: List[ExtractedCitedSource]


USER_PROMPT_TEMPLATE = (
    "Below is the contextual information:\n"
    "{summary}\n"
    "You are provided with content of the webpage: {webpage}. "
    "\n---CONTENT---\n{content}\n"
    "\n---END CONTENT---\n"
    "{user_prompt}"
)


def get_cited_sources(source):
    pattern = r"\[([^\]]+)\]"
    matches: List[str] = re.findall(pattern, source)
    return list(set([int(i) for i in matches if i.isnumeric()]))


def get_cited_content(sources: List[Source], citations: List[int]) -> List[CitedSource]:
    cited_content = []
    for citation in citations:
        if citation < 0 or citation >= len(sources):
            continue
        content = sources[citation].pageContent
        title = sources[citation].metadata.title
        url = sources[citation].metadata.url
        cited_content.append(
            CitedSource(citation_id=citation, title=title, content=content, url=url)
        )
    return cited_content


def call_api(query: str, history: list = None) -> Dict:
    if debug_mode:
        print("Debug mode is enabled. Returning Dummy response.")
        response = get_pydantic_dummy_instance(SearchResponse)
        return response.model_dump(mode="json")

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


def extract_data_from_sources(
    search_response: SearchResponse, user_prompt: str, system_prompt: str
) -> SearchResponseWithCitedData:
    def make_search_call():
        print("Record does not exist, making API call...")
        citations = get_cited_sources(search_response.message)
        if not citations:
            return []
        else:
            cited_sources = get_cited_content(search_response.sources, citations)
            print("cited sources", cited_sources)
            links = [source.url for source in cited_sources]
            message = search_response.message
            prompt = USER_PROMPT_TEMPLATE.format(
                summary=message,
                user_prompt=user_prompt,
                webpage="{webpage}",
                content="{content}",
            )
            data = extract_data_from_links(
                links, user_prompt=prompt, system_prompt=system_prompt
            )
            extracted_data_map = {source["link"]: source["data"] for source in data}
            extracted_cited_sources = [
                ExtractedCitedSource(
                    citation_id=citation.citation_id,
                    title=citation.title,
                    content=citation.content,
                    url=citation.url,
                    data=extracted_data_map[citation.url],
                ).model_dump(mode="json")
                for citation in cited_sources
                if citation.url in extracted_data_map
            ]
            return SearchResponseWithCitedData(
                message=message,
                extracted_sources=extracted_cited_sources,
            )

    num_retries = MAX_RETRIES
    while num_retries > 0:
        try:
            extracted_cited_sources = make_search_call()
            break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            num_retries -= 1
            if num_retries == 0:
                raise
            print("Retrying...")
            continue
    if not extracted_cited_sources:
        print("No data extracted from sources.")
        return []
    print("Data extracted from sources:", extracted_cited_sources)
    return extracted_cited_sources


def call_api_with_extracted_sources(
    query: str, user_prompt: str, system_prompt: str, history: list = None
) -> SearchResponseWithCitedData:
    if debug_mode:
        print("Debug mode is enabled. Returning Dummy response.")
        return get_pydantic_dummy_instance(ExtractedCitedSource).model_dump(mode="json")

    search_response = call_api(query, history)
    search_response = SearchResponse.model_validate(search_response)
    extracted_cited_sources = extract_data_from_sources(
        search_response, user_prompt, system_prompt
    )
    if not extracted_cited_sources:
        print("No data extracted from sources.")
        return SearchResponseWithCitedData(
            message=search_response.message,
            extracted_sources=[],
        )
    print("Data extracted from sources:", extracted_cited_sources)
    return extracted_cited_sources
