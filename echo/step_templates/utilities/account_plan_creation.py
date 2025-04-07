import enum
import concurrent.futures
from pydantic import BaseModel
import requests
import re
from typing import List, Optional
from echo.settings import MAX_RETRIES
from echo.tools.perplexity_search import call_api
from echo.tools.web_scraping import extract_data_from_links
from echo.indexing import (
    IndexType, 
    add_data, 
    check_metadata_exists, 
    get_data_from_index
)
from tqdm.auto import tqdm

# Replace with your actual port if different from 3000
API_URL = "http://localhost:3000/api/search"
MAX_CONCURRENT_REQUESTS = 5


class Metadata(BaseModel):
    title: str
    url: str


class Source(BaseModel):
    pageContent: str
    metadata: Metadata


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


class SearchResponse(BaseModel):
    message: str
    sources: List[Source]


class QueryTypes(enum.Enum):
    FMOD = "Financial Moddeling"
    STRATEGY = "Strategic Initiatives"
    COMPANALYSIS = "Competitor Analysis"
    RECENTNEWS = "Recent News & Events"


query_type_prompts = {
    QueryTypes.FMOD.value: {
        "description": (
            "Financial Modelling: This information captures the financial information about the company. "
            "It includes the size of the company, industry, revenue, and other financial metrics. "
            "This information is crucial for understanding the company's financial health and potential as a client."
        ),
        "search": "Gather financial information including size, industry, and revenue about the company - {buyer}",
        "system": (
            "You are a financial analyst. "
            "You need to gather financial information about the company that is relevant to the user query. "
            "The information should be accurate and ONLY from the content provided. "
            "DO NOT make up any information."
        ),
        "user": (
            "Below is the information about {buyer}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            "Extract the financial information like revenue, growth plan, profit about the company."
        ),
    },
    QueryTypes.STRATEGY.value: {
        "description": (
            "Strategic Initiatives: This information captures the strategic initiatives of the company. "
            "It includes the company's key business priorities, strategic initiatives, and future plans. "
            "This information is crucial for understanding the company's direction and potential as a client."
        ),
        "search": (
            "Identify the company's key business priorities and strategic initiatives for the company - {buyer}. "
            "Analyze 10-K reports and financial statements. "
            "Gain insights into the company's operations, products, services, and market position. "
            "Assess the company's revenue, profitability, and overall financial stability to gauge its potential as a client."
            "Understand the company's future plans and priorities. "
            "Identify challenges the company faces, enabling you to position your product or service as a solution to mitigate these risks. "
        ),
        "system": (
            "You are a strategic analyst. "
            "You need to gather strategic initiatives about the company that is relevant to the user query. "
            "Find the information from annual reports and financial statements of the company. "
            "The information should be accurate and ONLY from the content provided. "
            "DO NOT make up any information."
        ),
        "user": (
            "Below is the information about {buyer}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            "Identify the company's key business priorities and strategic initiatives for the company - {buyer}. "
            "Analyze 10-K reports and financial statements. "
            "Gain insights into the company's operations, products, services, and market position. "
            "Assess the company's revenue, profitability, and overall financial stability to gauge its potential as a client."
            "Understand the company's future plans and priorities. "
            "Identify challenges the company faces, enabling you to position your product or service as a solution to mitigate these risks. "
        ),
    },
    QueryTypes.RECENTNEWS.value: {
        "description": (
            "Recent News & Events: This information captures the recent news and events about the company. "
            "It includes any recent developments, announcements, or changes that may impact the company's operations or strategy. "
            "This information is crucial for understanding the company's current position and future outlook."
        ),
        "search": (
            "Gather recent news and events about the company - {buyer}. "
            "Identify any recent developments, announcements, or changes that may impact the company's operations or strategy. "
            "This information is crucial for understanding the company's current position and future outlook."
        ),
        "system": (
            "You are a news analyst. "
            "You need to gather recent news and events about the company that is relevant to the user query. "
            "The information should be accurate and ONLY from the content provided. "
            "DO NOT make up any information."
        ),
        "user": (
            "Below is the information about {buyer}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            "Gather recent news and events about the company - {buyer}. "
            "Identify any recent developments, announcements, or changes that may impact the company's operations or strategy. "
            "This information is crucial for understanding the company's current position and future outlook."
        ),
    },
    QueryTypes.COMPANALYSIS.value: {
        "description": (
            "Competitor Analysis: This information captures the competitors of the company. "
            "It includes the key players in the industry and their market positions. "
            "This information is crucial for understanding the competitive landscape and positioning your product or service."
        ),
        "search": (
            "Gather information about the competitors of the company - {buyer}. "
            "Identify key players in the industry and their market positions. "
            "Use websites like G2 or Crunchbase to find competitors. "
            "Analyze their strengths, weaknesses, and strategies to understand the competitive landscape."
        ),
        "system": (
            "You are a competitive analyst. "
            "You need to gather information about the competitors of the company that is relevant to the user query. "
            "The information should be accurate and ONLY from the content provided. "
            "DO NOT make up any information."
        ),
        "user": (
            "Below is the information about {buyer}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            "Gather information about the competitors of the company - {buyer}. "
            "Identify key players in the industry and their market positions. "
            "Analyze their strengths, weaknesses, and strategies to understand the competitive landscape."
        ),
    },
}


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


def extract_data_from_sources(buyer: str, query_type: str, search_response: SearchResponse):
    
    def make_search_call():
        print("Record does not exist, making API call...")
        citations = get_cited_sources(search_response.message)
        if not citations:
            return []
        else:
            cited_sources = get_cited_content(search_response.sources, citations)
            print("cited sources", cited_sources)
            links = [source.url for source in cited_sources]
            company_info = search_response.message
            system_prompt = query_type_prompts[query_type]["system"]
            user_prompt = query_type_prompts[query_type]["user"].format(
                buyer=buyer,
                company_info=company_info,
                webpage="{webpage}",
                content="{content}",
            )
            data = extract_data_from_links(
                links, 
                user_prompt=user_prompt, 
                system_prompt=system_prompt
            )
            extracted_data_map = {source["link"]: source["data"] for source in data}
            extracted_cited_sources = [
                ExtractedCitedSource(
                    citation_id=citation.citation_id,
                    title=citation.title,
                    content=citation.content,
                    url=citation.url,
                    data=extracted_data_map[citation.url],
                ).model_dump(mode='json')
                for citation in cited_sources
                if citation.url in extracted_data_map
            ]
        return extracted_cited_sources

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



def search_query(seller, buyer, query_type, history=None) -> SearchResponse:
    condition_dict = {"seller": seller, "buyer": buyer, "query_type": query_type}
    print("Checking if record exists in the database...")
    if check_metadata_exists(
        index_name=seller,
        index_type=IndexType.BUYER_ACCOUNT_PLAN,
        metadata=condition_dict,    
    ):  
        print("Record exists, fetching from the database...")
        record = get_data_from_index(
            index_name=seller,
            index_type=IndexType.BUYER_ACCOUNT_PLAN,
            metadata=condition_dict,
        )
        return record

    print("Record does not exist, making API call...")

    query = query_type_prompts[query_type]["search"].format(buyer=buyer)

    response = call_api(query=query, history=history)
    response_obj = SearchResponse(**response)
        
    description = query_type_prompts[query_type]["description"]
    response_obj.message = f"{description}\n{response['message']}"
    extracted_data_sources = extract_data_from_sources(
        buyer=buyer,
        query_type=query_type,
        search_response=response_obj,
    )
    
    
    add_data(
        data=response_obj.message,
        metadata={
            **response_obj.model_dump(mode='json'),
            'source_extracted_data': extracted_data_sources,
            "buyer": buyer,
            "query_type": query_type,
            "query": query,
        },
        index_name=seller,
        index_type=IndexType.BUYER_ACCOUNT_PLAN,
    )

    return response_obj


def create_account_plan(seller: str, buyer: str):
    """
    Multithreaded function to create an account plan for a buyer.
    Iterate over all query types and create an account plan for each one.
    """
    query_results = []
    
    query_types = [q for q in QueryTypes]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS
    ) as executor:
        # Submit all tasks and store futures in a dictionary.
        futures = {executor.submit(search_query, seller, buyer, query_type): query_type for query_type in set(query_types)}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Executing Account Plan Queries",
            unit="query",
        ):
            result = future.result()
            if result is not None:
                query_results.append(result)

    # Filter out None results
    query_results = [result for result in query_results if result is not None]
    return query_results
