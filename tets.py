"""
from echo.tools.web_scraping import extract_data_from_website
import asyncio

website_content = asyncio.run(extract_data_from_website(url='https://foundit.in'))
print(website_content)
"""

from echo import setup_db_tables
import nest_asyncio
from dotenv import load_dotenv
import asyncio
from echo.constants import DISCOVERY, DEMO, PRICING, NEGOTIATION
from echo.runner import create_or_get_seller
from echo.query_executor import QueryResponse

nest_asyncio.apply()

load_dotenv()


setup_db_tables()

NUM_BUYERS = 10

inputs = {
    "seller": "https://whatfix.com",
    "num_buyers": NUM_BUYERS,
}

seller_data = asyncio.run(create_or_get_seller(inputs))


# 2

from echo.runner import make_call


buyer_inputs = {
    **inputs,
    **seller_data,
    "call_id": 1,
    "stakeholders": ["Product Manager", "CFO", "VP of Product", "VP of Sales"],
}

clients = [
    # "https://www.synechron.com/",
    # "https://services.harman.com/",
    # "https://www.capgemini.com/",
    # "https://www.cognizant.com/us/en"
    "https://www.coca-cola.com/in/en",
    "https://www.manpowergroup.com/en",
]


# 3
import re


def split_camel_case(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s)


def process_text(word: str):
    return " ".join([t.title() for t in split_camel_case(word).split()])


process_text("ProductManager")


# 4

from echo import sqldb
from echo.indexing import IndexType

# sqldb.query_records(IndexType.BUYER_RESEARCH.value)[7]


# 5

discovery_calls_data = asyncio.run(make_call(DISCOVERY, clients, buyer_inputs))


# buyer_inputs['call_id'] = 2
# demo_calls_data = asyncio.run(make_call(DEMO, clients, buyer_inputs))

# buyer_inputs['call_id'] = 3
# pricing_calls_data = asyncio.run(make_call(PRICING, clients[:5], buyer_inputs))
# buyer_inputs['call_id'] = 4
# negotiations_calls_data = asyncio.run(make_call(NEGOTIATION, clients[:5], buyer_inputs))


# new account plan code
import enum
from pydantic import BaseModel
import requests
import json
import re
from typing import Dict, List, Optional, Tuple
from echo.settings import MAX_RETRIES
from echo.tools.web_scraping import extract_data_from_links
from echo import sqldb
from echo.indexing import add_data, IndexType


# Replace with your actual port if different from 3000
API_URL = "http://localhost:3000/api/search"


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
    buyer: str
    seller: str
    query_type: str
    query: str
    source_extracted_data: Optional[List[ExtractedCitedSource]] = None
    timestamp: Optional[str] = None


class SearchResult(BaseModel):
    message: str
    cited_content: List[str]


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


def curate_buyer_index_data_from_search_response(
    search_response: SearchResponse,
) -> Tuple[str, Dict]:
    data = []
    metadata = {
        "seller": search_response.seller,
        "buyer": search_response.buyer,
        "query_type": search_response.query_type,
        "query": search_response.query,
        "sources": [
            {"title": source.metadata.title, "url": source.metadata.url}
            for source in search_response.sources
        ],
    }
    data = search_response.message
    return data, metadata


def curate_buyer_index_sources_data_from_search_response(
    search_response: SearchResponse,
) -> Tuple[str, Dict]:
    data = "\n\n".join([source.pageContent for source in search_response.sources])
    metadata = {
        "seller": search_response.seller,
        "buyer": search_response.buyer,
        "query_type": search_response.query_type,
        "query": search_response.query,
        "sources": [
            {"title": source.metadata.title, "url": source.metadata.url}
            for source in search_response.sources
        ],
    }
    return data, metadata


def search_query(seller, buyer, query_type, history=None) -> SearchResponse:
    condition_dict = {"seller": seller, "buyer": buyer, "query_type": query_type}
    print("Checking if record exists in the database...")
    if sqldb.check_record_exists(
        IndexType.BUYER_FOUNDATIONAL_PLAN.value, condition_dict
    ):
        print("Record exists, fetching from the database...")
        return SearchResponse(
            **sqldb.get_record(IndexType.BUYER_FOUNDATIONAL_PLAN.value, condition_dict)
        )

    print("Record does not exist, making API call...")

    query = query_type_prompts[query_type]["search"].format(buyer=buyer)

    if history is None:
        history = [
            ["human", "Hi, how are you?"],
            ["assistant", "I am doing well, how can I help you today?"],
        ]

    headers = {"Content-Type": "application/json"}
    payload = {
        "chatModel": {"provider": "openai", "name": "gpt-4o-mini"},
        "embeddingModel": {"provider": "openai", "name": "text-embedding-3-large"},
        "optimizationMode": "speed",
        "focusMode": "webSearch",
        "query": query,
        "history": history,
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload)).json()

    # print("response = ")
    # print(response)
    response_obj = SearchResponse(
        **{
            **response,
            "buyer": buyer,
            "seller": seller,
            "query_type": query_type,
            "query": query,
        }
    )
    description = query_type_prompts[query_type]["description"]
    response_obj.message = f"{description}\n{response['message']}"

    sqldb.insert_record(
        IndexType.BUYER_FOUNDATIONAL_PLAN.value,
        {
            "buyer": buyer,
            "seller": seller,
            "query_type": query_type,
            "query": query,
            "message": response_obj.message,
            "sources": response["sources"],
        },
    )

    data, metadata = curate_buyer_index_data_from_search_response(response_obj)

    add_data(
        data=data,
        metadata=metadata,
        index_name=seller,
        index_type=IndexType.BUYER_FOUNDATIONAL_PLAN,
    )

    return response_obj


def get_cited_sources(source):
    print(source)
    pattern = r"\[([^\]]+)\]"
    matches: List[str] = re.findall(pattern, source)
    print("matches = ", matches)
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


def extract_data_from_sources(search_response: SearchResponse) -> SearchResponse:
    condition_dict = {
        "seller": search_response.seller,
        "buyer": search_response.buyer,
        "query_type": search_response.query_type,
    }
    print("Checking if record exists in the database...")
    if sqldb.check_record_exists(
        IndexType.BUYER_FOUNDATIONAL_PLAN.value, condition_dict
    ):
        print("Record exists, fetching from the database...")
        record = sqldb.get_record(
            IndexType.BUYER_FOUNDATIONAL_PLAN.value, condition_dict
        )
        if record["source_extracted_data"]:
            return SearchResponse(**record)

    def make_search_call():
        print("Record does not exist, making API call...")
        citations = get_cited_sources(search_response.message)
        print("Citations = ", citations)
        if not citations:
            data = []
            extracted_cited_sources = ExtractedCitedSource(CitedSource)
        else:
            cited_sources = get_cited_content(search_response.sources, citations)
            print("cited sources", cited_sources)
            links = [source.url for source in cited_sources]
            company_info = search_response.message
            system_prompt = query_type_prompts[search_response.query_type]["system"]
            user_prompt = query_type_prompts[search_response.query_type]["user"].format(
                buyer=search_response.buyer,
                company_info=company_info,
                webpage="{webpage}",
                content="{content}",
            )
            data = extract_data_from_links(
                links, user_prompt=user_prompt, system_prompt=system_prompt
            )
            extracted_data_map = {source["link"]: source["data"] for source in data}
            extracted_cited_sources = [
                ExtractedCitedSource(
                    citation_id=citation.citation_id,
                    title=citation.title,
                    content=citation.content,
                    url=citation.url,
                    data=extracted_data_map[citation.url],
                )
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

    sqldb.update_record(
        IndexType.BUYER_FOUNDATIONAL_PLAN.value,
        condition_dict,
        {
            "source_extracted_data": [
                d.model_dump(mode="json") for d in extracted_cited_sources
            ]
        },
    )

    sources_data, metadata = curate_buyer_index_sources_data_from_search_response(
        search_response
    )
    add_data(
        data=sources_data,
        metadata=metadata,
        index_name=search_response.seller,
        index_type=IndexType.BUYER_FOUNDATIONAL_PLAN,
    )

    return search_response.model_copy(
        update={"source_extracted_data": extracted_cited_sources}
    )


buyer = "https://www.manpowergroup.com/en"
seller = "https://whatfix.com"


from echo.tools.perplexity_search import create_account_plan
from echo.indexing import setup_db_tables

buyer = "https://www.coca-cola.com/in/en"
seller = "https://whatfix.com"

setup_db_tables()
create_account_plan(seller=seller, buyer=buyer)


# query part

from echo.queries import get_queries

queries = get_queries(seller=seller)
print(queries)

# account_plan_query = queries['prediscovery']
from echo.query_executor import arun_queries
from echo.query_executor import ResponseFormat
from echo.query_executor import ContextExtractionMode
import nest_asyncio
import asyncio

nest_asyncio.apply()


inputs = {"buyer": buyer, "company_size": "Enterprise"}

# single query endpoint
# same runs for multiple queries too
responses = asyncio.run(
    arun_queries(
        queries=queries,
        inputs=inputs,
        response_format=ResponseFormat.MARKDOWN,
        context_extraction_mode=ContextExtractionMode.QUERY_ENGINE,
    )
)
print(responses)
print(type(responses))
print("\n\n\n")

# summary endpoint for email

email_summary = {}
for query in responses.responses:
    email_summary[query] = {
        "response": responses.responses[query].response,
        "sub_queries_context": responses.responses[query].sub_queries_context,
        "summary": responses.responses[query].summary,
    }

email_summary = json.dumps(email_summary, indent=4)
print(email_summary)
print(type(email_summary))
