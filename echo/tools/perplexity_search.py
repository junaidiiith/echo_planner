import enum
from pydantic import BaseModel
import requests
import json
import re
from typing import List, Optional
from echo.tools.web_scraping import extract_data_from_links
from echo import sqldb

# Replace with your actual port if different from 3000
API_URL = "http://localhost:3000/api/search"

sqldb.create_table(
    '''CREATE TABLE IF NOT EXISTS search_results (
        company_name TEXT,
        query_type TEXT,
        query TEXT,
        message TEXT,
        sources TEXT,
        source_extracted_data TEXT DEFAULT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (company_name, query_type)
    );'''   
)


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
    company_name: str
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
        "search": "Gather financial information including size, industry, and revenue about the company - {company_name}",
        "system": (
            "You are a financial analyst. "
            "You need to gather financial information about the company that is relevant to the user query. "
            "The information should be accurate and ONLY from the content provided. "
            "DO NOT make up any information."
        ),
        "user": (
            "Below is the information about {company_name}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            "Extract the financial information like revenue, growth plan, profit about the company."
        )
    },
    QueryTypes.STRATEGY.value: {
        "search": (
            "Identify the company's key business priorities and strategic initiatives for the company - {company_name}. "
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
            "Below is the information about {company_name}:\n{company_info} "
            
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            
            "Identify the company's key business priorities and strategic initiatives for the company - {company_name}. "
            "Analyze 10-K reports and financial statements. "
            "Gain insights into the company's operations, products, services, and market position. "
            "Assess the company's revenue, profitability, and overall financial stability to gauge its potential as a client."
            "Understand the company's future plans and priorities. "
            "Identify challenges the company faces, enabling you to position your product or service as a solution to mitigate these risks. "
        )
    },
    QueryTypes.RECENTNEWS.value: {
        "search": (
            "Gather recent news and events about the company - {company_name}. "
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
            "Below is the information about {company_name}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            
            "Gather recent news and events about the company - {company_name}. "
            "Identify any recent developments, announcements, or changes that may impact the company's operations or strategy. "
            "This information is crucial for understanding the company's current position and future outlook."
        )    
    },
    QueryTypes.COMPANALYSIS.value: {
        "search": (
            "Gather information about the competitors of the company - {company_name}. "
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
            "Below is the information about {company_name}:\n{company_info} "
            "You are provided with content of the webpage: {webpage}. "
            "\n---CONTENT---\n{content}\n"
            "\n---END CONTENT---\n"
            
            "Gather information about the competitors of the company - {company_name}. "
            "Identify key players in the industry and their market positions. "
            "Analyze their strengths, weaknesses, and strategies to understand the competitive landscape."
        )
    }
}


def search_query(seller, company_name, query_type, history=None) -> SearchResponse:
    
    condition_dict = {
        "seller": seller, 
        "company_name": company_name, 
        "query_type": query_type
    }
    print("Checking if record exists in the database...")
    if sqldb.check_record_exists("search_results", condition_dict):
        print("Record exists, fetching from the database...")
        return SearchResponse(**sqldb.get_record("search_results", condition_dict))
    
    print("Record does not exist, making API call...")
    
    query = query_type_prompts[query_type]['search'].format(company_name=company_name)
    
    if history is None:
        history = [
            ["human", "Hi, how are you?"],
            ["assistant", "I am doing well, how can I help you today?"]
        ]
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "chatModel": {
            "provider": "openai",
            "name": "gpt-4o-mini"
        },
        "embeddingModel": {
            "provider": "openai",
            "name": "text-embedding-3-large"
        },
        "optimizationMode": "speed",
        "focusMode": "webSearch",
        "query": query,
        "history": history
    }
    
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload)).json()
    response_obj = SearchResponse(**{
        **response,
        "company_name": company_name,
        "seller": seller,
        "query_type": query_type,
        "query": query
    })
    
    sqldb.insert_record(
        "search_results",
        {
            "company_name": company_name,
            "seller": seller,
            "query_type": query_type,
            "query": query,
            "message": response['message'],
            "sources": response['sources'],
        }
    )
    return response_obj


def get_cited_sources(source):
    pattern = r'\[([^\]]+)\]'
    matches: List[str] = re.findall(pattern, source)
    return list(set([int(i) for i in matches if i.isnumeric()]))


def get_cited_content(sources: List[Source], citations: List[int]) -> List[CitedSource]:
    cited_content = []
    for citation in citations:
        content = sources[citation].pageContent
        title = sources[citation].metadata.title
        url = sources[citation].metadata.url
        cited_content.append(CitedSource(
            citation_id=citation,
            title=title,
            content=content,
            url=url
        ))
    return cited_content


def extract_data_from_sources(search_response: SearchResponse) -> SearchResponse:
    condition_dict = {
        "seller": search_response.seller, 
        "company_name": search_response.company_name, 
        "query_type": search_response.query_type
    }
    print("Checking if record exists in the database...")
    if sqldb.check_record_exists("search_results", condition_dict):
        print("Record exists, fetching from the database...")
        record = sqldb.get_record("search_results", condition_dict)
        if record['source_extracted_data']:
            return SearchResponse(**record)
    
    print("Record does not exist, making API call...")
    citations = get_cited_sources(search_response.message)
    if not citations:
        data = []
    else:
        cited_sources = get_cited_content(search_response.sources, citations)
        links = [source.url for source in cited_sources]
        company_info = search_response.message
        system_prompt = query_type_prompts[search_response.query_type]['system']
        user_prompt = query_type_prompts[search_response.query_type]['user'].format(
            company_name=search_response.company_name, company_info=company_info,
            webpage="{webpage}", content="{content}"
        )
        data = extract_data_from_links(links, user_prompt=user_prompt, system_prompt=system_prompt)
        extracted_data_map = {source['link']: source['data'] for source in data}
        extracted_cited_sources = [
            ExtractedCitedSource(
                citation_id=citation.citation_id,
                title=citation.title,
                content=citation.content,
                url=citation.url,
                data=extracted_data_map[citation.url]
            )
            for citation in cited_sources
            if citation.url in extracted_data_map
        ]
    
    sqldb.update_record(
        "search_results",
        condition_dict,
        {"source_extracted_data": [d.model_dump(mode='json') for d in extracted_cited_sources]}
    )
    
    return search_response.model_copy(update={"source_extracted_data": extracted_cited_sources})