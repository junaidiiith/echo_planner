from firecrawl import FirecrawlApp
import os
from crewai import Agent, Task
from pydantic import BaseModel, Field
from echo import sqldb
from echo.data.utils import get_seller_relevant_link_categories
from echo.echo_agent import EchoAgent
from echo.utils import format_response, get_crew_llm, json_to_markdown, set_new_firecrawl_key
from echo.data.indexes import IndexType
from echo.indexing import add_data, check_metadata_exists_in_db
import concurrent.futures
from typing import List
from langchain.document_loaders import SeleniumURLLoader
from tqdm.auto import tqdm
from urllib.parse import urlparse, ParseResult
import threading

# module‑level lock for guarding key file access
_key_file_lock = threading.Lock()

LANDING_PAGE_DATA_EXTRACTION_PROMPT = """
Extract all relevant information to an account executive about the product he or she is selling from the landing page. 
Extract details on the industry, product details, features it has, problems and pains it solves for its customers and any more details about its customers.
Also capture messaging on how the product presents itself to buyers.
"""

LINK_BATCH_SIZE = 10
LINKS_RELEVANCE_BATCH_SIZE = 50
MAX_CONCURRENT_WORKERS = 5


def get_firecrawl_app(api_key=None):
    """
    Get the Firecrawl app instance.
    """
    if api_key is None:
        if "FIRECRAWL_API_KEY" not in os.environ:
            raise ValueError("FIRECRAWL_API_KEY environment variable not set.")
        
        return FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    else:
        return FirecrawlApp(api_key=api_key)

def get_landing_page_data(url, prompt=LANDING_PAGE_DATA_EXTRACTION_PROMPT, enable_web_search=True):
    
    app = get_firecrawl_app()
    response = app.extract([url], {
        'prompt': prompt,
        'enableWebSearch': enable_web_search,
    })
    return response['data']


def get_seller_links(seller, only_top_level=True, only_https=False):
    """
    Get all links for a given seller.
    :param seller: The seller URL.
    :return: List of links.
    """
    # Fetch the links using the map_url method
    
    def check_only_https(link):
        if not only_https:
            return True
        parsed_url:ParseResult = urlparse(link)
        return parsed_url.scheme == 'https'
    
    parsed_seller_url: ParseResult = urlparse(seller)
    seller_url = f"{parsed_seller_url.scheme}://{parsed_seller_url.netloc}"
        
    print("Getting links for seller: ", seller_url)
    app = get_firecrawl_app()
    all_links = app.map_url(seller_url)
    
    print("Total links: ", len(all_links['links']))
    links = [
        link for link in all_links['links'] 
        if check_only_https(link) and link != seller
    ]
    print("Filtered links: ", len(links))
    top_level_links = list()
    for link in tqdm(links, desc="Filtering links", total=len(links)):
        parsed_link: ParseResult = urlparse(link)
        parsed_link_page = parsed_link.path or '/'
        if only_top_level and parsed_link_page.count('/') == 1:
            top_level_links.append(link)
            
    print("Top level links: ", len(top_level_links))
    return top_level_links



def extract_instruction_information_from_webpage_content(
    links_str: str,
    categories_str: str,
):
    
    class CategoryAssignment(BaseModel):
        url: str = Field(description="The URL of the webpage")
        category: str = Field(description="The category of the webpage")
    
    class AssignmentResponse(BaseModel):
        assignments : list[CategoryAssignment] = Field(description="The list of assignments")
        
    agent = Agent(
        role="Website Content Extraction Expert",
        goal="Extract out the information according to the instruction provided",
        backstory="You are an expert in extracting out the information according to the instruction provided",
        llm=get_crew_llm(),
    )
    
    task = Task(
        name="Extracting Information",
        description=(
            "Given a set of links and their title and description assign each link to one of the following categories that might be relevant for an Account Executive to understand the company's offerings and services.\n"
            "A link is relevant if it can belong to one of the categories below: \n"
            "If a link is not relevant to any of the categories, assign it to the 'Other' category.\n"
            "The categories are as follows:\n"
            "{categories}\n\n"
            
            "The links are as follows:\n"
            "{links}\n"
                        
            "Based on this information, extract out the information according to the instruction provided.\n"
            "You need to extract the relevant information that satisfies the instruction provided.\n"
            "You also need to provide a boolean value that indicates whether the instruction was satisfied or not.\n"
        ),
        expected_output=(
            "The response should conform to the provided schema."
            "You need to extract the following information in the following pydantic structure -\n"
            "{pydantic_structure}\n"
        ),
        output_pydantic=AssignmentResponse,
        agent=agent,
    )

    crew = EchoAgent(
        agents=[agent], 
        tasks=[task]
    )

    inputs = {
        "categories": categories_str,
        "links": links_str,
    }
    print("Inputs: ", inputs)

    response = crew.kickoff(inputs=inputs)
    response = format_response(response)
    return response


def extract_relevant_urls(
    links: List[str], 
    categories, 
    batch_size: int = LINKS_RELEVANCE_BATCH_SIZE
) -> List[dict]:
    print("Total number of links: ", len(links))
    prompts_args = list()
    print("Extracting Selenium Metadata...")
    final_docs = SeleniumURLLoader(urls=links).load()
    print("Extracted Selenium Metadata")
    
    for i in range(0, len(final_docs), batch_size):
        batch = final_docs[i:i + batch_size]
        links_str = "\n".join([f"URL:{doc.metadata['source']}\n{doc.metadata['title']}: {doc.metadata['description']}" for doc in batch])
        categories_str = "\n".join(
            f"{i+1}. Category: {category}\nPurpose: {r['Purpose']}\nExamples:{r['Examples'].replace("\n", ', ')}\n" for i, (category, r) in enumerate(categories.items())
        )
        prompts_args.append((links_str, categories_str))
    
    print("Total number of batches: ", len(prompts_args))
    print("Extracting relevant URLs...")
    extracted_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = {
            executor.submit(extract_instruction_information_from_webpage_content, links_str, categories_str): idx
            for idx, (links_str, categories_str) in enumerate(prompts_args)
        }
        for future in tqdm(concurrent.futures.as_completed(results), total=len(results), desc="Extracting relevant URLs"):
            idx = results[future]
            try:
                extracted_results.append(future.result())
            except Exception as exc:
                print(f"Prompt {idx} generated an exception: {exc}")

    relevant_links = list()
    for response in extracted_results:    
        if not isinstance(response, str):
            assignments = response['assignments']    
            for assignment in assignments:
                if assignment['category'].lower() != 'other':
                    relevant_links.append(assignment)
    
    links_by_category = {}
    for link in relevant_links:
        category = link['category']
        if category not in links_by_category:
            links_by_category[category] = {'prompt': categories[category]['Prompt'], 'links': []}
        links_by_category[category]['links'].append(link['url'])
    
    print("Links by category: ", links_by_category)
    for category, data in links_by_category.items():
        print(f"Category: {category}, Number of links: {len(data['links'])}")
    
    print("Total number of links: ", len(relevant_links))
    print("Total number of categories: ", len(links_by_category))
    return links_by_category


def get_data_from_links_by_category(seller, links_by_category):
    """
    Get data from links by category using Firecrawl.
    """
    
    def serialize_category_data(category_data):
        data_str = ""
        for category, category_data_links in tqdm(category_data.items(), desc="Adding data to index", total=len(category_data)):
            for category_data_response in category_data_links:
                data = json_to_markdown(category_data_response['data'])
                url = category_data_response['url']
                
                link_str = f"URLs: {url}\n\n{data}\n\n"
                data_str += link_str
                
        return data_str
    
    
    def call_firecrawl_app(link: str, prompt: str):
        retries = 5
        while retries > 0:
            app = get_firecrawl_app()
            try:
                response = app.extract([link], {
                    'prompt': prompt,
                    'enableWebSearch': True,
                })
                return response['data']
            except ValueError:
                with _key_file_lock:
                    set_new_firecrawl_key(app.api_key)

                print("ValueError: Retrying...")
                retries -= 1
                if retries == 0:
                    return None
    
    
    category_data = {category: [] for category in links_by_category.keys()}
    print("Getting data from links by category...")
    print("Total number of categories: ", len(links_by_category))
    for category, data in tqdm(links_by_category.items()):
        links = data['links']
        filtered_links = list()
        for link in links:
            if check_metadata_exists_in_db(
                seller,
                IndexType.SELLER_WEB_SEARCH,
                {'category': category, 'url': link},
            ):
                print(f"Category {category} data already exists in the database")
                record = sqldb.get_record(
                    seller, IndexType.SELLER_WEB_SEARCH.value,
                {'category': category, 'url': link})
                
                category_data[category].append({
                    'url': link,
                    'data': record['data'],
                })
            else:
                filtered_links.append(link)
        
        print("Filtered links: ", len(filtered_links))
        prompt = data['prompt']

        responses = [None]*len(filtered_links)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
            futures = {
                executor.submit(call_firecrawl_app, link, prompt): idx
                for idx, link in enumerate(filtered_links)
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting data from links"):
                idx = futures[future]
                responses[idx] = {
                    'url': filtered_links[idx],
                    'data': future.result(),
                }
                
                add_data(
                    json_to_markdown(responses[idx]['data']),
                    {
                        'category': category, 
                        'url': filtered_links[idx], 
                        'data': data
                    },
                    seller,
                    IndexType.SELLER_WEB_SEARCH,
                )
        
        for response in responses:
            category_data[category].append({
                'url': response['url'],
                'data': response['data'],
            })
            
        print(f"Processed {category} - {len(responses)} responses")
    
    # import json
    # with open("category_data.json", "w") as f:
    #     json.dump(category_data, f, indent=4)
        
    return serialize_category_data(category_data)



def add_landing_page_data_to_seller_index(seller):
    """
    Add landing page data to the seller index.
    """
    
    if check_metadata_exists_in_db(
        seller, 
        IndexType.SELLER_WEB_SEARCH,
        {'url': seller, 'category': 'Landing Page'},
    ):
        print("Seller web search index already exists")
        return
    
    
    data = get_landing_page_data(seller)
    if data:
        add_data(
            json_to_markdown(data),
            {'url': seller, 'category': 'Landing Page', 'data': data},
            seller,
            IndexType.SELLER_WEB_SEARCH,
        )
    else:
        print(f"No data found for URL: {seller}")
    
    return {
        'data': data,
    }            


def add_seller_web_search_data(seller: str):
    """
    Add seller web search data to the index.
    """
    links = get_seller_links(seller)
    all_exisitng_links = sum(
        [r['url'] if isinstance(r['url'], list) else [r['url']] 
        for r in sqldb.get_records(f'{seller}', IndexType.SELLER_WEB_SEARCH.value)], []
    )
    filtered_links = [
        link for link in links
        if link not in all_exisitng_links
    ]
    print("Filtered links: ", len(filtered_links))
    categories = get_seller_relevant_link_categories()
    links_by_category = extract_relevant_urls(filtered_links, categories)
    
    with open("links_by_category.json", "w") as f:
        import json
        json.dump(links_by_category, f, indent=4)
    # with open("links_by_category.json") as f:
    #     import json
    #     links_by_category = json.load(f)
    
    category_data = get_data_from_links_by_category(seller, links_by_category)
    # with open("category_data.json") as f:
    #     import json
    #     category_data = json.load(f)
        
    
    return {
        'data': category_data,
    }


