import pandas as pd
from echo.data.utils import get_buyer_insights_categories_df
from echo.indexing import add_data, check_metadata_exists_in_db
from echo.data.indexes import IndexType
import re
from typing import List
from langchain_community.document_loaders import SeleniumURLLoader
import concurrent.futures
from tqdm.auto import tqdm
from echo.settings import MAX_CONCURRENT_REQUESTS
from echo.llm_utils import run_openai_query


DOC_SUMMARIZATION_PROMPT = (
    "You are a strategic sales assistant. Given the document text below, "
    "summarize the key points and insights that are relevant to the buyer {buyer} based on the following context.\n"
    
    "Context:\n"
    "{context}\n"
    
    "Document Title:\n"
    "{doc_title}\n"
    
    "Document Text:\n"
    "{doc_text}\n"
    
    "Extract the relevant information according to the context\n"
)


def create_buyer_web_search_index(buyer: str, seller: str):
    def get_links(text):
        links = re.findall(r'\[.*?\]\((https?://.*?)\)', text)
        return links

    def load_pages(urls: List[str]):
        loader = SeleniumURLLoader(urls=urls)
        documents = loader.load()

        return documents
    
    def get_url_docs_summary(prompts: List[str]):
        print(f"Extracting data from {len(prompts)} URLs")
        concurrent_responses = dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_prompt = {executor.submit(run_openai_query, prompt): prompt for prompt in prompts}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_prompt),
                total=len(future_to_prompt),
                desc="Generating URL summaries",
            ):
                prompt = future_to_prompt[future]
                try:
                    data = future.result()
                    concurrent_responses[prompt] = data
                except Exception as exc:
                    print(f"Prompt {prompt} generated an exception: {exc}")
        
        return concurrent_responses

    
    if check_metadata_exists_in_db(seller, IndexType.BUYER_WEB_SEARCH, {'buyer': buyer}):
        print("Buyer web search index already exists")
        return
    
    df: pd.DataFrame = get_buyer_insights_categories_df()
    print("Extracting data for buyer: ", buyer)
    for _, r in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
        d_str: str = (
            "Task: " + r['LLM Prompt for Crawl'] + 
            "\n\nPurpose: " + r['Purpose'] + 
            "\nCategory: " + r['Category'] + "\n" + 
            "\nSources: " + r['Source'] + "\n" + 
            "Examples: " + r['Examples']
        )
        prompt = f"You need to extract information about {buyer}. \n{d_str.format(buyer=buyer)}\n\n"
        response = f"{prompt}\n\nResponse: " + run_openai_query(prompt, use_tools=True)
        urls_docs = load_pages(get_links(response))
        if len(urls_docs) > 0:
            doc_prompts = [
                DOC_SUMMARIZATION_PROMPT.format(
                    buyer=buyer,
                    context=d_str,
                    doc_title=doc.metadata['title'],
                    doc_text=doc.page_content
                ) for doc in urls_docs
            ]
            prompt_to_doc = {prompt: doc for prompt, doc in zip(doc_prompts, urls_docs)}
            doc_responses = get_url_docs_summary(doc_prompts)
            response += "\n\n Related Documents:\n\n" + "\n".join(
                [
                    (
                        f"Title: {prompt_to_doc[r].metadata['title']}\n"
                        "URL: " + prompt_to_doc[r].metadata['source'] + "\n"
                        f"Description: {prompt_to_doc[r].metadata['description']}\n"
                        f"Content: {doc_responses[r]}\n"
                    )
                    for r in doc_responses
                ]
            )
        
        add_data(
            response,
            {'buyer': buyer, 'data': response},
            seller,
            IndexType.BUYER_WEB_SEARCH,
        )
        