import pandas as pd
from echo.data.utils import get_buyer_insights_categories_df
from echo.indexing import add_data, check_metadata_exists_in_db, get_data_from_db
from echo.data.indexes import IndexType
from tqdm.auto import tqdm
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


def add_buyer_web_search_data(buyer: str, seller: str):
    
    def summarize(query: str):
        SUMMARIZATION_PROMPT = (
            "You are a strategic sales assistant. Given the document text below, "
            "Summarize the key points and insights that are relevant to the buyer {buyer} based on the following context.\n"
            "Extract the relevant information according to the context. \n"
            "In the end, mention all the URLs as the relevant sources.\n"
            "DO NOT MISS ANY SOURCE. Provide all of them in the end below as references.\n"
            "Below is the document text - \n"
            "{web_search_response}"
        )
        r = run_openai_query(SUMMARIZATION_PROMPT.format(buyer=buyer, web_search_response=query))
        return r.output_text
    

    df: pd.DataFrame = get_buyer_insights_categories_df()
    print("Extracting data for buyer: ", buyer)
    
    full_text = ""
    summarized_text = ""
    
    for _, r in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
        category = r['Category']
        
        if check_metadata_exists_in_db(
            seller, IndexType.BUYER_WEB_SEARCH, 
            {'buyer': buyer, 'category': category}
        ):
            print(f"Buyer web search index already exists for {buyer} and category {category}.")
            record = get_data_from_db(
                seller, IndexType.BUYER_WEB_SEARCH, 
                {'buyer': buyer, 'category': category}
            )
            summarized_text += "\n\n" + record['summary']
            full_text += "\n\n" + record['data']
            continue
        
        d_str: str = (
            "Task: " + r['LLM Prompt for Crawl'] + 
            "\n\nPurpose: " + r['Purpose'] + 
            "\nCategory: " + r['Category'] + "\n" + 
            "\nSources: " + r['Source'] + "\n" + 
            "Examples: " + r['Examples']
        )
        prompt = (
            f"You need to extract crucial sales information about {buyer} that can be used by relevant sellers of this company.\n\n"
            "---Instructions---\n"
            f"We need to provide the extracted information about the buyer to some seller company.\n"
            "Therefore, make sure to provide the extracted information in a way that is relevant and useful to the seller.\n"
            f"Remove any fluff, internal jargon, or non-client-relevant details.\n"
            "The final output should be clear and sales-oriented.\n"
            "---End of Instructions---\n"
            
            f"---Context---\n"
            f"\n{d_str.format(buyer=buyer)}\n\n"
        )
        print("Prompt: ", prompt)
        openai_response = run_openai_query(prompt, use_tools=True)
        overall_response = openai_response.output_text
        summary = summarize(overall_response)
        
        summarized_text += "\n\n" + summary
        full_text += "\n\n" + overall_response
        
        # print("Text response: ", overall_response)
        
        add_data(
            overall_response,
            {
                'buyer': buyer, 
                'category': category, 
                'data': overall_response, 
                'summary': summary
            },
            seller,
            IndexType.BUYER_WEB_SEARCH,
        )
    
    return {
        'full_text': full_text,
        'summarized_text': summarized_text
    }