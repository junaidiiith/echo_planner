from typing import List
from crewai import LLM
from pydantic import BaseModel

from echo.constants import COMPETITOR_EXTRACTION
from echo.data.indexes import IndexDataType, IndexType
from echo.echo_agent import EchoAgent, get_crew as get_crew_obj
from echo.indexing import add_data, check_metadata_exists
from echo.settings import N_COMPETITORS
from echo.tools.perplexity_search import call_api
from echo.tools.web_scraping import extract_data_from_website


class Competitor(BaseModel):
    name: str
    description: str
    url: str
    rationale: str

class CompetitorsExtractionResponse(BaseModel):
    competitors: list[Competitor]


COMPETITOR_EXTRACTION_SEARCH_QUERY = (
    "Provide me top {n_competitors} competitors of {company}."
    "Make sure to include the --\n"
    "1. competitors landing page URL, \n"
    "2. The description of the company\n"
    "3. why it is a competitor.\n"
    "Use the landing page of the competitor to extract the description.\n"
)

agent_templates = {
    COMPETITOR_EXTRACTION: dict(
        role="Competitor Research Agent",
        goal=(
            "You are an expert in extracting out the list of competitors of a sales company."
        ),
        backstory=(
            "A sales company is trying to sell its product to a customer."
            "The customer is asking for a list of competitors of the sales company."
            "You are an expert in extracting out the list of competitors of a sales company."
        ),
    )    
}

task_templates = {
    COMPETITOR_EXTRACTION: dict(
        name='Competitors Extraction',
        description=(
            "Given the following result from a search engine, extract the competitors of the company."
            "Make sure to include the competitors landing page URL, and the description of the company and why it is a competitor."
            "Below is the search engine result -\n"
            "{search_engine_result}\n"
        ),
        expected_output=(
            "A list of .\n"
            "The response should conform to the provided schema.\n"
            "You need to extract the following information in the following pydantic structure -\n"
            "{pydantic_structure}\n"
            "Make sure there are no comments in the response JSON and it should be a valid JSON."
        ),
        output_pydantic=CompetitorsExtractionResponse,
        agent="CompetitorExtractionAgent",
    )  
}


def get_crew(step: str, llm: LLM, **crew_config) -> EchoAgent:
    assert step in [COMPETITOR_EXTRACTION], (
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    )

    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config,
    )



async def agent_competitor_info(
    inputs: dict, llm: LLM, **crew_config
) -> CompetitorsExtractionResponse:
    seller = inputs["seller"]
    n_competitors = inputs.get("n_competitors", N_COMPETITORS)
    
    
    def save_competitor_data():
        print(f"Adding Competitors Data for {seller}")
        for competitor_data in competitors_website_data:
            print(f"Adding Competitor: {competitor_data['name']}")
            add_data(
                data=competitor_data['website_data'],
                metadata={
                    "data_type": IndexDataType.COMPETITOR_WEBSITE_DATA,
                    "data": competitor_data,
                },
                index_name=seller,
                index_type=IndexType.SELLER_RESEARCH,
            )
            
    
    if check_metadata_exists(
        index_name=seller,
        index_type=IndexType.SELLER_RESEARCH,
        metadata={"data_type": IndexDataType.COMPETITOR_WEBSITE_DATA}
    ):
        print("Competitors Data Already Exists")
        return
        
    
    prompt = COMPETITOR_EXTRACTION_SEARCH_QUERY.format(
        n_competitors=n_competitors, 
        company=seller
    )
    api_response = call_api(prompt)
    inputs["search_engine_result"] = api_response
    crew = get_crew(
        COMPETITOR_EXTRACTION, llm, **crew_config
    )
    response = await crew.kickoff_async(
        inputs={
            "search_engine_result": api_response,
        }
    )
    
    competitors: List[Competitor] = response.tasks_output[0].pydantic.competitors
    
    competitors_website_data = list()
    for competitor in competitors:
        print(f"Extracting data from {competitor.url}")
        website_data = await extract_data_from_website(competitor.url)
        competitors_website_data.append({
            **competitor.model_dump(mode="json"),
            "website_data": website_data,
        })
    
    save_competitor_data()
