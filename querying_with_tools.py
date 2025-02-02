import os
from echo.indexing import IndexType, get_vector_index
from crewai.tools.structured_tool import CrewStructuredTool
from pydantic import BaseModel, Field

from crewai import Crew, Task, Agent, LLM

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from echo.settings import SIMILARITY_TOP_K
from echo.step_templates.generic import FilledSections
from echo.utils import add_pydantic_structure


def get_llm():
    llm = LLM(
        model=os.getenv("FIREWORKS_MODEL_NAME"),
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )
    return llm


def take_input(label: str) -> str:
    while True:
        q = input(f"Enter {label}: \n")
        if q:
            if q == "exit":
                exit()
            return q
        print(f"Please enter a valid {label}.")


def get_inputs():
    call_type = take_input("call type").strip()
    buyer = take_input("buyer").strip()
    seller = take_input("seller").strip()
    query = take_input("query").strip()
    query = f"Input parameters for the query: \ncall_type = {call_type}\nbuyer={buyer}\nseller={seller}\nquery={query}"
    return {
        "query": query,
    }


class HistoricalIndexArguments(BaseModel):
    """Input schema for MyCustomTool."""

    query: str = Field(..., description="Query to retrieve from the index.")
    call_type: str = Field(..., description="Type of call")
    seller: str = Field(..., description="Name of the seller.")


def historical_relevant_calls_info_retriever(
    query: str, call_type: str, seller: str
) -> str:
    index = get_vector_index(seller, IndexType.HISTORICAL)

    filters_dict = {"call_type": call_type}

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key=k,
                value=v,
            )
            for k, v in filters_dict.items()
        ]
    )

    retrieved_nodes = index.as_retriever(
        filters=filters, similarity_top_k=SIMILARITY_TOP_K
    ).retrieve(query)
    return f"\n".join(
        [
            f"Historical Call: {i}\n{n.get_content()}"
            for i, n in enumerate(retrieved_nodes)
        ]
    )


historical_call_index_tool = CrewStructuredTool.from_function(
    name="Historical Calls Information Retriever",
    description=(
        """
        This tool retrieves relevant calls, i.e., calls with buyers that belong to similar industry, support similar use cases and so on.
        This tool uses seller and call type to retrieve the relevant calls.
        And then uses the query to retrieve the most relevant calls.
        This tool can be used to see what has worked in the past for similar deals.
        
        The relevant information about the current call includes the -
        
        The buyer information  such as - 
        Pain Points - The pain points of the buyer identified in the previous calls of other buyers.
		Objections - The objections raised by the buyer in the previous calls of the other buyers
		Time_lines - The time lines expressed by the buyer in the previous calls of the other buyers
		Success Indicators - The success indicators expressed by the buyer in the previous calls of the other buyers
		Budget Constraints - The budget constraints mentioned by the buyer in the previous calls of the other buyers 
		Competition - The competitors mentioned by the buyer in the previous calls of the other buyers
		Decision Committee - The members of the decision committee identified in the previous calls of the other buyers
		
		The seller information such as -
		Discovery Questions - The discovery questions asked by the seller in the previous calls of the other buyers
		Decision Making Process Questions - The decision making process questions asked by the seller in the previous calls of other buyers
		Objection Resolution Pairs - The objection resolution pairs identified in the call by the seller in the previous calls of the other buyers
        """
    ),
    args_schema=HistoricalIndexArguments,
    func=historical_relevant_calls_info_retriever,
)


class CurrentCallIndexArguments(BaseModel):
    """Input schema for MyCustomTool."""

    call_type: str = Field(..., description="Type of call")
    buyer: str = Field(..., description="Name of the buyer.")
    seller: str = Field(..., description="Name of the seller.")


def current_call_info_retriver(self, call_type: str, buyer: str, seller: str) -> str:
    index = get_vector_index(seller, IndexType.HISTORICAL)

    filters_dict = {"seller": seller, "buyer": buyer, "call_type": call_type}

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key=k,
                value=v,
            )
            for k, v in filters_dict.items()
        ]
    )

    retrieved_nodes = index.as_retriever(
        filters=filters, similarity_top_k=SIMILARITY_TOP_K
    ).retrieve("")
    return f"\n".join(
        [
            f"Current {call_type} Call Data: {i}\n{n.get_content()}"
            for i, n in enumerate(retrieved_nodes)
        ]
    )


current_call_info_retriver_tool = CrewStructuredTool.from_function(
    name="Current Call Information Retriever",
    description=(
        """
        This tool retrieves the relevant information about the current call from the vector index.
        The relevant information about the current call includes the -
        
        The buyer information  such as - 
        Pain Points - The pain points of the buyer identified in the previous call of the current buyer.
		Objections - The objections raised by the buyer in the previous call of the current buyer
		Time_lines - The time lines expressed by the buyer in the previous call of the current buyer
		Success Indicators - The success indicators expressed by the buyer in the previous call of the current buyer
		Budget Constraints - The budget constraints mentioned by the buyer in the previous call of the current buyer 
		Competition - The competitors mentioned by the buyer in the previous call of the current buyer
		Decision Committee - The members of the decision committee identified in the previous call of the current buyer
		
		The seller information such as -
		Discovery Questions - The discovery questions asked by the seller in the previous call of the current buyer
		Decision Making Process Questions - The decision making process questions asked by the seller in the previous call of the current buyer
		Objection Resolution Pairs - The objection resolution pairs identified in the call by the seller in the previous call of the current buyer
        """
    ),
    args_schema=CurrentCallIndexArguments,
    func=current_call_info_retriver,
)


class BuyerResearchIndexArguments(BaseModel):
    """Input schema for BuyerResearchIndexTool."""

    seller: str = Field(..., description="Name of the seller.")
    buyer: str = Field(..., description="Name of the buyer.")


def buyer_research_info_retriever(buyer: str, seller: str) -> str:
    index = get_vector_index(seller, IndexType.BUYER_RESEARCH)

    filters_dict = {
        "buyer": buyer,
    }

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key=k,
                value=v,
            )
            for k, v in filters_dict.items()
        ]
    )

    retrieved_nodes = index.as_retriever(
        filters=filters, similarity_top_k=SIMILARITY_TOP_K
    ).retrieve("")
    return f"\n".join(
        [
            f"Buyer Research Data: {i}\n{n.get_content()}"
            for i, n in enumerate(retrieved_nodes)
        ]
    )


buyer_research_index_tool = CrewStructuredTool.from_function(
    name="Buyer Research Information Retriever",
    description=(
        """
        This tool retrieves information about a particular buyer, including:
        - Description of the buyer
        - Industry the buyer belongs to
        - Company size (e.g., SMB, mid-market, enterprise)
        - Goals of the buyer to be successful
        - Use cases supported by the buyer
        - Stakeholders of the buyer
        - Competitors and their pros and cons
        """
    ),
    args_schema=BuyerResearchIndexArguments,
    func=buyer_research_info_retriever,
)


class SellerResearchIndexArguments(BaseModel):
    """Input schema for SellerResearchIndexTool."""

    seller: str = Field(..., description="Name of the seller.")


def seller_research_info_retriever(seller: str) -> str:
    index = get_vector_index(seller, IndexType.SELLER_RESEARCH)

    filters_dict = {
        "seller": seller,
    }

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key=k,
                value=v,
            )
            for k, v in filters_dict.items()
        ]
    )

    retrieved_nodes = index.as_retriever(
        filters=filters, similarity_top_k=SIMILARITY_TOP_K
    ).retrieve("")
    return f"\n".join(
        [
            f"Seller Research Data: {i}\n{n.get_content()}"
            for i, n in enumerate(retrieved_nodes)
        ]
    )


seller_research_index_tool = CrewStructuredTool.from_function(
    name="Seller Research Information Retriever",
    description=(
        """
        This tool retrieves information about a particular seller, including:
        - Name of the seller
        - Website
        - Description of the seller
        - Industry the seller belongs to
        - Products offered by the seller
        - Solutions provided by the seller
        - Use cases for the seller's products
        - Pricing models (e.g., free, subscription, premium)
        - Pricing descriptions, numbers, and durations
        """
    ),
    args_schema=SellerResearchIndexArguments,
    func=seller_research_info_retriever,
)


agent = Agent(
    role="Call Preparation Support",
    goal="Provide most useful information that helps a salesperson prepare for a call.",
    backstory=(
        "You are an expert in preparing for sales deals. "
        "Each deal involves different types of calls like discovery, demo, pricing and procurement. "
        "You have been tasked with preparing for a call with a buyer. "
    ),
    llm=get_llm(),
    tools=[
        historical_call_index_tool,
        current_call_info_retriver_tool,
        buyer_research_index_tool,
        seller_research_index_tool,
    ],
)


task = Task(
    name="Information Retriever for Sales Calls",
    description=(
        "You will be asked to retrieve relevant information for a sales call. "
        "Each query will require you to use your expertise in sales calls. "
        "To answer each query, you will need to carefully plan how to answer the query. "
        "You can break the query into smaller parts and define a sequence of actions to take. "
        "You are provided with tools that vector indexes to help you answer the queries. "
        "You can query these vector indexes to retrieve the relevant information. "
        "You MUST use the retrieved information to answer the queries without making your own answers. "
        "Below is the query that you need to answer. "
        "{query}"
    ),
    expected_output=(
        "A structured overview of the call, seller's client with distinct sections."
        "The response should conform to the provided schema."
        "You need to extract the following information in the following pydantic structure -\n"
        "{pydantic_structure}\n"
    ),
    output_pydantic=FilledSections,
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)

while True:
    inputs = get_inputs()
    add_pydantic_structure(crew, inputs)
    for k, v in inputs.items():
        print(f"{k}: {v}")
    print(crew.kickoff(inputs=inputs))
