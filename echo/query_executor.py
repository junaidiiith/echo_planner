import copy
import enum
from crewai import Agent, Task, Crew
from echo.indexing import get_vector_index, IndexType
from echo.utils import format_response, get_llm
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from echo.utils import add_pydantic_structure
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from tqdm.asyncio import tqdm as async_tqdm

from llama_index.core.schema import NodeWithScore
from echo.settings import SIMILARITY_TOP_K


class ContextExtractionMode(enum.Enum):
    RETRIEVER = "retriever"
    QUERY_ENGINE = "query_engine"


class ResponseFormat(enum.Enum):
    JSON = "json"
    MARKDOWN = "markdown"


class FilledSubsection(BaseModel):
    name: str = Field(
        ..., title="Subsection Name", description="The name of the subsection."
    )
    content: str = Field(
        ..., title="Subsection Content", description="The content of the subsection."
    )


class FilledSection(BaseModel):
    name: str = Field(..., title="Section Name", description="The name of the section.")
    content: str = Field(
        ..., title="Section Content", description="The content of the section."
    )
    subsections: List[FilledSubsection] = Field(
        ..., title="Subsections", description="The subsections of the section."
    )


class QueryResponse(BaseModel):
    sections: List[FilledSection] = Field(
        ..., title="Sections", description="The filled sections of the call transcript."
    )


class QueryMetadata(BaseModel):
    key: str = Field(..., title="Key", description="The key for the metadata.")
    value: str = Field(..., title="Value", description="The value for the metadata.")
    operator: FilterOperator = Field(
        ..., title="Operator", description="The operator for the metadata."
    )


class SubQuery(BaseModel):
    query: str = Field(
        ...,
        title="Sub Query",
        description="The sub query for which the response is needed.",
    )
    index_type: str = Field(
        ..., title="Index Type", description="The index type for the sub query."
    )
    inputs: Optional[Dict] = Field(
        default=None, title="Inputs", description="The inputs for the sub query."
    )
    context_tasks: Optional[List[int]] = Field(
        default=None,
        title="Context Tasks",
        description="The context tasks for the sub query.",
    )


class Query(BaseModel):
    seller: str = Field(..., title="Seller", description="The seller for the call.")
    call_type: str = Field(
        ...,
        title="Call Type",
        description="The call type for which the response is needed.",
    )
    query: str = Field(
        ..., title="Query", description="The query for which the response is needed."
    )
    sub_queries: List[SubQuery] = Field(
        ..., title="Sub Queries", description="The sub queries and their context."
    )


METADATA_KEYS_MAP = {
    IndexType.HISTORICAL.value: [
        {
            "key": "seller",
            "operator": FilterOperator.EQ,
        },
        {
            "key": "buyer",
            "operator": FilterOperator.NE,
        },
        {
            "key": "call_type",
            "operator": FilterOperator.EQ,
        },
        {
            "key": "company_size",
            "operator": FilterOperator.EQ,
        },
    ],
    IndexType.CURRENT_CALL.value: [
        {
            "key": "seller",
            "operator": FilterOperator.EQ,
        },
        {
            "key": "buyer",
            "operator": FilterOperator.EQ,
        },
        {
            "key": "call_type",
            "operator": FilterOperator.EQ,
        },
    ],
    IndexType.BUYER_RESEARCH.value: [
        {
            "key": "buyer",
            "operator": FilterOperator.EQ,
        }
    ],
    IndexType.SELLER_RESEARCH.value: [
        {
            "key": "seller",
            "operator": FilterOperator.EQ,
        }
    ],
    IndexType.SALES_PLAYBOOK.value: [],
    IndexType.TRANSCRIPT.value: [
        {
            "key": "seller",
            "operator": FilterOperator.EQ,
        },
        {
            "key": "buyer",
            "operator": FilterOperator.EQ,
        },
        {
            "key": "call_type",
            "operator": FilterOperator.EQ,
        },
    ],
}


def get_llama_metadata_filters(metadata: List[QueryMetadata]):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key=md.key, value=md.value, operator=md.operator)
            for md in metadata
        ],
    )
    return filters


def get_metadata_filters(index_type: str, metadata: Dict):
    index_keys = METADATA_KEYS_MAP[index_type]
    assert all([item["key"] in metadata for item in index_keys]), (
        f"Metadata keys missing for index type {index_type}: {[i['key'] for i in index_keys]}"
    )
    filters = [
        QueryMetadata(
            key=item["key"], value=metadata[item["key"]], operator=item["operator"]
        )
        for item in index_keys
        if metadata[item["key"]]
    ]
    filters = get_llama_metadata_filters(filters)
    return filters


def get_qe_crew(response_format: ResponseFormat = ResponseFormat.MARKDOWN):
    markdown_response_format = (
        "Your response should be in the form of a structured document with clear sections and subsections."
        "The document should contain all the relevant information that the sales agent needs to know to prepare for the call."
        "Provide the response in markdown format that is easy to read and understand."
    )

    json_response_format = (
        "Your response should be in the form of a structured document with clear sections and subsections."
        "The document should contain all the relevant information that the sales agent needs to know to prepare for the call."
        "Provide the response in JSON format that is easy to read and understand."
        "Each section should have a title and a description of the content and sub-sections."
        "Each subsection should have a title and a description of the content."
        "You need to extract the following information in the following pydantic structure -\n"
        "{pydantic_structure}\n"
        "Make sure there are no comments in the response JSON and it should be a valid JSON."
    )

    def get_response_format():
        if response_format == ResponseFormat.JSON:
            return {
                "expected_output": json_response_format,
                "output_pydantic": QueryResponse,
            }
        return {"expected_output": markdown_response_format}

    agent = Agent(
        role="Sales Call Preparation Specialist",
        backstory=(
            "You are a sales assistant who given specific queries to resolve, need to generate responses to the queries using the queries and context provided."
            "You need to help solve queries that help AE's prepare for calls using historical call info, current deals calls, and buyer and seller data"
            "You are an expert in helping sales agents prepare for their calls with potential buyers."
            "You have access to historical data from past deals, information about the potential buyer needs and seller goals."
            "Given a query, you can reason about the query and provide an answer to a specific query asked by the sales agent with the most relevant information."
            "MOST IMPORTANTLY: YOU NEED TO USE THE INFORMATION ONLY PROVIDED HERE AND NOT ANY PRIOR KNOWLEDGE."
        ),
        goal="You need to provide the sales agent with the most relevant information that helps them to understand the needs of the client and successfully close the deal.",
        llm=get_llm(),
    )

    task = Task(
        name="Sales Calls Query Engine",
        description=(
            "Provide a clear, helpful answer to below query that the sales agent needs and answer to - \n{query}"
            "You will be provided with the buyer research information and supporting context relevant to each of those sub-queries that aims to help you answer the query."
            "The research context, sub queries and their responses will help you to reason about the query and provide the most relevant information."
            "The buyer research, sub-queries and their context is based on the historical data from past deals is provided below - \n{context}"
            "\n\nNow, given the research information about the buyer and the relevant sub-queries as context, provide a clear and concise answer to the query."
            "You need to use ONLY this information as context to provide an answer to the query. You cannot use any other information."
            "Answer the following query: {query}"
        ),
        agent=agent,
        **get_response_format(),
    )

    qe_crew = Crew(name="Query Resolution Crew", agents=[agent], tasks=[task])

    return qe_crew


async def aget_qe_crew_response(
    query: str,
    sub_queries_context: List[Dict],
    response_format: ResponseFormat = ResponseFormat.MARKDOWN,
):
    print("Running final query", query)
    qe_crew = get_qe_crew(response_format=response_format)
    sub_queries_context_str = "\n".join(
        [f"{sq['query']}\n{sq['context']}" for sq in sub_queries_context]
    )
    inputs = {
        "query": query,
        "context": sub_queries_context_str
    }
    add_pydantic_structure(qe_crew, inputs)
    response = await qe_crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


def get_buyer_research(index_type: str, seller: str, buyer: str) -> str:
    metadata_filters = get_metadata_filters(index_type, {"buyer": buyer, "company_size": "Enterprise"})  # noqa: F821
    vector_index = get_vector_index(seller, index_type)
    retriever = vector_index.as_retriever(filters=metadata_filters)
    docs: List[NodeWithScore] = retriever.retrieve("")
    assert len(docs) > 0, f"No Buyer Research documents found for {buyer}"
    return docs[0].text


def get_sub_queries_context(
    query: Query,
    inputs: Dict[str, str],
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE,
    similarity_top_k=SIMILARITY_TOP_K,
    **kwargs,
) -> List[Dict]:
    def doc_data(doc):
        lambda doc: f"Document Text: {doc.text}\n"
        + f"Document Metadata: {doc.metadata}"
    

    def retrieve_content(query: str, filters: MetadataFilters):
        docs: List[NodeWithScore] = vector_index.as_retriever(
            filters=filters, similarity_top_k=similarity_top_k, **kwargs
        ).retrieve(query)
        return "Relevant Document Details\n" + "\n".join(
            [doc_data(doc) for doc in docs]
        )

    def query_content(query: str, filters: MetadataFilters):
        response = vector_index.as_query_engine(
            filters=filters, similarity_top_k=similarity_top_k, **kwargs
        ).query(query)
        return "Relevant Context:\n" + str(response)

    inputs.update({"seller": query.seller, "call_type": query.call_type})
    
    buyer_context = get_buyer_research(
        IndexType.BUYER_RESEARCH.value, 
        inputs["seller"], 
        inputs["buyer"]
    )
    
    sub_queries_context = [{
        "query": "Buyer Research Information",
        "context": buyer_context
    }]
    
    for sub_query in query.sub_queries:
        print("Running sub query", sub_query.query)
        sub_query_inputs = copy.deepcopy(inputs)
        if sub_query.inputs:
            sub_query_inputs.update(sub_query.inputs)

        metadata_filters = get_metadata_filters(sub_query.index_type, sub_query_inputs)
        vector_index = get_vector_index(query.seller, sub_query.index_type)
        if sub_query.context_tasks:
            assert all(
                [task < len(sub_queries_context) for task in sub_query.context_tasks]
            ), f"Incorrect dependencies for context tasks: {sub_query.context_tasks}"
            context_str = "Context: \n" + "\n".join(
                [
                    f"{sub_queries_context[task]['context']}"
                    for task in sub_query.context_tasks
                ]
            )
            sub_query.query = f"{sub_query.query}\n{context_str}"

        sub_queries_context.append(
            {
                "query": sub_query.query,
                "context": query_content(sub_query.query, metadata_filters)
                if context_extraction_mode == ContextExtractionMode.QUERY_ENGINE
                else retrieve_content(sub_query.query, metadata_filters),
            }
        )
        print("Sub query context", sub_queries_context[-1])

    return sub_queries_context


async def aget_query_response(
    query: Query,
    inputs: Dict[str, str],
    response_format=ResponseFormat.MARKDOWN,
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE,
    **kwargs,
):
    context = get_sub_queries_context(
        query, inputs, context_extraction_mode, **kwargs
    )
    response = await aget_qe_crew_response(query.query, context, response_format)
    return response, context


async def arun_queries(
    queries: Dict[str, Dict[str, Query]],
    inputs: Dict[str, str],
    response_format=ResponseFormat.MARKDOWN,
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE,
    **kwargs,
):
    responses = dict()
    for call_type, call_queries in queries.items():
        print(f"Running queries for call type {call_type}")
        for query_name, query in async_tqdm(call_queries.items(), desc="Running Queries"):
            response, sub_queries_context = await aget_query_response(
                query, inputs, response_format, context_extraction_mode, **kwargs
            )
            responses[query_name] = {
                "response": response,
                "sub_queries_context": sub_queries_context,
            }
        return responses
