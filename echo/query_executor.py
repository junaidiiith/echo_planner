import copy
import enum
from crewai import Agent, Task, Crew
from echo.data.indexes import get_echo_index
from echo.indexing import get_vector_index, IndexType
from echo.utils import format_response, get_crew_llm, get_variables_from_prompt
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
from tqdm.asyncio import tqdm as async_tqdm

from llama_index.core.schema import NodeWithScore, Document
from echo.settings import SIMILARITY_TOP_K
from echo.llm_utils import run_openai_query, summarize_text
from echo.query_kg import ask_kg
from echo.tools.perplexity_search import call_api, call_api_with_extracted_sources

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


class QEResponse(BaseModel):
    sections: List[FilledSection] = Field(
        ..., title="Sections", description="The filled sections of the call transcript."
    )


class SingleQueryResponse(BaseModel):
    query: str = Field(
        ..., title="Query", description="The query for which the response is needed."
    )
    
    response: str = Field(
        ..., title="Response", description="The response to the query."
    )
    sub_queries_context: List[Dict] = Field(
        ..., title="Sub Queries Context", description="The context for the sub queries."
    )
    summary: str = Field(
        ..., title="Summary", description="The summary of the response."
    )


class QueryResponse(BaseModel):
    summary: str = Field(
        ..., title="Summary", description="The summary of the responses."
    )
    responses: List[SingleQueryResponse] = Field(
        ..., title="Responses", description="The responses to the queries."
    )


class QueryMetadata(BaseModel):
    key: str = Field(..., title="Key", description="The key for the metadata.")
    value: Union[str, List[str]] = Field(
        ..., title="Value", description="The value for the metadata."
    )
    operator: FilterOperator = Field(
        ..., title="Operator", description="The operator for the metadata."
    )


class SubQuery(BaseModel):
    query: str = Field(
        ..., title="Sub Query", description="The sub query for which the response is needed."
    )
    output_name: str = Field(
        default=None,
        title="Output Name",
        description="The output name for the sub query.",
    )
    inputs: Optional[Dict] = Field(
        default=None, title="Inputs", description="The inputs for the sub query."
    )
    context_tasks: Optional[List[int]] = Field(
        default=None,
        title="Context Tasks",
        description="The context tasks for the sub query.",
    )
   

class PerplexicaSourceExtraction(BaseModel):
    system_prompt: str = Field(
        ..., title="System Prompt", description="The system prompt for the source extraction."
    )
    user_prompt: str = Field(
        ..., title="User Prompt", description="The user prompt for the source extraction."
    )
    

class PerplexicaSubQuery(SubQuery):
    source_extraction_prompts: Optional[PerplexicaSourceExtraction] = Field(
        default=None,
        title="Source Extraction Prompts",
        description="The source extraction prompts for the sub query.",
    )


class LlamaSubQuery(SubQuery):
    index_type: IndexType = Field(
        ..., title="Index Type", description="The index type for the sub query."
    )

class LLMSubQuery(SubQuery):
    use_web_search: bool = Field(
        default=False,
        title="Use Web Search",
        description="Whether to use web search for the sub query.",
    )

class KGSubQuery(SubQuery):
    pass

class Query(BaseModel):
    query: str = Field(
        ..., title="Query", description="The query for which the response is needed."
    )
    sub_queries: List[Union[
        LlamaSubQuery, 
        PerplexicaSubQuery,
        LLMSubQuery,
        KGSubQuery
    ]] = Field(
        ..., title="Sub Queries", description="The sub queries and their context."
    )
    output_name: str = Field(
        default=None,
        title="Output Name",
        description="The output name for the query.",
    )

class QueryChain(BaseModel):
    queries: List[Query] = Field(
        ..., title="Queries", description="The queries for the call."
    )


def get_llama_metadata_filters(metadata: List[QueryMetadata]):
    and_filters, or_filters = list(), list()

    for md in metadata:
        if isinstance(md.value, list):
            filters = [
                MetadataFilter(key=md.key, value=v, operator=md.operator)
                for v in md.value
            ]
            or_filters.append(
                MetadataFilters(filters=filters, condition=FilterCondition.OR)
            )
        else:
            and_filters.append(
                MetadataFilter(key=md.key, value=md.value, operator=md.operator)
            )

    and_filters = MetadataFilters(filters=and_filters, condition=FilterCondition.AND)
    if len(or_filters) == 0:
        return and_filters
    final_filters = MetadataFilters(
        filters=[and_filters] + or_filters, condition=FilterCondition.AND
    )
    return final_filters


def get_metadata_filters(index_type: IndexType, metadata: Dict):
    echo_index = get_echo_index(metadata["seller"], index_type)

    for item in echo_index.metadata_columns:
        if item.mandatory:
            assert item.key in metadata, (
                f"Metadata key missing for index type {index_type}: {item.key}"
            )

    filters = [
        QueryMetadata(key=item.key, value=metadata[item.key], operator=item.operator)
        for item in echo_index.metadata_columns
        if item.key in metadata
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
                "output_pydantic": QEResponse,
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
        llm=get_crew_llm(),
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
    inputs = {"query": query, "context": sub_queries_context_str}
    response = await qe_crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


def run_perplexica_subquery(perplexica_subquery: PerplexicaSubQuery):
    if perplexica_subquery.source_extraction_prompts:
        source_extraction_prompts = perplexica_subquery.source_extraction_prompts
        system_prompt = source_extraction_prompts.system_prompt
        user_prompt = source_extraction_prompts.user_prompt
        response = call_api_with_extracted_sources(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query=perplexica_subquery.query,
        )
        response_str = f"{response.message}\n\n" + "\n".join(
            [
                f"Title: {source.title}\nURL: {source.url}\nContent: {source.data}\n"
                for source in response.extracted_sources
            ]
        )
        return response_str
        
        
        
    response = call_api(query=perplexica_subquery.query)
    response_str = f"{response['message']}\n\n"
    return response_str


def run_llm_subquery(sub_query: LLMSubQuery):
    return run_openai_query(sub_query.query, sub_query.use_web_search)


def get_buyer_research(metadata: dict) -> str:
    metadata_filters = get_metadata_filters(IndexType.BUYER_RESEARCH, metadata)  # noqa: F821
    vector_index = get_vector_index(metadata["seller"], IndexType.BUYER_RESEARCH)
    retriever = vector_index.as_retriever(filters=metadata_filters)
    docs: List[NodeWithScore] = retriever.retrieve("")
    print(metadata_filters)
    print(metadata)
    assert len(docs) > 0, f"No Buyer Research documents found for {metadata['buyer']}"
    return "\n\n".join([d.text for d in docs])


def run_kg_subquery(
    sub_query: KGSubQuery, 
    sub_query_inputs: Dict[str, str]
):
    return ask_kg(
        sub_query_inputs['seller'],
        sub_query_inputs['buyer'],
        sub_query.query,
    )

def get_buyer_account_plan(metadata: dict) -> str:
    metadata_filters = get_metadata_filters(
        IndexType.BUYER_ACCOUNT_PLAN, metadata
    )  # noqa: F821
    vector_index = get_vector_index(metadata["seller"], IndexType.BUYER_ACCOUNT_PLAN)
    retriever = vector_index.as_retriever(filters=metadata_filters)
    docs: List[NodeWithScore] = retriever.retrieve("")
    assert len(docs) > 0, f"No Buyer Research documents found for {metadata['buyer']}"
    return "\n\n".join([d.text for d in docs])


def run_sub_queries(
    query: Query,
    inputs: Dict[str, str],
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE,
    similarity_top_k=SIMILARITY_TOP_K,
    **kwargs,
) -> List[Dict]:
    def doc_data(doc: Document):
        lambda doc: f"Document Text: {doc.text}\n"
        +f"Document Metadata: {doc.metadata}"

    def retrieve_content(query: str, filters: MetadataFilters):
        docs: List[NodeWithScore] = vector_index.as_retriever(
            filters=filters, similarity_top_k=similarity_top_k, **kwargs
        ).retrieve(query)
        return "Relevant Document Details\n" + "\n".join(
            [doc_data(doc) for doc in docs]
        )
    
    def replace_sub_query_variables():
        variables = get_variables_from_prompt(sub_query.query)
        variable_values = dict()
        for variable in variables:
            if variable in sub_query_outputs:
                variable_values[variable] = sub_query_outputs[variable]
            else:
                # variable_values[variable] = "{" + variable + "}"
                assert variable in sub_query_inputs, (
                    f"Input variable {variable} in sub query '{sub_query.query}'"
                    f" not found in inputs or outputs of previous: {sub_query_inputs}"
                )
                variable_values[variable] = sub_query_inputs[variable]
                
        sub_query.query = sub_query.query.format(**variable_values)
    
    def update_sub_query():
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


    def query_content(query: str, filters: MetadataFilters):
        response = vector_index.as_query_engine(
            filters=filters, similarity_top_k=similarity_top_k, **kwargs
        ).query(query)
        return "Relevant Context:\n" + str(response)

    
    assert all(a in inputs for a in ['buyer', 'seller']), (
        f"Buyer and Seller metadata not found in inputs: {inputs.keys()}"
    )
    sub_queries_context = []
    sub_query_outputs = dict()

    for sub_query in query.sub_queries:
        print("Running sub query", sub_query.query)
        #sub_query.query += f"{buyer_seller_context}"
        sub_query_inputs = copy.deepcopy(inputs)
        if sub_query.inputs:
            sub_query_inputs.update(sub_query.inputs)
        
        replace_sub_query_variables()
        update_sub_query()
        
        if isinstance(sub_query, LLMSubQuery):
            context = run_llm_subquery(sub_query)
        elif isinstance(sub_query, PerplexicaSubQuery):
            context = run_perplexica_subquery(sub_query)
        elif isinstance(sub_query, LlamaSubQuery):
            metadata_filters = get_metadata_filters(sub_query.index_type, sub_query_inputs)
            vector_index = get_vector_index(sub_query_inputs['seller'], sub_query.index_type)
            context = query_content(sub_query.query, metadata_filters)\
            if context_extraction_mode == ContextExtractionMode.QUERY_ENGINE\
            else retrieve_content(sub_query.query, metadata_filters)
        elif isinstance(sub_query, KGSubQuery):
            context = run_kg_subquery(sub_query, sub_query_inputs)
        else:
            raise ValueError((
                f"Unknown sub query type: {type(sub_query)}. "
                "Supported types are LlamaSubQuery and PerplexicaSubQuery."
            ))
        
        sub_query_outputs[sub_query.output_name] = context
            
        sub_queries_context.append({
            "query": sub_query.query,
            "context": context,
        })
        print("Sub query context", sub_queries_context[-1])

    return sub_queries_context


async def aget_query_response(
    echo_query: Query,
    inputs: Dict[str, str],
    response_format=ResponseFormat.MARKDOWN,
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE,
    **kwargs,
):
    context = run_sub_queries(echo_query, inputs, context_extraction_mode, **kwargs)
    response = await aget_qe_crew_response(echo_query.query, context, response_format)
    return response, context


async def arun_query_chain(
    query_chain: QueryChain,
    inputs: Dict[str, str],
    response_format: ResponseFormat = ResponseFormat.MARKDOWN,
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE
) -> QueryResponse:
    query_responses = list()
    query_outputs = dict()
    query_inputs = copy.deepcopy(inputs)
    for echo_query in query_chain.queries:
        print("Running query", echo_query.query)
        
        variables = get_variables_from_prompt(echo_query.query)
        variable_values = dict()
        for variable in variables:
            if variable in query_outputs:
                variable_values[variable] = query_outputs[variable]
            else:
                variable_values[variable] = "{" + variable + "}"
                assert variable in inputs, (
                    f"Input variable {variable} in query '{echo_query.query}' not found in inputs or outputs of previous: {inputs.keys()} | {query_outputs.keys()}"
                )
        
        
        echo_query.query = echo_query.query.format(**variable_values)
        response, sub_queries_context = await aget_query_response(
            echo_query, query_inputs, 
            response_format, context_extraction_mode
        )
        query_outputs[echo_query.output_name] = response
        query_inputs.update(query_outputs)
        
        query_responses.append(
            {
                "query": echo_query.query,
                "response": response,
                "response_summary": summarize_text(response),
                "sub_queries_context": sub_queries_context,
            }
        )
    summary = summarize_text(
        "\n\n".join(
            [
                f"{r['query']}: {r['response']}"
                for r in query_responses
            ]
        )
    )
    return QueryResponse(
        summary=summary,
        responses=[
            SingleQueryResponse(
                query=r["query"],
                response=r["response"],
                sub_queries_context=r["sub_queries_context"],
                summary=r["response_summary"],
            )
            for r in query_responses
        ],
    )


async def arun_queries(
    queries: Dict[str, Dict[str, Query]],
    inputs: Dict[str, str],
    response_format=ResponseFormat.MARKDOWN,
    context_extraction_mode: ContextExtractionMode = ContextExtractionMode.QUERY_ENGINE,
    **kwargs,
) -> QueryResponse:
    query_responses = list()
    for call_type, call_queries in queries.items():
        print(f"Running queries for call type {call_type}")
        for i, query_name in async_tqdm(enumerate(call_queries), desc="Running Queries"):
            query: Query = queries[query_name]
            
            response, sub_queries_context = await aget_query_response(
                query, inputs, 
                response_format, context_extraction_mode, **kwargs
            )
            query_responses.append({
                "query_name": query_name,
                "response": response,
                "response_summary": summarize_text(response),
                "sub_queries_context": sub_queries_context,
            })

    summary = summarize_text(
        "\n\n".join(
            [
                f"{r['query_name']}: {r['response']}"
                for r in query_responses
            ]
        )
    )
    return QueryResponse(
        summary=summary,
        responses=[
            SingleQueryResponse(
                query=r["query"],
                response=r["response"],
                sub_queries_context=r["sub_queries_context"],
                summary=r["response_summary"],
            )
            for r in query_responses
        ],
    )
