
from echo import setup_db_tables
import nest_asyncio
import asyncio
from echo.constants import (
    DISCOVERY,
    DEMO,
    PRICING,
    NEGOTIATION
)
from echo.runner import create_or_get_seller
from echo.query_executor import QueryResponse
nest_asyncio.apply()

# for queries implemented here
from echo.query_executor import (
    Query,
    SubQuery,
)

from echo.indexing import IndexType
from echo.step_templates.generic import CallType
from echo.tools.perplexity_search import QueryTypes

from echo.runner import create_or_get_seller, create_or_get_competitor
from echo.queries import get_queries
from echo.step_templates.generic import (
    CallType,
    Transcript,
    add_previous_call_analysis,
    aget_clients_call_data,
)


def generate_competitor_data(competitors):
    for competitor in competitors:
        # create or get the competitor data
        competitor_data = asyncio.run(create_or_get_competitor(competitor))


"""
Flow
1. competitior extraction and research
1.1 get top 3 competitors from llm using the seller name and context from seller index
1.2 query internet to find landing pages and crawl all info on them
1.3 see how to dump to an index (create new index type)


2. competitor value prop and differentiation

"""

#account_plan_query = queries['prediscovery']
from echo.query_executor import arun_queries, aget_query_response
from echo.query_executor import ResponseFormat
from echo.query_executor import ContextExtractionMode
import nest_asyncio
import asyncio

def get_competitors_comparison_value_prop(competitors, seller, buyer):

    nest_asyncio.apply()

    queries = get_queries(seller=seller)
    print(queries)
    value_prop_query_seller = queries[CallType.PREDISCOVERY.value]["Buyer Account Plan Value Prop"]
    queries_to_execute = {
        CallType.PREDISCOVERY.value:{
            "Buyer Account Plan Value Prop": value_prop_query_seller
        }
    }

    inputs = {"buyer": buyer, "company_size": "Enterprise", "seller": seller}

    # single query endpoint
    # same runs for multiple queries too
    responses = asyncio.run(
        # arun_queries(
        #     queries=queries_to_execute,
        #     inputs=inputs,
        #     response_format=ResponseFormat.MARKDOWN,
        #     context_extraction_mode=ContextExtractionMode.QUERY_ENGINE,
        # )
        aget_query_response(
            query=value_prop_query_seller,
            inputs=inputs,
            response_format=ResponseFormat.MARKDOWN,
            context_extraction_mode=ContextExtractionMode.QUERY_ENGINE,
        )
    )
    # capture response of above query
    value_prop_seller = str(responses[0])

    



    # get relevant query for competitor value_prop 
    # rewrite the query to include the competitor name and compariosn with above seller value prop  
    value_prop_competitors = {} 
    for competitor in competitors:
        value_prop_query_competitor = Query(
        query=f"""You need to help an account executive of our seller company. {seller} differentiate against the competitor {competitor} in relevance to the buyer {buyer}.
                The seller is trying to create a business case for the buyer and you need to help the account exectutive differentiate the sellers product from the competitors product.
                The seller is {seller}, the competitor {competitor} and the buyer is {buyer}.
                The top financial, strategic, competitive and priorities evident from news and media to craft top issues and focus points of the buyer are given.
                Next deeply understand the sellers product, the core problems it solves for its buyers.
                Next deeply understand the competitors product, the core porblems it solves for its buyers.
                Consider deep differentiation and not just surface level differentiation between the seller and the COMPETITOR IN CONTEXT TO THE BUYERS PRIORITIES.
                HIGHLIGHT WHERE THE SELLERS VALUE PROP CAN BE STRONGER AND WHY AND WHERE THE COMPETITORS VALUE PROP CAN BE STRONGER AND HOW TO TACKLE THAT IN THE BUSINESS CASE FOR THE BUYER.  
                Please make sure to properly align value prop to actual business cases and not just generic value prop. 
                Also understand deeply what the seller sells and the kind of impact it can have before answering. 
                Think deeply
                Now finally, craft a set of value propositions and business cases that the sellers product can solve in alignment with the buyers priorities identified. This will be used by an account executive to pitch the product to the buyer and align with their priorities. so be clear, detailed and specific.
                Use the sellers product info, testimonials, broad initiatives theyve tackled for other customers and how they can align with the buyers strategic, financial and competitive priorities. 
                Also include news and media about the buyer into consideration for further hints and signals on buyer priorities.
                USE SAME DATA FOR COMPETITORS TO IDENTIFY HOW THE SELLER CAN TACKLE THE COMPETITORS VALUE PROP AND HOW THEY CAN ALIGN WITH THE BUYERS PRIORITIES BETTER.
                """,
        seller=seller,
        call_type=CallType.PREDISCOVERY.value,
        sub_queries=[
            SubQuery(
                query="What is the details on the industry and products of the buyer account?",
                index_type=IndexType.BUYER_RESEARCH.value,
                #inputs={"query_type": QueryTypes.FMOD.value},
            ),
            SubQuery(
                query="What are the top 3 financial priorities for the account to solve for?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.FMOD.value},
            ),
            SubQuery(
                query="What are the top 3 competitors that buyer might be worried about and want to tackle",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            ),
            SubQuery(
                query="What is the most relevant news and recent media for the buyer account?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.RECENTNEWS.value},
            ),
            SubQuery(
                query="What are the top 3 strategic priorities for the account to solve for?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            SubQuery(
                query="What are the top value propositions of the sellers product and what pains do they solve for customers. Dont give generic answers, but deep pains and priotrities of their buyers theyve solved for",
                index_type=IndexType.SELLER_RESEARCH.value,
                #inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            SubQuery(
                query="What are the exhaustive use cases and benefits of the sellers product? Dont be generic, be specific and also include details of how the use cases are tackled by the sellers product",
                index_type=IndexType.SELLER_RESEARCH.value,
                #inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            SubQuery(
                query="What case studies and testimonials do we have for the sellers product? Please include details of the case studies and testimonials and how they align with the buyers priorities",
                index_type=IndexType.SELLER_RESEARCH.value,
                #inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            SubQuery(
                query= f"What are the top value propositions of the competitors {competitor} product and what pains do they solve for customers. Dont give generic answers, but deep pains and priotrities of their buyers theyve solved for",
                index_type=IndexType.COMPETITORS.value,
                #inputs={"seller":"https://www.pendo.io"}
                #inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            SubQuery(
                query= f"What are the exhaustive use cases and benefits of the competitor {competitor} product? Dont be generic, be specific and also include details of how the use cases are tackled by the sellers product",
                index_type=IndexType.COMPETITORS.value,
                #inputs={"seller":"https://www.pendo.io"},
                #inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            SubQuery(
                query=f"What case studies and testimonials do we have for the competitor {competitor} product? Please include details of the case studies and testimonials and how they align with the buyers priorities",
                index_type=IndexType.COMPETITORS.value,
                #inputs={"seller":"https://www.pendo.io"},
                #inputs={"query_type": QueryTypes.STRATEGY.value},
            )
        ],
        )
    
        # run the query
        inputs = {"buyer": buyer, "company_size": "Enterprise", "seller": seller,"competitor": competitor}

        # single query endpoint
        # same runs for multiple queries too
        responses = asyncio.run(
            # arun_queries(
            #     queries=value_prop_query_competitor,
            #     inputs=inputs,
            #     response_format=ResponseFormat.MARKDOWN,
            #     context_extraction_mode=ContextExtractionMode.QUERY_ENGINE,
            # )
            aget_query_response(
            query=value_prop_query_competitor,
            inputs=inputs,
            response_format=ResponseFormat.MARKDOWN,
            context_extraction_mode=ContextExtractionMode.QUERY_ENGINE,
        )
        )
        # capture response of above query
        value_prop_competitors[competitor] = str(responses[0])
    
    return value_prop_competitors, value_prop_seller
