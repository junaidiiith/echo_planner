from echo.query_executor import (
    Query,
    SubQuery,
)

from echo.indexing import IndexType
from echo.step_templates.generic import CallType
from echo.tools.perplexity_search import QueryTypes


def get_queries(seller):
    discovery_info_to_cover = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query=(
            "Given all the information needed to uncover and the current information uncovered till now - "
            "What all pieces of information are missing that should be uner and the current information uncovered till now"
            "What all pieces of information are missing that should be uncovered in upcoming calls?"
        ),
        sub_queries=[
            SubQuery(
                query="What all pieces of information about the buyer are uncovered in discovery calls of successful deals?",
                index_type=IndexType.ANALYSIS.value,
                inputs = {"stakeholder": "CFO"}
            ),
            SubQuery(
                query="What all piece of information have we learnt about the buyer from discovery till now?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
        ],
    )

    stakeholder_priorities = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query="What are the top priorities this quarter for the account to solve for?",
        sub_queries=[
            SubQuery(
                query="What are the top priorities for the account to solve for?",
                index_type=IndexType.BUYER_RESEARCH.value,
            )
        ],
    )

    value_proposition = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query=(
            "Given stakeholder priorities and historical ways we've presented product to similar buyers - "
            "How can I present the value proposition of this product that aligns with my prospects industry and current stakeholders priorities?"
        ),
        sub_queries=[
            SubQuery(
                query="What are the top priorities and possible pains for my prospect in the upcoming quarters?",
                index_type=IndexType.BUYER_RESEARCH.value,
            ),
            SubQuery(
                query="What are the historical ways to present the product value prop to similar buyers",
                index_type=IndexType.ANALYSIS.value,
            ),
        ],
    )

    discovery_questions = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query="What are the top relevant discovery questions that have been successful for similar buyers and the respective stakeholder?",
        sub_queries=[
            SubQuery(
                query="What are the top relevant discovery questions that have been successful for similar buyers and the respective stakeholder?",
                index_type=IndexType.ANALYSIS.value,
            )
        ],
    )

    competitor_analysis = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query="What are my top competitors and our differentiation?",
        sub_queries=[
            SubQuery(
                query="What are the top competitors for the account and what are the key differentiators?",
                index_type=IndexType.BUYER_RESEARCH.value,
            ),
            SubQuery(
                query="Who are my top competitors and how do i differentiate ourselves compared to each of them to better solve the buyers issues?",
                index_type=IndexType.ANALYSIS.value,
            ),
        ],
    )

    decision_makers = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query="What all pieces of information about the decision making committee is pending given the current information I have?",
        sub_queries=[
            SubQuery(
                query="What all questions and information about the decision making process do i need to gather from the buyer?",
                index_type=IndexType.ANALYSIS.value,
            ),
            SubQuery(
                query="What all pieces of information do i already have about the decision makers and buying committee?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
        ],
    )

    rapport_building = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query="what are some rapport building cues I could use with the prospect?",
        sub_queries=[
            SubQuery(
                query="Based on the historical calls with the buyer, what are some ways or topics to build rapport?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
            SubQuery(
                query="Based on similar buyers, what kind of talking points about the industry could help build rapport?",
                index_type=IndexType.ANALYSIS.value,
            ),
            SubQuery(
                query="Based on the external research of the buyer, what kind of talking points could help build rapport?",
                index_type=IndexType.BUYER_RESEARCH.value,
            ),
        ],
    )

    possible_objections = Query(
        seller=seller,
        call_type=CallType.DISCOVERY.value,
        query="What are some possible objections the prospect could raise regarding our offering based on similar buyers in the past. Also provide how to handle them?",
        sub_queries=[
            SubQuery(
                query="What are some possible objections the prospect could raise regarding our offering based on similar buyers in the past. Also provide how to handle them?",
                index_type=IndexType.ANALYSIS.value,
            ),
            SubQuery(
                query="What are some possible objections the buyer could raise given their details and pains?",
                index_type=IndexType.BUYER_RESEARCH.value,
            ),
        ],
    )

    top_pains_identified = Query(
        seller=seller,
        call_type=CallType.DEMO.value,
        query="What are the top pains identified for this account from discovery?",
        sub_queries=[
            SubQuery(
                query="What are the top pains identified for the buyer in the past?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": CallType.DISCOVERY.value},
            )
        ],
    )

    features_to_demo = Query(
        seller=seller,
        query="Based on historical call features presented and the features identified from the product info, collate responses on what features to present based on pains identified",
        call_type=CallType.DEMO.value,
        sub_queries=[
            SubQuery(
                query="What are the top pains identified from the discovery phase?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": CallType.DISCOVERY.value},
            ),
            SubQuery(
                query="What features align best with the pains identified?",
                index_type=IndexType.SELLER_RESEARCH.value,
                context_tasks=[0],
            ),
            SubQuery(
                query="What features best align with the pains identified for similar buyers?",
                index_type=IndexType.ANALYSIS.value,
                context_tasks=[0],
            ),
        ],
    )

    possible_demo_objections = Query(
        seller=seller,
        call_type=CallType.DEMO.value,
        query="Given objections from current deal and historical deals, How have these objections been handled successfully before in discovery and demo calls. Display output in pairs of objection and their successful response?",
        sub_queries=[
            SubQuery(
                query="What are the top objections that came up in discovery?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": CallType.DISCOVERY.value},
            ),
            SubQuery(
                query="What are the top objections that come up in demo calls for similar buyers?",
                index_type=IndexType.ANALYSIS.value,
            ),
            SubQuery(
                query=(
                    "Given objections from current deal and historical deals below, how have these objections been handled successfully before in discovery and demo calls"
                    "Display output in pairs of objection and their successful response "
                ),
                index_type=IndexType.ANALYSIS.value,
                inputs={"call_type": None},
                context_tasks=[0, 1],
            ),
        ],
    )

    missing_info_to_uncover = Query(
        query="Given pieces of information we need to uncover and the information we have till now - what pending information do I need to uncover?",
        seller=seller,
        call_type=CallType.PRICING.value,
        sub_queries=[
            SubQuery(
                query="What all pieces of information regarding pricing needs to be uncovered during pricing calls from similar buyers?",
                index_type=IndexType.ANALYSIS.value,
            ),
            SubQuery(
                query="What all pieces of information regarding pricing needs to be uncovered during discovery calls from similar buyers?",
                index_type=IndexType.ANALYSIS.value,
                inputs={"call_type": CallType.DISCOVERY.value},
            ),
            SubQuery(
                query="What all pieces of information regarding pricing needs to be uncovered during demo calls from similar buyers?",
                index_type=IndexType.ANALYSIS.value,
                inputs={"call_type": CallType.DEMO.value},
            ),
            SubQuery(
                query="what all pieces of information have we uncovered about the buyer in the current deal across discovery, demo and pricing stages?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": None},
            ),
        ],
    )

    pricing_levers = Query(
        query="Summarize the various pricing levers that will be used in a pricing call we have given the historical justifications used for similar buyers, the features that excited the buyer, and the pains mentioned by the buyer.",
        seller=seller,
        call_type=CallType.PRICING.value,
        sub_queries=[
            SubQuery(
                query="What pricing levers have been successful in the past for similar buyers?",
                index_type=IndexType.ANALYSIS.value,
            ),
            SubQuery(
                query="What features and product offerings were received positively in discovery calls?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": CallType.DISCOVERY.value},
            ),
            SubQuery(
                query="What features and product offerings were received positively in demo calls?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": CallType.DEMO.value},
            ),
            SubQuery(
                query="What are the top pains identified for the buyer in the past?",
                index_type=IndexType.CURRENT_CALL.value,
                inputs={"call_type": CallType.DISCOVERY.value},
            ),
        ],
    )

    relevant_pricing_plans = Query(
        query="Summarize the two responses around pricing plans relevant to the buyer that woukd be presented in a pricing sales call?",
        seller=seller,
        call_type=CallType.PRICING.value,
        sub_queries=[
            SubQuery(
                query="What are the top concerns and priorities of the buyer?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
            SubQuery(
                query="what are the top objections till now around the product ROI and value?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
            SubQuery(
                query="What pricing plans are relevant to buyers with pain points and priotities and top concerns around the product?",
                index_type=IndexType.ANALYSIS.value,
                context_tasks=[0, 1],
            ),
            SubQuery(
                query="What pricing plans are relevant to buyers with pain points and priotities and top concerns around the product?",
                index_type=IndexType.SELLER_RESEARCH.value,
                context_tasks=[0, 1],
            ),
        ],
    )

    roi_and_business_justification = Query(
        query="Given the pains and objections of the buyer, What are the top ways to make a business case and ROI justification to the buyer?",
        seller=seller,
        call_type=CallType.PRICING.value,
        sub_queries=[
            SubQuery(
                query="What are the top concerns and priorities of the buyer?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
            SubQuery(
                query="What are the top objections till now around the product ROI and value?",
                index_type=IndexType.CURRENT_CALL.value,
            ),
        ],
    )

    negotiation_pending_concerns = Query(
        query="What are pending concerns to be addressed before close?",
        seller=seller,
        call_type=CallType.NEGOTIATION.value,
        sub_queries=[
            SubQuery(
                query="What are pending concerns to be addressed before close?",
                index_type=IndexType.CURRENT_CALL.value,
            )
        ],
    )

    discounts_and_concessions = Query(
        query="What discounts and concessions can and have been offered?",
        seller=seller,
        call_type=CallType.NEGOTIATION.value,
        sub_queries=[
            SubQuery(
                query="What discounts and concessions can and have been offered during pricing calls?",
                index_type=IndexType.ANALYSIS.value,
                inputs={"call_type": CallType.PRICING.value, 'stakeholder': 'CFO'},
            ),
            SubQuery(
                query="What discounts and concessions can and have been offered during negotiation calls?",
                index_type=IndexType.ANALYSIS.value,
                inputs={"call_type": CallType.NEGOTIATION.value},
            ),
        ],
    )

    possible_legal_concerns = Query(
        query="What are the possible legal concerns that could come up during negotiation?",
        seller=seller,
        call_type=CallType.NEGOTIATION.value,
        sub_queries=[
            SubQuery(
                query="What are the procurement and legal concerns possible?",
                index_type=IndexType.ANALYSIS.value,
            )
        ],
    )

    closing_tactics = Query(
        query="What final closing tactics can be used for their account?",
        seller=seller,
        call_type=CallType.NEGOTIATION.value,
        sub_queries=[
            SubQuery(
                query="What are the closing tactics that have been successful in the past?",
                index_type=IndexType.ANALYSIS.value,
            )
        ],
    )
    
    account_plan = Query(
        query="What are the top 3 priorities to do for this account in the next quarter?",
        seller=seller,
        call_type=CallType.PREDISCOVERY.value,
        sub_queries=[
            SubQuery(
                query="What are the top 3 financial priorities for the account to solve for?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.FMOD.value},
            ),
            SubQuery(
                query="What are the top 3 competitors for the account that that client needs to consider?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            ),
            SubQuery(
                query="What is the most relevant news for the account?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.RECENTNEWS.value},
            ),
            SubQuery(
                query="What are the top 3 strategic priorities for the account to solve for?",
                index_type=IndexType.BUYER_FOUNDATIONAL_PLAN.value,
                inputs={"query_type": QueryTypes.STRATEGY.value},
            )
        ],
    )


    # new query for aligning value prop
    account_plan_value_prop = Query(
        query=f"""You need to generate a few value propositions and business cases that the seller Whatfix can then use to sell their solution to the buyer Manpower group. 
                The seller is whatfix and the buyer is manpower group
                First identify the top financial, strategic, competitive and priorities evident from news and media to craft top issues and focus points of the buyer.
                Next deeply understand the sellers product, the core problems it has and does solve for its buyers.
                Please make sure to properly align value prop to actual business cases and not just generic value prop. 
                Also understand deeply what the seller sells and the kind of impact it can have before answering. 
                Think deeply
                Now finally, craft a set of value propositions and business cases that the sellers product can solve in alignment with the buyers priorities identified. This will be used by an account executive to pitch the product to the buyer and align with their priorities. so be clear, detailed and specific.
                Use the sellers product info, testimonials, broad initiatives theyve tackled for other customers and how they can align with the buyers strategic, financial and competitive priorities. Also include news and media about the buyer into consideration for further hints and signals on buyer priorities
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
                query="What is the most relevant news and recent media for the buyeraccount?",
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
            )
        ],
    )

    # competitor value prop differentiator query
    competitor_differentiator_value_prop = Query(
        query=f"""You need to generate a few value propositions and business cases that the seller Whatfix can then use to sell their solution to the buyer Manpower group. 
                The seller is whatfix and the buyer is manpower group
                First identify the top financial, strategic, competitive and priorities evident from news and media to craft top issues and focus points of the buyer.
                Next deeply understand the sellers product, the core problems it has and does solve for its buyers.
                Please make sure to properly align value prop to actual business cases and not just generic value prop. 
                Also understand deeply what the seller sells and the kind of impact it can have before answering. 
                Think deeply
                Now finally, craft a set of value propositions and business cases that the sellers product can solve in alignment with the buyers priorities identified. This will be used by an account executive to pitch the product to the buyer and align with their priorities. so be clear, detailed and specific.
                Use the sellers product info, testimonials, broad initiatives theyve tackled for other customers and how they can align with the buyers strategic, financial and competitive priorities. Also include news and media about the buyer into consideration for further hints and signals on buyer priorities
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
                query="What is the most relevant news and recent media for the buyeraccount?",
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
            )
        ],
    )
    


    queries = {
        CallType.PREDISCOVERY.value: {
            #"Buyer Account Plan": account_plan,
            "Buyer Account Plan Value Prop": account_plan_value_prop,
            "Competitor Differentiator Value Prop": competitor_differentiator_value_prop,
        },   
    }
    """
    queries = {
        
        CallType.DISCOVERY.value: {
            "Info To Cover": discovery_info_to_cover,
            "Stakeholder Priorities": stakeholder_priorities,
            "Value Proposition": value_proposition,
            "Discovery Questions": discovery_questions,
            "Competitor Analysis": competitor_analysis,
            "Decision Makers": decision_makers,
            "Rapport Building": rapport_building,
            "Possible Objections": possible_objections,
        },
        CallType.DEMO.value: {
            "Top Pains Identified": top_pains_identified,
            "Features To Demo": features_to_demo,
            "Possible Demo Objections": possible_demo_objections,
            "Missing Info To Uncover": missing_info_to_uncover,
        },
        CallType.PRICING.value: {
            "Pricing Levers": pricing_levers,
            "Relevant Pricing Plans": relevant_pricing_plans,
            "ROI and Business Justification": roi_and_business_justification,
        },
        CallType.NEGOTIATION.value: {
            "Negotiation Pending Concerns": negotiation_pending_concerns,
            "Discounts and Concessions": discounts_and_concessions,
            "Possible Legal Concerns": possible_legal_concerns,
            "Closing Tactics": closing_tactics,
        },
        "Account Plan": {
            "Buyer Account Plan": account_plan,
        }
    }
    """
    return queries
