import echo.sqldb as sqldb
from echo.data.indexes import IndexDataType, IndexType
from echo.indexing import IndexType
from echo.query_executor import (  # type: ignore
    LlamaSubQuery,
    LLMSubQuery,
    PerplexicaSourceExtraction,
    PerplexicaSubQuery,
    Query,
    QueryChain,
)
from echo.step_templates.utilities.account_plan_creation import QueryTypes


def get_multithread_query_chain(seller, buyer):
    account_plan_value_prop = Query(
        query="""
    You are a strategic sales assistant. Given a set of company initiatives and the product profile of the sellers product below as context,
    identify which initiatives are *relevant* to what this product solves.

    For each initiative:
    - Mark as Relevant or Not Relevant
    - If Relevant: explain which product capability maps to it
    - If Not Relevant: explain why it's not a fit (e.g., not adjacent, unrelated)


    output format:
        "initiative": "...",
        "relevant": not_relevant/ mid / highly relevant,
        "mapped_to_product": "...",
        "reasoning": "..."
        "similar buyers and their roi': "...",

    """,
        sub_queries=[
            LlamaSubQuery(
                query="What is the details on the industry and products of the buyer account?",
                index_type=IndexType.BUYER_RESEARCH,
            ),
            LlamaSubQuery(
                query="What are the top 3 financial priorities for the account to solve for?",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.FMOD.value},
            ),
            LlamaSubQuery(
                query="What are the top 3 competitors that buyer might be worried about and want to tackle",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            ),
            LlamaSubQuery(
                query="What is the most relevant news and recent media for the buyeraccount?",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.RECENTNEWS.value},
            ),
            LlamaSubQuery(
                query="What are the top 3 strategic priorities for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
            LlamaSubQuery(
                query="What are the top value propositions of the sellers product and what pains do they solve for customers. Dont give generic answers, but deep pains and priotrities of their buyers theyve solved for",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
            LlamaSubQuery(
                query="What are the exhaustive use cases and benefits of the sellers product? Dont be generic, be specific and also include details of how the use cases are tackled by the sellers product",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
            LlamaSubQuery(
                query="What case studies and testimonials do we have for the sellers product? Please include details of the case studies and testimonials and how they align with the buyers priorities",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
        ],
        output_name="account_plan_value_prop",
    )
    multi_threading_team_gen = Query(
        query=(
            "You're an experienced enterprise seller. Given these company initiatives, for the ones marked relevant to the seller's product, "
            "Infer which internal team likely owns or sponsors each initiative. \n"
            "If multiple teams are involved, note primary and secondary.\n"
            "Input:\n"
            "{account_plan_value_prop}\n"
            "Return format:\n"
            "- Initiative: ...\n"
            "- Likely owning team(s): ...\n"
            "- Reasoning:\n"
        ),
        sub_queries=[
            LlamaSubQuery(
                query="What is the details on the industry and products of the buyer account?",
                index_type=IndexType.BUYER_RESEARCH,
                inputs={"data_type": IndexDataType.BUYER_RESEARCH_DATA.value},
            )
        ],
        output_name="multi_threading_team_gen",
    )

    perplexity_search_query = Query(
        query="""
    You are provided with the company initiatives and the owning team for each of them.
    
    {multi_threading_team_gen}.
    
    You are provided with the company initiatives and the relevant information about from the web about the potential team members and employees of the company that would be relevant to the company initiatives.
    Now, for each of the initiatives, Search for all possible employees and leaders from LinkedIn data who belong to that team and extract role, name, and background.
    """,
        sub_queries=[
            PerplexicaSubQuery(
                query=(
                    "You are provided with the company initiatives and the owning team for each of them\n"
                    "{multi_threading_team_gen}\n"
                    "Now, for each of the initiatives, Search for all possible employees and leaders from LinkedIn data who belong to that team and extract role, name, and background.\n"
                ),
                source_extraction_prompts=PerplexicaSourceExtraction(
                    system_prompt=(
                        "You are a strategic sales assistant. Given the list of initiatives, owning team and reasoning "
                        "You need to extract the team members and employees of the company that would be relevant to the company initiatives.\n"
                    ),
                    user_prompt=(
                        "You are provided with the company initiatives and the owning team for each of them\n"
                        "You need to extract the team members and employees of the company that would be relevant to the company initiatives.\n"
                        "Extract out the team members or employees from the below data\n"
                    ),
                ),
            )
        ],
        output_name="perplexity_search_query",
    )

    # 2) do a perplexica serach here using team name and buyer name FOR EACH INITIATIVE RETURNED FROM ABOVE
    # Search for all possible employees and leaders  from linkedin who belong to that team and extract role, name, and background

    # 3) multi threading ROLE AND PERSON EXTRACTOR - use above response also aas input below additionally

    multi_threading_person_extractor = Query(
        query="""
        You are a strategic sales assistant. Given the list of initiatives, owning team and reasoning 
        for every relavant initiative the company is pursuing.
        
        
        You are provided with the company initiatives and the owning team for each of them.
        {multi_threading_team_gen}.
        
        You are further provided with the relevant team members and employees of the company that would be relevant to the company initiatives as crawled from the web.
        {perplexity_search_query}
        
        Do the following:

        Given the company {buyer}, and the owning team of that initiative and the team members crawled from linkedin,
        return people who match titles commonly associated with owning this initiative.
        Focus on seniority, team fit, and tenure. Prioritize those with likely budget/influence.

        Also, Classify each as a champion, decision maker, gatekeeper and influncer within the team responsible for the initiative.
        champion - one who directly owns the pain and will want it solved
        decision maker- the one with power to purchase in the team and for the initiative
        gatekeeper - the one who will block the deal from happening or be tough to convince. This is the only role that could be outside the team like procurement , legal etc.
        influencer - the one who will influence the decision maker and champion to buy the product.

        Return:
        - initiative
        - Name
        - Title
        - Tenure
        - Team
        - Reason they likely own this initiative
        - Classification (champion, decision maker, gatekeeper, influencer) and why

        here is the list of initiatives and the owning team for each of them:
        {multi_threading_team_gen}
    """,
        sub_queries=[
            LlamaSubQuery(
                query="What is the details on the industry and products of the buyer account?",
                index_type=IndexType.BUYER_RESEARCH,
                inputs={
                    "data_type": IndexDataType.BUYER_RESEARCH_DATA.value
                },  ## Has demo data separately
            )
        ],
        output_name="multi_threading_person_extractor",
    )

    # 4) multi threading outreach generator

    multi_threading_outreach_generator = Query(
        query="""
        You're a strategic AE selling {seller}. 
        You are given a list of initiatives, persona to target, title, reasoning and initiative they are participating in. For each buyer in the list
        Do the following:


        Based on this buyer's title, initiative, 
        and recent activity, seller's product details and generate a 1st outreach email that aligns to their business goals and personal context.
        You are also given similar companies the seller has helped before below.

        Tone: Crisp, consultative, relevant.

        Return:
        - Subject line
        - Message body (under 100 words)
        - CTA
        - Persona
        - Title
        - Reasoning for message


        Input:
        {multi_threading_person_extractor}
        
    """,
        sub_queries=[
            LlamaSubQuery(
                query="what is the seller's product and what pains does it solve?",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
            LlamaSubQuery(
                query="what are some companies the sellers product has helped before? be specific and metric driven",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
        ],
        output_name="multi_threading_outreach_generator",
    )

    query_chain = QueryChain(
        queries=[
            account_plan_value_prop,
            multi_threading_team_gen,
            perplexity_search_query,
            multi_threading_person_extractor,
            multi_threading_outreach_generator,
        ]
    )
    return query_chain


def get_competitor_query_chain(seller, buyer):
    db = sqldb.get_records(
        seller,
        IndexType.SELLER_RESEARCH.value,
        condition_dict={"data_type": IndexDataType.COMPETITOR_WEBSITE_DATA.value},
    )
    competitors = []
    for record in db:
        competitors.append(record["data"]["url"])

    competitors = list(set(competitors))
    competitor_queries = []
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
            sub_queries=[
                LlamaSubQuery(
                    query="What is the details on the industry and products of the buyer account?",
                    index_type=IndexType.BUYER_RESEARCH,
                    inputs={"data_type": IndexDataType.BUYER_RESEARCH_DATA.value},
                ),
                LlamaSubQuery(
                    query="What are the top 3 financial priorities for the account to solve for?",
                    index_type=IndexType.BUYER_ACCOUNT_PLAN,
                    inputs={"query_type": QueryTypes.FMOD.value},
                ),
                LlamaSubQuery(
                    query="What are the top 3 competitors that buyer might be worried about and want to tackle",
                    index_type=IndexType.BUYER_ACCOUNT_PLAN,
                    inputs={"query_type": QueryTypes.COMPANALYSIS.value},
                ),
                LlamaSubQuery(
                    query="What is the most relevant news and recent media for the buyer account?",
                    index_type=IndexType.BUYER_ACCOUNT_PLAN,
                    inputs={"query_type": QueryTypes.RECENTNEWS.value},
                ),
                LlamaSubQuery(
                    query="What are the top 3 strategic priorities for the account to solve for?",
                    index_type=IndexType.BUYER_ACCOUNT_PLAN,
                    inputs={"query_type": QueryTypes.STRATEGY.value},
                ),
                LlamaSubQuery(
                    query="What are the top value propositions of the sellers product and what pains do they solve for customers. Dont give generic answers, but deep pains and priotrities of their buyers theyve solved for",
                    index_type=IndexType.SELLER_RESEARCH,
                    # inputs={"query_type": QueryTypes.STRATEGY.value},
                ),
                LlamaSubQuery(
                    query="What are the exhaustive use cases and benefits of the sellers product? Dont be generic, be specific and also include details of how the use cases are tackled by the sellers product",
                    index_type=IndexType.SELLER_RESEARCH,
                    inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
                ),
                LlamaSubQuery(
                    query="What case studies and testimonials do we have for the sellers product? Please include details of the case studies and testimonials and how they align with the buyers priorities",
                    index_type=IndexType.SELLER_RESEARCH,
                    inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
                    # inputs={"query_type": QueryTypes.STRATEGY.value},
                ),
                LlamaSubQuery(
                    query=f"What are the top value propositions of the competitors {competitor} product and what pains do they solve for customers. Dont give generic answers, but deep pains and priotrities of their buyers theyve solved for",
                    index_type=IndexType.SELLER_RESEARCH,
                    inputs={"data_type": IndexDataType.COMPETITOR_WEBSITE_DATA.value},
                    # inputs={"seller":"https://www.pendo.io"}
                    # inputs={"query_type": QueryTypes.STRATEGY.value},
                ),
                LlamaSubQuery(
                    query=f"What are the exhaustive use cases and benefits of the competitor {competitor} product? Dont be generic, be specific and also include details of how the use cases are tackled by the sellers product",
                    index_type=IndexType.SELLER_RESEARCH,
                    inputs={"data_type": IndexDataType.COMPETITOR_WEBSITE_DATA.value},
                ),
                LlamaSubQuery(
                    query=f"What case studies and testimonials do we have for the competitor {competitor} product? Please include details of the case studies and testimonials and how they align with the buyers priorities",
                    index_type=IndexType.SELLER_RESEARCH,
                    inputs={"data_type": IndexDataType.COMPETITOR_WEBSITE_DATA.value},
                ),
            ],
        )
        competitor_queries.append(value_prop_query_competitor)
    query_chain = QueryChain(
        queries=competitor_queries,
    )
    return query_chain


def get_query_account_plan_query_chain():
    account_plan_value_prop = Query(
        query="""
    You are a strategic sales assistant. Given a set of company initiatives and the product profile of the sellers product below as context,
    identify which initiatives are *relevant* to what this product solves.

    For each initiative:
    - Mark as Relevant or Not Relevant
    - If Relevant: explain which product capability maps to it
    - If Not Relevant: explain why it's not a fit (e.g., not adjacent, unrelated)


    output format:
        "initiative": "...",
        "relevant": not_relevant/ mid / highly relevant,
        "mapped_to_product": "...",
        "reasoning": "..."
        "similar buyers and their roi': "...",

    """,
        sub_queries=[
            # LlamaSubQuery(
            #     query="What is the details on the industry and products of the buyer account?",
            #     index_type=IndexType.BUYER_RESEARCH,
            # ),
            # LlamaSubQuery(
            #     query="What are the top 3 financial, strategic and competitive goals and pains for the buyer account to solve for?",
            #     index_type=IndexType.BUYER_RESEARCH,
            # ),
            # LlamaSubQuery(
            #     query="What are the top 3 financial priorities for the account to solve for?",
            #     index_type=IndexType.BUYER_ACCOUNT_PLAN,
            #     inputs={"query_type": QueryTypes.FMOD.value},
            # ),
            # LlamaSubQuery(
            #     query="What are the top 3 competitors that buyer might be worried about and want to tackle",
            #     index_type=IndexType.BUYER_ACCOUNT_PLAN,
            #     inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            # ),
            # LlamaSubQuery(
            #     query="What is the most relevant news and recent media for the buyeraccount?",
            #     index_type=IndexType.BUYER_ACCOUNT_PLAN,
            #     inputs={"query_type": QueryTypes.RECENTNEWS.value},
            # ),
            # LlamaSubQuery(
            #     query="What are the top 3 strategic priorities for the account",
            #     index_type=IndexType.BUYER_ACCOUNT_PLAN,
            #     inputs={"query_type": QueryTypes.STRATEGY.value},
            # ),
            LlamaSubQuery(
                query="What are the top value propositions of the sellers product and what pains do they solve for customers. Dont give generic answers, but deep pains and priotrities of their buyers theyve solved for",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
            LlamaSubQuery(
                query="What are the exhaustive use cases and benefits of the sellers product? Dont be generic, be specific and also include details of how the use cases are tackled by the sellers product",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
            LlamaSubQuery(
                query="What case studies and testimonials do we have for the sellers product? Please include details of the case studies and testimonials and how they align with the buyers priorities",
                index_type=IndexType.SELLER_RESEARCH,
                inputs={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
            ),
        ],
        output_name="account_plan_value_prop",
    )

    company_summary = Query(
        query="You're an analyst understanding the company and the industry of. {buyer}.\n"
        "You are given news, company financials, competitors and strategic priorities of the company.\n"
        "Extract an overview of the company and the industry.\n"
        "Also extract the company's product details and the core problems it solves for its buyers.\n"
        "Find its top clients and the top impacts it has had on them as well"
        "This information would be used by a seller to understand the company and the industry better.\n"
        "Also give signals on how well versed and tech centric the industry is, how regulatory it is and how much of a market leader the company is.\n"
        "Signals for the various initiatives are given below:\n"
        "\nVague Return format:\n"
        "- Insight: clear description\n"
        "- Supporting evidence: source\n",
        sub_queries=[
            LlamaSubQuery(
                query="Summarize all the finacial events and plans and details for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.FMOD.value},
            ),
            LlamaSubQuery(
                query="What are the top 3 competitors for the account and what goals would account have with respect to competitors. Include any media about competitors",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            ),
            LlamaSubQuery(
                query="summarize all  relevant news and events for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.RECENTNEWS.value},
            ),
            LlamaSubQuery(
                query="summarize all the strategic priorities and decisions being made for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
        ],
        output_name="company_summary",
    )

    account_plan = Query(
        query="You're a strategic analyst that needs to help sellers from the seller company {seller}"
        "can know how to frame their value prop of their product with respect to the buyer {buyer} initiatives"
        "Based on the following public signals about {buyer}.\n"
        "Extract 3-5 key initiatives or priorities the company is likely pursuing this year or quarter.\n"
        "Phrase each as a business goal. Do NOT include vague goals. Be specific.\n"
        "Signals for the various initiatives are given below:\n"
        "Dont talk about your role as a b2b seller and all"
        "\nReturn format:\n"
        "- Initiative: clear description\n"
        "- Supporting evidence: source\n",
        sub_queries=[
            LlamaSubQuery(
                query="Summarize all the finacial events and plans and details for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.FMOD.value},
            ),
            LlamaSubQuery(
                query="What are the top 3 competitors for the account and what goals would account have with respect to competitors. Include any media about competitors",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            ),
            LlamaSubQuery(
                query="summarize all  relevant news and events for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.RECENTNEWS.value},
            ),
            LlamaSubQuery(
                query="summarize all the strategic priorities and decisions being made for the account",
                index_type=IndexType.BUYER_ACCOUNT_PLAN,
                inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
        ],
        output_name="account_plan",
    )

    return QueryChain(
        queries=[
            company_summary,
            # account_plan,
            account_plan_value_prop,
        ]
    )


"""

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
                inputs={"stakeholder": "CFO"},
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
                inputs={"call_type": CallType.PRICING.value, "stakeholder": "CFO"},
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
                index_type=IndexType.BUYER_ACCOUNT_PLAN.value,
                inputs={"query_type": QueryTypes.FMOD.value},
            ),
            SubQuery(
                query="What are the top 3 competitors for the account that that client needs to consider?",
                index_type=IndexType.BUYER_ACCOUNT_PLAN.value,
                inputs={"query_type": QueryTypes.COMPANALYSIS.value},
            ),
            SubQuery(
                query="What is the most relevant news for the account?",
                index_type=IndexType.BUYER_ACCOUNT_PLAN.value,
                inputs={"query_type": QueryTypes.RECENTNEWS.value},
            ),
            SubQuery(
                query="What are the top 3 strategic priorities for the account to solve for?",
                index_type=IndexType.BUYER_ACCOUNT_PLAN.value,
                inputs={"query_type": QueryTypes.STRATEGY.value},
            ),
        ],
    )

    
    return []
"""
