import copy
from crewai import LLM
from crewai.crews import CrewOutput
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.data.indexes import IndexDataType
from echo.settings import MAX_TEXT_TOKENS
from echo.step_templates.utilities.competitor_analysis import add_competitor_info
from echo.step_templates.utilities.account_plan_creation import create_account_plan
from echo.tools.web_scraping import extract_data_from_website
from echo.utils import (
    dict_to_markdown,
    format_response,
    get_text_upto_tokens,
    json_to_markdown,
    get_data_str,
)
from echo.step_templates.generic import (
    CallType,
    Transcript,
    add_previous_call_analysis,
    aget_clients_call_data,
)
from echo.echo_agent import EchoAgent, get_crew as get_crew_obj
from tqdm.auto import tqdm
from echo.indexing import (
    IndexType,
    add_data,
    check_metadata_exists_in_db,
    get_data_from_db,
)

from echo.constants import (
    COMPETITOR_EXTRACTION,
    SELLER_RESEARCH,
    RESEARCH,
    SIMULATION,
    EXTRACTION,
    ANALYSIS,
)


class ValueProposition(BaseModel):
    product: str = Field(
        ...,
        title="Product",
        description="The name of the product offered by the seller.",
    )
    solution: str = Field(
        ...,
        title="Solution",
        description="The description the solution offered by the seller.",
    )
    use_cases: List[str] = Field(
        ...,
        title="Use Cases",
        description="The use case descriptions of the product offered by the seller.",
    )


class SellerInfo(BaseModel):
    name: str = Field(..., title="Seller's Name", description="The name of the seller.")
    website: str = Field(
        ..., title="Seller's Website", description="The website of the seller."
    )
    description: str = Field(
        ..., title="Seller's Description", description="A description of the seller."
    )
    industry: str = Field(
        ...,
        title="Seller's Industry",
        description="The industry that the seller belongs to.",
    )

    value_propositions: List[ValueProposition] = Field(
        ...,
        title="Seller's Value Propositions",
        description="The value propositions offered by the seller.",
    )


class SellerPricingModel(BaseModel):
    type: str = Field(
        ...,
        title="Seller's Pricing Models",
        description="The pricing models offered by the seller, such free, subscription, premium, etc.",
    )
    description: str = Field(
        ...,
        title="Seller's Pricing Description",
        description="A description of the pricing model.",
    )
    price: str = Field(
        ...,
        title="Seller's Pricing numbers",
        description="The pricing (NUMBERS) of the seller's products.",
    )
    duration: str = Field(
        ...,
        title="Seller's Pricing Duration",
        description="The duration of the pricing model.",
    )


class SellerPricingModels(BaseModel):
    pricing_models: List[SellerPricingModel] = Field(
        ...,
        title="Seller's Pricing Models",
        description="The pricing models offered by the seller.",
    )


class SellerClient(BaseModel):
    name: str = Field(
        ...,
        title="Seller's Client Name",
        description="The name of the client of the seller.",
    )
    website: str = Field(
        ...,
        title="Seller's Client Website",
        description="The website of the client of the seller.",
    )


class SellerClients(BaseModel):
    clients: List[SellerClient] = Field(
        ...,
        title="Seller's Clients",
        description="The clients of the seller.",
    )


class SellerResearchResponse(BaseModel):
    info: SellerInfo = Field(
        ..., title="Seller's Information", description="The information of the seller."
    )
    pricing: SellerPricingModels = Field(
        ...,
        title="Seller's Pricing Models",
        description="The pricing models offered by the seller.",
    )


class ClientResearchResponse(BaseModel):
    name: str = Field(..., title="Buyer's Name", description="The name of the client.")
    website: str = Field(
        ..., title="Buyer's Website", description="The website of the buyer."
    )
    description: str = Field(
        ..., title="Buyer's Description", description="A description of the buyer."
    )
    industry: str = Field(
        ...,
        title="Buyer's Industry",
        description="The industry that the buyer belongs to.",
    )
    company_size: str = Field(
        ...,
        title="Buyer's Company Size Type",
        description="The type of company size as: SMB, Mid-Market, Enterprise.",
    )
    goals: List[str] = Field(
        ..., title="Buyer's Goals", description="The goals of the buyer."
    )
    use_cases: List[str] = Field(
        ..., title="Buyer's Use Cases", description="The use cases of the buyer."
    )
    challenges: List[str] = Field(
        ...,
        title="Buyer's Challenges",
        description="The challenges faced by the buyer.",
    )
    stakeholders: List[str] = Field(
        ..., title="Buyer's Stakeholders", description="The stakeholders of the buyer."
    )


class Competitor(BaseModel):
    name: str
    description: str
    url: str
    rationale: str


class CompetitorsExtractionResponse(BaseModel):
    competitors: list[Competitor]


class CompetitorComparison(BaseModel):
    name: str = Field(
        ..., title="Competitor's Name", description="The name of the competitor."
    )
    pros: List[str] = Field(
        ..., title="Competitor's Pros", description="The pros of the competitor."
    )
    cons: List[str] = Field(
        ..., title="Competitor's Cons", description="The cons of the competitor."
    )
    differentiators: List[str] = Field(
        ...,
        title="Competitor's Differentiators",
        description="The differentiators of the seller against the competitor.",
    )


class SellerCompetitorAnalysisResponse(BaseModel):
    competitors: List[CompetitorComparison] = Field(
        ...,
        title="Competitor Analysis",
        description="The list of competitors of the seller.",
    )


class ObjectionResolutionPair(BaseModel):
    objection: str = Field(
        ..., title="Objection", description="The objection raised by the buyer."
    )
    resolution: str = Field(
        ..., title="Resolution", description="The resolution provided by the seller."
    )


class AnticipatedPainsAndObjections(BaseModel):
    pains: List[str] = Field(
        ..., title="Pain Points", description="The pain points identified in the call."
    )
    objections: List[str] = Field(
        ..., title="Objections", description="The objections identified in the call."
    )


class BuyerDataExtracted(BaseModel):
    pain_points: List[str] = Field(
        ..., title="Pain Points", description="The pain points identified in the call."
    )
    objections: List[str] = Field(
        ..., title="Objections", description="The objections identified in the call."
    )
    time_lines: List[str] = Field(
        ..., title="Time Lines", description="The time lines identified in the call."
    )
    success_indicators: List[str] = Field(
        ...,
        title="Success Indicators",
        description="The success indicators identified in the call.",
    )
    budget_constraints: List[str] = Field(
        ...,
        title="Budget Constraints",
        description="The budget constraints identified in the call.",
    )
    competition: List[str] = Field(
        ..., title="Competitors", description="The competitors identified in the call."
    )
    decision_committee: List[str] = Field(
        ...,
        title="Decision Committee Members",
        description="The members of the decision committee identified in the call.",
    )


class SellerDataExtracted(BaseModel):
    discovery_questions: List[str] = Field(
        ...,
        title="Discovery Questions",
        description="The discovery questions asked by the seller.",
    )
    decision_making_process_questions: List[str] = Field(
        ...,
        title="Decision Making Process Questions",
        description="The decision making process questions asked by the seller.",
    )
    objection_resolution_pairs: List[ObjectionResolutionPair] = Field(
        ...,
        title="Objection Resolution Pairs",
        description="The objection resolution pairs identified in the call.",
    )
    insights: List[str] = Field(
        ..., title="Insights", description="The insights identified in the call."
    )
    improvements: List[str] = Field(
        ...,
        title="Areas of Improvement",
        description="The areas of improvement identified in the call.",
    )


agent_templates = {
    SELLER_RESEARCH: {
        "SellerResearchAgent": dict(
            role="Seller Research Specialist",
            goal=(
                "Conduct in-depth research on {seller} to understand their value propositions."
                "You also need to find out their current (or potential) clients"
            ),
            backstory=(
                "You are an expert in generating detailed profile of a sales company, conducting in-depth research on sales companies."
                "You can also extract out the list of current clients of {seller}"
            ),
        )
    },
    RESEARCH: {
        "BuyerResearchAgent": dict(
            role="Sales Research Specialist",
            goal="Prepare for the sales call between a buyer and seller by conducting in-depth research about the buyer, seller, and competitive landscape.",
            backstory=(
                "You are an expert in generating detailed profile of a potential client, conducting in-depth research on sales companies, and analyzing competitors."
                "You curate detailed information about the buyer, seller, and competitive landscape to prepare for the sales call between a potential buyer and {seller}."
                "This information is supposed to help the sales team understand the buyer's needs, the {seller}'s offerings, and the competitive landscape."
            ),
        ),
        "CallPreparationAgent": dict(
            role="Sales Call Preparation Specialist",
            goal="Prepare for the sales call between buyer and {seller} by aligning the buyer's requirements and goals with the {seller}'s offerings.",
            backstory=(
                "You are an expert in preparing for sales calls."
                "Your goal is to check the requirements and goals that can be fulfilled by the {seller} and anticipate the questions, objections, pain points, and challenges that may arise during the call."
                "You are also responsible for providing potential resolutions to the anticipated questions, objections, pain points, and challenges."
            ),
        ),
    },
    SIMULATION: {
        "CallSimulationAgent": dict(
            role="Sales Call Simulation Specialist",
            goal="Simulate a very elaborated, detailed call between buyer and {seller}.",
            backstory=(
                "You are an expert in simulating realistic sales calls."
                "You have been tasked with simulating a detailed call between buyer and {seller}."
                "Your goal is to provide a realistic and engaging simulation of the call."
                "The sales call simulation should be structured, engaging, and informative."
            ),
        )
    },
    EXTRACTION: {
        "DataExtractionAgent": dict(
            role="Data Extraction Specialist",
            goal="Extract the required information from the call transcripts.",
            backstory=(
                "You are an expert in extracting information from research reports and call transcripts."
                "Your goal is to extract the required information from the call transcripts to provide insights to the sales team."
            ),
        )
    },
    ANALYSIS: {
        "DiscoveryCallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key pain points, challenges, objections, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing discovery sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is to identify the pain points and objections, areas of improvement, and potential strategies for future calls."
                ""
            ),
        )
    },
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
    ),
}

task_templates = {
    SELLER_RESEARCH: {
        "SellerIndustryResearchTask": dict(
            name="Seller Industry Research",
            description=(
                "You are providing with the summarized content of the website of {seller}. "
                "Using the website content, conduct an in-depth research on {seller} to understand their value propositions."
                "The value proposition provides information about the product, solution, and use cases of the seller."
                "Below is the website content of {seller}\n"
                "---{seller}'s Website Content---\n"
                "{seller_website_content}\n"
                "---END of Website Content---\n"
            ),
            expected_output=(
                "A comprehensive research report on {seller} detailing their value propositions.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            output_pydantic=SellerInfo,
            agent="SellerResearchAgent",
        ),
        "SellerPricingModelTask": dict(
            name="Seller Pricing Model Research",
            description=(
                "Conduct in-depth research on {seller} to understand their pricing models."
                "Below is the website content of {seller}\n"
                "---{seller}'s Website Content---\n"
                "{seller_website_content}\n"
                "---END of Website Content---\n"
                "Your response should include the pricing models offered by {seller}."
            ),
            expected_output=(
                "A comprehensive research report on {seller} detailing their pricing models.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            output_pydantic=SellerPricingModels,
            agent="SellerResearchAgent",
        ),
    },
    RESEARCH: {
        "BuyerResearcher": dict(
            name="Client for {seller}",
            description=(
                "You are provided by the website content of {buyer} that is a potential client of {seller}"
                "You need to extract the company size, industry, goals, use cases, challenges, etc."
                "You need to extract everything that can be useful for the sales team to understand the client and prepare for a discovery call."
                "---{buyer}'s Website Content---\n"
                "{buyer_website_content}\n"
                "---END of Website Content---\n"
                "---{seller}'s Research Information---\n"
                "You can use the following the information regarding {seller}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "---End of {seller}'s Research Information---\n"
            ),
            expected_output=(
                "A detailed profile of the {buyer}'s team including their company size, industry, goals.\n"
                "The response should conform to the schema of ClientResearchResponse.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="BuyerResearchAgent",
            output_pydantic=ClientResearchResponse,
        ),
        "CompetitorAnalysisTask": dict(
            name="Competitor Analysis",
            description=(
                "Search for the top competitors of {buyer}.\n"
                "Analyze the pros, cons, and differentiators of {buyer} compared to their competitors. "
                "---{buyer}'s Website Content---\n"
                "{buyer_website_content}\n"
                "---END of Website Content---\n"
                "---{seller}'s Research Information---\n"
                "You can use the following the information regarding {seller}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "---End of {seller}'s Research Information---\n"
            ),
            expected_output=(
                "A detailed analysis of the competitors of {seller} using their value propositions, and pricing model.\n"
                "Identify the strengths and weaknesses of {seller} compared to their competitors.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="BuyerResearchAgent",
            output_pydantic=SellerCompetitorAnalysisResponse,
        ),
        "BuyerPainsAndObjectionsDiscoveryTask": dict(
            name="Anticipating possible questions, objections, pain points, and challenges",
            description=(
                "Identify the possible questions, objections, pain points, and challenges that may arise during the sales discovery call between {buyer} and {seller}."
                "You should anticipate the questions, objections, pain points, and challenges (QOPCs) based on the buyer's goals, requirements, and the competitive landscape."
                "The QOPCs should be categorized as questions, objections, pain points, and challenges."
                "The anticipated QOPCs are supposed to help the sales team prepare for the discovery call and provide potential resolutions."
                "---{seller}'s Research Information---\n"
                "You can use the following the information regarding {seller}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "---End of {seller}'s Research Information---\n"
            ),
            expected_output=(
                "A list of possible questions, objections, pain points, and challenges for the sales call between {buyer} and {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            output_pydantic=AnticipatedPainsAndObjections,
            context=[
                "BuyerResearcher",
                "CompetitorAnalysisTask",
            ],
            agent="CallPreparationAgent",
        ),
    },
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate Discovery Call",
            description=(
                "Simulate a very elaborated, detailed discovery call between a {seller} and {buyer}'s stakeholders: {stakeholders}."
                "The {buyer}'s team is represented by {stakeholders} as stakeholders during the call.\n"
                "You need to simulate the call as a conversation between the {seller}'s sales person and ALL the {buyer}'s stakeholders.\n"
                "You are provided with the {buyer}'s and {seller}'s information as well as the competitive landscape.\n"
                "You need to use the following as context -"
                "\n1.) anticipated Questions, Objections, Pain Points, and Challenges\n"
                " to simulate the call.\n"
                "In the call, {seller}'s sales person will aim to discover the goals, requirements, potential pain points, challenges, and objections of the buyer."
                "In the call, person from {buyer}'s team will aim to provide their goals, requirements, potential pain points, challenges, and objections."
                "The {buyer}'s team will talk mostly in their own vocabulary that they use in their company."
                "The {seller} is supposed to be very confident, inquistive, empathetic and understanding."
                "The {seller} MUST NOT be aggressive, pushy, or rude. "
                "The {seller} MUST ADDRESS all the stakeholders in the call. "
                "All the stakeholders, i.e., {stakeholders} MUST voice their opinions, goals, requirements, pain points, challenges, and objections."
                "The call should be very detailed. "
                "Your goal is to provide a realistic and engaging simulation of the call. "
                "---Call Simulation Guidelines---:\n"
                "The sales call simuation should be a very realistic simulation of a discovery call. "
                "The sales call MUST clearly cover the {buyer}'s goals, requirements, pain points and objections. "
                "The sales call simulation MUST be tailored to address the buyer's specific goals, requirements, pain points, and objections."
                "The sales call simulation MUST find out the decision-making process and the committee involved for the buyer. "
                "The sales cal simulation MUST find out the competition that the buyer is considering. "
                "The sales call simulation MUST find out the success indicators for the buyer. "
                "The sales call simulation MUST find out the time lines for the buyer. "
                "The sales call simulation MUST find out the budget constraints for the buyer. "
                "The goal of the discovery call is to discover the goals, requirements, potential pain points, challenges, and objections of the buyer. "
                "The contents of each message should be as detailed and realistic as possible like a human conversation. "
                "The call should be structured and flow naturally like a real discovery call. "
                "The call should have a smooth flow and should be engaging and informative. "
                "The call should not end abruptly and should have a proper conclusion. "
                "---Call Simulation Context---:\n"
                "Previous call analysis details with other stakeholders\n"
                "---Previous Call Analysis Details---\n"
                "{previous_calls_analysis}\n"
                "---Previous Call Analysis Details---\n"
                "You are provided with the following context -\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nAnticipated questions, objections, pain points and challenges:\n{anticipated_qopcs}\n"
                "The call should be very detailed. "
            ),
            expected_output=(
                "A realistic simulation of the discovery call between {buyer} and {seller}."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="CallSimulationAgent",
            output_pydantic=Transcript,
        )
    },
    ANALYSIS: {
        "BuyerDataExtraction": dict(
            name="Analyze Discovery Call",
            description=(
                "Given a the discovery sales call transcript between {buyer} and {seller}, you need to identify the following key elements about the {buyer}'s data for the {stakeholder} of {buyer} present in the call -\n"
                "1. {buyer} {stakeholder}'s Pain Points - The pain points expressed by the buyer during the call.\n"
                "2. {buyer} {stakeholder}'s Objections - The objections raised by the buyer during the call.\n"
                "3. {buyer} {stakeholder}'s Time Lines - The time lines mentioned by the buyer during the call.\n"
                "4. {buyer} {stakeholder}'s Success Indicators - The success indicators mentioend by the buyer during the call.\n"
                "5. {buyer} {stakeholder}'s Budget Constraints - The budget constraints expressed by the buyer during the call.\n"
                "6. {buyer} {stakeholder}'s Decision Committee - The members of the decision mentioned by the buyer during the call.\n"
                "7. {buyer} {stakeholder}'s Competition - The competitors mentioned by the buyer during the call.\n"
                "If there is no information about any of the above points, you should provide response as - No information found. "
                "The goal is to provide insights to the sales team for improving future calls and addressing the buyer's needs effectively and also to prepare for the discovery call."
                "The discovery call transcript:\n{discovery_transcript}\n"
                "You are provided with the following context -\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Seller's Information:\n{competitive_info}\n"
            ),
            expected_output=(
                "An analysis of the sales call between {buyer} and {seller} identifying the key pain points, challenges, objections, insights, and areas of improvement."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="DiscoveryCallAnalysisAgent",
            output_pydantic=BuyerDataExtracted,
        ),
        "SellerDataExtraction": dict(
            name="Analyze Discovery Call",
            description=(
                "Given a the discovery sales call transcript between {buyer} and {seller} to identify the following key elements from the {seller}'s perspective for {buyer}'s {stakeholder} -\n"
                "1. {seller}'s Discovery Questions for {stakeholder} - The discovery questions asked by the seller during the call.\n"
                "2. {seller}'s Decision Making Process Questions for {stakeholder} - The decision making process questions asked by the seller during the call.\n"
                "3. {seller}'s Objection Resolution Pairs for {stakeholder} - The objection resolution pairs identified by the seller during the call, i.e., for each objection raised by {buyer}, the resolution that the {seller} provided. \n"
                "4. {seller}'s Insights for {stakeholder} - The insights identified by the seller during the call.\n"
                "5. {seller}'s Areas of Improvement for {stakeholder} - The areas of improvement identified by the seller during the call.\n"
                "The goal is to provide insights to the sales team for improving future calls and addressing the buyer's needs effectively and also to prepare for the discovery call."
                "The discovery call transcript:\n{discovery_transcript}\n"
                "You are provided with the following context -\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Seller's Information:\n{competitive_info}\n"
                "Below data provides the objections raised by the buyer - \n"
            ),
            expected_output=(
                "An analysis of the sales call between {buyer} and {seller} identifying the key pain points, challenges, objections, insights, and areas of improvement."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="DiscoveryCallAnalysisAgent",
            context=["BuyerDataExtraction"],
            output_pydantic=SellerDataExtracted,
        ),
    },
    COMPETITOR_EXTRACTION: dict(
        name="Competitors Extraction",
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
    ),
}


def get_buyer_research_data(data: Dict, string_format: bool = False):
    buyer_research_keys = {
        "buyer_research": ClientResearchResponse,
        "competitive_info": CompetitorComparison,
        "anticipated_qopcs": AnticipatedPainsAndObjections,
    }

    if string_format:
        return get_data_str(buyer_research_keys, data)
    return {k: data[k] for k in buyer_research_keys.keys()}


def get_seller_research_data(data: Dict, string_format: bool = False):
    seller_research_keys = {
        "seller_research": SellerResearchResponse,
        "seller_pricing": SellerPricingModels,
    }

    if string_format:
        return get_data_str(seller_research_keys, data)
    return {k: data[k] for k in seller_research_keys.keys()}


def get_analysis_data(data: Dict, string_format: bool = False):
    analysis_keys = {
        "discovery_analysis_buyer_data": BuyerDataExtracted,
        "discovery_analysis_seller_data": SellerDataExtracted,
    }

    if string_format:
        return get_data_str(analysis_keys, data)
    return {k: data[k] for k in analysis_keys.keys()}


def get_analysis_metadata(data: Dict):
    return {
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"],
    }


def process_seller_research_data_output(output: CrewOutput):
    seller_info = format_response(output.tasks_output[0])
    seller_pricing = format_response(output.tasks_output[1])

    return {
        "seller_research": seller_info,
        "seller_pricing": seller_pricing,
    }


def process_research_data_output(output: CrewOutput):
    buyer_research = format_response(output.tasks_output[0])
    competitive_info = format_response(output.tasks_output[1])
    qopcs = format_response(output.tasks_output[2])

    return {
        "buyer_research": buyer_research,
        "competitive_info": competitive_info,
        "anticipated_qopcs": qopcs,
    }


def process_analysis_data_output(output: CrewOutput):
    buyer_data = format_response(output.tasks_output[0])
    seller_data = format_response(output.tasks_output[1])

    return {
        "discovery_analysis_buyer_data": buyer_data,
        "discovery_analysis_seller_data": seller_data,
    }


def get_crew(step: str, llm: LLM, **crew_config) -> EchoAgent:
    assert step in [SELLER_RESEARCH, RESEARCH, SIMULATION, EXTRACTION, ANALYSIS], (
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    )

    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config,
    )


async def aget_seller_research_data(inputs: dict, llm: LLM, **crew_config):
    assert "seller" in inputs, "Invalid input data for research"
    seller = inputs["seller"]
    data = copy.deepcopy(inputs)

    async def get_seller_website_content():
        metadata = {"data_type": IndexDataType.SELLER_WEBSITE_DATA.value}
        if check_metadata_exists_in_db(
            index_name=seller, index_type=IndexType.SELLER_RESEARCH, metadata=metadata
        ):
            print(f"Found Seller: {seller} Website Content")
            record = get_data_from_db(
                index_name=seller,
                index_type=IndexType.SELLER_RESEARCH,
                metadata=metadata,
            )
            return record["data"]

        print(f"Website data not found. Extracting Seller: {seller} Website Content")
        website_content = await extract_data_from_website(seller)

        add_data(
            data=website_content,
            metadata={**metadata, "data": website_content},
            index_name=seller,
            index_type=IndexType.SELLER_RESEARCH,
        )

        return website_content

    def save_research_data():
        print(f"Adding Seller: {seller} Summarized Website Research Data")
        metadata = {
            "data_type": IndexDataType.SELLER_RESEARCH_DATA.value,
            "industry": data["seller_research"]["industry"],
            "data": get_seller_research_data(data),
        }

        print(f"Adding Seller: {seller} Research Data")
        add_data(
            data=get_seller_research_data(data, True),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.SELLER_RESEARCH,
        )

    if check_metadata_exists_in_db(
        index_name=seller,
        index_type=IndexType.SELLER_RESEARCH,
        metadata={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
    ):
        print("Seller Research Data Found")
        data["seller_website_content"] = await get_seller_website_content()
        record = get_data_from_db(
            index_name=seller,
            index_type=IndexType.SELLER_RESEARCH,
            metadata={"data_type": IndexDataType.SELLER_RESEARCH_DATA.value},
        )
        data.update(record["data"])
        return data

    print("Seller Research Data Not Found. Generating Data...")
    data["seller_website_content"] = await get_seller_website_content()
    data["seller_website_content"] = get_text_upto_tokens(
        data["seller_website_content"], MAX_TEXT_TOKENS
    )
    print("Website Content Extracted")

    print("Extracting Competitor Information")
    await add_competitor_info(data, llm=llm, **crew_config)
    print("Competitor Information Extracted")

    crew = get_crew(SELLER_RESEARCH, llm, **crew_config)

    response = await crew.kickoff_async(
        inputs={**dict_to_markdown(data), "call_type": CallType.DISCOVERY.value}
    )
    data.update(process_seller_research_data_output(response))
    save_research_data()
    return data


async def aget_research_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in ["seller", "buyer"]]), (
        "Invalid input data for research"
    )
    seller, client = inputs["seller"], inputs["buyer"]
    data = copy.deepcopy(inputs)

    data.update(await aget_seller_research_data({"seller": seller}, llm, **crew_config))

    async def get_website_content():
        metadata = {
            "buyer": client,
            "data_type": IndexDataType.BUYER_WEBSITE_DATA.value,
        }
        if check_metadata_exists_in_db(
            index_name=seller, index_type=IndexType.BUYER_RESEARCH, metadata=metadata
        ):
            record = get_data_from_db(
                index_name=seller,
                index_type=IndexType.BUYER_RESEARCH,
                metadata=metadata,
            )

            print(f"Found Buyer: {client} Website Content")
            return record["data"]

        website_content = await extract_data_from_website(client)

        add_data(
            data=website_content,
            metadata={
                **metadata,
                "data": website_content,
            },
            index_name=seller,
            index_type=IndexType.BUYER_RESEARCH,
        )

        return website_content

    def save_data():
        metadata = {
            "buyer": client,
            "industry": data["buyer_research"]["industry"],
            "company_size": data["buyer_research"]["company_size"],
            "data_type": IndexDataType.BUYER_RESEARCH_DATA.value,
            "data": get_buyer_research_data(data),
        }

        print(f"Adding Buyer: {client} Research Data")
        add_data(
            data=get_buyer_research_data(data, True),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.BUYER_RESEARCH,
        )

    if check_metadata_exists_in_db(
        index_name=seller,
        index_type=IndexType.BUYER_RESEARCH,
        metadata={
            "buyer": client,
            "data_type": IndexDataType.BUYER_RESEARCH_DATA.value,
        },
    ):
        print("Buyer Research Data Found")
        record = get_data_from_db(
            index_name=seller,
            index_type=IndexType.BUYER_RESEARCH,
            metadata={
                "buyer": client,
                "data_type": IndexDataType.BUYER_RESEARCH_DATA.value,
            },
        )
        data.update(record["data"])
        return data

    data["buyer_website_content"] = await get_website_content()
    data["buyer_website_content"] = get_text_upto_tokens(
        data["buyer_website_content"], MAX_TEXT_TOKENS
    )
    print("Website Content Extracted")

    create_account_plan(seller=seller, buyer=client)

    crew = get_crew(RESEARCH, llm, **crew_config)

    response = await crew.kickoff_async(
        inputs={**dict_to_markdown(data), "call_type": CallType.DISCOVERY.value}
    )
    data.update(process_research_data_output(response))
    save_data()
    return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert "stakeholders" in inputs, "No stakeholders found for simulation"
    seller, client = inputs["seller"], inputs["buyer"]
    call_id = inputs["call_id"]
    metadata = {
        "buyer": client,
        "call_type": CallType.DISCOVERY.value,
        "call_id": call_id,
    }

    data = copy.deepcopy(inputs)
    data.update(metadata)
    data.update(
        await aget_research_data_for_client(copy.deepcopy(inputs), llm, **crew_config)
    )

    add_previous_call_analysis(data)

    def save_data():
        print(f"Embedding Simulation Data for Client: {client}")
        add_data(
            data=json_to_markdown(data["discovery_transcript"]),
            metadata={**metadata, "transcript": data["discovery_transcript"]},
            index_name=seller,
            index_type=IndexType.CALL_TRANSCRIPTS,
        )

    if check_metadata_exists_in_db(
        index_name=seller,
        index_type=IndexType.CALL_TRANSCRIPTS,
        metadata=metadata,
    ):
        record = get_data_from_db(
            index_name=seller,
            index_type=IndexType.CALL_TRANSCRIPTS,
            metadata=metadata,
        )

        data["discovery_transcript"] = record["transcript"]
        return data

    crew = get_crew(SIMULATION, llm, **crew_config)

    response = await crew.kickoff_async(
        inputs={**dict_to_markdown(data), "call_type": CallType.DISCOVERY.value}
    )

    data.update({"discovery_transcript": format_response(response.tasks_output[0])})
    save_data()
    return data


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert "stakeholders" in inputs, "No stakeholders found for simulation"

    stakeholders = inputs["stakeholders"]
    client, seller = inputs["buyer"], inputs["seller"]
    call_id = inputs["call_id"]

    metadata = {
        "call_type": CallType.DISCOVERY.value,
        "buyer": client,
        "call_id": call_id,
    }

    data = copy.deepcopy(inputs)
    data.update(
        await aget_simulation_data_for_client(copy.deepcopy(inputs), llm, **crew_config)
    )

    def check_stakeholder_analysis_exist(stakeholder):
        return check_metadata_exists_in_db(
            index_name=seller,
            index_type=IndexType.ANALYSIS,
            metadata={**metadata, "stakeholder": stakeholder},
        )

    def save_stakeholder_analysis(stakeholder):
        add_data(
            data=get_analysis_data(data["discovery_analysis_data"][stakeholder], True),
            metadata={
                **metadata,
                **get_analysis_metadata(data),
                "stakeholder": stakeholder,
                "transcript": data["discovery_transcript"],
                "data": get_analysis_data(data["discovery_analysis_data"][stakeholder]),
            },
            index_name=seller,
            index_type=IndexType.ANALYSIS,
        )

    if all(
        check_stakeholder_analysis_exist(stakeholder) for stakeholder in stakeholders
    ):
        discovery_analysis_data = dict()
        for stakeholder in stakeholders:
            record = get_data_from_db(
                index_name=seller,
                index_type=IndexType.ANALYSIS,
                metadata={
                    "call_type": CallType.DISCOVERY.value,
                    "buyer": client,
                    "call_id": call_id,
                    "stakeholder": stakeholder,
                },
            )
            discovery_analysis_data[stakeholder] = record["data"]

        data.update({"discovery_analysis_data": discovery_analysis_data})
        return data
    print("No analysis data found for the stakeholders. Generating data...")

    crew = get_crew(ANALYSIS, llm, **crew_config)

    discovery_analysis_data = dict()

    for stakeholder in tqdm(stakeholders):
        print(f"Analyzing discovery data for stakeholder: {stakeholder}")

        response = await crew.kickoff_async(
            inputs={
                **dict_to_markdown(data),
                "call_type": CallType.DISCOVERY.value,
                "stakeholder": stakeholder,
            }
        )

        analysis_data = process_analysis_data_output(response)
        discovery_analysis_data[stakeholder] = analysis_data

    data.update({"discovery_analysis_data": discovery_analysis_data})

    for stakeholder in stakeholders:
        save_stakeholder_analysis(stakeholder)

    return data


async def aget_data_for_clients(
    task_type: str, clients: List[str], inputs: dict, llm: LLM, **crew_config
):
    assert task_type in [RESEARCH, SIMULATION, ANALYSIS], (
        f"Invalid task type: {task_type}"
    )
    task_to_data_extraction_fn = {
        RESEARCH: aget_research_data_for_client,
        SIMULATION: aget_simulation_data_for_client,
        ANALYSIS: aanalyze_data_for_client,
    }
    task_fn = task_to_data_extraction_fn[task_type]

    assert all([k in inputs for k in ["seller"]]), f"Invalid input data for {task_type}"
    print(f"Getting {task_type} Data")
    data = await aget_clients_call_data(task_fn, clients, inputs, llm, **crew_config)
    return data


async def aget_seller_data(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in ["seller"]]), "Invalid input data for research"
    return await aget_seller_research_data(inputs, llm, **crew_config)
