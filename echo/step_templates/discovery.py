import copy
from crewai_tools import SerperDevTool
from echo.agent import EchoAgent
from crewai import LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import (
    BUYER_RESEARCH,
    SELLER_RESEARCH,
    RESEARCH,
    SIMULATION,
    EXTRACTION,
    ANALYSIS,
    BUYER,
)
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.utils import add_pydantic_structure, format_response
from echo.step_templates.generic import (
    CallType,
    Transcript,
    aget_clients_call_data,
    save_transcript_data,
)
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils

from echo.indexing import IndexType, add_data


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
    products: List[str] = Field(
        ...,
        title="Seller's Products",
        description="The products offered by the seller.",
    )
    solutions: List[str] = Field(
        ...,
        title="Seller's Solutions",
        description="The solutions offered by the seller.",
    )
    use_cases: List[str] = Field(
        ...,
        title="Seller's Use Cases",
        description="The use cases of the seller's products.",
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


class SellerClients(BaseModel):
    clients: List[str] = Field(
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

    clients: SellerClients = Field(
        ...,
        title="Seller's Clients",
        description="The clients of the seller.",
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


tools = {"search_tool": SerperDevTool()}

agent_templates = {
    SELLER_RESEARCH: {
        "SellerResearchAgent": dict(
            role="Seller Research Specialist",
            goal=(
                "Conduct in-depth research on {seller} to understand their products, solutions, use cases, and goals. "
                "You also need to find out their current (or potential) clients"
            ),
            backstory=(
                "You are an expert in generating detailed profile of a sales company, conducting in-depth research on sales companies."
                "You can also extract out the list of current clients of {seller}"
            ),
            tools=[tools["search_tool"]],
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
            tools=[tools["search_tool"]],
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
}

task_templates = {
    SELLER_RESEARCH: {
        "SellerIndustryResearchTask": dict(
            name="Seller Industry Research",
            description=(
                "You are providing with the link to the landing page of a seller as - {seller}. "
                "Conduct in-depth research on {seller} to understand their products, solutions, use cases and goals."
            ),
            expected_output=(
                "A comprehensive research report on {seller} detailing their products, solutions, use cases, and goals.\n"
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
                "You may need to explore their website, documentation to find the pricing models."
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
        "SellerClientsTask": dict(
            name="Seller Clients Research",
            description=(
                "You need to find out the buyers of {seller} by conducting in-depth research about {seller} online.\n"
                "You need to get a list of upto {num_buyers} current clients of {seller}. "
                "The buyer of the {seller} MUST BE PRESENT on the website of the {seller} so you should not create your own list of buyers. "
                "If there are no current buyers, then you should get the list of potential buyers of {seller}."
            ),
            expected_output=(
                "A list of current or potential buyers of the {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            output_pydantic=SellerClients,
            agent="SellerResearchAgent",
            context=[
                "SellerIndustryResearchTask",
                "SellerPricingModelTask",
            ],
        ),
    },
    RESEARCH: {
        "BuyerResearcher": dict(
            name="Client for {seller}",
            description=(
                "You need to do a proper research for {buyer} over the internet that is a potential client of {seller}"
                "You need to search for their website, company size, industry, goals, use cases, challenges, etc."
                "You need extract a detailed profile including their company size, industry, goals."
                "You need to extract everything that can be useful for the sales team to understand the client and prepare for a discovery call."
                "Below is the information regarding {seller}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
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
                "Search for the top {n_competitors} of {seller}.\n"
                "Analyze the pros, cons, and differentiators of {seller} compared to their competitors. "
                "You can use the following the information regarding {seller}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
            ),
            expected_output=(
                "A detailed analysis of the competitors of {seller} using their products, solutions, use cases, and pricing model.\n"
                "Identify the strengths and weaknesses of {seller} compared to their competitors.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="BuyerResearchAgent",
            context=[
                "BuyerResearcher",
            ],
            output_pydantic=SellerCompetitorAnalysisResponse,
        ),
        "BuyerPainsAndObjectionsDiscoveryTask": dict(
            name="Anticipating possible questions, objections, pain points, and challenges",
            description=(
                "Identify the possible questions, objections, pain points, and challenges that may arise during the sales discovery call between {buyer} and {seller}."
                "You should anticipate the questions, objections, pain points, and challenges (QOPCs) based on the buyer's goals, requirements, and the competitive landscape."
                "The QOPCs should be categorized as questions, objections, pain points, and challenges."
                "The anticipated QOPCs are supposed to help the sales team prepare for the discovery call and provide potential resolutions."
                
                "You can use the following the information regarding {seller}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
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
                "Simulate a very elaborated, detailed discovery call between a {buyer} and {seller}."
                "The {buyer}'s team can be represented by one or more persons during the call.\n"
                "You need to simulate the call as a conversation between the {seller}'s sales person and the {buyer}'s team.\n"
                "You are provided with the {buyer}'s and {seller}'s information as well as the competitive landscape.\n"
                "You need to the following as context -"
                "\n1.) anticipated Questions, Objections, Pain Points, and Challenges\n"
                " to simulate the call.\n"
                "In the call, {seller}'s sales person will aim to discover the goals, requirements, potential pain points, challenges, and objections of the buyer."
                "In the call, person from {buyer}'s team will aim to provide their goals, requirements, potential pain points, challenges, and objections."
                "The {buyer}'s team will talk mostly in their own vocabulary that they use in their company."
                "The {seller}'s team person is supposed to be very confident, inquistive, empathetic and understanding."
                "Your goal is to provide a realistic and engaging simulation of the call. "
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
                "You are provided with the following context -\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nAnticipated questions, objections, pain points and challenges:\n{anticipated_qopcs}\n"
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
                "Given a the discovery sales call transcript between {buyer} and {seller} to identify the following key elements about the {buyer}'s data -\n"
                "1. Buyer's Pain Points - The pain points expressed by the buyer during the call.\n"
                "2. Buyer's Objections - The objections raised by the buyer during the call.\n"
                "3. Buyer's Time Lines - The time lines mentioned by the buyer during the call.\n"
                "4. Buyer's Success Indicators - The success indicators mentioend by the buyer during the call.\n"
                "5. Buyer's Budget Constraints - The budget constraints expressed by the buyer during the call.\n"
                "6. Buyer's Decision Committee - The members of the decision mentioned by the buyer during the call.\n"
                "7. Buyer's Competition - The competitors mentioned by the buyer during the call.\n"
                "The goal is to provide insights to the sales team for improving future calls and addressing the buyer's needs effectively and also to prepare for the demo call."
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
                "Given a the discovery sales call transcript between {buyer} and {seller} to identify the following key elements from the {seller}'s perspective -\n"
                "1. Seller's Discovery Questions - The discovery questions asked by the seller during the call.\n"
                "2. Seller's Decision Making Process Questions - The decision making process questions asked by the seller during the call.\n"
                "3. Seller's Objection Resolution Pairs - The objection resolution pairs identified by the seller during the call, i.e., for each objection raised by {buyer}, the resolution that the {seller} provided. \n"
                "4. Seller's Insights - The insights identified by the seller during the call.\n"
                "5. Seller's Areas of Improvement - The areas of improvement identified by the seller during the call.\n"
                "The goal is to provide insights to the sales team for improving future calls and addressing the buyer's needs effectively and also to prepare for the demo call."
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
}

inputs = {
    SELLER_RESEARCH: ["seller", "num_buyers"],
    BUYER_RESEARCH: ["seller", "buyer", "n_competitors"],
    SIMULATION: [
        "seller",
        "buyer",
        "seller_research",
        "seller_pricing",
        "buyer_research",
        "competitive_info",
        "anticipated_qopcs",
    ],
}


buyer_call_format = (
    "Buyer Info: {buyer_research}\n"
    "Objections raised by the buyer: {discovery_analysis_buyer_data.objections}\n"
    "Pain Points raised by the buyer: {discovery_analysis_buyer_data.pain_points}\n"
    "Time Lines mentioned by the buyer: {discovery_analysis_buyer_data.time_lines}\n"
    "Success Indicators mentioned by the buyer: {discovery_analysis_buyer_data.success_indicators}\n"
    "Budget Constraints mentioned by the buyer: {discovery_analysis_buyer_data.budget_constraints}\n"
    "Competition mentioned by the buyer: {discovery_analysis_buyer_data.competition}\n"
    "Decision Committee mentioned by the buyer: {discovery_analysis_buyer_data.decision_committee}\n"
)

buyer_keys = {
    "Buyer Research": "buyer_research",
    "Buyer Objections": "discovery_analysis_buyer_data.objections",
    "Pain Points Raised": "discovery_analysis_buyer_data.pain_points",
    "Time Lines mentioned": "discovery_analysis_buyer_data.time_lines",
    "Success Indicators mentioned": "discovery_analysis_buyer_data.success_indicators",
    "Budget Constraints mentioned": "discovery_analysis_buyer_data.budget_constraints",
    "Competition mentioned": "discovery_analysis_buyer_data.competition",
    "Decision Committee mentioned": "discovery_analysis_buyer_data.decision_committee",
}


seller_call_format = (
    "Seller Info: {seller_research}\n"
    "Seller Pricing: {seller_pricing}\n"
    "Discovery Questions asked by the seller: {discovery_analysis_seller_data.discovery_questions}\n"
    "Decision Making Process Questions asked by the seller: {discovery_analysis_seller_data.decision_making_process_questions}\n"
    "Objection Resolution Pairs: {discovery_analysis_seller_data.objection_resolution_pairs}\n"
)


seller_keys = {
    "Seller Research": "seller_research",
    "Seller Pricing": "seller_pricing",
    "Discovery Questions asked": "discovery_analysis_seller_data.discovery_questions",
    "Decision Making Process Questions asked": "discovery_analysis_seller_data.decision_making_process_questions",
    "Objection Resolution Pairs": "discovery_analysis_seller_data.objection_resolution_pairs",
}


def get_buyer_research_data(data: Dict):
    buyer_research_keys = {
        "buyer_research": ClientResearchResponse,
        "competitive_info": CompetitorComparison,
        "anticipated_qopcs": AnticipatedPainsAndObjections,
    }

    data_str = utils.get_data_str(buyer_research_keys, data)
    return data_str


def get_seller_research_data(data: Dict):
    seller_research_keys = {
        "seller_research": SellerResearchResponse,
        "seller_pricing": SellerPricingModels,
        "seller_clients": SellerClients,
    }

    data_str = utils.get_data_str(seller_research_keys, data)
    return data_str


def get_analysis_data(data: Dict):
    analysis_keys = {
        "discovery_analysis_buyer_data": BuyerDataExtracted,
        "discovery_analysis_seller_data": SellerDataExtracted,
    }

    data_str = utils.get_data_str(analysis_keys, data)
    return data_str


def get_analysis_metadata(data: Dict):
    return {
        "seller": data["seller"],
        "buyer": data["buyer"],
        "call_type": CallType.DISCOVERY.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"],
    }


def get_client_data_to_embed(client_name, user_type):
    data = utils.get_client_data(client_name)
    if user_type == BUYER:
        return utils.replace_keys_with_values(buyer_call_format, data)
    else:
        return utils.replace_keys_with_values(seller_call_format, data)


def get_client_data_to_save(client_name, user_type):
    data = utils.get_client_data(client_name)
    if user_type == BUYER:
        return utils.get_nested_key_values(buyer_keys, data)
    else:
        return utils.get_nested_key_values(seller_keys, data)


def process_seller_research_data_output(output: CrewOutput):
    seller_info = format_response(output.tasks_output[0])
    seller_pricing = format_response(output.tasks_output[1])
    seller_clients = format_response(output.tasks_output[2])

    return {
        "seller_research": seller_info,
        "seller_pricing": seller_pricing,
        "seller_clients": seller_clients,
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
    data = copy.deepcopy(inputs)

    def save_data():
        utils.save_client_data(f"{seller}/research", data)
        print(f"Adding Seller: {seller} Data")
        add_data(
            data=get_seller_research_data(data),
            metadata={
                "seller": seller,
            },
            index_name=seller,
            index_type=IndexType.SELLER_RESEARCH,
        )

    seller = inputs["seller"]
    if utils.check_data_exists(f"{seller}/research"):
        data.update(utils.get_client_data(f"{seller}/research"))
        save_data()
        return data

    crew = get_crew(SELLER_RESEARCH, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.DISCOVERY.value}
    )
    data.update(process_seller_research_data_output(response))
    save_data()
    return data


async def aget_research_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in ["seller", "n_competitors", "buyer"]]), (
        "Invalid input data for research"
    )
    data = copy.deepcopy(inputs)

    def save_data():
        print(f"Adding Buyer: {client} Data")
        add_data(
            data=get_buyer_research_data(data),
            metadata={
                "buyer": client,
            },
            index_name=data["seller"],
            index_type=IndexType.BUYER_RESEARCH,
        )

    client = inputs["buyer"]
    save_pth = f"{inputs['seller']}/{inputs['buyer']}"
    if utils.check_data_exists(save_pth):
        data.update(utils.get_client_data(save_pth))
        save_data()
        return data

    crew = get_crew(RESEARCH, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.DISCOVERY.value}
    )
    data.update(process_research_data_output(response))

    save_data()
    return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    save_pth = f"{inputs['seller']}/{inputs['buyer']}"
    data = copy.deepcopy(inputs)
    if utils.check_data_exists(save_pth):
        data.update(utils.get_client_data(save_pth))
        if "discovery_transcript" in data:
            save_transcript_data(data, CallType.DISCOVERY.value)
            return data
    else:
        print(f"Data not found for client: {inputs['buyer']}")
        data.update(await aget_research_data_for_client(inputs, llm, **crew_config))

    try:
        assert all(
            [
                k in data
                for k in [
                    "seller",
                    "buyer",
                    "seller_research",
                    "seller_pricing",
                    "buyer_research",
                    "competitive_info",
                    "anticipated_qopcs",
                ]
            ]
        ), f"Invalid input data for simulation call: {inputs.keys()}"
    except AssertionError:
        print("Research data not found for the client. Getting Research Data...")
        research_data = await aget_research_data_for_client(data, llm, **crew_config)
        data.update(research_data)

    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.DISCOVERY.value}
    )

    simulation_data = {
        "discovery_transcript": format_response(response.tasks_output[0])
    }

    data.update(simulation_data)
    save_transcript_data(data, CallType.DISCOVERY.value)

    return data


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    save_pth = f"{inputs['seller']}/{inputs['buyer']}"
    client, seller = inputs["buyer"], inputs["seller"]
    data = copy.deepcopy(inputs)

    def save_data():
        utils.save_client_data(save_pth, data)
        print("Adding Analysis Data to Vector Store")
        add_data(
            data=get_analysis_data(data),
            metadata=get_analysis_metadata(data),
            index_name=seller,
            index_type=IndexType.HISTORICAL,
        )

    client = inputs["buyer"]
    if utils.check_data_exists(save_pth):
        data.update(utils.get_client_data(save_pth))
        if "discovery_analysis_buyer_data" in data:
            save_data()
            return data
    else:
        print(f"Data not found for client: {client}")
        data.update(await aget_simulation_data_for_client(inputs, llm, **crew_config))

    try:
        assert all(
            [
                k in data
                for k in [
                    "seller",
                    "buyer",
                    "seller_research",
                    "seller_pricing",
                    "buyer_research",
                    "competitive_info",
                    "discovery_transcript",
                ]
            ]
        ), f"Invalid input data for simulation: {data.keys()}"
    except AssertionError:
        print("Simulation data not found for the client. Getting Simulation Data...")
        simulation_data = await aget_simulation_data_for_client(
            data, llm, **crew_config
        )
        data.update(simulation_data)

    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.DISCOVERY.value}
    )
    data.update(process_analysis_data_output(response))
    save_data()

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

    assert all([k in inputs for k in ["seller", "n_competitors"]]), (
        f"Invalid input data for {task_type}"
    )
    print(f"Getting {task_type} Data")
    data = await aget_clients_call_data(task_fn, clients, inputs, llm, **crew_config)
    return data


async def aget_seller_data(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in ["seller", "num_buyers"]]), (
        "Invalid input data for research"
    )
    return await aget_seller_research_data(inputs, llm, **crew_config)
