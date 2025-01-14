from tqdm.asyncio import tqdm
from crewai_tools import SerperDevTool
from crewai import Crew, LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import *
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.utils import add_pydanctic_structure, format_response
from echo.step_templates.section_extraction_and_mapping import FilledSections, Transcript
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils


class SellerInfo(BaseModel):
	name: str = Field(..., title="Seller's Name", description="The name of the seller.")
	website: str = Field(..., title="Seller's Website", description="The website of the seller.")
	description: str = Field(..., title="Seller's Description", description="A description of the seller.")
	industry: str = Field(..., title="Seller's Industry", description="The industry that the seller belongs to.")
	products: List[str] = Field(..., title="Seller's Products", description="The products offered by the seller.")
	solutions: List[str] = Field(..., title="Seller's Solutions", description="The solutions offered by the seller.")
	use_cases: List[str] = Field(..., title="Seller's Use Cases", description="The use cases of the seller's products.")

class SellerPricingModel(BaseModel):
	type: str = Field(..., title="Seller's Pricing Models", description="The pricing models offered by the seller, such free, subscription, premium, etc.")
	description: str = Field(..., title="Seller's Pricing Description", description="A description of the pricing model.")
	price: str = Field(..., title="Seller's Pricing numbers", description="The pricing (NUMBERS) of the seller's products.")
	duration: str = Field(..., title="Seller's Pricing Duration", description="The duration of the pricing model.")

class SellerPricingModels(BaseModel):
	pricing_models: List[SellerPricingModel] = Field(..., title="Seller's Pricing Models", description="The pricing models offered by the seller.")

class SellerResearchResponse(BaseModel):
	info: SellerInfo = Field(..., title="Seller's Information", description="The information of the seller.")
	pricing: SellerPricingModels = Field(..., title="Seller's Pricing Models", description="The pricing models offered by the seller.")


class ClientResearchResponse(BaseModel):
	name: str = Field(..., title="Buyer's Name", description="The name of the client.")
	website: str = Field(..., title="Buyer's Website", description="The website of the buyer.")
	description: str = Field(..., title="Buyer's Description", description="A description of the buyer.")
	industry: str = Field(..., title="Buyer's Industry", description="The industry that the buyer belongs to.")
	company_size: int = Field(..., title="Buyer's Company Size", description="The size of the buyer's company.")
	goals: List[str] = Field(..., title="Buyer's Goals", description="The goals of the buyer.")
	use_cases: List[str] = Field(..., title="Buyer's Use Cases", description="The use cases of the buyer.")
	challenges: List[str] = Field(..., title="Buyer's Challenges", description="The challenges faced by the buyer.")
	stakeholders: List[str] = Field(..., title="Buyer's Stakeholders", description="The stakeholders of the buyer.")


class CompetitorComparison(BaseModel):
	name: str = Field(..., title="Competitor's Name", description="The name of the competitor.")
	pros: List[str] = Field(..., title="Competitor's Pros", description="The pros of the competitor.")
	cons: List[str] = Field(..., title="Competitor's Cons", description="The cons of the competitor.")
	differentiators: List[str] = Field(..., title="Competitor's Differentiators", description="The differentiators of the seller against the competitor.")


class SellerCompetitorAnalysisResponse(BaseModel):
	competitors: List[CompetitorComparison] = Field(..., title="Competitor Analysis", description="The list of competitors of the seller.")


class QOPC(BaseModel):
	type_: str = Field(..., title="QOPC Type", description="The type of the QOPC ('question', 'objection', 'painpoint', 'challenge').")
	content: str = Field(..., title="QOPC Content", description="The content of the QOPC.")
	

class QOPCs(BaseModel):
	qopcs: List[QOPC] = Field(..., title="QOPCs", description="The list of possible questions, objections, pain points, and challenges for the call.")


class RequirementsAndGoals(BaseModel):
	requirements: List[str] = Field(..., title="Buyer's Requirements", description="The requirements of the buyer.")
	goals: List[str] = Field(..., title="Buyer's Goals", description="The goals of the buyer.")


class QOPCRGs(BaseModel):
	qopcs: QOPCs = Field(..., title="QOPCs", description="The list of possible questions, objections, pain points, and challenges for the call.")
	requirements: List[str] = Field(..., title="Buyer's Requirements", description="The requirements of the buyer.")
	goals: List[str] = Field(..., title="Buyer's Goals", description="The goals of the buyer.")

class AnalysisResults(BaseModel):
    pain_points: List[str] = Field(..., title="Pain Points", description="The pain points identified in the call.")
    challenges: List[str] = Field(..., title="Challenges", description="The challenges identified in the call.")
    objections: List[str] = Field(..., title="Objections", description="The objections identified in the call.")
    insights: List[str] = Field(..., title="Insights", description="The insights identified in the call.")
    improvements: List[str] = Field(..., title="Areas of Improvement", description="The areas of improvement identified in the call.")


tools = {
	"search_tool": SerperDevTool()
}

agent_templates = {
    RESEARCH: {
        "PreCallResearchAgent": dict(
            role="Sales Research Specialist",
            goal="Prepare for the sales call between a buyer and seller by conducting in-depth research about the buyer, seller, and competitive landscape.",
            backstory=(
                "You are an expert in generating detailed profiles of potential clients, conducting in-depth research on sales companies, and analyzing competitors."
                "You curate detailed information about the buyer, seller, and competitive landscape to prepare for the sales call between a potential buyer and {seller}."
                "This information is supposed to help the sales team understand the buyer's needs, the {seller}'s offerings, and the competitive landscape."
            ),
            tools=[tools["search_tool"]],
        ),
        "DiscoveryCallPreparationAgent": dict(
            role="Sales Call Preparation Specialist",
            goal="Prepare for the sales call between buyer and {seller} by aligning the buyer's requirements and goals with the {seller}'s offerings.",
            backstory=(
                "You are an expert in preparing for sales calls."
                "Your goal is to check the requirements and goals that can be fulfilled by the {seller} and anticipate the questions, objections, pain points, and challenges that may arise during the call."
                "You are also responsible for providing potential resolutions to the anticipated questions, objections, pain points, and challenges."
            )
        )
    },
	SIMULATION: {
        "CallSimulationAgent": dict(
            role="Sales Call Simulation Specialist",
            goal="Simulate a very elaborated, detailed call between buyer and {seller}.",
            backstory=(
                "You are an expert in simulating realistic sales calls."
                "You have been tasked with simulating a detailed call between buyer and {seller}."
                "Your goal is to provide a realistic and engaging simulation of the call."
                "The sales call simulation should be tailored to address the buyer's specific goals, requirements, pain points, and challenges."
            )
        )
    },
    EXTRACTION: {
        "DataExtractionAgent": dict(
            role="Data Extraction Specialist",
            goal="Extract the required information from the call transcripts.",
            backstory=(
                "You are an expert in extracting information from research reports and call transcripts based on the provided schema."
                "Your goal is to extract the required information from the call transcripts to provide insights to the sales team."
            )
        )
    },
    ANALYSIS: {
        "DiscoveryCallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key pain points, challenges, objections, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing discovery sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is identify the key pain points, challenges, objections, key insights, areas of improvement, and potential strategies for future calls."
            )
        )
    }
}

task_templates = {
    RESEARCH: {
        "SellerIndustryResearchTask": dict(
            name="Seller Industry Research",
            description=(
                "Conduct in-depth research on {seller} to understand their products, solutions, use cases and goals."
            ),
            expected_output=(
                "A comprehensive research report on {seller} detailing their products, solutions, use cases, and goals.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            output_pydantic=SellerInfo,
            agent="PreCallResearchAgent"	
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
            ),
            output_pydantic=SellerPricingModels,
            agent="PreCallResearchAgent"	
        ),
        "BuyerResearcher": dict(
            name="Client for {seller}",
            description=(
                "You need to do a proper research for {buyer} over the internet that is a potential client of {seller}"
                "You need to search for their website, company size, industry, goals, use cases, challenges, etc."
                "You need extract a detailed profile including their company size, industry, goals."
                "You need to extract everything that can be useful for the sales team to understand the client and prepare for a discovery call."
            ),
            expected_output=(
                "A detailed profile of the {buyer}'s team including their company size, industry, goals.\n"
                "The response should conform to the schema of ClientResearchResponse.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="PreCallResearchAgent",
            context=["SellerIndustryResearchTask", "SellerPricingModelTask"],
            output_pydantic=ClientResearchResponse,
        ),
        "CompetitorAnalysisTask": dict(
            name="Competitor Analysis",
            description=(
                "Search for the top {n_competitors} of {seller}.\n"
                "Analyze the pros, cons, and differentiators of {seller} compared to their competitors. "
            ),
            expected_output=(
                "A detailed analysis of the competitors of {seller} using their products, solutions, use cases, and pricing model.\n"
                "Identify the strengths and weaknesses of {seller} compared to their competitors.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="PreCallResearchAgent",
            context=["SellerIndustryResearchTask", "SellerPricingModelTask", "BuyerResearcher"],
            output_pydantic=SellerCompetitorAnalysisResponse,
        ),
        "QOPCDiscoveryTask": dict(
            name="Anticipating possible questions, objections, pain points, and challenges",
            description=(
                "Identify the possible questions, objections, pain points, and challenges that may arise during the sales discovery call between {buyer} and {seller}."
                "You should anticipate the questions, objections, pain points, and challenges (QOPCs) based on the buyer's goals, requirements, and the competitive landscape."
                "The QOPCs should be categorized as questions, objections, pain points, and challenges."
                "The anticipated QOPCs are supposed to help the sales team prepare for the discovery call and provide potential resolutions."
            ),
            expected_output=(
                "A list of possible questions, objections, pain points, and challenges for the sales call between {buyer} and {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            output_pydantic=QOPCs,
            context=[
                "SellerIndustryResearchTask", 
                "SellerPricingModelTask", 
                "BuyerResearcher",
                "CompetitorAnalysisTask"
            ],
            agent="DiscoveryCallPreparationAgent"
        ),
        "RequirementsAndGoalsDiscoveryTask": dict(
            name="Buyer's Requirements and Goals Discovery",
            description=(
                "Identify the buyer's requirements and goals for the sales discovery call between {buyer} and {seller}."
                "The buyer's requirements and goals are crucial for understanding the buyer's needs and aligning them with the {seller}'s offerings."
                "The requirements and goals are supposed to help the sales team tailor the discovery call to address the buyer's specific needs."
            ),
            expected_output=(
                "The buyer's requirements and goals for the sales call between {buyer} and {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            context=[
                "SellerIndustryResearchTask", 
                "SellerPricingModelTask", 
                "BuyerResearcher",
                "CompetitorAnalysisTask"
            ],
            output_pydantic=RequirementsAndGoals,
            agent="DiscoveryCallPreparationAgent"
        )
    },
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate Discovery Call",
            description=(
                "Simulate a very elaborated, detailed discovery call between {buyer} and {seller}."
                "You are provided with the {buyer}'s and {seller}'s information as well as the competitive landscape."
                
                "You need to the following as context -"
                "\n1.) anticipated Questions, Objections, Pain Points, and Challenges\n"
                " to simulate the call.\n"
                
                
                "In the call, {seller}'s sales person will aim to discover the goals, requirements, potential pain points, challenges, and objections of the buyer."
                "In the call, person from {buyer}'s team will aim to provide their goals, requirements, potential pain points, challenges, and objections."
                "The {buyer}'s team will talk mostly in their own vocabulary that they use in their company."
                "The {seller}'s team person is supposed to be very confident, inquistive, empathetic and understanding."

                "Your goal is to provide a realistic and engaging simulation of the call."
                "The sales call simuation should be a very realistic simulation of a discovery call."
                "The sales call MUST clearly cover the {buyer}'s goals, requirements, potential pain points, challenges, and objections."
                "The goal of the discovery call is to discover the goals, requirements, potential pain points, challenges, and objections of the buyer."
                "The contents of each message should be as detailed and realistic as possible like a human conversation."
                "The call should be structured and flow naturally like a real discovery call."
                "The call should have a smooth flow and should be engaging and informative."
                "The call should not end abruptly and should have a proper conclusion."
        
                "You are provided with the following context -\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nAnticipated questions, objections, pain points and challenges:\n{anticipated_qopcs}\n"
                "\nBuyer's requirements and goals:\n{requirements_goals}\n"
            ),
            expected_output=(
                "A realistic simulation of the discovery call between {buyer} and {seller}."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="CallSimulationAgent",
            output_pydantic=Transcript
        )
    },
    EXTRACTION: {
        "DataExtractionTask": dict(
            name="Extract Data from Call Transcript",
            description=(
                "Analyze a sales call transcript and map the content of the conversation to a predefined schema by extracting relevant information for each section and subsection. \n"
                "Using the provided schema as a template, extract relevant information from the sales call transcript and populate each section and subsection. \n"
                "The goal is to generate a structured, concise, and accurate report that reflects the content of the transcript."
                """
                Task Details:

                1. Input:
                    •	Schema Template: A pre-defined schema containing sections and subsections.
                    •	Transcript: A detailed transcript of a sales call.

                2. Requirements:
                    •	Section and Subsection Mapping: Extract information from the transcript and populate each section and subsection according to the schema.
                    •	The content of the section should be the summarized version of the contents of the subsections.
                    •	Conciseness and Clarity: Provide clear, concise, and relevant content without copying verbatim from the transcript unless necessary.
                    •	Completeness: Ensure that all sections and subsections are filled, even if the transcript has limited information (e.g., note “not mentioned” for missing content).
                    •	Context Preservation: Ensure the content reflects the context of the conversation accurately, especially for subjective parts (e.g., client goals and objections).
                    
                
                Below is the schema that you need to follow for the extraction of the data from the call transcript.
                {schema}
                
                And below is the transcript that you need to extract the data from.
                {discovery_transcript}
                """                
            ),
            expected_output=(
                "A structured representation of the transcript, with each section and subsection filled according to the provided schema.\n"
                "Each subsection should contain the relevant information based on the transcript.\n"
                "The content of the section should be the summarized version of the contents of the subsections.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="DataExtractionAgent",
            output_pydantic=FilledSections
        )
    },
    ANALYSIS: {
        "DiscoveryCallAnalysisTask": dict(
            name="Analyze Discovery Call",
            description=(
                "Analyze the sales call between {buyer} and {seller} to identify the key pain points, challenges, objections, insights, and areas of improvement."
                "The analysis should focus on the buyer's requirements, goals, anticipated QOPCs, and the seller's responses."
                "The goal is to provide insights to the sales team for improving future calls and addressing the buyer's needs effectively."
                "You are provided with the following context -\n"
                "The discovery call transcript:\n{discovery_transcript}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Seller's Information:\n{competitive_info}\n"
            ),
            expected_output=(
                "An analysis of the sales call between {buyer} and {seller} identifying the key pain points, challenges, objections, insights, and areas of improvement."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="DiscoveryCallAnalysisAgent",
            output_pydantic=AnalysisResults
        )
    }
}

inputs = {
    RESEARCH: [
        "seller",
        "buyer",
        "n_competitors"
    ],
    SIMULATION: [
        "seller",
        "buyer",
        "seller_research",
        "seller_pricing",
        "buyer_research",
        "competitive_info",
        "anticipated_qopcs",
        "requirements_goals"
    ],   
}

# def utils.check_data_exists(client_name):
#     if not os.path.exists(f"runs/{client_name}.json"):
#         return False
#     return True
    # data = json.load(open(f"runs/{client_name}.json"))
    # return all(
    #     [k in data for k in [
    #         "seller_research", 
    #         "seller_pricing", 
    #         "buyer_research", 
    #         "competitive_info", 
    #         "anticipated_qopcs", 
    #         "requirements_goals"
    #     ]])


def process_research_data_output(output: CrewOutput):
    seller_research = format_response(output.tasks_output[0])
    seller_pricing = format_response(output.tasks_output[1])
    buyer_research = format_response(output.tasks_output[2])
    competitive_info = format_response(output.tasks_output[3])
    qopcs = format_response(output.tasks_output[4])
    requirements_goals = format_response(output.tasks_output[5])
    
    return {
        "seller_research": seller_research,
        "seller_pricing": seller_pricing,
        "buyer_research": buyer_research,
        "competitive_info": competitive_info,
        "anticipated_qopcs": qopcs,
        "requirements_goals": requirements_goals,
    }
    

def get_crew(step: str, llm: LLM, **crew_config) -> Crew:
    assert step in [RESEARCH, SIMULATION, EXTRACTION, ANALYSIS],\
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    
    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config
    )


async def aget_research_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in ["seller", "n_competitors", "buyer"]]), "Invalid input data for research"
    if utils.check_data_exists(inputs['buyer']):
        return utils.get_client_data(inputs['buyer'])
    
    crew = get_crew(RESEARCH, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    return process_research_data_output(response)


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in [
        "seller", 
        "buyer", 
        "seller_research", 
        "seller_pricing", 
        "buyer_research", 
        "competitive_info", 
        "anticipated_qopcs", 
        "requirements_goals"
    ]]), f"Invalid input data for simulation call: {inputs.keys()}"
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        if 'discovery_transcript' in data:
            return data['discovery_transcript']
    
    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


async def aextract_data_from_transcript_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in [
        "seller", 
        "buyer", 
        "schema",
        "discovery_transcript"
    ]]), "Invalid input data for simulation"
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        if 'discovery_transcript_structured' in data:
            return data['discovery_transcript_structured']
    
    crew = get_crew(EXTRACTION, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in [
        "seller", 
        "buyer", 
        "seller_research", 
        "seller_pricing", 
        "buyer_research", 
        "competitive_info", 
        "discovery_transcript"
    ]]), f"Invalid input data for simulation: {inputs.keys()}"
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        if 'discovery_analysis_results' in data:
            return data['discovery_analysis_results']
    
    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


async def aget_research_data(clients: List[str], inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in ["seller", "n_competitors"]]), "Invalid input data for research"
    research_data = dict()
    for client in tqdm(clients, desc="Getting Research Data"):
        print(f"Getting Research Data for {client}")
        inputs['buyer'] = client
        research_data[client] = await aget_research_data_for_client(inputs, llm, **crew_config)
        utils.save_client_data(client, research_data[client])
    
    return research_data


async def asimulate_data(clients: List[str], inputs: dict, llm: LLM, **crew_config):
    research_data = await aget_research_data(clients, inputs, llm, **crew_config)
    simulation_data = dict()
    for client in tqdm(clients, desc="Simulating Data"):
        print(f"Simulating Data for {client}")
        data = research_data[client]
        inputs.update(data)
        data['discovery_transcript'] = await aget_simulation_data_for_client(inputs, llm, **crew_config)
        utils.save_client_data(client, data)
        simulation_data[client] = data

    return simulation_data


async def aextract_data_from_transcript(
    clients: List[str], 
    schema_json: Dict, 
    inputs: dict, 
    llm: LLM, 
    **crew_config
):
    extracted_data = dict()
    simulation_data = await asimulate_data(clients, inputs, llm, **crew_config)
    
    for client in tqdm(clients, desc="Extracting Data"):
        print(f"Extracting Data for {client}")
        data = simulation_data[client]
        inputs.update(data)
        inputs['schema'] = schema_json
        data['discovery_transcript_structured'] = await aextract_data_from_transcript_for_client(inputs, llm, **crew_config)
        utils.save_client_data(client, data)
        
        extracted_data[client] = data
    
    return extracted_data


async def aget_analysis(
    clients: List[str], 
    schema_json,
    inputs: dict, 
    llm: LLM, 
    **crew_config
):
    analysis_data = dict()
    extracted_data = await aextract_data_from_transcript(
        clients, schema_json, inputs, llm, **crew_config
    )
    
    for client in tqdm(clients, desc="Analyzing Data"):
        data: Dict = extracted_data[client]
        data.update(inputs)
        data['discovery_analysis_results'] = await aanalyze_data_for_client(data, llm, **crew_config)
        analysis_data[client] = data
    
    utils.save_clients_data(analysis_data)
    
    return analysis_data