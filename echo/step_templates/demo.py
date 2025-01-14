import json
import os
from tqdm.asyncio import tqdm
from crewai import Crew, LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import *
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.utils import add_pydanctic_structure, format_response
from echo.step_templates.section_extraction_and_mapping import FilledSections, Transcript
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils


class Feature(BaseModel):
	name: str = Field(..., title="Feature Name", description="The name of the feature of the product or service.")
	description: str = Field(..., title="Feature Description", description="A description of the feature.")

class GoalOrRequirementFeatureMap(BaseModel):
	type_: str = Field(..., title="Goal or Requirement", description="Either 'goal' or 'requirement'.")
	content: str = Field(..., title="Goal or Requirement Content", description="The content of the goal or requirement.")
	resolving_features: List[str] = Field(..., title="Resolving Features", description="The features that can fulfill the goal or requirement.")
	details: str = Field(..., title="Resolution details", description="The details of how the features fulfill the goal or requirement.")

class GoalOrRequirementFeatureMaps(BaseModel):
	grm: List[GoalOrRequirementFeatureMap] = Field(..., title="Goal or Requirement Feature Maps", description="The mapping of goals or requirements to features.")


class QOPCFeatureMap(BaseModel):
	type_: str = Field(..., title="QOPC Type", description="The type of the QOPC ('question', 'objection', 'painpoint', 'challenge').")
	content: str = Field(..., title="QOPC Content", description="The content of the QOPC.")
	resolving_features: List[str] = Field(..., title="Resolving Features", description="The features that can resolve the QOPC.")
	details: str = Field(..., title="QOPC Details", description="The details of how the features resolve the QOPC.")


class QOPCFeatureMaps(BaseModel):
	qfm: List[QOPCFeatureMap] = Field(..., title="QOPC Feature Maps", description="The mapping of questions, objections, pain points, and challenges to features.")


class ROIFeatureMap(BaseModel):
	resolving_features: List[str] = Field(..., title="Resolving Features", description="The features that provide a return on investment.")
	details: str = Field(..., title="ROI Details", description="The details of how the features provide a return on investment.")

class DemoFeatures(BaseModel):
	features: List[Feature] = Field(..., title="Features", description="The features of the product or service.")

class FeatureMaps(BaseModel):
	features: List[Feature] = Field(..., title="Features", description="The features of the product or service.")
	goal_or_requirement_feature_map: List[GoalOrRequirementFeatureMap] = Field(..., title="Goal or Requirement Feature Map", description="The mapping of goals or requirements to features.")
	qopc_feature_map: List[QOPCFeatureMap] = Field(..., title="QOPC Feature Map", description="The mapping of questions, objections, pain points, and challenges to features.")
	roi_feature_map: ROIFeatureMap = Field(..., title="ROI Feature Map", description="The mapping of features to return on investment.")


class AnalysisResults(BaseModel):
    key_features: List[str] = Field(..., title="Key Features", description="The key features identified in the call.")
    pain_points: List[str] = Field(..., title="Pain Points", description="The pain points identified in the call.")
    challenges: List[str] = Field(..., title="Challenges", description="The challenges identified in the call.")
    objections: List[str] = Field(..., title="Objections", description="The objections identified in the call.")
    insights: List[str] = Field(..., title="Insights", description="The insights identified in the call.")
    improvements: List[str] = Field(..., title="Areas of Improvement", description="The areas of improvement identified in the call.")


agent_templates = {
    RESEARCH: {
        "DemoPreparationAgent": dict(
            role="Sales Demo Calls Expert",
            goal="The goal of this agent is to prepare for the sales demo calls by mapping the features of the product or service to the goals, requirements, questions, objections, pain points, challenges, and return on investment.",
            backstory=(
                "You are an expert in preparing for sales demo calls. "
                "You have an in-depth understanding of how to sell the features of a product and service to potential customers."
                "You have the ability to sell the benefits of the product or service according to the requirements, goals, pain points of the customers such that they are satisfied and convinced to make a purchase."
                "Using these abilities, you will map the features of the product or service to the goals, requirements, questions, objections, pain points, challenges, and return on investment."
            )
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
        "DemoCallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key features, pain points, challenges, objections, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing demo sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is identify the key pain points, challenges, objections, key insights, areas of improvement, and potential strategies for future calls."
            )
        )
    }
}

task_templates = {
    RESEARCH: {
        "DemoFeatureExtraction": dict(
            name="Extract the features of the {seller}'s product",
            description=(
                "Extract the features of the {seller}'s product or service that will be demonstrated during the sales demo for the {buyer}\n"
                "You are supposed to extract the features from the information provided below.\n"
                "You are provided with the following context -\n"
                "Transcript of the discovery call between {buyer} and {seller}: {discovery_transcript}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nData from the analysis of previous discovery call:\n{discovery_analysis_results}\n"
            ),
            expected_output=(
                "A list of features of the {seller}'s product or service that will be demonstrated during the sales demo.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            output_pydantic=DemoFeatures,
            agent="DemoPreparationAgent"
        ),
        "Feature2GoalsOrRequirementAlignment": dict(
            name="Align the features of {seller}'s product with the goals and requirements of {buyer}",
            description=(
                "You are provided with the information about the {seller}, {seller}'s features and the discovered goals and requirements from the demo call.\n"
                "You task is to align the features of the {seller}'s product with the goals and requirements of the {buyer}.\n"
                "More specifically, you need to map the features of the product that can fulfill the goals and requirements of the buyer.\n"
                "The mapping should include the features that can address the specific goals and requirements of the buyer.\n"
                "The mapping should also include the details of how the features fulfill the goals and requirements.\n"

                "You are provided with the following context -\n"
                "Transcript of the discovery call between {buyer} and {seller}: {discovery_transcript}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nData from the analysis of previous discovery call:\n{discovery_analysis_results}\n"

                "You are also provided with the features of the {seller}'s product below.\n"
            ),
            expected_output=(
                "A mapping of the features of the {seller}'s product with the goals and requirements of the {buyer}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            context=["DemoFeatureExtraction"],
            output_pydantic=GoalOrRequirementFeatureMaps,
            agent="DemoPreparationAgent"
        ),
        "Feature2QOPCsAlignment": dict(
            name="Align the features of {seller}'s product with the pain points, challenges, objections and questions of {buyer}",
            description=(
                "You are provided with the information about the {seller} and the discovered questions, objections, pain points, and challenges (QOPCs) from the demo call.\n"
                "You task is to align the features of the {seller}'s product with the QOPCs of the {buyer}.\n"
                "More specifically, you need to map the features of the product that can resolve the QOPCs of the buyer.\n"
                "The mapping should include the features that can address the specific pain points, challenges, objections, and questions of the buyer.\n"
                "The mapping should also include the details of how the features resolve the QOPCs.\n"


                "You are provided with the following context -\n"
                "Transcript of the discovery call between {buyer} and {seller}: {discovery_transcript}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nData from the analysis of previous discovery call:\n{discovery_analysis_results}\n"

                "You are also provided with the features of the {seller}'s product below.\n"
            ),
            expected_output=(
                "A mapping of the features of the {seller}'s product with the pain points, challenges, objections, and questions of the {buyer}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            context=["DemoFeatureExtraction"],
            output_pydantic=QOPCFeatureMaps,
            agent="DemoPreparationAgent"
        ),
        "ROIFeatureMap": dict(
            name="Map features to Return on Investment (ROI)",
            description=(
                "You are provided with the features of the {seller}'s product and the discovered goals and requirements from the demo call.\n"
                "You task is to map the features of the {seller}'s product to the return on investment (ROI) for the {buyer}.\n"
                "More specifically, you need to identify the features that provide a return on investment for the buyer.\n"
                "The mapping should include the features that provide a return on investment for the buyer.\n"
                "The mapping should also include the details of how the features provide a return on investment.\n"

                "You are provided with the following context -\n"
                "Transcript of the discovery call between {buyer} and {seller}: {discovery_transcript}\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nData from the analysis of previous discovery call:\n{discovery_analysis_results}\n"

                "You are also provided with the features of the {seller}'s product below.\n"
            ),
            expected_output=(
                "A mapping of the features of the {seller}'s product with the return on investment (ROI) for the {buyer}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            context=["DemoFeatureExtraction"],
            output_pydantic=ROIFeatureMap,
            agent="DemoPreparationAgent"
        )
    },
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate demo Call",
            description=(
                "Simulate a very elaborated, detailed demo call between {buyer} and {seller}."
                "You are provided with the {buyer}'s and {seller}'s information as well as the competitive landscape."
                
                "You need to use the following as context -"
                "\n1.) The transcript of the demo call\n"
                "\n2.) The features of the {seller}'s product\n"
                "\n3.) The mapping of the features of the {seller}'s product with the goals and requirements of the {buyer}\n"
                "\n4.) The mapping of the features of the {seller}'s product with the pain points, challenges, objections, and questions of the {buyer}\n"
                " to simulate the call.\n"
                
                
                "In the call, {seller}'s sales person will aim to provide the features of the product or service that will be demonstrated during the sales demo."
                "The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the demo."
                "In order to simulate the demo call, you need to follow the following steps - "
                "1. Think about what is the objective of a general demo sales call after the demo call."
                "2. Use the context, i.e., {seller}'s features and buyer's requirements, pain points and their mappings to simulate the call that fulfills the objectives of a demo call.\n"

                "Your goal is to provide a realistic and engaging simulation of a demo call."
                "The sales call MUST clearly cover the {buyer}'s goals, requirements, potential pain points, challenges, and objections."
                "The contents of each message should be as detailed and realistic as possible like a human conversation."
                "The call should be structured and flow naturally like a real demo call."
                "The call should have a smooth flow and should be engaging and informative."
                "The call should not end abruptly and should have a proper conclusion with next steps according to how the call went."
        
                "You are provided with the following context -\n"
                "{seller}'s product features: {features}\n"
                "{seller}'s product to requirements and goals mappings: {goal_or_requirement_feature_map}\n"
                "{seller}'s product to QOPCs mappings: {qopc_feature_map}\n"
                "{seller}'s product to ROI mappings: {roi_feature_map}\n"
            ),
            expected_output=(
                "A realistic simulation of the demo call between {buyer} and {seller}."
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
                    •	Conciseness and Clarity: Provide clear, concise, and relevant content without copying verbatim from the transcript unless necessary.
                    •	Completeness: Ensure that all sections and subsections are filled, even if the transcript has limited information (e.g., note “not mentioned” for missing content).
                    •	Context Preservation: Ensure the content reflects the context of the conversation accurately, especially for subjective parts (e.g., client goals and objections).
                    
                
                Below is the schema that you need to follow for the extraction of the data from the call transcript.
                {schema}
                
                And below is the transcript that you need to extract the data from.
                {demo_transcript}
                """                
            ),
            expected_output=(
                "A structured representation of the transcript, with each section and subsection filled with relevant text or annotated as missing if not found in the transcript."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="DataExtractionAgent",
            output_pydantic=FilledSections
        )
    },
    ANALYSIS: {
        "DemoCallAnalysis": dict(
            name="Analyze the Demo Call",
            description=(
                "Analyze the sales call between {buyer} and {seller} to identify the key features, pain points, challenges, objections, insights, and areas of improvement.\n"
                "You are provided with the transcript of the demo call between {buyer} and {seller}.\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                "You need to identify the key features, pain points, challenges, objections, insights, and areas of improvement from the call.\n"
                "The analysis should be detailed and provide actionable insights to the sales team.\n"
                "The analysis should also include potential strategies for future calls.\n"
                "You are provided with the following context -\n"
                "{demo_transcript}\n"
            ),
            expected_output=(
                "An analysis of the sales call between {buyer} and {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            agent="DemoCallAnalysisAgent",
            output_pydantic=AnalysisResults
        )
    }
}


def get_crew(step: str, llm: LLM, **crew_config) -> Crew:
    assert step in [RESEARCH, SIMULATION, EXTRACTION, ANALYSIS], \
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    
    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config
    )


def process_research_data_output(response: CrewOutput):
    features = format_response(response.tasks_output[0])
    goal_or_requirement_feature_map = format_response(response.tasks_output[1])
    qopc_feature_map = format_response(response.tasks_output[2])
    roi_feature_map = format_response(response.tasks_output[3])
    
    return dict(
        features=features,
        goal_or_requirement_feature_map=goal_or_requirement_feature_map,
        qopc_feature_map=qopc_feature_map,
        roi_feature_map=roi_feature_map
    )


async def aget_research_data_for_client(inputs: dict, llm: LLM, **crew_config):
    if utils.check_data_exists(inputs['buyer']):
        data: Dict = utils.get_client_data(inputs['buyer'])
        if 'features' in data:
            return data
        else:
            inputs.update(data)
    else:
        raise ValueError(f"Data for client {inputs['buyer']} does not exist.")
    
    crew = get_crew(RESEARCH, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    data.update(process_research_data_output(response))
    return data


async def aget_research_data(
    clients: List[str],
    inputs: dict,
    llm: LLM,
    **crew_config
):
    response_data = dict()
    for client_name in tqdm(clients, desc="Get Research Data"):
        inputs['buyer'] = client_name
        response_data[client_name] = await aget_research_data_for_client(
            inputs,
            llm,
            **crew_config
        )
    
    utils.save_clients_data(response_data)
    return response_data


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
    ]]), "Invalid input data for simulation"
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        if 'demo_transcript' in data:
            return data['demo_transcript']
    
    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


async def aextract_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert all([k in inputs for k in [
        "seller", 
        "buyer", 
        "schema",
        "demo_transcript"
    ]]), "Invalid input data for simulation"
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        if 'demo_transcript_structured' in data:
            return data['demo_transcript_structured']
    
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
        "demo_transcript"
    ]]), "Invalid input data for simulation"
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        if 'demo_analysis_results' in data:
            return data['demo_analysis_results']
    
    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydanctic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    return format_response(response.tasks_output[0])


async def asimulate_data(clients: List[str], inputs: dict, llm: LLM, **crew_config):
    research_data = await aget_research_data(clients, inputs, llm, **crew_config)
    simulation_data = dict()
    for client in tqdm(clients, desc="Simulating Data"):
        data = research_data[client]
        inputs.update(data)
        data['demo_transcript'] = await aget_simulation_data_for_client(inputs, llm, **crew_config)
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
        data = simulation_data[client]
        inputs.update(data)
        inputs['schema'] = schema_json
        data['demo_transcript_structured'] = await aextract_data_for_client(inputs, llm, **crew_config)
        utils.save_client_data(client, data)
        
        extracted_data[client] = data
    
    return extracted_data


async def aget_analysis(
    clients: List[str], 
    schema_json: Dict, 
    inputs: dict, 
    llm: LLM, 
    **crew_config
):
    analysis_data = dict()
    extracted_data = await aextract_data_from_transcript(clients, schema_json, inputs, llm, **crew_config)
    
    for client in tqdm(clients, desc="Analyzing Data"):
        data: Dict = extracted_data[client]
        data.update(inputs)
        data['demo_analysis_results'] = await aanalyze_data_for_client(data, llm, **crew_config)
        analysis_data[client] = data
    
    utils.save_clients_data(analysis_data)
    return analysis_data