import copy
from echo.agent import EchoAgent
from crewai import LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import RESEARCH, SIMULATION, ANALYSIS, EXTRACTION
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.indexing import IndexType, add_data
from echo.utils import add_pydantic_structure, format_response, json_to_markdown
from echo.step_templates.generic import (
    CallType,
    Transcript,
    add_previous_call_analysis,
    aget_clients_call_data,
    add_buyer_research,
    add_seller_research
)
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils

import echo.sqldb as sqldb

# ------------------------------------------------------------------------------------------------ #
## Seller Data
# ------------------------------------------------------------------------------------------------ #


class Feature(BaseModel):
    name: str = Field(
        ...,
        title="Feature Name",
        description="The name of the feature of the product or service.",
    )
    description: str = Field(
        ..., title="Feature Description", description="A description of the feature."
    )


class FeatureWithPainPoint(BaseModel):
    name: str = Field(
        ...,
        title="Feature Name",
        description="The name of the feature of the product or service.",
    )
    description: str = Field(
        ..., title="Feature Description", description="A description of the feature."
    )
    pain_point: str = Field(
        ...,
        title="Pain Point",
        description="The pain point or challenge that the feature addresses.",
    )


class FeaturesAnticipated(BaseModel):
    features: List[Feature] = Field(
        ...,
        title="Features",
        description="A list of features of the product or service that will be demonstrated during the sales demo.",
    )


class SkepticalFeature(Feature):
    name: str = Field(
        ...,
        title="Feature Name",
        description="The name of the feature of the product or service.",
    )
    description: str = Field(
        ..., title="Feature Description", description="A description of the feature."
    )
    objection: str = Field(
        ...,
        title="Objection",
        description="The objection or concern raised by the buyer.",
    )


class ObjectionResolution(BaseModel):
    objection: str = Field(
        ..., title="Objection", description="The objection raised by the buyer."
    )
    resolution: str = Field(
        ..., title="Resolution", description="The resolution provided by the seller."
    )
    status: bool = Field(
        ...,
        title="Status",
        description="The status of the resolution, True if resolved, False if not resolved.",
    )


class SellerDataExtracted(BaseModel):
    demo_features: List[Feature] = Field(
        ...,
        title="Features",
        description="A list of features of the product or service that will be demonstrated during the sales demo.",
    )
    objection_handling: List[ObjectionResolution] = Field(
        ...,
        title="Objections",
        description="The objections raised by the buyer and their status.",
    )
    success_stories: List[str] = Field(
        ...,
        title="Success Stories",
        description="A list of success stories or case studies related to the product or service.",
    )
    assets_used: List[str] = Field(
        ...,
        title="Assets Used",
        description="A list of assets used during the sales demo, such as presentations, videos, or documents.",
    )
    customizations: List[str] = Field(
        ...,
        title="Customizations",
        description="A list of customizations or personalized features offered to the buyer.",
    )
    engagement_questions: List[str] = Field(
        ...,
        title="Engagement Questions",
        description="A list of questions to engage the buyer during the sales demo.",
    )


# ------------------------------------------------------------------------------------------------ #
## Buyer Data
# ------------------------------------------------------------------------------------------------ #


class StakeholderEngagement(BaseModel):
    stakeholder: str = Field(
        ..., title="Stakeholder", description="The stakeholder's name."
    )
    engagement_level: str = Field(
        ...,
        title="Engagement Level",
        description="The engagement level of the stakeholder during the sales demo - high, medium, or low.",
    )


class BuyerDataExtracted(BaseModel):
    interested_features: List[FeatureWithPainPoint] = Field(
        ...,
        title="Interested Features",
        description="The features of the product or service that stakeholders from the buyer are interested in.",
    )
    skeptical_features: List[SkepticalFeature] = Field(
        ...,
        title="Skeptical Features",
        description="The features of the product or service that stakeholders from the buyer are skeptical or have objections, concerns about.",
    )
    objections_raised: List[str] = Field(
        ..., title="Objections", description="A list of objections raised by the buyer."
    )
    stakeholder_engagement: List[StakeholderEngagement] = Field(
        ...,
        title="Stakeholders' Engagement",
        description="The engagement level of the stakeholders during the sales demo.",
    )
    feedback: str = Field(
        ...,
        title="Feedback",
        description="The feedback provided by the buyer after the sales demo.",
    )


agent_templates = {
    RESEARCH: {
        "PreparationAgent": dict(
            role="Sales Demo Calls Expert",
            goal="The goal of this agent is to prepare for the sales demo calls by mapping the features of the product or service to the goals, requirements, questions, objections, pain points, challenges, and return on investment.",
            backstory=(
                "You are an expert in preparing for sales demo calls. "
                "You have an in-depth understanding of how to sell the features of a product and service to potential customers."
                "You have the ability to sell the benefits of the product or service according to the requirements, goals, pain points of the customers such that they are satisfied and convinced to make a purchase."
                "Using these abilities, you will map the features of the product or service to the goals, requirements, questions, objections, pain points, challenges, and return on investment."
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
                "The sales call simulation should be tailored to address providing demo of {seller}'s features in ordet to take care of the buyer's specific pain points, and challenges."
            ),
        )
    },
    ANALYSIS: {
        "CallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key features, pain points, challenges, objections, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing demo sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is identify the key pain points, challenges, objections, key insights, areas of improvement, and potential strategies for future calls."
            ),
        )
    },
}

task_templates = {
    RESEARCH: {
        "FeatureExtraction": dict(
            name="Extract the features of the {seller}'s product",
            description=(
                "Extract the features of the {seller}'s product or service that will be demonstrated during the sales demo for the {buyer}\n"
                "You are supposed to extract the features from the information provided below.\n"
                "You are provided with the following context -\n"
                "{seller}'s research information {seller_research}\n"
                "{seller}'s pricing model {seller_pricing}\n"
                "\n{buyer}'s research information\n{buyer_research}\n"
                "\nCompetitive Information:\n{competitive_info}\n"
                "\nContext Data from the analysis of previous calls:\n{previous_calls_analysis}\n"
            ),
            expected_output=(
                "A list of features of the {seller}'s product or service that will be demonstrated during the sales demo.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            output_pydantic=FeaturesAnticipated,
            agent="PreparationAgent",
        )
    },
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate demo Call",
            description=(
                "Simulate a very elaborated, detailed demo call between a {seller} and {buyer}'s stakeholders: {stakeholders}."
                "The {buyer}'s team is represented by {stakeholders} as stakeholders during the call.\n"
                "You are provided with the {buyer}'s and {seller}'s information as well as the discovery call information."
                "The information from the discovery call will be used to simulate the demo call."

                
                "In the call, {seller}'s sales person will aim to provide the features of the product or service that will be demonstrated during the sales demo."
                "The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the demo."
                "In order to simulate the demo call, you need to follow the following steps - "
                "1. Think about what is the objective of a general demo sales call after the demo call. "
                "2. Use the context, i.e., {seller}'s features and buyer's requirements, pain points and their mappings to simulate the call that fulfills the objectives of a demo call.\n"
                
                "The simulation MUST cover the following aspects -\n"
                "1. The {seller}'s product features that MUST be clearly demonstrated during the sales demo.\n"
                "2. The {buyer} raises some objections during the sales demo and they MUST be handled by the {seller}.\n"
                "3. The {seller} MUST mention some of their relevant success stories or case studies related to the product or service.\n"
                "4. The {seller} should clarify the used during the demo assets like presentations, videos, or documents during the sales demo.\n"
                "5. The {seller} MUST offer customizations or personalized features to the buyer based on {buyer}'s objections and pain points which the existing features do not cover.\n"
                "6. The {seller} MUST engage the buyer with relevant questions. "
                
                "---Call Simulation Guidelines---:\n"
                "The {seller} MUST ADDRESS all the stakeholders in the call. "
                "All the stakeholders, i.e., {stakeholders} MUST voice their opinions, requirements, pain points and objections."
                "The call should be very detailed. "
                "Your goal is to provide a realistic and engaging simulation of a demo call."
                "The sales call MUST clearly cover the {buyer}'s goals, pain points and objections. "
                "The contents of each message should be as detailed and realistic as possible like a human conversation. "
                "The call should be structured and flow naturally like a real demo call. "
                "The call should have a smooth flow and should be engaging and informative. "
                "The call should not end abruptly and should have a proper conclusion with next steps according to how the call went. "
                
                
                "---Call Simulation Context---:\n"
                "You are provided with the following context -\n"
                "\nAnalysis Data from the analysis of previous discovery call:\n{previous_calls_analysis}\n"
                "You are also provided with {seller}'s product features that can address pain points of the buyer below - \n{demo_features}\n"
                "---End of Call Simulation Context---\n"
            ),
            expected_output=(
                "A realistic simulation of the demo call between {buyer} and {seller}."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="CallSimulationAgent",
            output_pydantic=Transcript,
        )
    },
    ANALYSIS: {
        "BuyerAnalysis": dict(
            name="Analyze the Demo Call and extract Buyer data",
            description=(
                "Analyze the sales call between {buyer} and {seller} to extract the buyer's data specifically for {buyer}'s {stakeholder}\n"
                "You are provided with the transcript of the demo call between {buyer} and {seller}.\n"
                "Your goal is to analyze the call and provide insights  specifically for {buyer}'s {stakeholder}.\n"
                "The analysis should include -\n"
                "1. The features of the product or service that {buyer}'s {stakeholder} is interested in. \n"
                "2. The features of the product or service that {buyer}'s {stakeholder} is skeptical or have objections, concerns about. \n"
                "3. The objections raised by {buyer}'s {stakeholder} \n"
                "4. Engagement level of {buyer}'s {stakeholder}\n"
                "5. Feedback provided by the {buyer}'s {stakeholder}.\n"
                "You are provided with the following context -\n"
                "{demo_transcript}\n"
            ),
            expected_output=(
                "An analysis of the sales call between {buyer} and {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            agent="CallAnalysisAgent",
            output_pydantic=BuyerDataExtracted,
        ),
        "SellerAnalysis": dict(
            name="Analyze the Demo Call and extract Seller data",
            description=(
                "Given a the demo sales call transcript between {buyer} and {seller} to identify the following key elements from the {seller}'s perspective for {buyer}'s {stakeholder} -\n"
                "Your goal is to analyze the call and provide insights to the {seller} specifically for {buyer}'s {stakeholder}.\n"
                "The analysis should include -\n"
                "1. All the features of the {seller}'s product or service that were demonstrated during the sales demo that {buyer}'s {stakeholder} showed interest in.\n"
                "2. The objections raised by {buyer}'s {stakeholder} and how they were handled by {seller}.\n"
                "3. {seller}'s Success stories or case studies related to the product or service.\n"
                "4. {seller}'s Assets used during the sales demo.\n"
                "5. {seller}'s Customizations or personalized features offered to the {buyer}'s {stakeholder}.\n"
                "6. Engagement questions to engage {buyer}'s {stakeholder} during the sales demo.\n"
                "You are provided with the following context -\n"
                "{demo_transcript}\n"
            ),
            expected_output=(
                "An analysis of the sales call between {buyer} and {seller}.\n"
                "The response should conform to the provided schema.\n"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
                "Make sure there are no comments in the response JSON and it should be a valid JSON."
            ),
            context=["BuyerAnalysis"],
            agent="CallAnalysisAgent",
            output_pydantic=SellerDataExtracted,
        ),
    },
}

def get_analysis_data(data: Dict, string_format=False):
    analysis_keys = {
        "demo_analysis_buyer_data": BuyerDataExtracted,
        "demo_analysis_seller_data": SellerDataExtracted,
    }

    if string_format:
        data_str = utils.get_data_str(analysis_keys, data)
        return data_str
    
    return {k: v for k, v in data.items() if k in analysis_keys}


def get_analysis_metadata(data: Dict):
    return {
        "seller": data["seller"],
        "buyer": data["buyer"],
        "call_type": CallType.DEMO.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"],
    }


def get_research_data(data: Dict, string_format=False):
    research_keys = {
        "demo_features": FeaturesAnticipated,
    }

    if string_format:
        data_str = utils.get_data_str(research_keys, data)
        return data_str
    return {k: v for k, v in data.items() if k in research_keys}


def get_research_metadata(data: Dict):
    return {
        "seller": data["seller"],
        "buyer": data["buyer"],
        "call_type": CallType.DEMO.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"],
    }


def get_crew(step: str, llm: LLM, **crew_config) -> EchoAgent:
    assert step in [RESEARCH, SIMULATION, EXTRACTION, ANALYSIS], (
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    )

    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config,
    )


def process_research_data_output(response: CrewOutput):
    features = format_response(response.tasks_output[0])
    return {"demo_features": features}


def process_analysis_data_output(response: CrewOutput):
    data = {
        "demo_analysis_buyer_data": format_response(response.tasks_output[0]),
        "demo_analysis_seller_data": format_response(response.tasks_output[1]),
    }
    return data


async def aget_research_data_for_client(inputs: dict, llm: LLM, **crew_config):
    data = copy.deepcopy(inputs)
    client, seller = inputs["buyer"], inputs["seller"]
    
    assert sqldb.check_record_exists(
        IndexType.BUYER_RESEARCH.value, 
        {"buyer": client, 'seller': seller}
    ), (
        f"Demo Data for client {client} does not exist."
    )

    demo_buyer_record = sqldb.get_record(
        IndexType.BUYER_RESEARCH.value, 
        {"buyer": client, 'seller': seller}
    )
    
    def save_data():
        print(f"Adding Buyer: {client} Data")
        buyer_research_data = {
            **get_research_data(data),
            **demo_buyer_record["data"],    
        }
        
        condition_attributes = {
            "buyer": client,
            "seller": seller,
            "industry": demo_buyer_record["industry"],
            "company_size": demo_buyer_record["company_size"]
        }
        
        update_dict = {
            "data": buyer_research_data,
        }
        
        sqldb.update_record(
            IndexType.BUYER_RESEARCH.value, 
            condition_attributes, 
            update_dict
        )
        
        add_data(
            data=get_research_data(data, True),
            metadata={**condition_attributes, **update_dict},
            index_name=seller,
            index_type=IndexType.BUYER_RESEARCH,
        )
    add_previous_call_analysis(data)
    add_seller_research(data)
    add_buyer_research(data)
    
    if "demo_features" in demo_buyer_record['data']:
        data.update(demo_buyer_record['data'])
        save_data()
        return data


    crew = get_crew(RESEARCH, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.DEMO.value}
    )
    data.update(process_research_data_output(response))
    save_data()
    return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert "stakeholders" in inputs, "No stakeholders found for simulation"
    

    seller, client = inputs["seller"], inputs["buyer"]
    call_id = inputs["call_id"]
    data = copy.deepcopy(inputs)
    data.update(await aget_research_data_for_client(copy.deepcopy(inputs), llm, **crew_config))
    
    def save_data():
        metadata = {
            "seller": seller,
            "buyer": client,
            "call_type": CallType.DEMO.value,
            "call_id": call_id,
            "transcript": data['demo_transcript'],
        }
        
        sqldb.insert_record(
            IndexType.CALL_TRANSCRIPTS.value,
            metadata
        )
        print(f"Embedding Simulation Data for Client: {client}")
        add_data(
            data=json_to_markdown(data['demo_transcript']),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.CALL_TRANSCRIPTS,
        )
            
    
    if sqldb.check_record_exists(
        IndexType.CALL_TRANSCRIPTS.value, 
        {"call_id": inputs['call_id'], "buyer": inputs["buyer"], "seller": inputs["seller"]}
    ):
        data['demo_transcript'] = sqldb.get_record(
            IndexType.CALL_TRANSCRIPTS.value,
            {"call_id": inputs['call_id'], "buyer": inputs["buyer"], "seller": inputs["seller"]}
        )['transcript']
        save_data()
        return data
    

    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.DEMO.value}
    )
    response_data = {"demo_transcript": format_response(response.tasks_output[0])}
    data.update(response_data)
    save_data()
    return data


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert "stakeholders" in inputs, "No stakeholders found for simulation"
    stakeholders = inputs["stakeholders"]
    client, seller = inputs["buyer"], inputs["seller"]
    call_id = inputs["call_id"]
    data = copy.deepcopy(inputs)
    data.update(await aget_simulation_data_for_client(inputs, llm, **crew_config))


    def check_stakeholder_analysis_exist(stakeholder):
        return sqldb.check_record_exists(
            IndexType.ANALYSIS.value,
            {
                "seller": seller,
                "buyer": client,
                "call_id": call_id,
                "stakeholder": stakeholder
            }
        )
    
    def save_stakeholder_analysis(stakeholder):
        metadata = get_analysis_metadata(data)
        metadata.update({
            "call_id": call_id,
            "stakeholder": stakeholder,
            "transcript": data['demo_transcript'],
            "data": get_analysis_data(data['demo_analysis_data'][stakeholder])
        })
        sqldb.insert_record(
            IndexType.ANALYSIS.value,
            metadata
        )
        print(f"Adding Analysis Data for Stakeholder: {stakeholder}")
        add_data(
            data=get_analysis_data(data['demo_analysis_data'][stakeholder], True),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.ANALYSIS,
        )
    

    if all(
        check_stakeholder_analysis_exist(stakeholder) 
        for stakeholder in stakeholders
    ):
        demo_analysis_data = dict()
        for stakeholder in stakeholders:
            demo_analysis_data[stakeholder] = sqldb.get_record(
                IndexType.ANALYSIS.value,
                {
                    "seller": seller,
                    "buyer": client,
                    "call_id": call_id,
                    "stakeholder": stakeholder
                }
            )['data']
        
        data.update({"demo_analysis_data": demo_analysis_data})
        
        for stakeholder in stakeholders:
            save_stakeholder_analysis(stakeholder)
        
        return data
    

    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydantic_structure(crew, data)
    
    demo_analysis_data = dict()
    
    for stakeholder in stakeholders:
        print(f"Analyzing demo data for stakeholder: {stakeholder}")
        
        response = await crew.kickoff_async(
            inputs={**data, "call_type": CallType.DEMO.value, "stakeholder": stakeholder}
        )
        
        analysis_data = process_analysis_data_output(response)
        demo_analysis_data[stakeholder] = analysis_data
        
    data.update({"demo_analysis_data": demo_analysis_data})
    
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

    assert all([k in inputs for k in ["seller", "n_competitors"]]), (
        f"Invalid input data for {task_type}"
    )
    print(f"Getting {task_type} Data")
    data = await aget_clients_call_data(task_fn, clients, inputs, llm, **crew_config)
    return data
