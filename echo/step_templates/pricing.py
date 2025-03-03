import copy
from echo import sqldb
from echo.agent import EchoAgent
from crewai import LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import (
    SIMULATION,
    ANALYSIS,
)
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.indexing import IndexType, add_data
from echo.utils import add_pydantic_structure, format_response, json_to_markdown
from echo.step_templates.generic import (
    CallType,
    Transcript,
    add_buyer_research,
    add_previous_call_analysis,
    add_seller_research,
    aget_clients_call_data,
)
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils


class BuyerDataExtracted(BaseModel):
    concerns: List[str] = Field(
        ..., description="The concerns raised by the buyer during the sales call."
    )
    budget_constraints: List[str] = Field(
        ...,
        description="The budget constraints raised by the buyer during the sales call.",
    )
    timelines_to_close: List[str] = Field(
        ...,
        description="The timelines to close mentioned by the buyer during the sales call.",
    )
    preferred_pricing_models: List[str] = Field(
        ...,
        description="The preferred pricing models mentioned by the buyer during the sales call.",
    )
    KPIs: List[str] = Field(
        ...,
        description="The Key Performance Indicators (KPIs) mentioned by the buyer during the sales call.",
    )
    company_financial_priorities: List[str] = Field(
        ...,
        description="The company's financial priorities mentioned by the buyer during the sales call.",
    )


class SellerDataExtracted(BaseModel):
    pricing_options: List[str] = Field(
        ...,
        description="The pricing options presented by the seller during the sales call.",
    )
    pricing_levers: List[str] = Field(
        ..., description="The pricing levers used by the seller to sell value."
    )
    negotiation_tactics: List[str] = Field(
        ...,
        description="The negotiation tactics used by the seller during the sales call.",
    )
    assets_used: List[str] = Field(
        ..., description="The assets used by the seller to make their pitch."
    )
    ROI_calculators: List[str] = Field(
        ..., description="The ROI calculators used by the seller during the sales call."
    )
    historical_case_studies: List[str] = Field(
        ...,
        description="The historical case studies used by the seller to sell their product or service.",
    )


agent_templates = {
    SIMULATION: {
        "CallSimulationAgent": dict(
            role="Sales Call Simulation Specialist",
            goal="Simulate a very elaborated, detailed call between buyer and {seller}.",
            backstory=(
                "You are an expert in simulating realistic sales pricing calls."
                "You have been tasked with simulating a detailed call between buyer and {seller}."
                "Your goal is to provide a realistic and engaging simulation of the call."
                "The sales call simulation should be tailored to addressing the buyer's pricing related requirements."
            ),
        )
    },
    ANALYSIS: {
        "CallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key buyer concerns around pricing, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing pricing sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is identify the key concerns around 1. Price, 2. Budget constraints, 3. Timelines to close mentioned, 4. Pricing model preferred 5. KPI's to measure success 6. Company priorities financially"
            ),
        )
    },
}

task_templates = {
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate Pricing Call",
            description=(
                "Simulate a very elaborated, detailed pricing call between a {seller} and {buyer}'s stakeholders: {stakeholders}."
                "The {buyer}'s team is represented by {stakeholders} as stakeholders during the call.\n"
                "You need to simulate the call as a conversation between the {seller}'s sales person and ALL the {buyer}'s stakeholders.\n"
                "You are provided with the {buyer}'s and {seller}'s information from previous calls as well."
                "The information from the previous calls will be used to simulate the pricing call."
                
                "\n{buyer}'s Data from the analysis of previous calls:\n{previous_calls_analysis}\n"
                "Below is the pricing models of the buyer - \n---Pricing Model---\n{seller_pricing}---End of Pricing Model---\n\n"
                
                "In the call, {seller}'s sales person will aim to provide the pricing models for the product or service that was demonstrated during the sales demo."
                "The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the demo."
                "In order to simulate the pricing call, you need to follow the following steps - "
                "1. Think about what is the objective of a general pricing sales call after the demo call."
                "2. Use the context, i.e., {seller}'s pricing model and previous calls analysis summary to simulate the call that fulfills the objectives of a pricing call.\n"
                
                "---Pricing Call Objectives---\n"
                "The simulation MUST cover the following instructions -\n"
                "1. The {seller}'s pricing models that MUST be clearly presented and explained during the sales pricing call.\n"
                "2. {buyer}'s {stakeholders} raises some concerns during the pricing call and they MUST be handled by the {seller}.\n"
                "3. The {buyer}'s {stakeholders} provide some constraints during the pricing call and they MUST be handled by the {seller} by providing a corresponding suitable plan\n"
                "4. The {buyer}'s {stakeholders} provide some timelines to close during the pricing call and they MUST be acknowledged by the {seller} during the call\n"
                "5. The {buyer}'s {stakeholders} provide their preferred pricing models and the {seller} should build on it and try to sell it better. \n"
                "6. The {buyer}'s {stakeholders} provide their Key Performance Indicators (KPIs) to measure success and the {seller} should build on it. \n"
                "7. The {buyer}'s {stakeholders} provide their company's finanical priorities and the {seller} should build on it. \n"
                "8. {seller}'s team should pricing levers to sell value.\n"
                "9. {seller}'s team should negotiate with a clear pitch.\n"
                "10. {seller}'s can use several different assets to make their pitch.\n"
                "11. {seller}'s team MUST take care of their ROI and MUST USE historical case studies to sell their product service\n"
                "---End of Pricing Call Objectives---\n"
                
                
                "---Call Simulation Guidelines---:\n"
                "Your goal is to provide a realistic and engaging simulation of a pricing call."
                "The contents of each message should be as detailed and realistic as possible like a human conversation. "
                "The call should be structured and flow naturally like a real pricing call. "
                "The call should have a smooth flow and should be engaging and informative. "
                "The call should not end abruptly and should have a proper conclusion with next steps according to how the call went. "
                "---End of Call Simulation Guidelines---"
            ),
            expected_output=(
                "A realistic simulation of the pricing call between {buyer} and {seller}."
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
            name="Analyze the Pricing Call and extract Buyer data",
            description=(
                "Analyze the sales call between {buyer}'s {stakeholder} and {seller} to extract the {buyer}'s {stakeholder} data.\n"
                "You are provided with the transcript of the pricing call between {seller} and the stakeholders of {buyer} including the {stakeholder}.\n"
                "Your goal is to analyze the call and provide insights to the sales team specific to the {buyer}'s {stakeholder}\n"
                "The analysis should include -\n"
                "1. Concers raised by the {buyer}'s {stakeholder} during the sales pricing call.\n"
                "2. The {buyer}'s {stakeholder} budget constraints and how they were addressed.\n"
                "3. The {buyer}'s {stakeholder} timelines to close and how they were acknowledged.\n"
                "4. The {buyer}'s {stakeholder} preferred pricing models and how they were handled.\n"
                "5. The {buyer}'s {stakeholder} Key Performance Indicators (KPIs) to measure success and how they were addressed.\n"
                "6. The {buyer}'s {stakeholder} company's financial priorities and how they were addressed.\n"
                "You are provided with the following context -\n"
                "{pricing_transcript}\n"
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
            name="Analyze the Pricing Call and extract Seller data",
            description=(
                "Analyze the sales call between {seller} and {buyer}'s {stakeholder} to extract {seller}'s data.\n"
                "You are provided with the transcript of the pricing call between {seller} and {buyer}'s stakeholders include {stakeholder}\n"
                "Your goal is to analyze the call and provide insights to the sales team specifically for {buyer}'s {stakeholder}\n"
                "The analysis should include -\n"
                "1. The pricing options presented by the seller during the sales pricing that appealed the {buyer}'s {stakeholder}\n"
                "2. The pricing levers used by the seller to sell value that appealed the {buyer}'s {stakeholder}\n"
                "3. The negotiation tactics used by the seller that appealed the {buyer}'s {stakeholder}\n"
                "4. The assets used by the seller to make their pitch that appealed the {buyer}'s {stakeholder}\n"
                "5. The ROI calculators in the call. \n"
                "6. The historical case studies used by the seller to sell their product or service.\n"
                "You are provided with the following context -\n"
                "{pricing_transcript}\n"
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


def get_analysis_data(data: Dict, string_format: bool = False):
    analysis_keys = {
        "pricing_analysis_buyer_data": BuyerDataExtracted,
        "pricing_analysis_seller_data": SellerDataExtracted,
    }

    if string_format:
        data_str = utils.get_data_str(analysis_keys, data)
        return data_str
    
    return {k: v for k, v in data.items() if k in analysis_keys}


def get_analysis_metadata(data: Dict):
    return {
        "seller": data["seller"],
        "buyer": data["buyer"],
        "call_type": CallType.PRICING.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"],
    }


def get_crew(step: str, llm: LLM, **crew_config) -> EchoAgent:
    assert step in [SIMULATION, ANALYSIS], (
        f"Invalid step type: {step} Must be one of 'simulation', 'analysis'"
    )

    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config,
    )


def process_analysis_data_output(response: CrewOutput):
    data = {
        "pricing_analysis_buyer_data": format_response(response.tasks_output[0]),
        "pricing_analysis_seller_data": format_response(response.tasks_output[1]),
    }
    return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert "stakeholders" in inputs, "No stakeholders found for simulation"
    seller, client = inputs["seller"], inputs["buyer"]
    call_id = inputs["call_id"]
    
    data = copy.deepcopy(inputs)
    add_previous_call_analysis(data)
    add_seller_research(data)
    
    
    def save_data():
        metadata = {
            "seller": seller,
            "buyer": client,
            "call_type": CallType.PRICING.value,
            "call_id": call_id,
            "transcript": data["pricing_transcript"],
        }

        sqldb.insert_record(IndexType.CALL_TRANSCRIPTS.value, metadata)
        print(f"Embedding Simulation Data for Client: {client}")
        add_data(
            data=json_to_markdown(data["pricing_transcript"]),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.CALL_TRANSCRIPTS,
        )

    if sqldb.check_record_exists(
        IndexType.CALL_TRANSCRIPTS.value,
        {
            "call_id": inputs["call_id"],
            "buyer": inputs["buyer"],
            "seller": inputs["seller"],
        },
    ):
        data["pricing_transcript"] = sqldb.get_record(
            IndexType.CALL_TRANSCRIPTS.value,
            {
                "call_id": inputs["call_id"],
                "buyer": inputs["buyer"],
                "seller": inputs["seller"],
            },
        )["transcript"]
        save_data()
        return data
    

    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(
        inputs={**data, "call_type": CallType.PRICING.value}
    )

    data.update({"pricing_transcript": format_response(response.tasks_output[0])})

    save_data()
    return data



async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    assert "stakeholders" in inputs, "No stakeholders found for simulation"
    stakeholders = inputs["stakeholders"]
    client, seller = inputs["buyer"], inputs["seller"]
    call_id = inputs["call_id"]
    data = copy.deepcopy(inputs)
    data.update(await aget_simulation_data_for_client(copy.deepcopy(inputs), llm, **crew_config))

    add_buyer_research(data)

    def check_stakeholder_analysis_exist(stakeholder):
        return sqldb.check_record_exists(
            IndexType.ANALYSIS.value,
            {
                "seller": seller,
                "buyer": client,
                "call_id": call_id,
                "stakeholder": stakeholder,
            },
        )

    def save_stakeholder_analysis(stakeholder):
        metadata = get_analysis_metadata(data)
        metadata.update(
            {
                "call_id": call_id,
                "stakeholder": stakeholder,
                "transcript": data["pricing_transcript"],
                "data": get_analysis_data(data["pricing_analysis_data"][stakeholder]),
            }
        )
        sqldb.insert_record(IndexType.ANALYSIS.value, metadata)
        print(f"Adding Analysis Data for Stakeholder: {stakeholder}")
        add_data(
            data=get_analysis_data(data["pricing_analysis_data"][stakeholder], True),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.ANALYSIS,
        )

    if all(
        check_stakeholder_analysis_exist(stakeholder) for stakeholder in stakeholders
    ):
        pricing_analysis_data = dict()
        for stakeholder in stakeholders:
            pricing_analysis_data[stakeholder] = sqldb.get_record(
                IndexType.ANALYSIS.value,
                {
                    "seller": seller,
                    "buyer": client,
                    "call_id": call_id,
                    "stakeholder": stakeholder,
                },
            )["data"]

        data.update({"pricing_analysis_data": pricing_analysis_data})
        for stakeholder in stakeholders:
            save_stakeholder_analysis(stakeholder)
        return data
    
    
    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydantic_structure(crew, data)

    pricing_analysis_data = dict()

    for stakeholder in stakeholders:
        print(f"Analyzing pricing data for stakeholder: {stakeholder}")

        response = await crew.kickoff_async(
            inputs={
                **data,
                "call_type": CallType.PRICING.value,
                "stakeholder": stakeholder,
            }
        )

        analysis_data = process_analysis_data_output(response)
        pricing_analysis_data[stakeholder] = analysis_data

    data.update({"pricing_analysis_data": pricing_analysis_data})

    for stakeholder in stakeholders:
        save_stakeholder_analysis(stakeholder)

    return data



async def aget_data_for_clients(
    task_type: str, clients: List[str], inputs: dict, llm: LLM, **crew_config
):
    assert task_type in [SIMULATION, ANALYSIS], f"Invalid task type: {task_type}"
    task_to_data_extraction_fn = {
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
