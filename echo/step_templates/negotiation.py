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
    final_objections: List[str] = Field(
        ..., title="Final objections raised by the buyer during the Negotiation call."
    )
    key_decision_makers: List[str] = Field(
        ..., title="Key decision-makers involved in the final decision."
    )
    remaining_concerns: List[str] = Field(
        ..., title="Remaining unresolved concerns of the buyer."
    )
    contract_terms: List[str] = Field(
        ..., title="Contract terms and procurement concerns of the buyer."
    )


class SellerDataExtracted(BaseModel):
    concessions_discounts: List[str] = Field(
        ..., title="Concessions and discounts provided by the seller to the buyer."
    )
    objection_handling: List[str] = Field(
        ..., title="Objection handling by the seller's team."
    )
    stakeholder_consensus: List[str] = Field(
        ..., title="Stakeholder consensus tactics and efforts."
    )
    contract_procurement: List[str] = Field(
        ..., title="Contract and procurement concern handling."
    )
    closing_tactics: List[str] = Field(..., title="Closing tactics and next steps.")
    assests_used: List[str] = Field(..., title="Assests used by the seller's team.")


agent_templates = {
    SIMULATION: {
        "CallSimulationAgent": dict(
            role="Sales Call Simulation Specialist",
            goal="Simulate a very elaborated, detailed call between buyer and {seller}.",
            backstory=(
                "You are an expert in simulating realistic sales negotiation calls after pricing is discussed."
                "You have been tasked with simulating a detailed call between buyer and {seller}."
                "Your goal is to provide a realistic and engaging simulation of the call."
                "The sales call simulation should be tailored to addressing the buyer's needs."
            ),
        )
    },
    ANALYSIS: {
        "CallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key buyer concerns around negotiations, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing negotiation sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is identify the key concerns raised by the buyer and provide insights to the sales team."
            ),
        )
    },
}

task_templates = {
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate Negotiation Call",
            description=(
                "Simulate a very elaborated, detailed negotiation call between a {seller} and {buyer}'s stakeholders: {stakeholders}."
                "The {buyer}'s team is represented by {stakeholders} as stakeholders during the call.\n"
                "You need to simulate the call as a conversation between the {seller}'s sales person and ALL the {buyer}'s stakeholders.\n"
                "The information from the previous calls will be used to simulate the Negotiation call."
                "\nData from the analysis of previous discovery call:\n{previous_calls_analysis}\n"
                
                
                "The simulation MUST cover the following instructions -\n"
                "The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the Negotiation."
                "In order to simulate the Negotiation call, you need to follow the following steps - "
                "1. Think about what is the objective of a general Negotiation sales call after the pricing call."
                "2. Use the context, i.e., {seller}'s previous calls analysis summary to simulate the call that fulfills the objectives of a Negotiation call.\n"
                "The simulation MUST cover the following instructions -\n"
                "1. The {buyer}'s {stakeholders} raise some final objections during the Negotiation call and they MUST be handled by the {seller}.\n"
                "2. The {buyer}'s {stakeholders} mentions the key decision-makers involved in the final decision and the {seller} should build on it.\n"
                "3. The {buyer}'s {stakeholders} provides their contract terms and procurement concerns and the {seller} should build on it.\n"
                "4. {seller}'s team should provide discounts and concessations to take care of the {buyer}'s concerns.\n"
                "5. {seller}'s team should do very clear and helpful objection handling\n"
                "6. {seller}'s team should use stakeholder consensus tactics and efforts.\n"
                "7. {seller}'s team should handle contract and procurement concern handling.\n"
                "Your goal is to provide a realistic and engaging simulation of a negotiation call."
                "The contents of each message should be as detailed and realistic as possible like a human conversation. "
                "The call should be structured and flow naturally like a real negotiation call. "
                "The call should have a smooth flow and should be engaging and informative. "
                "The call should not end abruptly and should have a proper conclusion with next steps according to how the call went. "
            ),
            expected_output=(
                "A realistic simulation of the negotiation call between {buyer} and {seller}."
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
            name="Analyze the Negotiation Call and extract Buyer data",
            description=(
                "Analyze the sales call between {buyer} and {seller} to extract the {buyer}'s {stakeholder} data..\n"
                "You are provided with the transcript of the Negotiation call between {seller} and the stakeholders of {buyer} including the {stakeholder}.\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                "The analysis should include -\n"
                "1. The final objections raised by the {buyer}'s {stakeholder} during the Negotiation call.\n"
                "2. The key decision-makers involved in the final decision mentioned by {buyer}'s {stakeholder}\n"
                "3. The remaining unresolved concerns of the {buyer}'s {stakeholder} \n"
                "4. The contract terms and procurement concerns of the {buyer}'s {stakeholder}\n"
                "You are provided with the following context -\n"
                "{negotiation_transcript}\n"
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
            name="Analyze the Negotiation Call and extract Seller data",
            description=(
                "Analyze the sales call between {seller} and {buyer}'s {stakeholder} to extract {seller}'s data specific to {buyer}'s {stakeholder}\n"
                "You are provided with the transcript of the pricing call between {seller} and {buyer}'s stakeholders include {stakeholder}\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                "The analysis should include -\n"
                "1. Concessions and discounts provided by the seller for {buyer}'s {stakeholder}.\n"
                "2. Objections from {buyer}'s {stakeholder} handled by the seller's team.\n"
                "3. {buyer}'s {stakeholder} consensus tactics and efforts.\n"
                "4. Contract and procurement concern handling.\n"
                "5. Closing tactics and next steps.\n"
                "6. Assests used by the seller's team.\n"
                "You are provided with the following context -\n"
                "{negotiation_transcript}\n"
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
        "negotiation_analysis_buyer_data": BuyerDataExtracted,
        "negotiation_analysis_seller_data": SellerDataExtracted,
    }

    if string_format:
        data_str = utils.get_data_str(analysis_keys, data)
        return data_str
    
    return {k: v for k, v in data.items() if k in analysis_keys}


def get_analysis_metadata(data: Dict):
    return {
        "seller": data["seller"],
        "buyer": data["buyer"],
        "call_type": CallType.NEGOTIATION.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"],
    }


def get_crew(step: str, llm: LLM, **crew_config) -> EchoAgent:
    assert step in [SIMULATION, ANALYSIS], (
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    )

    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config,
    )


def process_analysis_data_output(response: CrewOutput):
    data = {
        "negotiation_analysis_buyer_data": format_response(response.tasks_output[0]),
        "negotiation_analysis_seller_data": format_response(response.tasks_output[1]),
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
            "call_type": CallType.NEGOTIATION.value,
            "call_id": call_id,
            "transcript": data["negotiation_transcript"],
        }

        sqldb.insert_record(IndexType.CALL_TRANSCRIPTS.value, metadata)
        print(f"Embedding Simulation Data for Client: {client}")
        add_data(
            data=json_to_markdown(data["negotiation_transcript"]),
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
        data["negotiation_transcript"] = sqldb.get_record(
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
        inputs={**data, "call_type": CallType.NEGOTIATION.value}
    )

    data.update({"negotiation_transcript": format_response(response.tasks_output[0])})

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
                "transcript": data["negotiation_transcript"],
                "data": get_analysis_data(data["negotiation_analysis_data"][stakeholder]),
            }
        )
        sqldb.insert_record(IndexType.ANALYSIS.value, metadata)
        print(f"Adding Analysis Data for Stakeholder: {stakeholder}")
        add_data(
            data=get_analysis_data(data["negotiation_analysis_data"][stakeholder], True),
            metadata=metadata,
            index_name=seller,
            index_type=IndexType.ANALYSIS,
        )

    if all(
        check_stakeholder_analysis_exist(stakeholder) for stakeholder in stakeholders
    ):
        negotiation_analysis_data = dict()
        for stakeholder in stakeholders:
            negotiation_analysis_data[stakeholder] = sqldb.get_record(
                IndexType.ANALYSIS.value,
                {
                    "seller": seller,
                    "buyer": client,
                    "call_id": call_id,
                    "stakeholder": stakeholder,
                },
            )["data"]

        data.update({"negotiation_analysis_data": negotiation_analysis_data})
        for stakeholder in stakeholders:
            save_stakeholder_analysis(stakeholder)
        return data
    
    
    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydantic_structure(crew, data)

    negotiation_analysis_data = dict()

    for stakeholder in stakeholders:
        print(f"Analyzing negotiation data for stakeholder: {stakeholder}")

        response = await crew.kickoff_async(
            inputs={
                **data,
                "call_type": CallType.NEGOTIATION.value,
                "stakeholder": stakeholder,
            }
        )

        analysis_data = process_analysis_data_output(response)
        negotiation_analysis_data[stakeholder] = analysis_data

    data.update({"negotiation_analysis_data": negotiation_analysis_data})

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
