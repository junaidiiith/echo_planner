import copy
from tqdm.asyncio import tqdm
from crewai import Crew, LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import *
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.indexing import IndexType, add_data
from echo.utils import add_pydantic_structure, format_response
from echo.step_templates.generic import CallType, Transcript, save_transcript_data
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils

from echo.step_templates.generic import Transcript


class BuyerDataExtracted(BaseModel):
    final_objections: List[str] = Field(
        ..., 
        title="Final objections raised by the buyer during the Negotiation call."
    )
    key_decision_makers: List[str] = Field(
        ..., 
        title="Key decision-makers involved in the final decision."
    )
    remaining_concerns: List[str] = Field(
        ..., 
        title="Remaining unresolved concerns of the buyer."
    )
    contract_terms: List[str] = Field(
        ..., 
        title="Contract terms and procurement concerns of the buyer."
    )


class SellerDataExtracted(BaseModel):
    concessions_discounts: List[str] = Field(
        ..., 
        title="Concessions and discounts provided by the seller to the buyer."
    )
    objection_handling: List[str] = Field(
        ..., 
        title="Objection handling by the seller's team."
    )
    stakeholder_consensus: List[str] = Field(
        ..., 
        title="Stakeholder consensus tactics and efforts."
    )
    contract_procurement: List[str] = Field(
        ..., 
        title="Contract and procurement concern handling."
    )
    closing_tactics: List[str] = Field(
        ..., 
        title="Closing tactics and next steps."
    )
    assests_used: List[str] = Field(
        ..., 
        title="Assests used by the seller's team."
    )


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
            )
        )
    },
    ANALYSIS: {
        "CallAnalysisAgent": dict(
            role="Sales Call Analysis Specialist",
            goal="Analyze the sales call between buyer and {seller} to identify the key buyer concerns around pricing, insights and areas of improvement.",
            backstory=(
                "You are an expert in analyzing pricing sales calls and identifying key insights."
                "Your goal is to analyze the sales call between buyer and {seller}."
                "Your goal is identify the key concerns raised by the buyer and provide insights to the sales team."
            )
        )
    }
}

task_templates = {
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate Negotiation Call",
            description=(
                "Simulate a very elaborated, detailed Negotiation call between {buyer} and {seller}. "
                "Assume multiple stakeholder involved from the {buyer}'s side. "
                "You are provided with the {buyer}'s and {seller}'s information as well as the discovery call information."
    
                "The information from the discovery call will be used to simulate the Negotiation call."
                "\n{buyer}'s Data from the analysis of previous discovery call:\n{discovery_analysis_buyer_data}\n"
                "\n{seller}'s Data from the analysis of previous discovery call:\n{discovery_analysis_seller_data}\n"
       
                "\n{buyer}'s Data from the analysis of previous demo call:\n{demo_analysis_buyer_data}\n"
                "\n{seller}'s Data from the analysis of previous demo call:\n{demo_analysis_seller_data}\n"

                "\n{buyer}'s Data from the analysis of previous pricing call:\n{pricing_analysis_buyer_data}\n"
                "\n{seller}'s Data from the analysis of previous pricing call:\n{pricing_analysis_seller_data}\n"
                
                "The simulation MUST cover the following instructions -\n"
                
                "The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the Negotiation."
                "In order to simulate the Negotiation call, you need to follow the following steps - "
                "1. Think about what is the objective of a general Negotiation sales call after the pricing call."
                "2. Use the context, i.e., {seller}'s previous calls analysis summary to simulate the call that fulfills the objectives of a Negotiation call.\n"
    
                "The simulation MUST cover the following instructions -\n"
                "1. The {buyer} raises some final objections during the Negotiation call and they MUST be handled by the {seller}.\n"
                "2. The {buyer} mentions the key decision-makers involved in the final decision and the {seller} should build on it.\n"
                "3. The {buyer} provides their contract terms and procurement concerns and the {seller} should build on it.\n"
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
            output_pydantic=Transcript
        )
    },
    ANALYSIS: {
        "BuyerAnalysis": dict(
            name="Analyze the Negotiation Call and extract Buyer data",
            description=(
                "Analyze the sales call between {buyer} and {seller} to extract the buyer's data.\n"
                "You are provided with the transcript of the Negotiation call between {buyer} and {seller}.\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                
                "The analysis should include -\n"
                "1. The final objections raised by the buyer during the Negotiation call.\n"
                "2. The key decision-makers involved in the final decision.\n"
                "3. The remaining unresolved concerns of the buyer.\n"
                "4. The contract terms and procurement concerns of the buyer.\n"
                
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
            output_pydantic=BuyerDataExtracted
        ),
        "SellerAnalysis": dict(
            name="Analyze the Negotiation Call and extract Seller data",
            description=(
                "Analyze the sales call between {buyer} and {seller} to extract the seller's data.\n"
                "You are provided with the transcript of the Negotiation call between {buyer} and {seller}.\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                "The analysis should include -\n"
        
                "1. Concessions and discounts provided by the seller to the buyer.\n"
                "2. Objection handling by the seller's team.\n"
                "3. Stakeholder consensus tactics and efforts.\n"
                "4. Contract and procurement concern handling.\n"
                "5. Closing tactics and next steps.\n"
                "6. Assests used by the seller's team.\n"


                
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
            output_pydantic=SellerDataExtracted
        )
    }
}

buyer_call_format = (
    "Objections Raised: {negotiation_analysis_buyer_data.final_objections}\n"
    "Key Decision Makers: {negotiation_analysis_buyer_data.key_decision_makers}\n"
    "Remaining Concerns: {negotiation_analysis_buyer_data.remaining_concerns}\n"
    "Contract Terms: {negotiation_analysis_buyer_data.contract_terms}\n"
)

seller_call_format = (
    "Concessions & Discounts: {negotiation_analysis_seller_data.concessions_discounts}\n"
    "Objection Handling: {negotiation_analysis_seller_data.objection_handling}\n"
    "Stakeholder Consensus: {negotiation_analysis_seller_data.stakeholder_consensus}\n"
    "Contract Procurement: {negotiation_analysis_seller_data.contract_procurement}\n"
    "Closing Tactics: {negotiation_analysis_seller_data.closing_tactics}\n"
    "Assests Used: {negotiation_analysis_seller_data.assests_used}\n"
)

buyer_keys = {
    "Final Objections Raised": "final_objections",
    "Key Decision Makers": "key_decision_makers",
    "Remaining Concerns": "remaining_concerns",
    "Contract Terms": "contract_terms"
}

seller_keys = {
    "Concessions & Discounts": "concessions_discounts",
    "Objection Handling": "objection_handling",
    "Stakeholder Consensus": "stakeholder_consensus",
    "Contract Procurement": "contract_procurement",
    "Closing Tactics": "closing_tactics",
    "Assests Used": "assests_used"
}


def get_analysis_data(data: Dict):
    analysis_keys = {
        "negotiation_analysis_buyer_data": BuyerDataExtracted,
        "negotiation_analysis_seller_data": SellerDataExtracted,
    }
    
    data_str = utils.get_data_str(analysis_keys, data)
    return data_str
    
    
def get_analysis_metadata(data: Dict):
    return {
        "seller": data['seller'],
        "buyer": data['buyer'],
        "call_type": CallType.NEGOTIATION.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"]
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



def get_crew(step: str, llm: LLM, **crew_config) -> Crew:
    assert step in [SIMULATION, ANALYSIS], \
        f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
    
    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config
    )


def process_analysis_data_output(response: CrewOutput):
    data = {
        "negotiation_analysis_buyer_data": format_response(response.tasks_output[0]),
        "negotiation_analysis_seller_data": format_response(response.tasks_output[1])
    }
    return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        data.update(inputs)
        if 'negotiation_transcript' in data:
            save_transcript_data(data, CallType.NEGOTIATION.value)
            return data
        else:
            print("Data exists but pricing transcript not found")
    
    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydantic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
 
    data = {
        "negotiation_transcript": format_response(response.tasks_output[0])
    }
    data.update(inputs)
    save_transcript_data(data, CallType.NEGOTIATION.value)
    return data


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    client, seller = inputs['buyer'], inputs['seller']
    def save_data():
        utils.save_client_data(client, data)
        print("Adding Analysis Data to Vector Store")
        add_data(
            data=get_analysis_data(data),
            metadata=get_analysis_metadata(data),
            index_name=seller,
            index_type=IndexType.HISTORICAL
        )

    
    if utils.check_data_exists(inputs['buyer']):
        data = utils.get_client_data(inputs['buyer'])
        data.update(inputs)
        if 'negotiation_analysis_buyer_data' in data:
            save_data()
            return data
        else:
            print("Data exists but pricing transcript not found")
            
    
    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydantic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    data = process_analysis_data_output(response)
    data.update(inputs)
    save_data()
    return data


async def asimulate_data(clients: List[str], inputs: dict, llm: LLM, **crew_config):
    simulation_data: Dict[str, Dict] = dict()
    for client in tqdm(clients, desc="Simulating Data"):
        print(f"Simulating Data for {client}")
        data = copy.deepcopy(inputs)
        data['buyer'] = client
        response = await aget_simulation_data_for_client(data, llm, **crew_config)
        data.update(response)
        utils.save_client_data(client, data)
        simulation_data[client] = data

    return simulation_data


async def aget_analysis(
    clients: List[str], 
    inputs: dict, 
    llm: LLM, 
    **crew_config
):
    analysis_data = dict()
    simulated_data = await asimulate_data(clients, inputs, llm, **crew_config)
    
    for client in tqdm(clients, desc="Analyzing Data"):
        data: Dict = copy.deepcopy(simulated_data[client])
        data.update(inputs)
        data['buyer'] = client
        response = await aanalyze_data_for_client(data, llm, **crew_config)
        data.update(response)
        utils.save_client_data(client, data)
        analysis_data[client] = data
      
    return analysis_data