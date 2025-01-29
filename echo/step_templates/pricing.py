import copy
from tqdm.asyncio import tqdm
from crewai import Crew, LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import *
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.indexing import IndexType, add_data
from echo.utils import add_pydantic_structure, format_response
from echo.step_templates.generic import CallType, Transcript, aget_clients_call_data, save_transcript_data
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils

from echo.step_templates.generic import Transcript


class BuyerDataExtracted(BaseModel):
    concerns: List[str] = Field(
        ...,
        description="The concerns raised by the buyer during the sales call."
    )
    budget_constraints: List[str] = Field(
        ...,
        description="The budget constraints raised by the buyer during the sales call."
    )
    timelines_to_close: List[str] = Field(
        ...,
        description="The timelines to close mentioned by the buyer during the sales call."
    )
    preferred_pricing_models: List[str] = Field(
        ...,
        description="The preferred pricing models mentioned by the buyer during the sales call."
    )
    KPIs: List[str] = Field(
        ...,
        description="The Key Performance Indicators (KPIs) mentioned by the buyer during the sales call."
    )
    company_financial_priorities: List[str] = Field(
        ...,
        description="The company's financial priorities mentioned by the buyer during the sales call."
    )
    
class SellerDataExtracted(BaseModel):
    pricing_options: List[str] = Field(
        ...,
        description="The pricing options presented by the seller during the sales call."
    )
    pricing_levers: List[str] = Field(
        ...,
        description="The pricing levers used by the seller to sell value."
    )
    negotiation_tactics: List[str] = Field(
        ...,
        description="The negotiation tactics used by the seller during the sales call."
    )
    assets_used: List[str] = Field(
        ...,
        description="The assets used by the seller to make their pitch."
    )
    ROI_calculators: List[str] = Field(
        ...,
        description="The ROI calculators used by the seller during the sales call."
    )
    historical_case_studies: List[str] = Field(
        ...,
        description="The historical case studies used by the seller to sell their product or service."
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
                "Your goal is identify the key concerns around 1. Price, 2. Budget constraints, 3. Timelines to close mentioned, 4. Pricing model preferred 5. KPI's to measure success 6. Company priorities financially"
            )
        )
    }
}

task_templates = {
    SIMULATION: {
        "CallSimulationTask": dict(
            name="Simulate Pricing Call",
            description=(
                "Simulate a very elaborated, detailed pricing call between {buyer} and {seller}. "
                "Assume multiple stakeholder involved from the {buyer}'s side. "
                "You are provided with the {buyer}'s and {seller}'s information as well as the discovery call information."
    
                "The information from the discovery call will be used to simulate the pricing call."
                "\n{buyer}'s Data from the analysis of previous discovery call:\n{discovery_analysis_buyer_data}\n"
                "\n{seller}'s Data from the analysis of previous discovery call:\n{discovery_analysis_seller_data}\n"
                "\n{buyer}'s Data from the analysis of previous demo call:\n{demo_analysis_buyer_data}\n"
                "\n{seller}'s Data from the analysis of previous demo call:\n{demo_analysis_seller_data}\n"
                
                "You are also provided with pricing models of the buyer - \n{seller_pricing}\n"
    
                
                "In the call, {seller}'s sales person will aim to provide the pricing models of the product or service that was demonstrated during the sales demo."
                "The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the demo."
                "In order to simulate the pricing call, you need to follow the following steps - "
                "1. Think about what is the objective of a general pricing sales call after the demo call."
                "2. Use the context, i.e., {seller}'s pricing model and previous calls analysis summary to simulate the call that fulfills the objectives of a pricing call.\n"
    
                "The simulation MUST cover the following instructions -\n"
                "1. The {seller}'s pricing models that MUST be clearly presented and explained during the sales pricing call.\n"
                "2. The {buyer} raises some concerns during the pricing call and they MUST be handled by the {seller}.\n"
                "3. The {buyer} provides some budget constraints during the pricing call and they MUST be handled by the {seller} by providing a corresponding suitable plan\n"
                "4. The {buyer} provides some timelines to close during the pricing call and they MUST be acknowledged by the {seller} during the call\n"
                "5. The {buyer} provides their preferred pricing models and the {seller} should build on it and try to sell it better. \n"
                "6. The {buyer} provides their Key Performance Indicators (KPIs) to measure success and the {seller} should build on it. \n"
                "7. The {buyer} provides their company's finanical priorities and the {seller} should build on it. \n"
                "8. {seller}'s team should clearly present the pricing model and explain it in detail.\n"
                "9. {seller}'s team should pricing levers to sell value.\n"
                "10. {seller}'s team should negotiate with a clear pitch.\n"
                "11. {seller}'s can use several different assets to make their pitch.\n"
                "12. {seller}'s team MUST take care of their ROI and MUST USE historical case studies to sell their product service\n"
                

                "Your goal is to provide a realistic and engaging simulation of a pricing call."
                "The contents of each message should be as detailed and realistic as possible like a human conversation. "
                "The call should be structured and flow naturally like a real pricing call. "
                "The call should have a smooth flow and should be engaging and informative. "
                "The call should not end abruptly and should have a proper conclusion with next steps according to how the call went. "
                
            ),
            expected_output=(
                "A realistic simulation of the pricing call between {buyer} and {seller}."
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
            name="Analyze the Pricing Call and extract Buyer data",
            description=(
                "Analyze the sales call between {buyer} and {seller} to extract the buyer's data.\n"
                "You are provided with the transcript of the pricing call between {buyer} and {seller}.\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                
                "The analysis should include -\n"
                "1. Concers raised by the buyer during the sales pricing call.\n"
                "2. The {buyer}'s budget constraints and how they were addressed.\n"
                "3. The {buyer}'s timelines to close and how they were acknowledged.\n"
                   "4. The {buyer}'s preferred pricing models and how they were handled.\n"
                  "5. The {buyer}'s Key Performance Indicators (KPIs) to measure success and how they were addressed.\n"
                "6. The {buyer}'s company's financial priorities and how they were addressed.\n"
                
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
            name="Analyze the Pricing Call and extract Seller data",
            description=(
                "Analyze the sales call between {buyer} and {seller} to extract the seller's data.\n"
                "You are provided with the transcript of the pricing call between {buyer} and {seller}.\n"
                "Your goal is to analyze the call and provide insights to the sales team.\n"
                "The analysis should include -\n"
        
                "1. The pricing options presented by the seller during the sales pricing.\n"
                "2. The pricing levers used by the seller to sell value.\n"
                "3. The negotiation tactics used by the seller.\n"
                "4. The assets used by the seller to make their pitch.\n"
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
            output_pydantic=SellerDataExtracted
        )
    }
}


buyer_call_format = (
    "Concerns: {pricing_analysis_buyer_data.concerns}\n"
    "Budget constraints: {pricing_analysis_buyer_data.budget_constraints}\n"
    "Timelines to close: {pricing_analysis_buyer_data.timelines_to_close}\n"
    "Preferred pricing models: {pricing_analysis_buyer_data.preferred_pricing_models}\n"
    "Success KPIs: {pricing_analysis_buyer_data.KPIs}\n"
    "Company's Financial Priorities: {pricing_analysis_buyer_data.company_financial_priorities}\n"
)

seller_call_format = (
    "Pricing Options: {pricing_analysis_seller_data.pricing_options}\n"
    "Pricing Levers: {pricing_analysis_seller_data.pricing_levers}\n"
    "Negotiation Tactics: {pricing_analysis_seller_data.negotiation_tactics}\n"
    "Assets Used: {pricing_analysis_seller_data.assets_used}\n"
    "ROI Calculators: {pricing_analysis_seller_data.ROI_calculators}\n"
    "Historical Case Studies: {pricing_analysis_seller_data.historical_case_studies}\n"
)

buyer_keys = {
    "Concerns": "pricing_analysis_buyer_data.concerns",
    "Budget constraints": "pricing_analysis_buyer_data.budget_constraints",
    "Timelines to close": "pricing_analysis_buyer_data.timelines_to_close",
    "Preferred pricing models": "pricing_analysis_buyer_data.preferred_pricing_models",
    "Success KPIs": "pricing_analysis_buyer_data.KPIs",
    "Company's Financial Priorities": "pricing_analysis_buyer_data.company_financial_priorities"
}

seller_keys = {
    "Pricing Options": "pricing_analysis_seller_data.pricing_options",
    "Pricing Levers": "pricing_analysis_seller_data.pricing_levers",
    "Negotiation Tactics": "pricing_analysis_seller_data.negotiation_tactics",
    "Assets Used": "pricing_analysis_seller_data.assets_used",
    "ROI Calculators": "pricing_analysis_seller_data.ROI_calculators",
    "Historical Case Studies": "pricing_analysis_seller_data.historical_case_studies"
}


def get_analysis_data(data: Dict):
    analysis_keys = {
        "pricing_analysis_buyer_data": BuyerDataExtracted,
        "pricing_analysis_seller_data": SellerDataExtracted,
    }
    
    data_str = utils.get_data_str(analysis_keys, data)
    return data_str
    

def get_analysis_metadata(data: Dict):
    return {
        "seller": data['seller'],
        "buyer": data['buyer'],
        "call_type": CallType.PRICING.value,
        "company_size": data["buyer_research"]["company_size"],
        "industry": data["buyer_research"]["industry"],
        "description": data["buyer_research"]["description"]
    }


def get_client_data_to_embed(client_name, user_type):
    data = utils.get_client_data(client_name)
    if user_type == BUYER:
        return utils.replace_keys_with_values(buyer_call_format, data)
    return utils.replace_keys_with_values(seller_call_format, data)


def get_client_data_to_save(client_name, user_type):
    data = utils.get_client_data(client_name)
    if user_type == BUYER:
        return utils.get_nested_key_values(buyer_keys, data)
    return utils.get_nested_key_values(seller_keys, data)



def get_crew(step: str, llm: LLM, **crew_config) -> Crew:
    assert step in [SIMULATION, ANALYSIS], \
        f"Invalid step type: {step} Must be one of 'simulation', 'analysis'"
    
    return get_crew_obj(
        agent_templates=agent_templates[step],
        task_templates=task_templates[step],
        llm=llm,
        **crew_config
    )


def process_analysis_data_output(response: CrewOutput):
    data = {
        "pricing_analysis_buyer_data": format_response(response.tasks_output[0]),
        "pricing_analysis_seller_data": format_response(response.tasks_output[1])
    }
    return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
    data = copy.deepcopy(inputs)
    if utils.check_data_exists(inputs['buyer']):
        data.update(utils.get_client_data(inputs['buyer']))
        if 'pricing_transcript' in data:
            save_transcript_data(data, CallType.PRICING.value)
            return data
        else:
            print("Data exists but pricing transcript not found")
    else:
        raise Exception("Cannot simulate data without buyer data")
   
   
    crew = get_crew(SIMULATION, llm, **crew_config)
    add_pydantic_structure(crew, inputs)
    response = await crew.kickoff_async(inputs=inputs)
    data.update({
        "pricing_transcript": format_response(response.tasks_output[0])
    })
    save_transcript_data(data, CallType.PRICING.value)
    
    return data


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
    data = copy.deepcopy(inputs)
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
        data.update(utils.get_client_data(inputs['buyer']))
        if 'pricing_analysis_buyer_data' in data:
            save_data()
            return data
        
    try:
        assert all([k in data for k in [
            "pricing_transcript"
        ]]), "Invalid input data for simulation"
    except AssertionError as e:
        simulation_data = await aget_simulation_data_for_client(data, llm, **crew_config)
        data.update(simulation_data)
    
    
    crew = get_crew(ANALYSIS, llm, **crew_config)
    add_pydantic_structure(crew, data)
    response = await crew.kickoff_async(inputs=data)
    analysis_data = process_analysis_data_output(response)
    data.update(analysis_data)
    save_data()
    return data


async def aget_data_for_clients(
    task_type: str, 
    clients: List[str], 
    inputs: dict, 
    llm: LLM, 
    **crew_config
):
    assert task_type in [SIMULATION, ANALYSIS], f"Invalid task type: {task_type}"
    task_to_data_extraction_fn = {
        SIMULATION: aget_simulation_data_for_client,
        ANALYSIS: aanalyze_data_for_client
    }
    task_fn = task_to_data_extraction_fn[task_type]
    
    assert all([k in inputs for k in ["seller", "n_competitors"]]), f"Invalid input data for {task_type}"
    print(f"Getting {task_type} Data")
    data = await aget_clients_call_data(task_fn, clients, inputs, llm, **crew_config)
    return data
