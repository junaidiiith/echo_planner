import copy
import enum
from typing import Dict, List
from pydantic import BaseModel
from crewai import LLM
from tqdm.asyncio import tqdm
from echo.data.indexes import IndexDataType
from echo.indexing import IndexType, get_data_from_index
from echo.utils import (
    json_to_markdown,
)

class CallType(enum.Enum):
    PREDISCOVERY = "prediscovery"
    DISCOVERY = "discovery"
    DEMO = "demo"
    PRICING = "pricing"
    NEGOTIATION = "negotiation"


class Message(BaseModel):
    user: str
    content: str


class Transcript(BaseModel):
    messages: List[Message]



def get_analysis_by_stakeholders(inputs: dict):
    assert all(
        key in inputs for key in ["buyer", "seller"]
    ), "Please provide the required data for the analysis"
    
    records = get_data_from_index(
        index_name=inputs['seller'],
        index_type=IndexType.ANALYSIS,
        metadata={
            "buyer": inputs["buyer"],
        },
        fetch_all=True,
    )
    
    stakeholder_records: Dict[str, List] = dict()
    for record in records:
        if record["stakeholder"] not in stakeholder_records:
            stakeholder_records[record["stakeholder"]] = []
        stakeholder_records[record["stakeholder"]].append(record)
    return stakeholder_records


def get_analysis_by_stakeholder_str(inputs: dict):
    stakeholder_records = get_analysis_by_stakeholders(inputs)
    analysis_str = ""
    
    for stakeholder_name, records in stakeholder_records.items():
        analysis_str += f"\n\n{stakeholder_name} Analysis:\n"
        call_type_records = dict()
        for call_type in {record["call_type"] for record in records}:
            call_type_records[call_type] = [record['data'] for record in records if record["call_type"] == call_type]
        
        for call_type, records_data in call_type_records.items():
            analysis_str += f"\n\n{call_type} Analysis:\n"
            for data in records_data:
                analysis_str += f"{json_to_markdown(data)}\n"
        
    
    return analysis_str


def add_previous_call_analysis(inputs: dict):
    assert all(
        key in inputs for key in ["buyer", "seller"]
    ), "Please provide the required data for the analysis"
    
    previous_call_analysis_str = get_analysis_by_stakeholder_str(inputs)
    inputs['previous_calls_analysis'] = previous_call_analysis_str


def add_seller_research(inputs: dict):
    assert all(
        key in inputs for key in ["seller"]
    ), "Please provide the required data for the analysis"
    
    record = get_data_from_index(
        index_name=inputs["seller"],
        index_type=IndexType.SELLER_RESEARCH,
        metadata={
            "data_type": IndexDataType.SELLER_RESEARCH_DATA.value
        }
    )
    inputs.update(record['data'])


def add_buyer_research(inputs: dict):
    assert all(
        key in inputs for key in ["buyer", "seller"]
    ), "Please provide the required data for the analysis"
    
    record = get_data_from_index(
        index_name=inputs["buyer"],
        index_type=IndexType.BUYER_RESEARCH,
        metadata={
            "buyer": inputs["buyer"],
            "data_type": IndexDataType.BUYER_RESEARCH_DATA.value
        }
    )
    inputs.update(record['data'])


async def aget_clients_call_data(
    task_fn: callable, clients: List[str], inputs: dict, llm: LLM, **crew_config
):
    print("Number of Clients: ", len(clients))
    task_data: Dict[str, Dict] = dict()
    for client in tqdm(clients, desc="Getting Data"):
        print(f"Getting Data for {client}")
        data: Dict = copy.deepcopy(inputs)
        data["buyer"] = client
        response = await task_fn(data, llm, **crew_config)
        data.update(response)
        task_data[client] = data

    return task_data
