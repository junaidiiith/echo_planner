import json
from crewai import LLM
from dotenv import load_dotenv
import os
from echo.step_templates.discovery import asimulate_data as asimulate_discovery_data
from echo.step_templates.discovery import aextract_data_from_transcript as aextract_discovery_data_from_transcript
from echo.step_templates.discovery import aget_analysis as aget_discovery_analysis

from echo.step_templates.demo import asimulate_data as asimulate_demo_data
from echo.step_templates.demo import aextract_data_from_transcript as aextract_demo_data_from_transcript
from echo.step_templates.demo import aget_analysis as aget_demo_analysis
from echo.step_templates.section_extraction_and_mapping import aget_call_structure as aget_call_structure_from_transcripts

from echo.constants import *


load_dotenv()

def get_llm():
    llm = LLM(
        model=os.getenv("FIREWORKS_MODEL_NAME"),
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY")
    )
    return llm


test_clients = [
	"Shell",
	"Schneider electric"
]


train_clients = [
	"ICICI bank",
	"Infosys",	
	"University of Illinois",
	"Marks and spencer",
	"Mercedes Benz",
]


inputs = {
    'call_type': 'discovery', 
    'seller': 'Whatfix',
    'n_competitors': 3
}

call_fns = {
    DISCOVERY: {
        SIMULATION: asimulate_discovery_data,
        EXTRACTION: aextract_discovery_data_from_transcript,
        ANALYSIS: aget_discovery_analysis
    },
    DEMO: {
        SIMULATION: asimulate_demo_data,
        EXTRACTION: aextract_demo_data_from_transcript,
        ANALYSIS: aget_demo_analysis
    }
}

async def aget_call_structure(call_type, clients, inputs, llm=None, **crew_config):
    """
    Required that the transcripts are already simulated/created
    """
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    if all([os.path.exists(f"runs/{client_name}.json") for client_name in clients])\
        and all([f'{call_type}_transcript' in json.load(open(f"runs/{client_name}.json")) for client_name in clients]):
        transcripts = [
            json.load(open("runs/" + client_name + ".json"))[f'{call_type}_transcript'] 
            for client_name in clients
        ]
        call_structure = await aget_call_structure_from_transcripts(transcripts, llm, inputs, **crew_config)
        return call_structure
    
    await call_fns[call_type][SIMULATION](clients, inputs, llm, **crew_config)
    transcripts = [
        json.load(open("runs/" + client_name + ".json"))[f'{call_type}_transcript'] 
        for client_name in clients
    ]
    call_structure = await aget_call_structure_from_transcripts(transcripts, llm, inputs, **crew_config)
    return call_structure


async def simulate_calls(call_type, clients, inputs, llm=None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    simulated_call_data = await call_fns[call_type][SIMULATION](clients, inputs, llm, **crew_config)
    return simulated_call_data


async def extract_data_from_transcript(call_type, clients, schema_json, inputs, llm=None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    extracted_data = await call_fns[call_type][EXTRACTION](clients, schema_json, inputs, llm=llm, **crew_config)
    return extracted_data


async def get_analysis(call_type, clients, schema_json, inputs, llm=None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    analysed_call_data = await call_fns[call_type][ANALYSIS](clients, schema_json, inputs, llm, **crew_config)
    return analysed_call_data


async def make_call(call_type, clients, schema_json, inputs, llm=None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    await call_fns[call_type][SIMULATION](clients, inputs, llm, **crew_config)
    await call_fns[call_type][EXTRACTION](clients, schema_json, inputs, llm=llm, **crew_config)
    analysed_call_data = await call_fns[call_type][ANALYSIS](clients, schema_json, inputs, llm, **crew_config)
    return analysed_call_data
