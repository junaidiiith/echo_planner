import json
from crewai import LLM
import os
from echo.utils import get_llm

from echo.step_templates.discovery import asimulate_data as asimulate_discovery_data
from echo.step_templates.discovery import aget_analysis as aget_discovery_analysis

from echo.step_templates.demo import asimulate_data as asimulate_demo_data
from echo.step_templates.demo import aget_analysis as aget_demo_analysis
from echo.step_templates.generic import aget_call_structure as aget_call_structure_from_transcripts

from echo.step_templates.discovery import get_client_data_to_embed as get_discovery_data_to_embed
from echo.step_templates.demo import get_client_data_to_embed as get_demo_data_to_embed

from echo.step_templates.discovery import get_client_data_to_save as get_discovery_data_to_save
from echo.step_templates.demo import get_client_data_to_save as get_demo_data_to_save



from echo.constants import *



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
        ANALYSIS: aget_discovery_analysis,
        EMBED_DATA: get_discovery_data_to_embed,
        SAVE_DATA: get_discovery_data_to_save
    },
    DEMO: {
        SIMULATION: asimulate_demo_data,
        ANALYSIS: aget_demo_analysis,
        EMBED_DATA: get_demo_data_to_embed,
        SAVE_DATA: get_demo_data_to_save
    }
}

async def aget_call_structure(call_type, clients, inputs, llm: LLM = None, **crew_config):
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


async def simulate_calls(call_type, clients, inputs, llm: LLM = None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    simulated_call_data = await call_fns[call_type][SIMULATION](clients, inputs, llm, **crew_config)
    return simulated_call_data


async def get_analysis(call_type, clients, inputs, llm: LLM = None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
        
    analysed_call_data = await call_fns[call_type][ANALYSIS](clients, inputs, llm, **crew_config)
    return analysed_call_data


async def make_call(call_type, clients, inputs, llm: LLM = None, **crew_config):
    assert call_type in [DISCOVERY, DEMO], f"call_type must be one of {DISCOVERY}, {DEMO}"
    if llm is None:
        llm = get_llm()
    
    print(f"Making {call_type} call for {clients}")
    await call_fns[call_type][SIMULATION](clients, inputs, llm, **crew_config)
    analysed_call_data = await call_fns[call_type][ANALYSIS](clients, inputs, llm, **crew_config)
    return analysed_call_data
