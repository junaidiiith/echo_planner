from crewai import LLM
from echo.utils import get_llm


from echo.step_templates.discovery import (
    aget_data_for_clients as aget_discovery_data_for_clients,
    aget_seller_data,
)
from echo.step_templates.demo import aget_data_for_clients as aget_demo_data_for_clients
from echo.step_templates.pricing import (
    aget_data_for_clients as aget_pricing_data_for_clients,
)
from echo.step_templates.negotiation import (
    aget_data_for_clients as aget_negotiation_data_for_clients,
)

from echo.constants import (
    DISCOVERY,
    DEMO,
    PRICING,
    NEGOTIATION,
    SIMULATION,
    ANALYSIS,
)


call_fns = {
    DISCOVERY: aget_discovery_data_for_clients,
    DEMO: aget_demo_data_for_clients,
    PRICING: aget_pricing_data_for_clients,
    NEGOTIATION: aget_negotiation_data_for_clients,
}


async def simulate_calls(call_type, clients, inputs, llm: LLM = None, **crew_config):
    assert call_type in [DISCOVERY, DEMO, PRICING, NEGOTIATION], (
        f"call_type must be one of {DISCOVERY}, {DEMO}, {PRICING}, {NEGOTIATION}"
    )
    if llm is None:
        llm = get_llm()

    simulated_call_data = await call_fns[call_type](
        SIMULATION, clients, inputs, llm, **crew_config
    )
    return simulated_call_data


async def get_analysis(call_type, clients, inputs, llm: LLM = None, **crew_config):
    assert call_type in [DISCOVERY, DEMO, PRICING, NEGOTIATION], (
        f"call_type must be one of {DISCOVERY}, {DEMO}, {PRICING}, {NEGOTIATION}"
    )
    if llm is None:
        llm = get_llm()

    analysed_call_data = await call_fns[call_type](
        ANALYSIS, clients, inputs, llm, **crew_config
    )
    return analysed_call_data


async def make_call(call_type, clients, inputs, llm: LLM = None, **crew_config):
    assert call_type in [DISCOVERY, DEMO, PRICING, NEGOTIATION], (
        f"call_type must be one of {DISCOVERY}, {DEMO}, {PRICING}, {NEGOTIATION}"
    )
    if llm is None:
        llm = get_llm()

    print(f"Making {call_type} call for {clients}")
    analysed_call_data = await call_fns[call_type](
        ANALYSIS, clients, inputs, llm, **crew_config
    )
    return analysed_call_data


async def create_or_get_seller(inputs, llm: LLM = None, **crew_config):
    if llm is None:
        llm = get_llm()

    print(f"Creating seller for {inputs['seller']}")
    seller_data = await aget_seller_data(
        inputs, llm, **crew_config
    )
    return seller_data
