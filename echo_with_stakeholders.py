from echo import setup_db_tables
from dotenv import load_dotenv
import asyncio
from echo.constants import (
    DISCOVERY,
    DEMO,
    PRICING,
    NEGOTIATION
)
from echo.runner import (
    create_or_get_seller, 
    make_call
)


load_dotenv()
setup_db_tables()


NUM_BUYERS = 10

inputs = {
    "seller": "https://foundit.in",
    "num_buyers": NUM_BUYERS,
}

seller_data = asyncio.run(create_or_get_seller(inputs))

buyer_inputs = {
    **inputs,
    **seller_data,
    'call_id': 1,
    'stakeholders': [
        "Product Manager",
        "CFO",
        "VP of Product",
        "VP of Sales"
    ]
}

clients = [
    # "https://www.synechron.com/",
    "https://services.harman.com/",
    "https://www.capgemini.com/",
    "https://www.cognizant.com/us/en"
]


discovery_calls_data = asyncio.run(make_call(DISCOVERY, clients, buyer_inputs))
buyer_inputs['call_id'] = 2
demo_calls_data = asyncio.run(make_call(DEMO, clients, buyer_inputs))

buyer_inputs['call_id'] = 3
pricing_calls_data = asyncio.run(make_call(PRICING, clients, buyer_inputs))

buyer_inputs['call_id'] = 4
negotiations_calls_data = asyncio.run(make_call(NEGOTIATION, clients, buyer_inputs))