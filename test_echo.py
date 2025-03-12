# from echo import setup_db_tables
# from dotenv import load_dotenv
# import asyncio
# from echo.constants import (
#     DISCOVERY,
#     DEMO,
#     PRICING,
#     NEGOTIATION
# )
# from echo.runner import create_or_get_seller
# from echo.runner import make_call
# from echo.indexing import add_data, IndexType

# nest_asyncio.apply()

# load_dotenv()


# setup_db_tables()

# NUM_BUYERS = 10

# inputs = {
#     "seller": "https://foundit.in",
#     "num_buyers": NUM_BUYERS,
# }

# seller_data = asyncio.run(create_or_get_seller(inputs))



# buyer_inputs = {
#     **inputs,
#     **seller_data,
#     'call_id': 1,
#     'stakeholders': [
#         "Product Manager",
#         "CFO",
#         "VP of Product",
#         "VP of Sales"
#     ]
# }

# clients = [
#     "https://www.synechron.com/",
#     "https://www.eaton.com/",
#     "https://services.harman.com/",
#     "https://www.capgemini.com/",
#     "https://www.cognizant.com/us/en"
# ]

# discovery_calls_data = asyncio.run(make_call(DISCOVERY, clients[:5], buyer_inputs))


# data = open('t.txt').read()
# add_data(
#     data=data,
#     metadata=dict(
#         seller="foundit.in",
#         industry="SaaS"
#     ),
#     index_name="foundit.in",
#     index_type=IndexType.SELLER_RESEARCH
# )