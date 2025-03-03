BUYER_RESEARCH = "buyer_research"
SELLER_RESEARCH = "seller_research"
RESEARCH = "research"
SIMULATION = "simulation"
EXTRACTION = "extraction"
ANALYSIS = "analysis"
SAVE_DATA = "save_data"
EMBED_DATA = "embed_data"

SECTION_EXTRACTION = "extraction"
SECTION_MERGER = "merger"

DISCOVERY = "discovery"
DEMO = "demo"
PRICING = "pricing"
NEGOTIATION = "negotiation"

BUYER = "buyer"
SELLER = "seller"

ALLOWED_KEYS = [
    "seller_research",
    "seller_pricing",
    "seller_clients",
    "buyer_research",
    "competitive_info",
    "anticipated_qopcs",
    "discovery_transcript",
    "discovery_analysis_buyer_data",
    "discovery_analysis_seller_data",
    "demo_features",
    "demo_transcript",
    "demo_analysis_buyer_data",
    "demo_analysis_seller_data",
    "pricing_transcript",
    "pricing_analysis_buyer_data",
    "pricing_analysis_seller_data",
    "negotiation_transcript",
    "negotiation_analysis_buyer_data",
    "negotiation_analysis_seller_data",
]

SELLER_RESEARCH_KEYS = [
    "seller_research",
    "seller_pricing",
    "seller_clients",
]

BUYER_RESEARCH_KEYS = [
    "buyer_research",
    "competitive_info",
    "anticipated_qopcs",    
]

ANALYSIS_KEYS = [
    "discovery_analysis_buyer_data",
    "discovery_analysis_seller_data",
    "demo_analysis_buyer_data",
    "demo_analysis_seller_data",
    "pricing_analysis_buyer_data",
    "pricing_analysis_seller_data",
    "negotiation_analysis_buyer_data",
    "negotiation_analysis_seller_data",
]

SIMULATION_KEYS = [
    "discovery_transcript",
    "demo_transcript",
    "pricing_transcript",
    "negotiation_transcript",    
]

EMBED_STRING_HASH_KEY = "embed_string_hash"


OPENAI = "openai"
FIREWORKS = "fireworks"

LLM_TYPE = "LLM_TYPE"
EMBED_MODEL_TYPE = "EMBED_MODEL_TYPE"

OPENAI_MODEL = "OPENAI_MODEL"
FIREWORKS_LLM = "FIREWORKS_LLM"

OPENAI_API_KEY = "OPENAI_API_KEY"
FIREWORKS_API_KEY = "FIREWORKS_API_KEY"
EMBED_BATCH_SIZE = "EMBED_BATCH_SIZE"
