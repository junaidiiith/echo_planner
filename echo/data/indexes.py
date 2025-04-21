from enum import Enum
from pydantic import BaseModel, Field
from llama_index.core.vector_stores import (
    FilterOperator,
)


class IndexType(Enum):
    HISTORICAL = "historical"
    ANALYSIS = "analysis"
    BUYER_RESEARCH = "buyer_research"
    SELLER_RESEARCH = "seller_research"
    BUYER_WEB_SEARCH = "buyer_web_search"
    SELLER_WEB_SEARCH = "seller_web_search"
    BUYER_ACCOUNT_PLAN = "buyer_account_plan"
    SELLER_ACCOUNT_PLAN = "seller_account_plan"
    WEBSITE_CONTENT = "website_content"
    CURRENT_CALL = "current_call"
    CALL_TRANSCRIPTS = "transcripts"
    SALES_PLAYBOOK = "sales_playbook"
    COMPETITORS = "competitors"


class IndexDataType(Enum):
    BUYER_WEBSITE_DATA = "buyer_website_data"
    SELLER_WEBSITE_DATA = "seller_website_data"
    COMPETITOR_WEBSITE_DATA = "competitor_website_data"
    BUYER_RESEARCH_DATA = "buyer_research_data"
    SELLER_RESEARCH_DATA = "seller_research_data"
    BUYER_DEMO_RESEARCH_DATA = "buyer_demo_research_data"


class MetadataColumn(BaseModel):
    key: str
    operator: FilterOperator
    mandatory: bool = Field(default=False)

    def __str__(self):
        return self.key

    def __repr__(self):
        return self.key


class EchoIndex(BaseModel):
    """Base class for all indexes."""

    create_table_query: str
    index_type: IndexType
    name: str
    metadata_columns: list[MetadataColumn]
    data_columns: list[str]


indices_map = {
    IndexType.CALL_TRANSCRIPTS: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.CALL_TRANSCRIPTS.value} (
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            buyer TEXT,
            call_id INTEGER,
            call_type TEXT,
            transcript TEXT,
            PRIMARY KEY (call_id, buyer)
        );""",
        "metadata_columns": [
            {
                "key": "buyer",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "call_type",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "call_id",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        "data_columns": [
            "transcript",
        ],
        "index_type": IndexType.CALL_TRANSCRIPTS,
    },
    IndexType.ANALYSIS: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.ANALYSIS.value} (
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            call_id INTEGER,
            buyer TEXT,
            call_type TEXT,
            stakeholder TEXT,
            company_size TEXT,
            industry TEXT,
            description TEXT,
            transcript TEXT,
            data TEXT,
            PRIMARY KEY (call_id, buyer, stakeholder)
        );""",
        "metadata_columns": [
            {
                "key": "buyer",
                "operator": FilterOperator.NE,
                "mandatory": True,
            },
            {
                "key": "call_type",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "call_id",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "stakeholder",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "company_size",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "industry",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "description",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        "data_columns": [
            "data",
            "transcript",
        ],
        "index_type": IndexType.ANALYSIS,
    },
    IndexType.BUYER_RESEARCH: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.BUYER_RESEARCH.value} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            buyer TEXT,
            data_type TEXT,
            industry TEXT,
            company_size TEXT,
            data TEXT
        );""",
        "metadata_columns": [
            {
                "key": "buyer",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "industry",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "company_size",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "data_type",  # Can be website data, or buyer research data
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        "data_columns": [
            "data",
        ],
        "index_type": IndexType.BUYER_RESEARCH,
    },
    IndexType.SELLER_RESEARCH: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.SELLER_RESEARCH.value} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            industry TEXT,
            data_type TEXT,
            data TEXT
        );""",
        "metadata_columns": [
            {
                "key": "industry",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "data_type",  # Can be website data, competitor data or seller research data
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        "data_columns": [
            "data",
        ],
        "index_type": IndexType.SELLER_RESEARCH,
    },
    IndexType.BUYER_ACCOUNT_PLAN: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.BUYER_ACCOUNT_PLAN.value} (
            buyer TEXT,
            query_type TEXT,
            query TEXT,
            message TEXT,
            sources TEXT,
            source_extracted_data TEXT DEFAULT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (buyer, query_type)
        );""",
        "metadata_columns": [
            {
                "key": "buyer",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "query_type",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "query",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        "data_columns": [
            "message",
            "sources",
            "source_extracted_data",
        ],
        "index_type": IndexType.BUYER_ACCOUNT_PLAN,
    },
    IndexType.SELLER_ACCOUNT_PLAN: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.SELLER_ACCOUNT_PLAN.value} (
            call_type TEXT,
            query_type TEXT,
            query TEXT,
            message TEXT,
            sources TEXT,
            source_extracted_data TEXT DEFAULT NULL,
            competitors TEXT DEFAULT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (query_type)
        );""",
        "metadata_columns": [
            {
                "key": "buyer",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "call_type",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "query_type",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            }
        ],
        "data_columns": ["message", "sources", "source_extracted_data", "competitors"],
        "index_type": IndexType.SELLER_ACCOUNT_PLAN,
    },
    IndexType.BUYER_WEB_SEARCH: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.BUYER_WEB_SEARCH.value} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            buyer TEXT,
            category TEXT,
            summary TEXT,
            data TEXT
        );""",
        "metadata_columns": [
            {
                "key": "buyer",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "category",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        "data_columns": [
            "data",
            "summary"
        ],
        "index_type": IndexType.BUYER_WEB_SEARCH,
    },
    IndexType.SELLER_WEB_SEARCH: {
        "create_table_query": f"""CREATE TABLE IF NOT EXISTS {IndexType.SELLER_WEB_SEARCH.value} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            category TEXT,
            url TEXT,
            data TEXT
        );""",
        "metadata_columns": [
            {
                "key": "category",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
            {
                "key": "url",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            }
        ],
        "data_columns": [
            "data",
        ],
        "index_type": IndexType.SELLER_WEB_SEARCH,
    }
}


def get_echo_index(seller: str, index_type: IndexType):
    """
    Create indices based on the provided index tables.
    """
    index = indices_map.get(index_type)
    if not index:
        raise ValueError(f"Index type {index_type} not found.")

    # Create the index table if it doesn't exist
    create_table_query = index["create_table_query"]
    metadata_columns = index["metadata_columns"]
    data_columns = index["data_columns"]

    # Create the EchoIndex object
    echo_index = EchoIndex(
        create_table_query=create_table_query,
        index_type=index_type,
        name=f"{seller}",
        metadata_columns=metadata_columns,
        data_columns=data_columns,
    )

    return echo_index
