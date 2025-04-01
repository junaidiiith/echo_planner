from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import enum
from typing import Dict
from llama_index.core.node_parser import SentenceSplitter
from echo.llama_llm_embed_utils import get_embed_model
from echo.utils import db_storage_path
from echo.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from echo.sqldb import get_table_columns_for_embeddings
from echo.constants import (
    SELLER_RESEARCH_KEYS,
    BUYER_RESEARCH_KEYS,
    ANALYSIS_KEYS,
    SIMULATION_KEYS,
)

from echo.utils import (
    serialize_dict,
)

from dotenv import load_dotenv

load_dotenv()


class IndexType(enum.Enum):
    HISTORICAL = "historical"
    ANALYSIS = "analysis"
    BUYER_RESEARCH = "buyer_research"
    SELLER_RESEARCH = "seller_research"
    BUYER_FOUNDATIONAL_PLAN = "buyer_foundational_plan"
    SELLER_FOUNDATIONAL_PLAN = "seller_foundational_plan"
    WEBSITE_CONTENT = "website_content"
    CURRENT_CALL = "current_call"
    CALL_TRANSCRIPTS = "transcripts"
    SALES_PLAYBOOK = "sales_playbook"


tables = {
    IndexType.CALL_TRANSCRIPTS: f"""CREATE TABLE IF NOT EXISTS {IndexType.CALL_TRANSCRIPTS.value} (
        call_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        buyer TEXT,
        seller TEXT,
        call_type TEXT,
        transcript TEXT,
        PRIMARY KEY (call_id, buyer, seller)
    );""",
    IndexType.ANALYSIS: f"""CREATE TABLE IF NOT EXISTS {IndexType.ANALYSIS.value} (
        call_id INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        buyer TEXT,
        seller TEXT,
        call_type TEXT,
        stakeholder TEXT,
        company_size TEXT,
        industry TEXT,
        description TEXT,
        transcript TEXT,
        data TEXT,
        PRIMARY KEY (call_id, buyer, seller, stakeholder)
    );""",
    IndexType.BUYER_RESEARCH: f"""CREATE TABLE IF NOT EXISTS {IndexType.BUYER_RESEARCH.value} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        seller TEXT,
        buyer TEXT,
        industry TEXT,
        company_size TEXT,
        raw BOOLEAN DEFAULT FALSE,
        data TEXT
    );""",
    IndexType.SELLER_RESEARCH: f"""CREATE TABLE IF NOT EXISTS {IndexType.SELLER_RESEARCH.value} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        seller TEXT,
        industry TEXT,
        raw BOOLEAN DEFAULT FALSE,
        data TEXT
    );""",
    IndexType.WEBSITE_CONTENT: f"""CREATE TABLE IF NOT EXISTS {IndexType.WEBSITE_CONTENT.value} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        type TEXT,
        name TEXT,
        data TEXT
    );""",
    IndexType.BUYER_FOUNDATIONAL_PLAN: f'''CREATE TABLE IF NOT EXISTS {IndexType.BUYER_FOUNDATIONAL_PLAN.value} (
        seller TEXT,
        buyer TEXT,
        call_type TEXT,
        query_type TEXT,
        query TEXT,
        message TEXT,
        sources TEXT,
        source_extracted_data TEXT DEFAULT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (buyer, query_type)
    );'''  
}

table_to_metadata_columns_map = {
    IndexType.CALL_TRANSCRIPTS: [
        "buyer",
        "seller",
        "call_type",
    ],
    IndexType.ANALYSIS: [
        "buyer",
        "seller",
        "call_type",
        "stakeholder",
        "company_size",
        "industry",
        "description"
    ],
    IndexType.BUYER_RESEARCH: [
        "seller",
        "buyer",
        "industry",
        "company_size",
    ],
    IndexType.SELLER_RESEARCH: [
        "seller",
        "industry",
    ],
    IndexType.WEBSITE_CONTENT: [
        "type",
        "name",
    ],
    IndexType.BUYER_FOUNDATIONAL_PLAN: [
        "seller",
        "buyer",
        "query_type",
        "call_type",
    ]
}


def get_query_index_keys(index_type: str):
    index_keys = {
        IndexType.ANALYSIS.value: [
            {
                "key": "seller",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
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
                "key": "stakeholder",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        IndexType.CURRENT_CALL.value: [
            {
                "key": "seller",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
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
                "key": "stakeholder",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        IndexType.BUYER_RESEARCH.value: [
            {
                "key": "buyer",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "seller",
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
        ],
        IndexType.SELLER_RESEARCH.value: [
            {
                "key": "seller",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            },
            {
                "key": "industry",
                "operator": FilterOperator.EQ,
                "mandatory": False,
            },
        ],
        IndexType.SALES_PLAYBOOK.value: [],
        IndexType.BUYER_FOUNDATIONAL_PLAN.value: [
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
            },
            {
                "key": "seller",
                "operator": FilterOperator.EQ,
                "mandatory": True,
            }
        ]
    }[index_type]

    index_db_keys = get_table_columns_for_embeddings(
        index_type
        if index_type != IndexType.CURRENT_CALL.value
        else IndexType.ANALYSIS.value
    )
    assert all([k["key"] in index_db_keys for k in index_keys]), (
        f"Missing metadata keys for {index_type}.",
        f"\nRequired keys: {index_db_keys}. ",
        f"\nProvided keys: {[k['key'] for k in index_keys]}",
    )

    return index_keys


def get_index_keys(index_type: IndexType):
    if index_type == IndexType.SELLER_RESEARCH:
        return SELLER_RESEARCH_KEYS
    if index_type == IndexType.BUYER_RESEARCH:
        return BUYER_RESEARCH_KEYS
    if index_type == IndexType.ANALYSIS:
        return ANALYSIS_KEYS
    if index_type == IndexType.CALL_TRANSCRIPTS:
        return SIMULATION_KEYS
    return []


def get_filtered_data(data: Dict[str, str], index_type: IndexType):
    keys = get_index_keys(index_type)
    return {k: v for k, v in data.items() if k in keys}


def get_vector_index(index_name: str, index_type: str):
    index_type = (
        IndexType.ANALYSIS.value
        if index_type == IndexType.CURRENT_CALL.value
        else index_type
    )
    index_name = index_name.replace("https://", "").replace("http://", "")
    chroma_db_path = db_storage_path(index_name)
    db = chromadb.PersistentClient(path=str(chroma_db_path))
    chroma_collection = db.get_or_create_collection(f"{index_type}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=get_embed_model()
    )

    return index


def get_metadatas(index: VectorStoreIndex):
    return index.storage_context.vector_store._collection.get()["metadatas"]


def get_response(
    query: str,
    index: VectorStoreIndex,
    filters: Dict[str, str] = None,
    filter_operator: FilterOperator = FilterOperator.EQ,
    condition: FilterCondition = FilterCondition.AND,
):
    filters = filters or {}
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key=k.lower(), value=v.lower(), operator=filter_operator)
            for k, v in filters.items()
        ],
        condition=condition,
    )

    return index.as_query_engine(filters=filters).query(query)


def get_nodes_from_documents(data: str, metadata: Dict[str, str]):
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # print("Splitting text into documents", data)
    docs = splitter.split_text(data)

    return [Document(text=doc, metadata=metadata) for doc in docs]


def check_node_exists(data: str, metadata: Dict[str, str], index: VectorStoreIndex):
    metadatas = index.storage_context.vector_store._collection.get()["metadatas"]
    docs = index.storage_context.vector_store._collection.get()["documents"]
    if not any(all(md[k] == metadata[k] for k in metadata) for md in metadatas):
        return False

    if not any(doc in data for doc in docs):
        return False

    print("Node already exists in index")
    return True


def add_data(
    data: str, metadata: Dict[str, str], index_name: str, index_type: IndexType
):
    metadata = serialize_dict(metadata)
    metadata_columns = get_table_columns_for_embeddings(index_type.value, table_to_metadata_columns_map[index_type])
    assert all([k in metadata for k in metadata_columns]), (
        f"Missing metadata keys for {index_type}. \nRequired keys: {metadata_columns}. \nProvided keys: {metadata.keys()}"
    )
    filtered_metadata = {k: v for k, v in metadata.items() if k in metadata_columns}

    index = get_vector_index(index_name, index_type.value)

    node = Document(text=data, metadata=filtered_metadata)
    if not check_node_exists(data, filtered_metadata, index):
        index.insert(node)
        print(f"Added node to index: {index_name}")


def setup_db_tables():
    from echo.sqldb import create_table

    for table in tables.values():
        # print("Creating Table: ", table)
        create_table(table)
    print("Tables Created Successfully")
