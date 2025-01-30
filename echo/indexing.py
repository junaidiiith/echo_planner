from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import enum
from typing import Dict
from llama_index.core.node_parser import SentenceSplitter
from echo.utils import db_storage_path
from echo.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,    
)


class IndexType(enum.Enum):
    HISTORICAL = "historical"
    BUYER_RESEARCH = "buyer_research"
    SELLER_RESEARCH = "seller_research"
    CURRENT_CALL = "current_call"
    TRANSCRIPT = "transcripts"
    SALES_PLAYBOOK = "sales_playbook"


metadata_keys = {
    IndexType.HISTORICAL: [
        "seller", 
        "buyer", 
        "call_type", 
        "company_size", 
        "industry", 
        "description"
    ],
    IndexType.BUYER_RESEARCH: ["buyer"],
    IndexType.SELLER_RESEARCH: ["seller"],
    IndexType.SALES_PLAYBOOK: [],
    IndexType.TRANSCRIPT: ["seller", "buyer", "call_type"]
}


def get_vector_index(
    index_name: str, 
    index_type: str
):
    index_type = IndexType.HISTORICAL.value if index_type == IndexType.CURRENT_CALL.value else index_type
    chroma_db_path = db_storage_path(index_name)
    db = chromadb.PersistentClient(path=str(chroma_db_path))
    chroma_collection = db.get_or_create_collection(f"{index_type}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index


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
            MetadataFilter(
                key=k.lower(), 
                value=v.lower(),
                operator=filter_operator
            ) for k, v in filters.items()
        ],
        condition=condition,
    )

    return index.as_query_engine(filters=filters).query(query)


def get_nodes_from_documents(data: str, metadata: Dict[str, str]):
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    print("Splitting text into documents", data)
    docs = splitter.split_text(data)
    return [
        TextNode(
            text=doc,
            metadata=metadata
        )
        for doc in docs
    ]

def check_node_exists(
    data: str, 
    metadata: Dict[str, str], 
    index: VectorStoreIndex
):
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key=k, 
                value=v,
            ) for k, v in metadata.items()
        ],
    )
    retriever = index.as_retriever(filters=filters, similarity_top_k=3)
    if any([n.text == data for n in retriever.retrieve("")]):
        print(f"Node already exists in index: {metadata}")
        return True
    return False


def add_data(
    data: str, 
    metadata: Dict[str, str], 
    index_name: str, 
    index_type: IndexType
):
    assert all([k in metadata for k in metadata_keys[index_type]]), f"Missing metadata keys for {index_type}. \nRequired keys: {metadata_keys[index_type]}. \nProvided keys: {metadata.keys()}"
    filtered_metadata = {
        k: v for k, v in metadata.items() 
        if k in metadata_keys[index_type]
    }
    # filtered_metadata['data_json'] = data
    index = get_vector_index(index_name, index_type.value)
    nodes = get_nodes_from_documents(data, metadata=filtered_metadata)
    filtered_nodes = [
        n for n in nodes 
        if not check_node_exists(n.text, filtered_metadata, index)
    ]
    index.insert_nodes(filtered_nodes)