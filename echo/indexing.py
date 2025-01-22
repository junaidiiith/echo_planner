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
    SALES_PLAYBOOK = "sales_playbook"


def get_vector_index(
    index_name: str, 
    index_type: IndexType
):
    chroma_db_path = db_storage_path(index_name)
    db = chromadb.PersistentClient(path=str(chroma_db_path))
    chroma_collection = db.get_or_create_collection(f"{index_type.value}")
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
    docs = splitter.split_text(data)
    return [
        TextNode(
            text=doc,
            metadata=metadata
        )
        for doc in docs
    ]