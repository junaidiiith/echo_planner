from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import enum
from typing import Dict
from llama_index.core.node_parser import SentenceSplitter
from echo.utils import db_storage_path
import json
import os
import echo.utils as echo_utils


CHUNK_SIZE = 8192
CHUNK_OVERLAP = 128


class IndexType(enum.Enum):
    HISTORICAL = "historical"
    CURRENT_DEAL = "current_deal"
    BUYER_RESEARCH = "buyer_research"
    SELLER_RESEARCH = "seller_research"
    SALES_PLAYBOOK = "sales_playbook"


class CallType(enum.Enum):
    DISCOVERY = "discovery"
    DEMO = "demo"
    PRICING = "pricing"
    PROCUREMENT = "procurement"


def get_vector_index(index_name: str, index_type: IndexType):
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
            MetadataFilter(key=k.lower(), value=v.lower(), operator=filter_operator)
            for k, v in filters.items()
        ],
        condition=condition,
    )

    return index.as_query_engine(filters=filters).query(query)


def get_nodes_from_documents(data: str, metadata: Dict[str, str]):
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_text(data)
    return [TextNode(text=doc, metadata=metadata) for doc in docs]


metadata_keys = {
    IndexType.HISTORICAL: [
        "seller",
        "buyer",
        "call_type",
        "company_size",
        "industry",
        "description",
    ],
    IndexType.BUYER_RESEARCH: ["buyer"],
    IndexType.SELLER_RESEARCH: ["seller"],
    IndexType.SALES_PLAYBOOK: [],
}


def add_data(
    data: str, metadata: Dict[str, str], index_name: str, index_type: IndexType
):
    assert all([k in metadata for k in metadata_keys[index_type]]), (
        f"Missing metadata keys for {index_type}. \nRequired keys: {metadata_keys[index_type]}. \nProvided keys: {metadata.keys()}"
    )
    filtered_metadata = {
        k: v for k, v in metadata.items() if k in metadata_keys[index_type]
    }
    index = get_vector_index(index_name, index_type)
    nodes = get_nodes_from_documents(data, metadata=filtered_metadata)
    index.insert_nodes(nodes)


all_data = {
    client.split(".")[0]: json.load(open("runs/" + client))
    for client in os.listdir("runs")
}

print("\n".join(all_data["ICICI bank"].keys()))


seller = "Whatfix"

for client in all_data:
    add_data(
        data=echo_utils.json_to_markdown(all_data[client]["seller_research"])
        + "\n"
        + echo_utils.json_to_markdown(all_data[client]["seller_pricing"]),
        metadata={
            "seller": seller,
        },
        index_name=seller,
        index_type=IndexType.SELLER_RESEARCH,
    )

    add_data(
        data=echo_utils.json_to_markdown(all_data[client]["buyer_research"])
        + "\n"
        + echo_utils.json_to_markdown(all_data[client]["competitive_info"]),
        metadata={
            "buyer": client,
        },
        index_name=seller,
        index_type=IndexType.BUYER_RESEARCH,
    )

    add_data(
        data=echo_utils.json_to_markdown(
            all_data[client]["discovery_analysis_buyer_data"]
        )
        + "\n"
        + echo_utils.json_to_markdown(
            all_data[client]["discovery_analysis_seller_data"]
        ),
        metadata={
            "seller": seller,
            "buyer": client,
            "call_type": CallType.DISCOVERY.value,
            "company_size": all_data[client]["buyer_research"]["company_size"],
            "industry": all_data[client]["buyer_research"]["industry"],
            "description": all_data[client]["buyer_research"]["description"],
        },
        index_name=seller,
        index_type=IndexType.HISTORICAL,
    )

    add_data(
        data=echo_utils.json_to_markdown(all_data[client]["demo_analysis_buyer_data"])
        + "\n"
        + echo_utils.json_to_markdown(all_data[client]["demo_analysis_seller_data"]),
        metadata={
            "seller": seller,
            "buyer": client,
            "call_type": CallType.DEMO.value,
            "company_size": all_data[client]["buyer_research"]["company_size"],
            "industry": all_data[client]["buyer_research"]["industry"],
            "description": all_data[client]["buyer_research"]["description"],
        },
        index_name=seller,
        index_type=IndexType.HISTORICAL,
    )


index = get_vector_index(seller, IndexType.HISTORICAL)

filters_dict = {"seller": seller, "buyer": "ICICI bank", "call_type": "discovery"}

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key=k,
            value=v,
        )
        for k, v in filters_dict.items()
    ]
)

retriever = index.as_retriever(similarity_top_k=5)
print(len(retriever.retrieve("What are the top pain points of ICICI bank?")))
