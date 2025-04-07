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
from typing import Dict
from echo.data.indexes import IndexType, get_echo_index
from echo.llama_llm_embed_utils import get_embed_model
from echo.utils import db_storage_path

from echo.utils import (
    serialize_dict,
)


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


def get_documents(index: VectorStoreIndex):
    return index.storage_context.vector_store._collection.get()["documents"]


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
        

def check_metadata_exists(index_name: str, index_type: IndexType, metadata: Dict):
    echo_index = get_echo_index(index_name, index_type)
    metadata_columns = echo_index.metadata_columns
    assert all([mc.key in metadata for mc in metadata_columns if mc.mandatory]), (
        f"Missing metadata keys for {index_type}."
        f"\nRequired keys: {metadata_columns}."
        f"\nProvided keys: {metadata.keys()}"
    )
    filtered_metadata = {
        mc.key: metadata[mc.key] 
        for mc in metadata_columns 
        if mc.key in metadata
    }

    index = get_vector_index(index_name, index_type.value)
    metadatas = get_metadatas(index)
    return any(all(md[k] == filtered_metadata[k] for k in filtered_metadata) for md in metadatas)


def get_data_from_index(index_name: str, index_type: IndexType, metadata: Dict, fetch_all: bool = False):
    assert check_metadata_exists(index_name, index_type, metadata)
    echo_index = get_echo_index(index_name, index_type)
    metadata_columns = echo_index.metadata_columns
    data_columns = echo_index.data_columns
    filtered_metadata = {
        mc.key: metadata[mc.key] 
        for mc in metadata_columns 
        if mc.key in metadata
    }
    index = get_vector_index(index_name, index_type.value)
    metadatas = get_metadatas(index)
    
    retrieved_metadatas = [
        md for md in metadatas
        if all(md[k] == filtered_metadata[k] for k in filtered_metadata)
    ]
    if not fetch_all:
        return {
            k: retrieved_metadatas[0][k]
            for k in data_columns
            if k in retrieved_metadatas[0]
        }

    all_columns = [mc.key for mc in metadata_columns] + data_columns
    retrieved_metadatas = [{c: md[c] for c in all_columns} for md in retrieved_metadatas]
    retrieved_metadatas = list({str(rmd): rmd for rmd in retrieved_metadatas}.values())
    return retrieved_metadatas
    

def check_node_exists(data: str, metadata: Dict[str, str], index: VectorStoreIndex):
    metadatas = get_metadatas(index)
    docs = get_documents(index)
    if not any(all(md[k] == metadata[k] for k in metadata) for md in metadatas):
        return False

    if not any(doc in data for doc in docs):
        return False

    print("Node already exists in index")
    return True


def add_data(
    data: str, metadata: Dict[str, str], index_name: str, index_type: IndexType
):
    echo_index = get_echo_index(index_name, index_type)
    metadata = serialize_dict(metadata)
    metadata_columns = echo_index.metadata_columns
    data_columns = echo_index.data_columns
    assert all([mc.key in metadata for mc in metadata_columns if mc.mandatory]), (
        f"Missing metadata keys for {index_type}. \nRequired keys: {metadata_columns}. \nProvided keys: {metadata.keys()}"
    )
    
    assert all([dc in metadata for dc in data_columns]), (
        f"Missing metadata keys for {index_type}. \nRequired keys: {data_columns}. \nProvided keys: {metadata.keys()}"
    )
    filtered_metadata = {k: v for k, v in metadata.items() if k in metadata_columns}

    index = get_vector_index(index_name, index_type.value)

    node = Document(text=data, metadata=filtered_metadata)
    if not check_node_exists(data, filtered_metadata, index):
        index.insert(node)
        print(f"Added node to index: {index_name}")
