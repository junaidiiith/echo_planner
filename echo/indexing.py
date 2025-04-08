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
from echo.data.indexes import IndexType, get_echo_index, indices_map
from echo.llama_llm_embed_utils import get_embed_model
from echo.utils import db_storage_path

from echo.utils import (
    serialize_dict,
    url_to_sql_name
)
import echo.sqldb as sqldb


def get_vector_index(index_name: str, index_type: IndexType):
    index_type = (
        IndexType.ANALYSIS
        if index_type == IndexType.CURRENT_CALL
        else index_type
    )
    index_name = url_to_sql_name(index_name)
    chroma_db_path = db_storage_path(index_name)
    db = chromadb.PersistentClient(path=str(chroma_db_path))
    chroma_collection = db.get_or_create_collection(f"{index_type.value}")
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


def create_tables(index_name: str):
    for index_type, index_data in indices_map.items():
        query = index_data["create_table_query"]
        sqldb.create_table(path=index_name, query=query)

    print(f"Created tables for {index_name}")


def check_metadata_exists_in_index(
    index_name: str,
    index_type: IndexType,
    metadata: Dict[str, str],
) -> bool:
    assert check_metadata_exists_in_db(
        index_name=index_name,
        index_type=index_type,
        metadata=metadata,
    )
    echo_index = get_echo_index(index_name, index_type)
    metadata_columns = echo_index.metadata_columns
    filtered_metadata = {
        mc.key: metadata[mc.key] 
        for mc in metadata_columns 
        if mc.key in metadata
    }
    
    index = get_vector_index(index_name, index_type)
    metadatas = get_metadatas(index)
    print("Searching for metadata in index")
    print("Filtered metadata:", filtered_metadata)
    return any(all(md[k] == filtered_metadata[k] for k in filtered_metadata if k in md) for md in metadatas)
        

def check_metadata_exists_in_db(index_name: str, index_type: IndexType, metadata: Dict):
    echo_index = get_echo_index(index_name, index_type)
    metadata_columns = echo_index.metadata_columns
    assert all([mc.key in metadata for mc in metadata_columns if mc.mandatory]), (
        f"Missing metadata keys for {index_type}."
        f"\nRequired keys: {metadata_columns}."
        f"\nProvided keys: {metadata.keys()}"
    )
    filtered_metadata = {
        mc.key: metadata[mc.key] for mc in metadata_columns if mc.key in metadata
    }

    return sqldb.check_record_exists(
        path=index_name, table_name=index_type.value, condition_dict=filtered_metadata
    )


def get_data_from_db(
    index_name: str, index_type: IndexType, metadata: Dict, fetch_all: bool = False
):
    assert check_metadata_exists_in_db(index_name, index_type, metadata)
    echo_index = get_echo_index(index_name, index_type)
    metadata_columns = echo_index.metadata_columns
    data_columns = echo_index.data_columns
    filtered_metadata = {
        mc.key: metadata[mc.key] for mc in metadata_columns if mc.key in metadata
    }
    all_columns = [mc.key for mc in metadata_columns] + data_columns
    records = sqldb.get_records(
        path=index_name,
        table_name=index_type.value,
        condition_dict=filtered_metadata,
    )
    if not fetch_all:
        return {k: records[0][k] for k in all_columns if k in records[0]}
    return [{c: record[c] for c in all_columns} for record in records]


def add_data_to_db(
    index_name: str,
    index_type: IndexType,
    metadata: Dict[str, str],
):
    metadata = serialize_dict(metadata)
    echo_index = get_echo_index(index_name, index_type)
    metadata_columns = echo_index.metadata_columns
    data_columns = echo_index.data_columns
    assert all([mc.key in metadata for mc in metadata_columns if mc.mandatory]), (
        f"Missing metadata keys for {index_type}."
        f"\nRequired keys: {metadata_columns}. "
        f"\nProvided keys: {metadata.keys()}"
    )

    assert all([dc in metadata for dc in data_columns]), (
        f"Missing metadata keys for {index_type}."
        f"\nRequired keys: {data_columns}."
        f"\nProvided keys: {metadata.keys()}"
    )

    filtered_metadata = {
        **{mc.key: metadata[mc.key] for mc in metadata_columns if mc.key in metadata},
        **{dc: metadata[dc] for dc in data_columns if dc in metadata},
    }

    sqldb.insert_record(
        path=index_name,
        table_name=index_type.value,
        attributes=filtered_metadata,
    )
    print(f"Added data to db: {index_name} with metadata: {filtered_metadata}")


def check_index_node_exists(
    data: str, metadata: Dict[str, str], index: VectorStoreIndex
):
    metadatas = get_metadatas(index)
    docs = get_documents(index)
    if not any(all(md[k] == metadata[k] for k in metadata if k in md) for md in metadatas):
        return False

    if not any(doc in data for doc in docs):
        return False

    print("Node already exists in index")
    return True


def add_data(
    data: str, 
    metadata: Dict[str, str], 
    index_name: str, 
    index_type: IndexType
):
    # print(f"Adding data to index: {index_name} with metadata: {metadata}")
    # print("Data:", data)
    echo_index = get_echo_index(index_name, index_type)
    metadata = serialize_dict(metadata)
    metadata_columns = echo_index.metadata_columns
    data_columns = echo_index.data_columns
    assert all([mc.key in metadata for mc in metadata_columns if mc.mandatory]), (
        f"Missing metadata keys for {index_type}."
        f"\nRequired keys: {metadata_columns}. "
        f"\nProvided keys: {metadata.keys()}"
    )

    assert all([dc in metadata for dc in data_columns]), (
        f"Missing metadata keys for {index_type}."
        f"\nRequired keys: {data_columns}."
        f"\nProvided keys: {metadata.keys()}"
    )

    filtered_metadata = {
        mc.key: metadata[mc.key] for mc in metadata_columns if mc.key in metadata
    }

    complete_metadata = {
        **{mc.key: metadata[mc.key] for mc in metadata_columns if mc.key in metadata},
        **{dc: metadata[dc] for dc in data_columns if dc in metadata},
    }

    add_data_to_db(
        index_name=index_name,
        index_type=index_type,
        metadata=complete_metadata,
    )

    # with open('t.json') as f:
    #     import json
    #     data = json.load(f)[0]['content']
        
    if not check_metadata_exists_in_index(
        index_name=index_name, 
        index_type=index_type, 
        metadata=complete_metadata
    ):
        index = get_vector_index(index_name, index_type)
        
        node = Document(text=data, metadata=filtered_metadata)
        if not check_index_node_exists(data, filtered_metadata, index):
            index.insert(node)
            print(f"Added node to index: {index_name} with metadata: {filtered_metadata}")
