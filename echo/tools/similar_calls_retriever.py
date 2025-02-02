from echo.indexing import IndexType, get_vector_index
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from echo.settings import SIMILARITY_TOP_K


class HistoricalIndexArguments(BaseModel):
    """Input schema for MyCustomTool."""

    query: str = Field(..., description="Query to retrieve from the index.")
    call_type: str = Field(..., description="Type of call")
    buyer: str = Field(..., description="Name of the buyer.")
    seller: str = Field(..., description="Name of the seller.")


class HistoricalCallIndex(BaseTool):
    name: str = "Historical Calls Vector Index"
    description: str = (
        "This tool retrieves relevant the historical calls data from the vector index for a given deal."
        "This tool uses buyer, seller and type to retrieve the relevant calls."
        "And then uses the query to retrieve the most relevant calls."
        "This tool can be used to see what has worked in the past for similar deals."
    )
    args_schema: Type[BaseModel] = HistoricalIndexArguments

    def _run(self, query: str, call_type: str, buyer: str, seller: str) -> str:
        index = get_vector_index(seller, IndexType.HISTORICAL)

        filters_dict = {"seller": seller, "buyer": buyer, "call_type": call_type}

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key=k,
                    value=v,
                )
                for k, v in filters_dict.items()
            ]
        )

        retrieved_nodes = index.as_retriever(
            filters=filters, similarity_top_k=SIMILARITY_TOP_K
        ).retrieve(query)
        return "\n".join(
            [
                f"Historical Call: {i}\n{n.get_content()}"
                for i, n in enumerate(retrieved_nodes)
            ]
        )


class CurrentCallIndexArguments(BaseModel):
    """Input schema for MyCustomTool."""

    call_type: str = Field(..., description="Type of call")
    buyer: str = Field(..., description="Name of the buyer.")
    seller: str = Field(..., description="Name of the seller.")


class CurrentCallIndex(BaseTool):
    name: str = "Historical Calls Vector Index"
    description: str = (
        "This tool retrieves relevant documents related to the current call."
        "This tool uses buyer, seller and type to retrieve the relevant calls to retrieve the relevant call records."
    )
    args_schema: Type[BaseModel] = CurrentCallIndexArguments

    def _run(self, call_type: str, buyer: str, seller: str) -> str:
        index = get_vector_index(seller, IndexType.HISTORICAL)

        filters_dict = {"seller": seller, "buyer": buyer, "call_type": call_type}

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key=k,
                    value=v,
                )
                for k, v in filters_dict.items()
            ]
        )

        retrieved_nodes = index.as_retriever(
            filters=filters, similarity_top_k=SIMILARITY_TOP_K
        ).retrieve("")
        return "\n".join(
            [
                f"Current {call_type} Call Data: {i}\n{n.get_content()}"
                for i, n in enumerate(retrieved_nodes)
            ]
        )
