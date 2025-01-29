GENERATION_QUERY_RESPONSE_FORMAT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in markdown format in a well-formatted, easily readable way.\n"
    "Query: {query_str}\n"
    "Answer: "
)