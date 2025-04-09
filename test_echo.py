# from echo.data.indexes import IndexType
# from echo.indexing import get_vector_index
from echo.query_executor import run_perplexica_subquery, PerplexicaSubQuery, PerplexicaSourceExtraction

seller = 'https://whatfix.com/'
buyer = 'https://www.manpowergroup.com'


# vector_index = get_vector_index(seller, IndexType.BUYER_ACCOUNT_PLAN)

response = run_perplexica_subquery(
    PerplexicaSubQuery(
        query="",
        output_name="xyz"
    )
)