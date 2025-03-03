import os


save_dir = os.getenv("RUN_STORAGE_DIR", "runs")
buyer_db_name = "buyer_data"
seller_db_name = "seller_data"
db_name = "echo.db"

CHUNK_SIZE = 8192
CHUNK_OVERLAP = 128
SIMILARITY_TOP_K = 3
MAX_RETRIES = 3
CALL_HISTORY_LIMIT = 4

debug = os.getenv("DEBUG", False)