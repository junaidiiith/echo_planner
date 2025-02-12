import os


save_dir = os.getenv("RUN_STORAGE_DIR", "runs")
buyer_db_name = "buyer_data"
seller_db_name = "seller_data"

CHUNK_SIZE = 8192
CHUNK_OVERLAP = 128
SIMILARITY_TOP_K = 3
MAX_RETRIES = 3
