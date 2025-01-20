import logging
import os
import shutil
import uuid

from typing import Any, Dict, List, Optional
from chromadb.api import ClientAPI
from abc import ABC, abstractmethod
from chromadb import EmbeddingFunction
from chromadb.api.types import validate_embedding_function
import os
from pathlib import Path

import appdirs
import contextlib
import io

import json
import sqlite3
from typing import Any, Dict, List, Optional
import hashlib
import time

from echo.constants import EMBED_STRING_HASH_KEY


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def get_current_time():
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    return timestamp

class Printer:
    def print(self, content: str, color: Optional[str] = None):
        if color == "purple":
            self._print_purple(content)
        elif color == "red":
            self._print_red(content)
        elif color == "bold_green":
            self._print_bold_green(content)
        elif color == "bold_purple":
            self._print_bold_purple(content)
        elif color == "bold_blue":
            self._print_bold_blue(content)
        elif color == "yellow":
            self._print_yellow(content)
        elif color == "bold_yellow":
            self._print_bold_yellow(content)
        else:
            print(content)

    def _print_bold_purple(self, content):
        print("\033[1m\033[95m {}\033[00m".format(content))

    def _print_bold_green(self, content):
        print("\033[1m\033[92m {}\033[00m".format(content))

    def _print_purple(self, content):
        print("\033[95m {}\033[00m".format(content))

    def _print_red(self, content):
        print("\033[91m {}\033[00m".format(content))

    def _print_bold_blue(self, content):
        print("\033[1m\033[94m {}\033[00m".format(content))

    def _print_yellow(self, content):
        print("\033[93m {}\033[00m".format(content))

    def _print_bold_yellow(self, content):
        print("\033[1m\033[93m {}\033[00m".format(content))



@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)



def db_storage_path():
    app_name = get_project_directory_name()
    app_author = "Echo"

    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    return data_dir


def get_project_directory_name():
    project_directory_name = os.environ.get("ECHO_STORAGE_DIR")

    if project_directory_name:
        return project_directory_name
    else:
        cwd = Path.cwd()
        project_directory_name = cwd.name
        return project_directory_name



class BaseRAGStorage(ABC):
    """
    Base class for RAG-based Storage implementations.
    """

    app: Any | None = None

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Optional[Any] = None,
    ):
        self.type = type
        self.allow_reset = allow_reset
        self.embedder_config = embedder_config
        
    @abstractmethod
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a value with metadata to the storage."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        """Search for entries in the storage."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the storage."""
        pass

    @abstractmethod
    def _generate_embedding(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Generate an embedding for the given text and metadata."""
        pass

    @abstractmethod
    def _initialize_app(self):
        """Initialize the vector db."""
        pass

    def setup_config(self, config: Dict[str, Any]):
        """Setup the config of the storage."""
        pass

    def initialize_client(self):
        """Initialize the client of the storage. This should setup the app and the db collection"""
        pass


class EmbeddingConfigurator:
    def __init__(self):
        self.embedding_functions = {
            "openai": self._configure_openai,
            "azure": self._configure_azure,
            "ollama": self._configure_ollama,
            "vertexai": self._configure_vertexai,
            "google": self._configure_google,
            "cohere": self._configure_cohere,
            "bedrock": self._configure_bedrock,
            "huggingface": self._configure_huggingface
        }

    def configure_embedder(
        self,
        embedder_config: Dict[str, Any] | None = None,
    ) -> EmbeddingFunction:
        """Configures and returns an embedding function based on the provided config."""
        if embedder_config is None:
            return self._create_default_embedding_function()

        provider = embedder_config.get("provider")
        config = embedder_config.get("config", {})
        model_name = config.get("model")

        if isinstance(provider, EmbeddingFunction):
            try:
                validate_embedding_function(provider)
                return provider
            except Exception as e:
                raise ValueError(f"Invalid custom embedding function: {str(e)}")

        if provider not in self.embedding_functions:
            raise Exception(
                f"Unsupported embedding provider: {provider}, supported providers: {list(self.embedding_functions.keys())}"
            )

        return self.embedding_functions[provider](config, model_name)

    @staticmethod
    def _create_default_embedding_function():
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model_name="text-embedding-3-small"
        )

    @staticmethod
    def _configure_openai(config, model_name):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            model_name=config.get("model") or model_name,
        )

    @staticmethod
    def _configure_azure(config, model_name):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            api_type=config.get("api_type", "azure"),
            api_version=config.get("api_version"),
            model_name=config.get("model") or model_name,
        )

    @staticmethod
    def _configure_ollama(config, model_name):
        from chromadb.utils.embedding_functions.ollama_embedding_function import (
            OllamaEmbeddingFunction,
        )

        return OllamaEmbeddingFunction(
            url=config.get("url", "http://localhost:11434/api/embeddings"),
            model_name=config.get("model") or model_name,
        )

    @staticmethod
    def _configure_vertexai(config, model_name):
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleVertexEmbeddingFunction,
        )

        return GoogleVertexEmbeddingFunction(
            model_name=config.get("model") or model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_google(config, model_name):
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleGenerativeAiEmbeddingFunction,
        )

        return GoogleGenerativeAiEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_cohere(config, model_name):
        from chromadb.utils.embedding_functions.cohere_embedding_function import (
            CohereEmbeddingFunction,
        )

        return CohereEmbeddingFunction(
            model_name=config.get("model") or model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_bedrock(config, model_name):
        from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
            AmazonBedrockEmbeddingFunction,
        )

        return AmazonBedrockEmbeddingFunction(
            session=config.get("session"),
            model_name=config.get("model") or model_name,
        )

    @staticmethod
    def _configure_huggingface(config, model_name):
        from chromadb.utils.embedding_functions.huggingface_embedding_function import (
            HuggingFaceEmbeddingFunction,
        )

        return HuggingFaceEmbeddingFunction(
            api_key=os.getenv("HF_ACCESS_TOKEN"),
            model_name=config.get("model") or model_name,
        )


class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    app: ClientAPI | None = None

    def __init__(self, type, allow_reset=True, embedder_config=None, path=None):
        super().__init__(type, allow_reset, embedder_config)
        self.type = type
        self.allow_reset = allow_reset
        self.path = path
        self._initialize_app()

    def _set_embedder_config(self):
        configurator = EmbeddingConfigurator()
        self.embedder_config = configurator.configure_embedder(self.embedder_config)

    def _initialize_app(self):
        import chromadb
        from chromadb.config import Settings

        self._set_embedder_config()
        chroma_client = chromadb.PersistentClient(
            path=self.path if self.path else f"{db_storage_path()}{os.sep}{self.type}",
            settings=Settings(allow_reset=self.allow_reset),
        )

        self.app = chroma_client

        try:
            self.collection = self.app.get_collection(
                name=self.type, embedding_function=self.embedder_config
            )
        except Exception:
            self.collection = self.app.create_collection(
                name=self.type, embedding_function=self.embedder_config
            )

    def check_text_embedded(self, text: str, metadata: Dict) -> bool:
        text_hash = sha256_hash(text)
        try:
            filter_criteria = {EMBED_STRING_HASH_KEY: text_hash}
            filter_criteria.update(metadata)
            results = self.collection.get(
                where=filter_criteria, include=["metadatas"]
            )
            return len(results["metadatas"]) > 0
        except Exception as e:
            logging.error(f"Error checking if text is embedded: {str(e)}")
            return False
    
    
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()
        
        try:
            self._generate_embedding(value, metadata)
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging():
                response = self.collection.query(query_texts=query, n_results=limit)

            results = []
            for i in range(len(response["ids"][0])):
                result = {
                    "id": response["ids"][0][i],
                    "metadata": response["metadatas"][0][i],
                    "context": response["documents"][0][i],
                    "score": response["distances"][0][i],
                }
                if result["score"] >= score_threshold:
                    results.append(result)

            if filter is not None:
                results = [
                    result
                    for result in results
                    if all(
                        [
                            result["metadata"].get(key) == value
                            for key, value in filter.items()
                        ]
                    )
                ]
            
            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> None:  # type: ignore
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        if self.check_text_embedded(text, metadata):
            print(f"Text already embedded in {self.type}")
            logging.info(f"Text already embedded in {self.type}")
            return
        metadata = metadata or {}
        metadata[EMBED_STRING_HASH_KEY] = sha256_hash(text)
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())],
        )

    def reset(self) -> None:
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
            if self.app:
                self.app.reset()
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                # Ignore this specific error
                pass
            else:
                raise Exception(
                    f"An error occurred while resetting the {self.type} memory: {e}"
                )

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )



class LTMSQLiteStorage:
    """
    An updated SQLite storage class for LTM data storage.
    """

    def __init__(
        self, 
        db_type: str = f"long_term_memory_storage.db",
        reset: bool = False,
    ) -> None:
        self.db_path = f"{db_storage_path()}/{db_type}.db"
        self._printer: Printer = Printer()
        self._initialize_db(reset)

    def _initialize_db(self, reset=False):
        """
        Initializes the SQLite database and creates LTM table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if reset:
                    print("Deleting existing table")
                    conn.execute("DROP TABLE IF EXISTS seller_data")
                    conn.commit()
                
                conn.execute('PRAGMA foreign_keys = ON;')
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS seller_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        seller TEXT,
                        buyer TEXT,
                        call_type TEXT,
                        data JSON,
                        datetime TEXT
                    )
                """
                )

                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                color="red",
            )


    def check_if_exists(self, seller, buyer, call_type, metadata: Dict) -> int:
        """
        Checks if a row exists in the LTM table
        return:
            0: if the row does not exist
            1: if the row exists but metadata does not match
            2: if the row exists and metadata matches
        """
        query = "SELECT * FROM seller_data WHERE seller = ? AND buyer = ? AND call_type = ?"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (seller, buyer, call_type))
            row = cursor.fetchone()
            if row:
                try:
                    row_metadata: Dict = json.loads(row[4])
                except json.JSONDecodeError:
                    import ast
                    try:
                        row_metadata = json.loads(json.dumps(ast.literal_eval(row[4])))
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(f"Could not decode the value: {row[4]} with error: {e}")
                    
                if all(
                    [
                        key in row_metadata
                        for key in metadata
                    ]
                ):
                    print(f"Row already exists!")
                    return True
        return False
                
        
    def save(
        self,
        seller: str,
        buyer: str,
        call_type: str,
        data: Dict[str, Any]
    ) -> None:
        datetime = get_current_time()
        """Saves data to the LTM table with error handling."""
        record_exists =  self.check_if_exists(seller, buyer, call_type, data)
        assert record_exists in [0, 1, 2], f"Invalid record_exists value: {record_exists}"
        if record_exists == 2:
            logging.info(f"Record already exists for seller: {seller}, buyer: {buyer}, call_type: {call_type}")
            return 
        else:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    if record_exists == 1:
                        query = "UPDATE seller_data SET data = ?, datetime = ? WHERE seller = ? AND buyer = ? AND call_type = ?"
                        cursor.execute(query, (json.dumps(data), datetime, seller, buyer, call_type))
                    
                    else:
                        query = "INSERT INTO seller_data (seller, buyer, call_type, data, datetime) VALUES (?, ?, ?, ?, ?)"
                        cursor.execute(query, (seller, buyer, call_type, json.dumps(data), datetime))
                    conn.commit()
            except sqlite3.Error as e:
                self._printer.print(
                    content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                    color="red",
                )
        return None

    def load(
        self, 
        buyer: str,
        seller: str, 
        call_type: str, 
        data: Dict=None, 
        latest_n: int
    =-1) -> Optional[List[Dict[str, Any]]]:
        """Queries the LTM table by task description with error handling."""
        latest_n = latest_n if latest_n > 0 else 3
        query = 'SELECT seller, buyer, call_type, data FROM seller_data WHERE seller = ? AND buyer = ? AND call_type = ? '
        params = [seller, buyer, call_type]
        if data:
            for key, value in data.items():
                query += f' AND json_extract(data, \'$.{key}\') = ?'
                params.append(value)
        
        query += f' ORDER BY datetime DESC LIMIT {latest_n}'
        
        try:
            rows = self.run_query(query, params)
            if rows:
                return [
                    {
                        "seller": row[0],
                        "buyer": row[1],
                        "call_type": row[2],
                        "data": json.loads(row[3]),
                    }
                    for row in rows
                ]
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        return None
    
    
    def run_query(
        self, 
        query: str, 
        params: List[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Queries the LTM table by task description with error handling."""
        if not params:
            params = []
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            if rows:
                return [r for r in rows]
        
        return []
        
    def reset(
        self,
    ) -> None:
        """Resets the LTM table with error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()

        except sqlite3.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                color="red",
            )
        return None