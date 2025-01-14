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
    def _sanitize_role(self, role: str) -> str:
        """Sanitizes agent roles to ensure valid directory names."""
        pass

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

            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> None:  # type: ignore
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

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
