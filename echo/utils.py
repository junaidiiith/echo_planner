import ast
import sys
import logging
import os
from pathlib import Path
import re
import json
import time
import tiktoken
from typing import (
    List, 
    Dict,
    Union, 
)
from crewai import LLM
from crewai.tasks import TaskOutput
from echo.settings import (
    db_name
)


def format_response(x: TaskOutput):
    if x.pydantic:
        return x.pydantic.model_dump()
    try:
        print("Trying to parse with Ast...")
        data = ast.literal_eval(x.raw)
        print("Parsed successfully with Ast...")
        return data
    except Exception:
        return x.raw


def get_crew_llm():
    llm = LLM(
        model=os.getenv("FIREWORKS_MODEL_NAME"),
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )
    return llm


def serialize_dict(obj: dict):
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in obj.items()
    }


def deserialize_dict(obj: dict):
    def is_json(v):
        return isinstance(v, str) and (v.startswith("[") or v.startswith("{"))

    return {k: json.loads(v) if is_json(v) else v for k, v in obj.items()}


def get_db_name():
    return get_project_directory_name() / db_name


def get_current_time():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


def db_storage_path(suffix: str = None):
    data_dir = get_project_directory_name()
    if suffix:
        data_dir = data_dir / suffix
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_project_directory_name() -> Path:
    project_directory = os.environ.get("ECHO_STORAGE_DIR")

    if project_directory:
        # Convert the environment variable path to an absolute path
        return Path(project_directory).resolve()
    else:
        # If ECHO_STORAGE_DIR isn't set, use the current working directory's absolute path
        return Path.cwd().resolve() / ".echo_storage"


def split_camel_case(s):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s)


def process_text(word: str):
    return " ".join([t.title() for t in split_camel_case(word).split()])


def json_to_markdown(json_obj: Union[Dict, List], bullet_position: int = 0):
    markdown = ""
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            markdown += f"{'#' * (bullet_position + 1)}" + f" {process_text(key)}\n"
            markdown += json_to_markdown(value, bullet_position + 1)
    elif isinstance(json_obj, list):
        for item in json_obj:
            markdown += f"{'  ' * bullet_position} - "
            markdown += json_to_markdown(item, bullet_position + 1)
    else:
        markdown += f"{json_obj}\n"

    return markdown


def snake_to_camel(snake_str: str):
    components = snake_str.split("_")
    return " ".join(x.title() for x in components)


def dict_to_markdown(data: Dict):
    new_dict = dict()
    for k in data:
        new_dict[k] = (
            json_to_markdown(data[k]) if isinstance(data[k], (dict, list)) else data[k]
        )
    return new_dict


def get_text_upto_tokens(text, limit, model="gpt-4o"):
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text into tokens
    tokens = encoding.encode(text)
    # Return the text upto the specified limit
    return encoding.decode(tokens[:limit])


def get_num_tokens(text, model="gpt-4o"):
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text into tokens
    tokens = encoding.encode(text)
    # Return the number of tokens
    return len(tokens)


def redirect_print_to_logger():
    
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logs_dir = os.getenv("LOGS_DIR", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"{current_time}.log")

    class LoggerWriter:
        def __init__(self, logger: logging.Logger, level):
            self.logger = logger
            self.level = level

        def write(self, message: str):
            message = message.strip()
            if message:
                self.logger.log(self.level, message)

        def flush(self):
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.__stdout__)
        ]
    )

    logger = logging.getLogger("PrintLogger")
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)


def get_variables_from_prompt(prompt: str):
    """
    Extracts variable names from the prompt string.
    """
    # Regular expression to match variable names (e.g., {variable_name})
    pattern = r"\{(\w+)\}"
    matches = re.findall(pattern, prompt)
    
    # Return the list of variable names
    return matches
