import ast
import datetime
import random
import string
import sys
import logging
import os
from pathlib import Path
import re
import json
import time
from urllib.parse import urlparse
from pydantic import BaseModel
import tiktoken
from typing import (
    List, 
    Dict,
    Union,
    get_args,
    get_origin, 
)
from crewai import LLM, Crew
from crewai.crews import CrewOutput
from crewai.tasks import TaskOutput
from echo.settings import (
    db_name
)


def dummy_value(field_type):
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is list:
        return [dummy_value(args[0])] if args else []
    elif origin is dict:
        return {dummy_value(args[0]): dummy_value(args[1])} if args else {}
    elif origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        return dummy_value(non_none[0]) if non_none else None

    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        return get_pydantic_dummy_instance(field_type)
    if field_type is str:
        return "".join(random.choices(string.ascii_letters, k=8))
    if field_type is int:
        return random.randint(0, 100)
    if field_type is float:
        return random.uniform(0.0, 100.0)
    if field_type is bool:
        return random.choice([True, False])
    if field_type is datetime.date:
        return datetime.date.today()
    if field_type is datetime.datetime:
        return datetime.datetime.now()

    return None


def get_pydantic_dummy_instance(model_cls: BaseModel) -> BaseModel:
    field_values = {
        name: dummy_value(field.annotation)
        for name, field in model_cls.model_fields.items()
    }
    return model_cls(**field_values)


def create_dummy_crew_output(crew: Crew):
    """
    Create a dummy output for the crew.
    """
    task_outputs = [
        TaskOutput(
            description=t.description,
            name=t.name,
            pydantic=get_pydantic_dummy_instance(t.output_pydantic),
            agent="Dummy Agent",
        )
        for t in crew.tasks
    ]
    crew_output = CrewOutput(tasks_output=task_outputs)
    return crew_output


def get_model_code_with_comments(model: BaseModel) -> str:
    def resolve_type(annotation):
        """
        Resolves the type of an attribute, handling primitive types, composite types,
        and nested models.
        """
        origin = get_origin(annotation)
        args = get_args(annotation)

        # If the type is a Pydantic model
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return resolve_model(annotation)

        # If the type is a List, Dict, etc., resolve its arguments
        elif origin in [list, List]:
            inner_type = resolve_type(args[0]) if args else "Any"
            return f"List[{inner_type}]"
        elif origin in [dict, Dict]:
            key_type = resolve_type(args[0]) if len(args) > 0 else "Any"
            value_type = resolve_type(args[1]) if len(args) > 1 else "Any"
            return f"Dict[{key_type}, {value_type}]"

        # If it's a primitive type
        elif origin is None:
            return annotation.__name__

        # Default to string representation
        return str(annotation)

    def resolve_model(model: BaseModel) -> str:
        """
        Resolves a Pydantic model's attributes into a formatted representation with comments.
        """
        fields = model.__annotations__
        resolved_fields = []
        for field, field_type in fields.items():
            comment = (
                model.model_fields[field].description
                if field in model.model_fields
                else ""
            )
            comment_str = f" # {comment}" if comment else ""
            resolved_fields.append(
                f"\t\t{field}: {resolve_type(field_type)}{comment_str}"
            )
        return "{\n" + "\n".join(resolved_fields) + "\n\t}"

    # Resolve attributes from the base class(es) first
    base_classes = [
        base
        for base in model.__bases__
        if issubclass(base, BaseModel) and base is not BaseModel
    ]
    resolved_base_classes = [
        get_model_code_with_comments(base) for base in base_classes
    ]

    # Top-level model resolution
    fields = model.__annotations__
    resolved_fields = []
    for field, field_type in fields.items():
        comment = (
            model.model_fields[field].description if field in model.model_fields else ""
        )
        comment_str = f" # {comment}" if comment else ""
        resolved_fields.append(f"\t{field}: {resolve_type(field_type)}{comment_str}")

    return (
        f"class {model.__name__}(BaseModel):\n"
        + "\n".join(resolved_base_classes)
        + "\n"
        + "\n".join(resolved_fields)
        + "\n"
    )


def add_pydantic_structure(t_crew: Crew, inputs: dict):
    for i, task in enumerate(t_crew.tasks):
        if "{pydantic_structure}" in task.expected_output:
            pyd = "{pydantic_structure" + f"_{i}" + "}"
            task.expected_output = task.expected_output.replace(
                "{pydantic_structure}", pyd
            )
            inputs[pyd[1:-1]] = get_model_code_with_comments(task.output_pydantic)


def get_data_str(key_items: Dict, data):
    data_str = "\n".join(
        [
            f"{snake_to_camel(k)} Fields Description: {get_model_code_with_comments(v)}\n{json_to_markdown(data[k])}"
            for k, v in key_items.items()
        ]
    )
    return data_str


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


def get_db_name(pth: str = None) -> Path:
    return db_storage_path(suffix=pth) / db_name


def get_current_time():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


def db_storage_path(suffix: str = None):
    data_dir = get_project_directory_name()
    if suffix:
        data_dir = data_dir / suffix
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_project_directory_name(pth: str = None) -> Path:
    pth = Path(pth) if pth else None
    project_directory = os.environ.get("ECHO_STORAGE_DIR")

    if project_directory:
        # Convert the environment variable path to an absolute path
        return Path(project_directory).resolve() / pth if pth else Path(project_directory).resolve()
    else:
        # If ECHO_STORAGE_DIR isn't set, use the current working directory's absolute path
        return Path.cwd().resolve() / ".echo_storage" / pth if pth else Path.cwd().resolve() / ".echo_storage"


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


def get_dummy_string(word_size: int = 9, sentence_size: int = 50, line_size: int = 20):
    import random
    import string
    return "\n".join([" ".join(
        [''.join(
            random.choices(string.ascii_letters, k=word_size)) 
            for _ in range(sentence_size)
        ]) 
        for _ in range(line_size)
    ])
    
    
def url_to_sql_name(url):
    parsed = urlparse(url)
    domain = parsed.netloc + parsed.path
    table_name = re.sub(r'\W+', '_', domain).strip('_').lower()
    return table_name

