import os
from pathlib import Path
import re
import ast
import json
import time
from typing import List, Dict, Union, get_origin, get_args
from dotenv import load_dotenv
from pydantic import BaseModel
from crewai import LLM, Agent, Task, Crew
from echo.agent import EchoAgent
from crewai.tasks.task_output import TaskOutput
from echo.constants import ALLOWED_KEYS, BUYER, SELLER
from echo.settings import save_dir, buyer_db_name, seller_db_name


load_dotenv()


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


# format_response = lambda x: x.pydantic.model_dump_json(indent=2) if x.pydantic else x.raw


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

    def resolve_model(model):
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


def get_crew(
    agent_templates: Dict[str, Dict],
    task_templates: Dict[str, Dict],
    llm: LLM,
    **crew_config,
):
    agents = {
        agent_name: Agent(llm=llm, **v) for agent_name, v in agent_templates.items()
    }
    tasks = dict()
    for task_name, v in task_templates.items():
        d = v.copy()
        d["agent"] = agents[v["agent"]]
        context = v.get("context", [])
        if context:
            d["context"] = [tasks[i] for i in context]
        tasks[task_name] = Task(**d)

    crew = EchoAgent(agents=list(agents.values()), tasks=list(tasks.values()), **crew_config)

    return crew


def add_pydantic_structure(t_crew: Crew, inputs: dict):
    for i, task in enumerate(t_crew.tasks):
        if "{pydantic_structure}" in task.expected_output:
            pyd = "{pydantic_structure" + f"_{i}" + "}"
            task.expected_output = task.expected_output.replace(
                "{pydantic_structure}", pyd
            )
            inputs[pyd[1:-1]] = get_model_code_with_comments(task.output_pydantic)


def get_save_path(save_path_str: str):
    local_dir = f"{os.sep}".join(save_path_str.split(os.sep)[:-1])
    # print("Saving Path", local_dir)
    save_pth_dir = os.path.join(get_project_directory_name(), save_dir, local_dir)
    os.makedirs(save_pth_dir, exist_ok=True)

    file_name = save_path_str.split(os.sep)[-1]
    save_pth = os.path.join(save_pth_dir, file_name) + ".json"

    return save_pth


def check_data_exists(client_name):
    save_path = get_save_path(client_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        return False
    return True


def get_client_data(client_name) -> Dict:
    save_path = get_save_path(client_name)
    with open(save_path) as f:
        return json.load(f)


def get_refined_data(client_name):
    data = get_client_data(client_name)
    return {k: data[k] for k in data if k in ALLOWED_KEYS}


def save_client_data(client_name: str, data):
    save_pth = get_save_path(client_name)
    ndata = {k: data[k] for k in data if k in ALLOWED_KEYS}
    with open(save_pth, "w") as f:
        json.dump(ndata, f, indent=2)



def remove_keys(keys: List[str]):
    for file in os.listdir(save_dir):
        with open(f"{save_dir}/{file}") as f:
            data = json.load(f)
        for key in keys:
            if key in data:
                del data[key]
        with open(f"{save_dir}/{file}", "w") as f:
            json.dump(data, f, indent=2)


def get_nested_value(key, nested_dict):
    def get_structured_value(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass
        return value

    """Extract value from nested dictionary using dot-separated keys."""
    keys = key.split(".")
    value: Dict = (
        json.loads(nested_dict) if isinstance(nested_dict, str) else nested_dict
    )
    for k in keys:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    value = json.loads(json.dumps(ast.literal_eval(value)))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Could not decode the value: {value} with error: {e}"
                    )

        if k in value:
            value = value[k]
        else:
            raise KeyError(
                f"Key '{key}' not found in the dictionary with keys: {value.keys()}"
            )
    return get_structured_value(value)


def replace_keys_with_values(string, dictionary):
    def get_printable_value(value):
        if isinstance(value, list):
            return "\n\t" + "\n\t".join(
                [f"{i + 1}. {get_printable_value(v)}" for i, v in enumerate(value)]
            )
        if isinstance(value, dict):
            return ",\n".join([f"{k} - {get_printable_value(value[k])}" for k in value])
        return value

    # Regex to find keys enclosed in curly braces
    pattern = r"\{([^{}]+)\}"

    # Replace keys with their corresponding values
    def replacer(match):
        key = match.group(1)  # Extract the key inside curly braces
        try:
            value = get_nested_value(key, dictionary)
            value = get_printable_value(value)
            return value
        except KeyError:
            print(f"Key '{key}' not found in the dictionary")
            return match.group(0)  # If key is not found, keep it as is

    return re.sub(pattern, replacer, string)


def get_nested_key_values(key_location_dict: Dict, dictionary: Dict):
    return {k: get_nested_value(v, dictionary) for k, v in key_location_dict.items()}


def get_llm():
    llm = LLM(
        model=os.getenv("FIREWORKS_MODEL_NAME"),
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )
    return llm



def get_db_type(user_type):
    if user_type == BUYER:
        return buyer_db_name
    elif user_type == SELLER:
        return seller_db_name
    else:
        raise ValueError("User type must be one of 'buyer' or 'seller'")


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
        return (Path.cwd().resolve() / ".echo_storage")

def json_to_markdown(json_obj: Union[Dict, List], bullet_position: int = 0):
    markdown = ""
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            markdown += f"{'#' * (bullet_position + 1)}" + f" {key.title()}\n"
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


def get_data_str(key_items: Dict, data):
    data_str = "\n".join(
        [
            f"{snake_to_camel(k)} Fields Description: {get_model_code_with_comments(v)}\n{json_to_markdown(data[k])}"
            for k, v in key_items.items()
        ]
    )
    return data_str
