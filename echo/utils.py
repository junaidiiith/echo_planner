import json
import os
from typing import List, Dict, get_origin, get_args
from pydantic import BaseModel
from crewai import LLM, Agent, Task, Crew
from echo.constants import ALLOWED_KEYS
from echo.settings import save_dir


format_response = lambda x: x.pydantic.model_dump_json(indent=2) if x.pydantic else x.raw

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
            comment = model.model_fields[field].description if field in model.model_fields else ""
            comment_str = f" # {comment}" if comment else ""
            resolved_fields.append(f"\t\t{field}: {resolve_type(field_type)}{comment_str}")
        return "{\n" + "\n".join(resolved_fields) + "\n\t}"

    # Resolve attributes from the base class(es) first
    base_classes = [base for base in model.__bases__ if issubclass(base, BaseModel) and base is not BaseModel]
    resolved_base_classes = [get_model_code_with_comments(base) for base in base_classes]

    # Top-level model resolution
    fields = model.__annotations__
    resolved_fields = []
    for field, field_type in fields.items():
        comment = model.model_fields[field].description if field in model.model_fields else ""
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
    **crew_config
):
    agents = {
        agent_name: Agent(llm=llm, **v) 
        for agent_name, v in agent_templates.items()
    }
    tasks = dict()
    for task_name, v in task_templates.items():
        d = v.copy()
        d['agent'] = agents[v["agent"]]
        context = v.get("context", [])
        if context:
            d["context"] = [tasks[i] for i in context]
        tasks[task_name] = Task(**d)
    
    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        **crew_config
    )

    return crew


def add_pydanctic_structure(t_crew: Crew, inputs: dict):
    for i, task in enumerate(t_crew.tasks):
        if '{pydantic_structure}' in task.expected_output:
            pyd = "{pydantic_structure" + f"_{i}" + "}"
            task.expected_output = task.expected_output.replace(
                '{pydantic_structure}',
                pyd
            )
            inputs[pyd[1:-1]] = get_model_code_with_comments(task.output_pydantic)



def check_data_exists(client_name):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(f"{save_dir}/{client_name}.json"):
        return False
    return True

def get_client_data(client_name):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{client_name}.json") as f:
        return json.load(f)

def save_client_data(client_name, data):
    os.makedirs(save_dir, exist_ok=True)
    ndata = {k: data[k] for k in data if k in ALLOWED_KEYS}
    with open(f"{save_dir}/{client_name}.json", "w") as f:
        json.dump(ndata, f, indent=2)


def save_clients_data(response_data: Dict):
    client_data = dict()
    for client_name, client_response in response_data.items():
        save_client_data(client_name, client_response)
        
    return client_data
