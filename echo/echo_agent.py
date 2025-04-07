import datetime
import random
import string
from typing import Dict, List, Union, get_args, get_origin
from crewai import LLM, Agent, Crew, Task
from crewai.crews import CrewOutput
from crewai.tasks import TaskOutput

from pydantic import BaseModel, Field
from echo.settings import MAX_RETRIES, debug
from echo.utils import (
    get_crew_llm,
    snake_to_camel,
    json_to_markdown,
    get_variables_from_prompt
)

exclude_variables = [
    "pydantic_structure",    
]

def create_dummy_crew_output(crew: Crew):
    """
    Create a dummy output for the crew.
    """
    task_outputs = [
        TaskOutput(
            description=t.description, 
            name=t.name, 
            pydantic=get_pydantic_dummy_instance(t.output_pydantic), 
            agent='Dummy Agent'
        ) for t in crew.tasks
    ]
    crew_output = CrewOutput(tasks_output=task_outputs)
    return crew_output

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
        return ''.join(random.choices(string.ascii_letters, k=8))
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


def get_pydantic_dummy_instance(model_cls: BaseModel):
    field_values = {
        name: dummy_value(field.annotation)
        for name, field in model_cls.model_fields.items()
    }
    return model_cls(**field_values)


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



class EchoAgent(Crew):
    max_retries: int = Field(MAX_RETRIES, description="Maximum number of retries")
    
    async def kickoff_async(self, inputs):
        check_variables_presence(self, inputs)
        add_pydantic_structure(self, inputs)
        if debug:
            print("Debug mode is enabled. Returning Dummy response.")
            return create_dummy_crew_output(self)
        
        retries = self.max_retries
        while retries > 0:
            try:
                response = await super().kickoff_async(inputs)  # Use `super()` to call parent method
                return response
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                print(f"Retrying due to error: {e}")
        
    def kickoff(self, inputs):
        check_variables_presence(self, inputs)
        add_pydantic_structure(self, inputs)
        
        if debug:
            print("Debug mode is enabled. Returning Dummy response.")
            return create_dummy_crew_output(self)
        
        retries = self.max_retries
        while retries > 0:
            try:
                response = super().kickoff(inputs)  # Use `super()` to call parent method
                return response
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                print(f"Retrying due to error: {e}")


def get_crew(
    agent_templates: Dict[str, Dict],
    task_templates: Dict[str, Dict],
    llm: LLM = None,
    **crew_config,
):
    if llm is None:
        llm = get_crew_llm()

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

    crew = EchoAgent(
        agents=list(agents.values()), tasks=list(tasks.values()), **crew_config
    )

    return crew


def get_object_variables(obj: Union[Agent, Task]):
    if isinstance(obj, Task):
        check_attrs = ['description', 'name', 'expected_output']
        obj_type = "Task"
    elif isinstance(obj, Agent):
        check_attrs = ['goal', 'backstory', 'role']
        obj_type = "Agent"
    else:
        raise ValueError("Object must be either an Agent or a Task.")
    
    assert all(hasattr(obj, attr) for attr in check_attrs), f"{obj_type}: {obj} is missing attributes: {check_attrs}"
    variables = {
        attr: get_variables_from_prompt(getattr(obj, attr)) 
        for attr in check_attrs
    }
    variables = {
        k: [i for i in v if i not in exclude_variables] 
        for k, v in variables.items()
    }
    return variables
    
    
def get_crew_variables(crew: Crew):
    """
    Get the crew variables from the crew.
    """
    agent_variables = [get_object_variables(a) for a in crew.agents]
    task_variables = [get_object_variables(t) for t in crew.tasks]
    return {
        "agents": agent_variables,
        "tasks": task_variables,
    }


def check_variables_presence(crew: Crew, inputs: Dict):
    """
    Check if all variables are present in the inputs.
    """
    crew_variables = get_crew_variables(crew)
    for agent_idx, agent in enumerate(crew_variables["agents"]):
        for k, v in agent.items():
            not_present_vars = [i for i in v if i not in inputs]
            if not_present_vars:
                raise ValueError(f"Missing variable(s) {not_present_vars} in inputs for {k} of agent {crew.agents[agent_idx]}")
            
    for task_idx, task in enumerate(crew_variables["tasks"]):
        for k, v in task.items():
            not_present_vars = [i for i in v if i not in inputs]
            if not_present_vars:
                raise ValueError(f"Missing variable(s) {not_present_vars} in inputs for {k} of task {crew.tasks[task_idx]}")
    
    return True