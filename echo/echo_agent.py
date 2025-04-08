from typing import Dict, Union
from crewai import LLM, Agent, Crew, Task

from pydantic import Field
from echo.settings import MAX_RETRIES
from echo.utils import (
    get_crew_llm,
    get_variables_from_prompt,
    create_dummy_crew_output,
    add_pydantic_structure
)

from echo.settings import debug_mode

exclude_variables = [
    "pydantic_structure",
]


# format_response = lambda x: x.pydantic.model_dump_json(indent=2) if x.pydantic else x.raw


class EchoAgent(Crew):
    max_retries: int = Field(MAX_RETRIES, description="Maximum number of retries")

    async def kickoff_async(self, inputs):
        check_variables_presence(self, inputs)
        add_pydantic_structure(self, inputs)
        if debug_mode:
            print("Debug mode is enabled. Returning Dummy response.")
            return create_dummy_crew_output(self)

        retries = self.max_retries
        while retries > 0:
            try:
                response = await super().kickoff_async(
                    inputs
                )  # Use `super()` to call parent method
                return response
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                print(f"Retrying due to error: {e}")

    def kickoff(self, inputs):
        check_variables_presence(self, inputs)
        add_pydantic_structure(self, inputs)

        if debug_mode:
            print("Debug mode is enabled. Returning Dummy response.")
            return create_dummy_crew_output(self)

        retries = self.max_retries
        while retries > 0:
            try:
                response = super().kickoff(
                    inputs
                )  # Use `super()` to call parent method
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
    print(
        f"Creating crew with {len(agent_templates)} agents and {len(task_templates)} tasks"
    )
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
        check_attrs = ["description", "name", "expected_output"]
        obj_type = "Task"
    elif isinstance(obj, Agent):
        check_attrs = ["goal", "backstory", "role"]
        obj_type = "Agent"
    else:
        raise ValueError("Object must be either an Agent or a Task.")

    assert all(hasattr(obj, attr) for attr in check_attrs), (
        f"{obj_type}: {obj} is missing attributes: {check_attrs}"
    )
    variables = {
        attr: get_variables_from_prompt(getattr(obj, attr)) for attr in check_attrs
    }
    variables = {
        k: [i for i in v if i not in exclude_variables] for k, v in variables.items()
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
    def is_prefix(var: str, prefixes: list):
        """
        Check if the variable is a prefix of any of the prefixes.
        """
        return any(var.startswith(prefix) for prefix in prefixes)
    
    
    crew_variables = get_crew_variables(crew)
    for agent_idx, agent in enumerate(crew_variables["agents"]):
        for k, v in agent.items():
            not_present_vars = [i for i in v if i not in inputs]
            if not_present_vars:
                raise ValueError(
                    f"Missing variable(s) {not_present_vars} in inputs for {k} of agent {crew.agents[agent_idx]}"
                )

    for task_idx, task in enumerate(crew_variables["tasks"]):
        for k, v in task.items():
            not_present_vars = [i for i in v if i not in inputs and not is_prefix(i, exclude_variables)]
            if not_present_vars:
                raise ValueError(
                    f"Missing variable(s) {not_present_vars} in inputs for {k} of task {crew.tasks[task_idx]}"
                )

    return True
