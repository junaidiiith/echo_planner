from abc import ABC, abstractmethod
import json
import re
from typing import (
    Any, 
    Dict, 
    List,
    Optional, 
    Tuple,
    Type,
    Union,
    get_args,
    get_origin
)

from pydantic import (
    BaseModel, 
    Field,
    ValidationError
)

from echo.utils import get_llm



class Printer:
    """Handles colored console output formatting."""

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
        elif color == "cyan":
            self._print_cyan(content)
        elif color == "bold_cyan":
            self._print_bold_cyan(content)
        elif color == "magenta":
            self._print_magenta(content)
        elif color == "bold_magenta":
            self._print_bold_magenta(content)
        elif color == "green":
            self._print_green(content)
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

    def _print_cyan(self, content):
        print("\033[96m {}\033[00m".format(content))

    def _print_bold_cyan(self, content):
        print("\033[1m\033[96m {}\033[00m".format(content))

    def _print_magenta(self, content):
        print("\033[35m {}\033[00m".format(content))

    def _print_bold_magenta(self, content):
        print("\033[1m\033[35m {}\033[00m".format(content))

    def _print_green(self, content):
        print("\033[32m {}\033[00m".format(content))


class PydanticSchemaParser(BaseModel):
    model: Type[BaseModel]

    def get_schema(self) -> str:
        """
        Public method to get the schema of a Pydantic model.

        :return: String representation of the model schema.
        """
        return "{\n" + self._get_model_schema(self.model) + "\n}"

    def _get_model_schema(self, model: Type[BaseModel], depth: int = 0) -> str:
        indent = " " * 4 * depth
        lines = [
            f"{indent}    {field_name}: {self._get_field_type(field, depth + 1)}"
            for field_name, field in model.model_fields.items()
        ]
        return ",\n".join(lines)

    def _get_field_type(self, field, depth: int) -> str:
        field_type = field.annotation
        origin = get_origin(field_type)

        if origin in {list, List}:
            list_item_type = get_args(field_type)[0]
            return self._format_list_type(list_item_type, depth)

        if origin in {dict, Dict}:
            key_type, value_type = get_args(field_type)
            return f"Dict[{key_type.__name__}, {value_type.__name__}]"

        if origin is Union:
            return self._format_union_type(field_type, depth)

        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_schema = self._get_model_schema(field_type, depth)
            nested_indent = " " * 4 * depth
            return f"{field_type.__name__}\n{nested_indent}{{\n{nested_schema}\n{nested_indent}}}"

        return field_type.__name__

    def _format_list_type(self, list_item_type, depth: int) -> str:
        if isinstance(list_item_type, type) and issubclass(list_item_type, BaseModel):
            nested_schema = self._get_model_schema(list_item_type, depth + 1)
            nested_indent = " " * 4 * (depth)
            return f"List[\n{nested_indent}{{\n{nested_schema}\n{nested_indent}}}\n{nested_indent}]"
        return f"List[{list_item_type.__name__}]"

    def _format_union_type(self, field_type, depth: int) -> str:
        args = get_args(field_type)
        if type(None) in args:
            # It's an Optional type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                inner_type = self._get_field_type_for_annotation(
                    non_none_args[0], depth
                )
                return f"Optional[{inner_type}]"
            else:
                # Union with None and multiple other types
                inner_types = ", ".join(
                    self._get_field_type_for_annotation(arg, depth)
                    for arg in non_none_args
                )
                return f"Optional[Union[{inner_types}]]"
        else:
            # General Union type
            inner_types = ", ".join(
                self._get_field_type_for_annotation(arg, depth) for arg in args
            )
            return f"Union[{inner_types}]"

    def _get_field_type_for_annotation(self, annotation, depth: int) -> str:
        origin = get_origin(annotation)
        if origin in {list, List}:
            list_item_type = get_args(annotation)[0]
            return self._format_list_type(list_item_type, depth)
        if origin in {dict, Dict}:
            key_type, value_type = get_args(annotation)
            return f"Dict[{key_type.__name__}, {value_type.__name__}]"
        if origin is Union:
            return self._format_union_type(annotation, depth)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_schema = self._get_model_schema(annotation, depth)
            nested_indent = " " * 4 * depth
            return f"{annotation.__name__}\n{nested_indent}{{\n{nested_schema}\n{nested_indent}}}"
        return annotation.__name__



class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class OutputConverter(BaseModel, ABC):
    """
    Abstract base class for converting task results into structured formats.

    This class provides a framework for converting unstructured text into
    either Pydantic models or JSON, tailored for specific agent requirements.
    It uses a language model to interpret and structure the input text based
    on given instructions.

    Attributes:
        text (str): The input text to be converted.
        llm (Any): The language model used for conversion.
        model (Any): The target model for structuring the output.
        instructions (str): Specific instructions for the conversion process.
        max_attempts (int): Maximum number of conversion attempts (default: 3).
    """

    text: str = Field(description="Text to be converted.")
    llm: Any = Field(description="The language model to be used to convert the text.")
    model: Any = Field(description="The model to be used to convert the text.")
    instructions: str = Field(description="Conversion instructions to the LLM.")
    max_attempts: int = Field(
        description="Max number of attempts to try to get the output formatted.",
        default=3,
    )

    @abstractmethod
    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        pass

    @abstractmethod
    def to_json(self, current_attempt=1):
        """Convert text to json."""
        pass


class Converter(OutputConverter):
    """Class that converts text into either pydantic or json."""

    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_pydantic()
            else:
                response = self.llm.call(
                    [
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": self.text},
                    ]
                )
                return self.model.model_validate_json(response)
        except ValidationError as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to the following validation error: {e}"
            )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_pydantic(current_attempt + 1)
            raise ConverterError(
                f"Failed to convert text into a Pydantic model due to the following error: {e}"
            )

    def to_json(self, current_attempt=1):
        """Convert text to json."""
        try:
            if self.llm.supports_function_calling():
                return self._create_instructor().to_json()
            else:
                return json.dumps(
                    self.llm.call(
                        [
                            {"role": "system", "content": self.instructions},
                            {"role": "user", "content": self.text},
                        ]
                    )
                )
        except Exception as e:
            if current_attempt < self.max_attempts:
                return self.to_json(current_attempt + 1)
            return ConverterError(f"Failed to convert text into JSON, error: {e}.")


    def _create_instructor(self):
        """Create an instructor."""
        from crewai.utilities import InternalInstructor

        inst = InternalInstructor(
            llm=self.llm,
            model=self.model,
            content=self.text,
        )
        return inst

    def _convert_with_instructions(self):
        """Create a chain."""
        from crewai.utilities.crew_pydantic_output_parser import (
            CrewPydanticOutputParser,
        )

        parser = CrewPydanticOutputParser(pydantic_object=self.model)
        result = self.llm.call(
            [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": self.text},
            ]
        )
        return parser.parse_result(result)


def validate_model(
    result: str, model: Type[BaseModel], is_json_output: bool
) -> Union[dict, BaseModel]:
    exported_result = model.model_validate_json(result)
    if is_json_output:
        return exported_result.model_dump()
    return exported_result


def handle_partial_json(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    match = re.search(r"({.*})", result, re.DOTALL)
    if match:
        try:
            exported_result = model.model_validate_json(match.group(0))
            if is_json_output:
                return exported_result.model_dump()
            return exported_result
        except json.JSONDecodeError:
            pass
        except ValidationError:
            pass
        except Exception as e:
            Printer().print(
                content=f"Unexpected error during partial JSON handling: {type(e).__name__}: {e}. Attempting alternative conversion method.",
                color="red",
            )

    return convert_with_instructions(
        result, model, is_json_output, converter_cls
    )


def convert_with_instructions(
    result: str,
    model: Type[BaseModel],
    is_json_output: bool,
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    llm = get_llm()
    instructions = get_conversion_instructions(model, llm)
    converter = create_converter(
        converter_cls=converter_cls,
        llm=llm,
        text=result,
        model=model,
        instructions=instructions,
    )
    exported_result = (
        converter.to_pydantic() if not is_json_output else converter.to_json()
    )

    if isinstance(exported_result, ConverterError):
        Printer().print(
            content=f"{exported_result.message} Using raw output instead.",
            color="red",
        )
        return result

    return exported_result


def get_conversion_instructions(model: Type[BaseModel], llm: Any) -> str:
    instructions = "Please convert the following text into valid JSON."
    if llm.supports_function_calling():
        model_schema = PydanticSchemaParser(model=model).get_schema()
        instructions += (
            f"\n\nThe JSON should follow this schema:\n```json\n{model_schema}\n```"
        )
    else:
        model_description = generate_model_description(model)
        instructions += f"\n\nThe JSON should follow this format:\n{model_description}"
    return instructions


def create_converter(
    agent: Optional[Any] = None,
    converter_cls: Optional[Type[Converter]] = None,
    *args,
    **kwargs,
) -> Converter:
    if agent and not converter_cls:
        if hasattr(agent, "get_output_converter"):
            converter = agent.get_output_converter(*args, **kwargs)
        else:
            raise AttributeError("Agent does not have a 'get_output_converter' method")
    elif converter_cls:
        converter = converter_cls(*args, **kwargs)
    else:
        raise ValueError("Either agent or converter_cls must be provided")

    if not converter:
        raise Exception("No output converter found or set.")

    return converter


def generate_model_description(model: Type[BaseModel]) -> str:
    """
    Generate a string description of a Pydantic model's fields and their types.

    This function takes a Pydantic model class and returns a string that describes
    the model's fields and their respective types. The description includes handling
    of complex types such as `Optional`, `List`, and `Dict`, as well as nested Pydantic
    models.
    """

    def describe_field(field_type):
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union or (origin is None and len(args) > 0):
            # Handle both Union and the new '|' syntax
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return f"Optional[{describe_field(non_none_args[0])}]"
            else:
                return f"Optional[Union[{', '.join(describe_field(arg) for arg in non_none_args)}]]"
        elif origin is list:
            return f"List[{describe_field(args[0])}]"
        elif origin is dict:
            key_type = describe_field(args[0])
            value_type = describe_field(args[1])
            return f"Dict[{key_type}, {value_type}]"
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return generate_model_description(field_type)
        elif hasattr(field_type, "__name__"):
            return field_type.__name__
        else:
            return str(field_type)

    fields = model.__annotations__
    field_descriptions = [
        f'"{name}": {describe_field(type_)}' for name, type_ in fields.items()
    ]
    return "{\n  " + ",\n  ".join(field_descriptions) + "\n}"



def convert_to_model(
    result: str,
    output_pydantic: Optional[Type[BaseModel]],
    output_json: Optional[Type[BaseModel]],
    converter_cls: Optional[Type[Converter]] = None,
) -> Union[dict, BaseModel, str]:
    model = output_pydantic or output_json
    if model is None:
        return result
    try:
        escaped_result = json.dumps(json.loads(result, strict=False))
        return validate_model(escaped_result, model, bool(output_json))
    except json.JSONDecodeError:
        return handle_partial_json(
            result, model, bool(output_json), converter_cls
        )

    except ValidationError:
        return handle_partial_json(
            result, model, bool(output_json), converter_cls
        )

    except Exception as e:
        Printer().print(
            content=f"Unexpected error during model conversion: {type(e).__name__}: {e}. Returning original result.",
            color="red",
        )
        return result



def export_output(
    result: str,
    output_pydantic: Optional[Type[BaseModel]],
    output_json: Optional[Type[BaseModel]],
    converter_cls: Optional[Type[Converter]] = None,
) -> Tuple[Optional[BaseModel], Optional[Dict[str, Any]]]:
    pydantic_output: Optional[BaseModel] = None
    json_output: Optional[Dict[str, Any]] = None

    if output_pydantic or output_json:
        model_output = convert_to_model(
            result,
            output_pydantic,
            output_json,
            converter_cls
        )

        if isinstance(model_output, BaseModel):
            pydantic_output = model_output
        elif isinstance(model_output, dict):
            json_output = model_output
        elif isinstance(model_output, str):
            try:
                json_output = json.loads(model_output)
            except json.JSONDecodeError:
                json_output = None

    return pydantic_output, json_output