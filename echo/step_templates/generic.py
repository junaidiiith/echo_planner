import enum
import os
from typing import Dict, List
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import *
from tqdm.asyncio import tqdm
from echo.memory import LTMSQLiteStorage, RAGStorage
from echo.utils import add_pydantic_structure, get_crew, format_response
from echo.utils import (
    get_db_type, 
    get_sql_client, 
    get_rag_client, 
)


class CallType(enum.Enum):
    DISCOVERY = "discovery"
    DEMO = "demo"
    PRICING = "pricing"
    PROCUREMENT = "procurement"



class Message(BaseModel):
	user: str
	content: str

class Transcript(BaseModel):
	messages: List[Message]


class Section(BaseModel):
    name: str = Field(..., title="Section Name", description="The name of the section.")
    subsection_names: List[str]

class Sections(BaseModel):
    sections: List[Section] = Field(..., title="Sections", description="The segmented sections of the call transcript.")

class FilledSubsection(BaseModel):
    name: str = Field(..., title="Subsection Name", description="The name of the subsection.")
    content: str = Field(..., title="Subsection Content", description="The content of the subsection.")

class FilledSection(FilledSubsection):
    name: str = Field(..., title="Section Name", description="The name of the section.")
    content: str = Field(..., title="Section Content", description="The content of the section.")
    subsections: List[FilledSubsection] = Field(..., title="Subsections", description="The subsections of the section.")

class FilledSections(BaseModel):
    sections: List[FilledSection] = Field(..., title="Sections", description="The filled sections of the call transcript.")


agent_templates = {
    SECTION_EXTRACTION: {
        "CallStructureExtractor": dict(
            role="Call Transcripts Expert Analyst",
            goal="Responsible for extracting the generalized structure from a sales call or a set of schemas of calls.",
            backstory=(
                "You are an expert in analyzing sales call transcripts and extracting the generalized structure of the conversation.\n"
                "Given a set of call transcripts, you can identify common sections, unique fields, and patterns to create a consolidated schema.\n"
                "Given a set of schemas extracted from sales call transcripts, you can also come up with a generalized schema that captures common and essential elements.\n"
                "The segmented sections should be clearly labeled and organized to provide a structured overview of the conversation."
            )
        )   
    },
    SECTION_MERGER: {
        "GeneralizedSchemaGenerator": 
            dict(
            role="Call Structure Generalization Expert",
            goal="Responsible for extracting the generalized structure from a sales call or a set of schemas of calls.",
            backstory=(
                "You are an expert in analyzing sales call transcripts and extracting the generalized structure of the conversation.\n"
                "Given a set of schemas extracted from sales call transcripts, you can also come up with a generalized schema that captures common and essential elements.\n"
                "The segmented sections should be clearly labeled and organized to provide a structured overview of the conversation."
            )
        )
    }
}

task_templates = {
    SECTION_EXTRACTION: {
        "CallStructureExtraction": dict(
            name="Extract {call_type} Call Structure",
            description=(
                "Generate a generalized and representative schema by analyzing multiple transcripts of {call_type} sales calls conducted by a sales agent.\n"
                "The schema should abstract and consolidate the structure and key elements of the {call_type} calls conducted by {seller}, reflecting how a sales agent from {seller} typically conducts a call while capturing variations and maintaining a logical flow.\n"

                "Task Details:\n"
                "Input:\n"
                "A set of transcripts from {call_type} sales calls conducted by a sales agent. Each transcript contains --"
                "Messaging data (e.g., messages exchanged between the sales agent and the client)\n"

                "Requirements for Schema Consolidation::\n"
                "1. Identify Common Sections: Recognize recurring sections that structure the calls (e.g., greetings, goal setting, pricing discussion).\n"
                "2. Generalize Key Elements: Consolidate common subsections and key talking points (e.g., “Business Challenges” vs. “Pain Points” → “Challenges”).\n"
                "3. Abstract Sales Agent's Flow: Identify and preserve the typical flow of the conversation, such as whether they handle objections early, probe budgets mid-call, etc\n"
                "4. Preserve Logical Structure: Ensure that the consolidated schema reflects the logical phases of the sales call (e.g., Opening → Needs Assessment → Solution Discussion → Next Steps).\n"
                "5. Avoid Overfitting: The schema should not be a union of all points but a meaningful abstraction that reflects general patterns.\n"

                "A more general goal, invariant goal is that the schema should be such that it eases comprehension and helps the sales team to understand the needs, pain points, challenges of the buyer and successfully close the deal."
        
                "Output - "
                "A single, consolidated sales call schema that represents the sales agent's typical call structure, consisting of:\n"
                "Sections: Key phases of the call (e.g., Introduction, Business Needs, Solution Discussion, Closing)\n"
                "Subsections: Representative discussion points within each section\n"
                
                "Below are the transcripts of the {call_type} calls that you need to analyze and extract the structure.\n"
                "{transcripts}"
            ),
            expected_output=(
                "A structured overview of the {call_type} call {seller}'s client with distinct sections."
                "The response should conform to the provided schema."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            output_pydantic=Sections,
            agent="CallStructureExtractor"    
        ),
    },
    SECTION_MERGER: {
        "GeneralizedSchemaGeneration": dict(
            name="Generalized Schema Generator",
            description=(
                "Create a generalized representative schema by consolidating multiple schemas extracted from {call_type} sales call transcripts.\n"
                "The goal is to produce a combined schema that captures common and essential elements across all provided schemas, while accounting for variability and ensuring coverage of diverse information.\n"

                "Task Details:\n"
                "Input:\n"
                "A set of schemas extracted from a set of sales call transcripts. Each schema contains --"
                "Sections (e.g., Introduction, Needs Assessment, Solution Discussion, etc.)\n"
                "Subsections Subsections (up to 3 per section, e.g., Key Pain Points, Business Goals, Decision-Making Process, etc.)\n"
                "Requirements:\n"
                "Identify common sections across schemas (e.g., introduction, needs assessment, objections handling, next steps).\n"
                "Extract unique fields and merge similar fields by recognizing patterns (e.g., “pain points” vs “challenges” should be combined if contextually similar)\n"
                "Maintain critical information while avoiding redundancy—ensure the combined schema is not an exhaustive union but a meaningful generalization.\n"
                "Preserve structure: Group related elements together and follow logical flow. Ensure that the combined schema follows a logical structure, typically reflecting the flow of a sales conversation (e.g., Introduction → Needs Assessment → Objections → Closing).\n"
                "Add flexibility for optional fields where certain schemas might have unique data points (e.g., pricing discussions, competitor mentions).\n"
        
                "A more general goal, invariant goal is that the schema should be such that it eases comprehension and helps the sales team to understand the needs, pain points, challenges of the buyer and successfully close the deal."
                "{example}"
                "Below are the schemas extracted from the {call_type} calls that you need to consolidate and generate a generalized schema.\n"
		        "{schemas}"
                
            ),
            expected_output=(
                "A single consolidated structured overview of the {call_type} call that is representative of the common structure followed by {seller} in their sales calls.\n"
                "The schema should have same properties i.e., sections and upto 3 subsections per section"
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            output_pydantic=Sections,
            agent="GeneralizedSchemaGenerator"
        ),
    }
}

example_merge = \
"""
Example:
Input:
    Schema 1:
    {
        "Introduction": ["Greeting", "Purpose of Meeting"],
        "Business Needs": ["Key Challenges", "Budget Discussion"],
        "Next Steps": ["Timeline", "Follow-Up"]
    }
    Schema 2:
    {
        "Greeting": ["Welcome", "Purpose of Call"],
        "Problem Areas": ["Pain Points", "Competitor Mention"],
        "Closing": ["Next Steps", "Stakeholder Confirmation"]
    }
    
Generalized Schema Output:
    {
        "Introduction": ["Greeting", "Purpose of Meeting/Call"],
        "Needs Assessment": ["Business Challenges (Key Challenges/Pain Points)", "Budget Discussion", "Competitor Mention"],
        "Next Steps": ["Timeline", "Follow-Up", "Stakeholder Confirmation"]
    }
"""

def get_section_crew(type: str, llm: LLM, **crew_config):
    assert type in [SECTION_EXTRACTION, SECTION_MERGER], \
        f"Invalid Section type: {type}: Must be one of {SECTION_EXTRACTION}, {SECTION_MERGER}"
    
    return get_crew(
        agent_templates=agent_templates[type],
        task_templates=task_templates[type],
        llm=llm,
        **crew_config
    )


async def extract_call_structure(transcripts: List[str], llm: LLM, inputs_dict, group_size=4):
    crew = get_section_crew(SECTION_EXTRACTION, llm)
    responses = list()
    for i in tqdm(range(0, len(transcripts), group_size), desc="Extracting Call Structure"):
        group = transcripts[i:i+group_size]
        inputs_dict['transcripts'] = "\n".join([f"Transcript: {i + 1}\n{s}" for i, s in enumerate(group)])
        add_pydantic_structure(crew, inputs_dict)
        response = await crew.kickoff_async(inputs_dict)
        responses.append(response)	
    return responses


async def merge_call_structures(
    sections: List[CrewOutput], 
    llm: LLM,
    inputs_dict, 
    group_size=4
) -> List[CrewOutput]:
    
    if len(sections) == 1:
        return sections
    
    responses = list()
    
    for i in tqdm(range(0, len(sections), group_size), desc="Merging Call Structures"):
        group = sections[i:i + group_size]
        inputs_dict['schemas'] = "\n".join([f"Schema: {i + 1}\n{s.tasks_output[0].raw}" for i, s in enumerate(group)])
        inputs_dict['example'] = example_merge
        crew = get_section_crew(SECTION_MERGER, llm)    
        add_pydantic_structure(crew, inputs_dict)
        response = await crew.kickoff_async(inputs=inputs_dict)
        responses.append(response)
    
    return await merge_call_structures(responses, llm, inputs_dict, group_size)



async def aget_call_structure(
    transcripts, 
    llm,
    inputs_dict: Dict
) -> CrewOutput:
    call_type = inputs_dict['call_type']
    save_path = f"runs/{inputs_dict['seller']}_{call_type}_call_structure.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return f.read()

    
    print("Total Transcripts: ", len(transcripts))
    call_structure_responses: List[CrewOutput] = await extract_call_structure(
        transcripts, 
        llm,
        inputs_dict, 
        group_size=4
    )
    call_structure = await merge_call_structures(
        call_structure_responses, 
        llm,
        inputs_dict, 
        group_size=4
    )
    
    structure = format_response(call_structure[0])
    with open(f"{save_path}", "w") as f:
        f.write(structure)
    return structure


def embed_clients_data(user_type, clients, metadata):
    assert user_type in [BUYER, SELLER], f"user_type must be one of {BUYER}, {SELLER}"
    data = {
        DISCOVERY: dict(),
        DEMO: dict()
    }
    for call_type in [DISCOVERY, DEMO]:
        print(f"Saving {user_type} call data for {call_type}")
        for client in clients:
            data[call_type][client] = embed_client_call_data(
                client, 
                user_type, 
                call_type,
                metadata
            )
            
    print(f"Saved {user_type} call data for {clients}")
    
    return data


def save_clients_data(user_type, clients, metadata):
    assert user_type in [BUYER, SELLER], f"user_type must be one of {BUYER}, {SELLER}"
    data = {
        DISCOVERY: dict(),
        DEMO: dict()
    }
    for call_type in [DISCOVERY, DEMO]:
        print(f"Saving {user_type} call data for {call_type}")
        for client in clients:
            data[call_type][client] = save_client_call_data(
                client, 
                user_type, 
                call_type,
                metadata
            )
            
    print(f"Saved {user_type} call data for {clients}")
    
    return data
    


def save_client_call_data(client_name, user_type, call_type, inputs, data):
    assert user_type in [BUYER, SELLER], f"user_type must be one of {BUYER}, {SELLER}"
    print(f"Saving {user_type} call data for {client_name}")
    sql_client: LTMSQLiteStorage = get_sql_client(db_type=get_db_type(user_type))
    buyer, seller = inputs['buyer'], inputs['seller']
    sql_client.save(
        buyer=buyer, 
        seller=seller, 
        call_type=call_type, 
        data=data, 
    )
    
    return data


def embed_client_call_data(client_name, user_type, call_type, metadata, data):
    assert user_type in [BUYER, SELLER], f"user_type must be one of {BUYER}, {SELLER}"
    print(f"Saving {user_type} call data for {client_name}")
    rag_client: RAGStorage = get_rag_client(db_type=get_db_type(user_type))
    metadata['call_type'] = call_type
    rag_client.save(
        value=data, 
        metadata=metadata, 
    )
    
    return data