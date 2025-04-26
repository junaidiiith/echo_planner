import copy
import networkx as nx
import math
import networkx as nx
from typing import List, Dict, Any
import math
import openai
import matplotlib.pyplot as plt
from pyvis.network import Network
from collections import defaultdict
import plotly.graph_objects as go

from echo.data.indexes import IndexType
from echo.data.index_enums import SellerIndexQueryTypes, BuyerIndexQueryTypes

from echo.query_executor import Query, LlamaSubQuery

from echo.query_executor import arun_query_chain
# from echo.query_executor import aget_query_response
import asyncio
from echo.query_executor import Query, LlamaSubQuery, QueryChain
import re
import nest_asyncio
import requests
nest_asyncio.apply()

def create_value_prop(buyer, seller, buyer_initiatives, seller_info):


  value_align_prompt_system = f"""You are a strategic sales executive at {seller}\n
  You are trying to find the initiatives of the buyer {buyer} and align them to your product."""


  value_align_prompt_user= f"""The top initiatives of the buyer and the details of your product are given. Of all initiatives, find the most relevant ones
  the sellers product can help solve and achieve. rate the relevance and problem solution fit as well. if all seem important, its okay to mark as relevant but make sure you assess properly.
  Output format in json:
  Buyer Initiative or pain
  Why the seller can help?
  Relevance of seller to solving the pain or initiative

  Sample output:
  {{
    "Strategic Initiatives": [
      {{
        "Buyer Initiative or pain": "International Expansion (hiring 100+ engineers in Bengaluru, scaling globally)",
        "Why the seller can help?": "Whatfix’s in-app guidance and onboarding flows accelerate time‑to‑productivity for new hires across geographies. Localized self‑help modules, task lists, and smart tips ensure consistent training and process adherence, reducing reliance on instructor‑led sessions and enabling Rippling to scale its workforce efficiently in India and beyond.",
        "Relevance of seller to solving the pain or initiative": "High"
      }},
      {{
        "Buyer Initiative or pain": "Investment in R&D (building new products, integrating AI, enhancing existing platform)",
        "Why the seller can help?": "Whatfix streamlines internal adoption of newly developed tools and features by embedding contextual walkthroughs and in‑app prompts. This ensures Rippling’s engineers and early adopters can instantly learn and validate new product capabilities, accelerating feedback loops and reducing friction in beta testing and rollout phases.",
        "Relevance of seller to solving the pain or initiative": "Medium"
      }},
      {{
        "Buyer Initiative or pain": "Enhancing IT Security and Automation (data privacy, compliance, identity & device management, reducing manual tasks)",
        "Why the seller can help?": "Whatfix embeds real‑time guidance and compliance checks directly within Rippling’s IT and security workflows. By guiding users through standardized processes, enforcing policy steps via task lists, and capturing process analytics, Whatfix helps minimize human error, automate repetitive tasks, and strengthen overall security posture.",
        "Relevance of seller to solving the pain or initiative": "High"
      }}
    ]
  }}

  Here is the initiative on buyer
  {buyer_initiatives}

  Here is the sellers details and pains they solve for their customers
  {seller_info}

  Output only the json and nothing else. Dont make anything up and answer from context provided.
  """
  client = openai.OpenAI()

  response_value_prop = client.responses.create(
      model="o4-mini",
      reasoning={"effort": "medium"},
      input=[
          {

              "role": "system",
              "content": value_align_prompt_system
          },
          {

              "role": "user",
              "content": value_align_prompt_user
          },
      ]
  )

  return response_value_prop.output_text


# Find teams directly owning the initiative as well as additional teams who can aid or block the process and will have an important say.
# When picking cross functional teams, make sure they have an impactful say that will be considered by leadership and team that owns the initiative.


def find_teams(buyer, seller, response_value_prop):
  best_fit_team_system_prompt = f"""

  You are a sales executive at {seller} who is an expert at mapping the buying committees for different buyer initiatives.
  You have the top relevant initiatives of buyer {buyer} and info on how the seller can solve these. For each initiative thats relevant (dont pick poor relevant ones),
  You need to find the teams and titles that would be involved in making the purchase decision.


  1. Predict all relevant teams that may be involved in the buying committee.
  2. For each team:
    a. Include job titles that would likely participate in the buying process.
    b. Capture alternate title variations commonly used on LinkedIn (e.g., "VP of Talent Acquisition", "Head of Talent", "Director - TA"). For each team except Direct Owner Team, limit to max of 5 titles. For Direct Owner Team, include max 10 titles.
    c. Keep titles as general as possible and as short as possible. For example - dont use titles like Learning Experience Designer or Learning Experience Manager. Use titles like Learning Manager or Learning Designer.
    d. Focus on senior titles and decision influencers. for Direct owner teams, keep some senior IC's but not too many.
    e. Maintain realistic seniority: include ICs in titles only if they are decision influencers or operational champions.
    f. Prioritize titles used in mid-sized to large tech companies (500+ headcount).
  3. Return the output structured by:
    - Direct Owner Team
    - Economic Buyer
    - Cross-Functional Reviewers
    - Internal Influencers
    - Champions

  Sample output:
  {{
    "Strategic Initiatives": [
      {{
        "Buyer Initiative or pain": "International Expansion (hiring 100+ engineers in Bengaluru, scaling globally)",
        "Why the seller can help?": "Whatfix’s in-app guidance and onboarding flows accelerate time‑to‑productivity for new hires across geographies. Localized self‑help modules, task lists, and smart tips ensure consistent training and process adherence, reducing reliance on instructor‑led sessions and enabling Rippling to scale its workforce efficiently in India and beyond.",
        "Relevance of seller to solving the pain or initiative": "High",
        {{
    "Direct Owner Team": [
      {{"team": "Sales", "Relevance":"why its relevant","titles": ["VP of Sales", "Director of Sales", "Sales Enablement Manager"]}}
    ],
    "Economic Buyer": [
      {{"team": "Finance", "Relevance":"why its relevant","titles": ["CFO", "VP of Finance"]}}
    ],
    "Cross-Functional Reviewers": [
      {{"team": "IT", "Relevance":"why its relevant","titles": ["IT Manager", "Director of Security"]}},
      {{"team": "Legal", "Relevance":"why its relevant","titles": ["Legal Counsel"]}},
      {{"team": "Procurement", "Relevance":"why its relevant","titles": ["Procurement Manager"]}}
    ],
    "Internal Influencers": [
      {{"team": "Customer Success", "Relevance":"why its relevant","titles": ["VP of Customer Success", "Director of CS"]}},
      {{"team": "RevOps", "Relevance":"why its relevant","titles": ["Revenue Operations Manager"]}}
    ],
    "Champions": [
      {{"team": "Sales", "Relevance":"why its relevant","titles": ["Account Executive"]}},
      {{"team": "Enablement", "Relevance":"why its relevant","titles": ["Sales Enablement Lead"]}}
    ]
  }}
      }},
      {{
        "Buyer Initiative or pain": "Investment in R&D (building new products, integrating AI, enhancing existing platform)",
        "Why the seller can help?": "Whatfix streamlines internal adoption of newly developed tools and features by embedding contextual walkthroughs and in‑app prompts. This ensures Rippling’s engineers and early adopters can instantly learn and validate new product capabilities, accelerating feedback loops and reducing friction in beta testing and rollout phases.",
        "Relevance of seller to solving the pain or initiative": "Medium",
        {{
    "Direct Owner Team": [
      {{"team": "Sales","Relevance":"why its relevant", "titles": ["VP of Sales", "Director of Sales", "Sales Enablement Manager"]}}
    ],
    "Economic Buyer": [
      {{"team": "Finance", "Relevance":"why its relevant","titles": ["CFO", "VP of Finance"]}}
    ],
    "Cross-Functional Reviewers": [
      {{"team": "IT", "Relevance":"why its relevant","titles": ["IT Manager", "Director of Security"]}},
      {{"team": "Legal", "Relevance":"why its relevant","titles": ["Legal Counsel"]}},
      {{"team": "Procurement", "Relevance":"why its relevant","titles": ["Procurement Manager"]}}
    ],
    "Internal Influencers": [
      {{"team": "Customer Success", "Relevance":"why its relevant","titles": ["VP of Customer Success", "Director of CS"]}},
      {{"team": "RevOps", "Relevance":"why its relevant","titles": ["Revenue Operations Manager"]}}
    ],
    "Champions": [
      {{"team": "Sales", "Relevance":"why its relevant","titles": ["Account Executive"]}},
      {{"team": "Enablement","Relevance":"why its relevant", "titles": ["Sales Enablement Lead"]}}
    ]
  }}
      }},
    ]
  }}
  """
  client = openai.OpenAI()

  best_fit_team_user_prompt = f"""
  here are the initiaitves identified for the buyer and how well the seller {seller} fits to solve them - {response_value_prop}
  Output just the json and nothing else.
  """
  response_team_guess_reasoning = client.responses.create(
      model="o4-mini",
      reasoning={"effort": "medium"},
      input=[
          {

              "role": "system",
              "content": best_fit_team_system_prompt
          },
          {

              "role": "user",
              "content": best_fit_team_user_prompt
          },
      ]
  )

  # response_team_guess = client.responses.create(
  #     model="o4-mini",
  #     #reasoning={"effort": "medium"},
  #     input=[
  #         {

  #             "role": "system",
  #             "content": best_fit_team_system_prompt
  #         },
  #         {

  #             "role": "user",
  #             "content": best_fit_team_user_prompt
  #         },
  #     ]
  # )
  return response_team_guess_reasoning.output_text

# print(response_team_guess_reasoning.output_text)
# print("\n\n\n")
# print(response_team_guess.output_text)



# using python



dummy_json_title_data = """
{
  "Strategic Initiatives": [
    {
      "Buyer Initiative or pain": "International Expansion (hiring 100+ engineers in Bengaluru, scaling globally)",
      "Why the seller can help?": "Whatfix’s in‑app guidance, task lists and smart tips onboard new hires across locations with contextual walkthroughs and self‑help modules, reducing time‑to‑productivity and training/support costs while ensuring consistent process adherence globally.",
      "Relevance of seller to solving the pain or initiative": "High",
      "Direct Owner Team": [
        {
          "team": "People Operations",
          "Relevance": "owns global hiring processes and onboarding",
          "titles": ["Head of People Operations", "VP of People Operations", "Director of People Operations", "People Operations Manager"]
        },
        {
          "team": "Talent Acquisition",
          "Relevance": "manages recruitment and initial candidate experience",
          "titles": ["VP of Talent Acquisition", "Director of Talent Acquisition", "Senior Manager - Talent Acquisition", "Talent Acquisition Manager"]
        },
        {
          "team": "Learning & Development",
          "Relevance": "designs and delivers onboarding training programs",
          "titles": ["Head of Learning & Development", "Director of L&D", "L&D Manager", "Learning Program Manager"]
        }
      ],
      "Economic Buyer": [
        {
          "team": "Finance",
          "Relevance": "controls budget for HR and training technologies",
          "titles": ["CFO", "VP of Finance", "Finance Director"]
        }
      ],
      "Cross-Functional Reviewers": [
        {
          "team": "IT",
          "Relevance": "ensures integration with HR systems and platform stability",
          "titles": ["IT Director", "IT Manager", "Head of IT Operations", "Infrastructure Manager"]
        },
        {
          "team": "Security",
          "Relevance": "validates compliance with company security policies",
          "titles": ["Director of Information Security", "Security Manager", "Head of IT Security"]
        },
        {
          "team": "Legal",
          "Relevance": "reviews contractual terms and data privacy implications",
          "titles": ["Legal Counsel", "Corporate Counsel"]
        },
        {
          "team": "Procurement",
          "Relevance": "manages vendor selection and contract negotiations",
          "titles": ["Procurement Manager", "Senior Procurement Specialist"]
        }
      ],
      "Internal Influencers": [
        {
          "team": "HRIS",
          "Relevance": "configures and maintains HR technology stack, influences tool adoption",
          "titles": ["HRIS Manager", "HR Systems Analyst", "HR Technology Manager"]
        },
        {
          "team": "Learning & Development Specialists",
          "Relevance": "use the tool for creating and delivering onboarding content",
          "titles": ["Training Specialist", "L&D Specialist", "Instructional Designer"]
        }
      ],
      "Champions": [
        {
          "team": "HR Operations",
          "Relevance": "operational champion driving day-to-day onboarding efficiency",
          "titles": ["HR Operations Manager", "People Ops Manager"]
        },
        {
          "team": "Talent Acquisition",
          "Relevance": "advocates for tools that improve time-to-hire and candidate experience",
          "titles": ["Talent Acquisition Manager", "TA Manager"]
        }
      ]
    },
    {
      "Buyer Initiative or pain": "Investment in R&D (building new products, integrating AI, enhancing existing platform)",
      "Why the seller can help?": "Whatfix embeds contextual walkthroughs and in‑app prompts into new or enhanced applications, driving feature adoption, capturing real‑time feedback via in‑app surveys, and accelerating internal user validation and iterative improvement cycles.",
      "Relevance of seller to solving the pain or initiative": "Medium",
      "Direct Owner Team": [
        {
          "team": "Product Management",
          "Relevance": "responsible for feature adoption and product feedback loops",
          "titles": ["VP of Product", "Director of Product Management", "Product Manager", "Senior Product Manager"]
        },
        {
          "team": "Engineering",
          "Relevance": "builds and maintains platforms where in‑app guidance will be embedded",
          "titles": ["VP of Engineering", "Director of Engineering", "Engineering Manager"]
        },
        {
          "team": "R&D",
          "Relevance": "drives development of new capabilities and requires user feedback",
          "titles": ["Head of R&D", "R&D Manager"]
        }
      ],
      "Economic Buyer": [
        {
          "team": "Technology Leadership",
          "Relevance": "approves budget for R&D and product innovation tools",
          "titles": ["CTO", "Chief Technology Officer"]
        }
      ],
      "Cross-Functional Reviewers": [
        {
          "team": "IT",
          "Relevance": "ensures tool deployment in development and staging environments",
          "titles": ["IT Manager", "Director of IT"]
        },
        {
          "team": "Security",
          "Relevance": "evaluates security of in‑app integrations and data handling",
          "titles": ["Information Security Manager", "Director of Security"]
        },
        {
          "team": "Procurement",
          "Relevance": "coordinates purchasing processes for software tools",
          "titles": ["Procurement Manager", "Senior Procurement Specialist"]
        },
        {
          "team": "Legal",
          "Relevance": "assesses contractual and IP considerations",
          "titles": ["Legal Counsel", "Corporate Counsel"]
        }
      ],
      "Internal Influencers": [
        {
          "team": "DevOps",
          "Relevance": "implements and maintains deployment pipelines for integrations",
          "titles": ["DevOps Engineer", "DevOps Lead"]
        },
        {
          "team": "UX/Design",
          "Relevance": "designs user flows and influences adoption of guidance tools",
          "titles": ["UX Manager", "User Experience Lead"]
        },
        {
          "team": "Quality Assurance",
          "Relevance": "tests integration points and ensures tool reliability",
          "titles": ["QA Manager", "Quality Assurance Lead"]
        }
      ],
      "Champions": [
        {
          "team": "Product Management",
          "Relevance": "drives user adoption and feedback within product teams",
          "titles": ["Product Manager", "PM"]
        },
        {
          "team": "Engineering",
          "Relevance": "supports operational rollout and developer advocacy",
          "titles": ["Engineering Manager", "Software Engineering Manager"]
        }
      ]
    },
    {
      "Buyer Initiative or pain": "Enhancing IT Security and Automation (data privacy, compliance, identity & device management, reducing manual tasks)",
      "Why the seller can help?": "Whatfix delivers real‑time guidance, compliance checkpoints and process analytics within IT and security workflows, enforcing standardized policy steps, minimizing human error in identity/device management, and automating repetitive tasks.",
      "Relevance of seller to solving the pain or initiative": "High",
      "Direct Owner Team": [
        {
          "team": "Security/InfoSec",
          "Relevance": "owns security policies and compliance requirements",
          "titles": ["CISO", "Chief Information Security Officer", "Director of Security Operations", "Head of IT Security"]
        },
        {
          "team": "IT Operations",
          "Relevance": "manages day‑to‑day IT infrastructure and process automation",
          "titles": ["VP of IT Operations", "Director of IT Operations", "IT Operations Manager"]
        }
      ],
      "Economic Buyer": [
        {
          "team": "Technology Leadership",
          "Relevance": "allocates budget for security and IT automation initiatives",
          "titles": ["CIO", "Chief Information Officer"]
        }
      ],
      "Cross-Functional Reviewers": [
        {
          "team": "Compliance",
          "Relevance": "ensures solutions meet regulatory and internal compliance standards",
          "titles": ["Compliance Manager", "Director of Compliance"]
        },
        {
          "team": "Legal",
          "Relevance": "reviews legal risks related to data privacy and vendor contracts",
          "titles": ["Legal Counsel", "Security & Privacy Counsel"]
        },
        {
          "team": "Procurement",
          "Relevance": "oversees vendor selection and contract management",
          "titles": ["Procurement Manager", "Senior Procurement Specialist"]
        },
        {
          "team": "Risk Management",
          "Relevance": "evaluates risk mitigation and governance implications",
          "titles": ["Risk Manager", "Risk & Compliance Director"]
        }
      ],
      "Internal Influencers": [
        {
          "team": "IAM Team",
          "Relevance": "operationalizes identity and access workflows, uses automation tools",
          "titles": ["IAM Engineer", "Identity & Access Management Lead"]
        },
        {
          "team": "SOC Team",
          "Relevance": "relies on real‑time guidance for incident response processes",
          "titles": ["SOC Analyst", "Security Operations Center Analyst"]
        },
        {
          "team": "IT Automation",
          "Relevance": "drives process automation and workflow standardization",
          "titles": ["Automation Engineer", "IT Process Automation Specialist"]
        }
      ],
      "Champions": [
        {
          "team": "Security Architecture",
          "Relevance": "advocates for architecture‑level integration of guidance and compliance checks",
          "titles": ["Security Architect", "Enterprise Security Architect"]
        },
        {
          "team": "Systems Administration",
          "Relevance": "implements and supports daily use and administration of the solution",
          "titles": ["Systems Administrator", "IT Administrator"]
        }
      ]
    }
  ]
}


"""
# parse prev response
import json
import pickle
from sentence_transformers import SentenceTransformer, util
import datetime
from rapidfuzz import fuzz
import pickle
import rapidfuzz

def get_people_titles_to_search(strategy_data:str):
    if "json" in strategy_data:
        strategy_data = strategy_data.split("```json")[1].strip("`")
    print("step 3 data")
    print(strategy_data)
    print("trying to load json")
    print("\n")
    title_data = json.loads(strategy_data)
    # this is what we need
    final_strategy_data = {}
    all_titles_to_search = []
    for initiative in title_data["Strategic Initiatives"]:
        final_strategy_data[initiative["Buyer Initiative or pain"]] = {}
        #print(initiative["Buyer Initiative or pain"])
        #print("\n")
        keys = ["Economic Buyer","Internal Influencers","Champions","Cross-Functional Reviewers","Direct Owner Team"]
        titles_data = {}
        for tag in keys:
            final_strategy_data[initiative["Buyer Initiative or pain"]][tag] = []
            for teams in initiative[tag]:
                titles_data[teams["team"]] = {"Relevance":teams["Relevance"],"titles":teams["titles"]}
                all_titles_to_search = all_titles_to_search +teams["titles"]
                final_strategy_data[initiative["Buyer Initiative or pain"]][tag].append(titles_data[teams["team"]] | {"Team":teams["team"]})

    return final_strategy_data, all_titles_to_search
    # 1) now search using crustdata on all_titles_t_search
# make sure to get all data using total_result_count and page_number


# match both sets of data here

def get_current_tenure(employer_data):
  tenure = 0.0
  for experience in employer_data:
    if not experience.get("end_date"):
      start_date = datetime.datetime.fromisoformat(experience["start_date"])
      end_date = datetime.datetime.fromisoformat(datetime.datetime.now().isoformat())
      delta = end_date - start_date
      tenure += int(delta.days)
  return tenure // 365

def get_industry_experience(employer_data):
  experience_days = 0
  for experience in employer_data:
    start_date = datetime.datetime.fromisoformat(experience["start_date"])
    end_date = datetime.datetime.fromisoformat(experience.get("end_date") or datetime.datetime.now().isoformat())
    # Compute the difference
    delta = end_date - start_date
    experience_days+=int(delta.days)
    # Get the number of days, years, etc.
    #print("Difference in days:", delta.days)

  return experience_days // 365




def generate_value_props_for_stakeholders(graph, top_targets: dict, initiative,buyer, seller):
  value_props = []
  email_snippets = []

  name_to_node_id = {
      data.get("name"): node_id
      for node_id, data in graph.graph.nodes(data=True)
  }

  for tag, stakeholders in top_targets.items():
      for full_name, score in stakeholders:
        node_id = name_to_node_id.get(full_name)
        if not node_id:
            continue
        node = graph.graph.nodes[node_id]
        role = node.get("default_position_title", "Unknown Role")
        org = node.get("role_enriched", {}).get("Org Unit", "")
        function = node.get("role_enriched", {}).get("Function Type", "")
        initiative = initiative
        influence = round(node.get("influence_score", 50))

        # Prepare prompt context
        stakeholder_context = f"""
        - Name: {full_name}
        - Role: {role}
        - Tag: {tag}
        - Org Unit: {org}
        - Function Type: {function}
        - Influence Score: {influence}
        - Known Initiatives: {initiative }
        """

        
        value_prop_alignment = Query(
              query=f""" Following is the buyer initiative at {buyer} that an enterprise seller at {seller} is selling to.
                    He is trying to sell to "title" at {buyer} company. Given the buyer background, sellers value add and pains it solves
                    and buyer initiatives, what is the business case and value prop to build for the stakeholder based on his / her title, buyer industry and 
                    buyer product.
                    here are details of the stakeholder in {buyer} company: 

                    {stakeholder_context}
                    Provide the output format in a proper and short way.
                    **Markdown format:**

                    ### {full_name} ({tag})
                    **Full Value Prop:** <your value prop here>  
                    **1-liner Email Summary:** <your summary here>
                    **Possible Objections:** <anticipated objections the stakeholder of that role and buyer initiative could raise>
                    **Their role and KPIs:** <What would their role be in this initiative and what KPI's would they care about and how to address them>
              """,
              sub_queries=[
                  LlamaSubQuery(
                      query="what are the top strategic goals and pains the buyer {buyer} is trying to solve for?",
                      index_type=IndexType.BUYER_WEB_SEARCH,
                      inputs={"category": BuyerIndexQueryTypes.STRATEGIC_INITIATIVES.value},
                  ),
                  LlamaSubQuery(
                      query="What are the top initiatives and focus areas by the leadership of {buyer}",
                      index_type=IndexType.BUYER_WEB_SEARCH,
                      inputs={"query_type": BuyerIndexQueryTypes.LEADERSHIP_MOVEMENTS.value},
                  ),
                  LlamaSubQuery(
                      query="What are the pains and challenges does {seller} solve for its customers",
                      index_type=IndexType.SELLER_WEB_SEARCH,
                      #inputs={"query_type": BuyerIndexQueryTypes.RECENTNEWS.value},
                  ),
                  LlamaSubQuery(
                      query="What all products and features does {seller} offer for its customers",
                      index_type=IndexType.SELLER_WEB_SEARCH,
                      #inputs={"query_type": BuyerIndexQueryTypes.STRATEGY.value},
                  ),
                  LlamaSubQuery(
                      query="how does {seller} pitch its product to the customers. what is the value prop and the vocaubulary used",
                      index_type=IndexType.SELLER_WEB_SEARCH,
                      #inputs={"query_type": BuyerIndexQueryTypes.STRATEGY.value},
                  )
              ],
              output_name="value_prop_alignement"
        )
        inputs = {
           "seller":seller,
           "buyer":buyer,
        }
        person_role_response = asyncio.run(arun_query_chain(query_chain=QueryChain(queries=[value_prop_alignment]), inputs=inputs))
        value_props.append(person_role_response.responses[0].response)

        # Extract just the 1-liner
        one_liner = re.search(r'\*\*1-liner Email Summary:\*\* (.+)', person_role_response.responses[0].response)
        email_snippets.append(f"- {full_name} ({tag}): {one_liner.group(1) if one_liner else '...' }")

  final_md = "\n\n".join(value_props)
  summary_md = "### Value Prop Email Snippets\n" + "\n".join(email_snippets)
  return final_md, summary_md

# extract relevant fields only from the person data dict provided
def extract_relevant_person_data(person_data, target_company):
  person_data_relevant = {}
  fields = ["name","default_position_title","num_of_connections",]
  person_data_relevant = {k:person_data[k] for k in fields}
  person_data_relevant["industry_experience"] = get_industry_experience(person_data["employer"])
  person_data_relevant["tenure"] = get_current_tenure(person_data["employer"])
  client = openai.OpenAI()

  # call llm here only ?

  enrich_profile_llm_system = f"""
  You are an expert data analyst. Given an api response data on various employees in a company,
  you need to extract data from the response per employee from the json provided.

  Given the person's title and headline, infer their org unit (e.g., Sales, Marketing, RevOps, Enablement, Executive), suborganization unit as well (say Engineering at Azure Cloud vs Engineering at Bing Ads or Human Resources Hiring or Human resources Talent Recruiting)
  and seniority level on a 1–7 scale (1 = junior IC, 7 = C-level) and Function type (strategic, tactical, IC).

  Here's what you need to extract finally and output as a json
  Final Output format -
  {{
  "Org Unit" - infer from title
  "Suborg Unit" - Say sales enablement , sales ops instead of just sales (infer from title)
  "Seniority Level" (1–7)
  "Function Type" (Strategic, Tactical, IC)
  }}

  OUtput only json with the 4 fields and nothing else.
  """

  enrich_profile_llm_user = f"""
  here is the title - {person_data["default_position_title"]} and here is the headline - {person_data["headline"]}

  """

  person_role_response = client.responses.create(
    model="gpt-4o",
    #reasoning={"effort": "medium"},
    input=[
        {

            "role": "system",
            "content": enrich_profile_llm_system
        },
        {

            "role": "user",
            "content": enrich_profile_llm_user
        },
    ]
  )


  """
  Alt endpoint - faster and cheaper
  .ChatCompletion.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": enrich_profile_llm_system },
        {"role": "user", "content": enrich_profile_llm_user }
    ],
    temperature=0,  # deterministic
    top_p=0.0,      # use for deterministic/simple tasks
)

print(response['choices'][0]['message']['content'])

  """
  print(person_role_response.output_text)
  try:
    person_data_relevant["role_enriched"] = json.loads(person_role_response.output_text.split("```json")[1].strip("`"))
    return person_data_relevant
  except:
    return {}


def fuzzy_match(title1, title2, threshold=85):
  print(title1)
  print(title2)
  score = fuzz.partial_ratio(title1.lower().strip(), title2.lower().strip())
  if score >= threshold:
    return True
  return False


def semantic_match(model,title1, title2, threshold=0.7):


  emb1 = model.encode(title1.lower().strip(), convert_to_tensor=True)
  emb2 = model.encode(title2.lower().strip(), convert_to_tensor=True)
  score = util.cos_sim(emb1, emb2).item()
  if score >= threshold:
    return True
  return False

# extract relevant data from people api responses and match with the titles needes for that strategy
def match_people_strategy_data(people_data, strategic_data, buyer):
  model = SentenceTransformer("all-MiniLM-L6-v2")
  # iterate strategy data
  strategic_data_temp = {}
  for strategy in strategic_data:
    strategic_data_temp[strategy] = {}
    for tag in strategic_data[strategy]:
      strategic_data_temp[strategy][tag] = []
      for team in strategic_data[strategy][tag]:
        persons = []
        person_names = set()
        titles_needed = team["titles"]
        for title in titles_needed:
          for people in people_data["profiles"]:
            if semantic_match(model,title, people["default_position_title"],0.70) or ((float(rapidfuzz.fuzz.token_set_ratio(title, people["default_position_title"])) / 100.0) > 0.90):
            #if fuzzy_match(title, people["default_position_title"],90):
              #print("true")
              print(title)
              print(people["default_position_title"])
              if people["name"] not in person_names:
                person_names.add(people["name"])
                persons.append(extract_relevant_person_data(people,buyer))
        team |= {"people":persons}
        strategic_data_temp[strategy][tag].append(team)

  return strategic_data_temp


#print(people_data)
#print(final_strategy_data)

def join_people_strategy_data(buyer, people_data, final_strategy_data):
  headers = {
          "Content-Type": "application/json",
          "Authorization": "Token 8582455305237735a32d0be5b74dda9b22dc9857"
        },
  endpoint = "https://api.crustdata.com/screener/person/search",
  buyer = "rippling.com"
  #people_data = fetch_all_profiles(endpoint, headers, buyer, all_titles_to_search)
  # we have a sample pickle dict for people_data
  # with open('./sample_data/people_data.pkl', 'rb') as f:
  #   people_data = pickle.load(f)
  #all_titles_to_search=list(set(all_titles_to_search)) # type: ignore
  people_strategy_data = match_people_strategy_data(people_data,final_strategy_data,buyer)
  #print(result)
  return people_strategy_data


def get_person_strategy_key(person, strategy):
    return f"{strategy}_{person.get('name')}_{person.get('title')}_{person.get('tenure', 0)}"

def get_influence_score(strategy_people_data):
  strategy_people_data_temp={}
  processed_person_strategy = {}
  total_calls = 0

  for strategy in strategy_people_data:
    strategy_people_data_temp[strategy]={}
    for tag in strategy_people_data[strategy]:
      strategy_people_data_temp[strategy][tag]=[]
      for team in strategy_people_data[strategy][tag]:

        copied_team = copy.deepcopy(team)
        copied_team["people"] = []  # re-init

        for person in team["people"]:
            key = get_person_strategy_key(person, strategy)

            if key in processed_person_strategy:
               person_enriched = person | processed_person_strategy[key]
            else:
                try:
                    influence_score_prompt_system = """
                    Given this stakeholder’s title, company initiative, tenure at the company,
                    organization of the stakeholder ,total career tenure in industry,
                    Seniority score in the buyers company and type of role (Strategic, IC , manager)
                    and social reach (follower or connections count),
                    predict their influence score from 0 to 100 in the company’s decision-making process. Consider all factors given,
                    Relevance of initiative to the stakeholders team and title and Score according to the weight given below.
                    

                    Weigh:

                    Seniority (30%)

                    Relevance of initiative to seller org - (10%)

                    Tenure at company (20%)

                    Industry tenure (20%)

                    Follower count (5%)

                    Suborganization relevance to initiative - 10%

                    Role (strategic, IC etc) - 5%

                    Input:


                    Title: Head of Revenue Operations

                    Tenure in company: 5 years

                    Career: 15 years

                    Connection count: 400

                    Industry Experience : 10 years

                    org_unit

                    suborg_unit - Say sales enablement , sales ops instead of just sales

                    seniority_level (1–7)

                    function_type (Strategic, Tactical, IC)

                    Outputformat:
                    Output a json with
                    {
                        "influence_score": 100,
                        "Reason": "Explain the influence and the reasons for high or low influence to an account executive. Dont talk about internal scoring alogirithm mentioned but explain the influence to the seller"
                    }

                    """

                    influence_score_prompt_user = f"""
                    here is persona data - {str(person)} and here is the buyer company's initiative - {str(strategy)}
                    """
                    client = openai.OpenAI()


                    node_influence = client.responses.create(
                        model="gpt-4o",
                        #reasoning={"effort": "medium"},
                        input=[
                            {
                                "role": "system",
                                "content": influence_score_prompt_system
                            },
                            {
                                "role": "user",
                                "content": influence_score_prompt_user
                            },
                        ]
                        )
                    total_calls += 1
                    print(f"Total GPT Calls: {total_calls}")
                    #print(response.output_text, "\n\n")
                    print(node_influence.output_text)
                    print("\n\n\n\n")
                    # node_influence = """
                    # ```json
                    # {
                    #     "influence_score": 100,
                    #     "Reason": "Reasoning for the score"
                    # }
                    # """
                    
                    node_data = json.loads(node_influence.output_text.split("```json")[1].strip("`").strip())
                    person_enriched = person | node_data
                    processed_person_strategy[key] = node_data
                    
                except:
                    print("error")
                    person_enriched = person | {
                        "influence_score": 50,
                        "Reason": "Default Score"
                    }
            copied_team["people"].append(person_enriched)

        strategy_people_data_temp[strategy][tag].append(copied_team)
  return strategy_people_data_temp



class StakeholderGraph:
    def __init__(self, initiative_name, people_data):
        self.initiative = initiative_name
        self.graph = nx.DiGraph()
        self.people = people_data
        self.max_connections = max(p.get("num_of_connections", 1) for p in people_data)
        self._build_graph()
        #self.edge_threshold = edge_threshold

    def _should_create_edge(self, a, b):
      seniority_a = a["role_enriched"].get("Seniority Level", 0)
      seniority_b = b["role_enriched"].get("Seniority Level", 0)
      influence_score_a = a.get("influence_score", 50)
      influence_score_b = b.get("influence_score", 50)

      seniority_diff = abs(seniority_a - seniority_b)
      influence_diff = influence_score_a - influence_score_b

      # Prune based on seniority gap
      if seniority_diff > 2:
          return False

      # Only connect from more influential to less
      if influence_score_a <= influence_score_b:
          return False

      # Minimum threshold for influence to create edge
      if influence_score_a < 30:
          return False

      # Allow edge if they're close in seniority and same org
      same_org = (
          a["role_enriched"].get("Org Unit") == b["role_enriched"].get("Org Unit")
      )
      if seniority_diff == 1 and same_org:
          return True

      # Otherwise only allow if influence diff is significant
      return influence_diff >= 20


    def _build_graph(self):
        for person in self.people:
            self.graph.add_node(person["name"], **person)

        raw_weights = []
        edge_candidates = []

        for i, a in enumerate(self.people):
            for j, b in enumerate(self.people):
                if a.get("tag", "") == "Economic Buyer":
                  continue  # DMs are sinks only

                if a["name"] == b["name"]:
                    continue  # Skip self

                if a["influence_score"] <= b["influence_score"]:
                    continue
                if not self._should_create_edge(a, b):
                    continue
                if i != j: #and a["name"] != b["name"]:
                    weight = self._calculate_edge_weight(a, b)
                    raw_weights.append(weight)
                    edge_candidates.append((a["name"], b["name"], weight))

        max_raw_weight = max(raw_weights) if raw_weights else 1
        avg_raw_weight = (1.0*(sum(raw_weights)/len(raw_weights)))
        normalized_avg_weight = round(avg_raw_weight / max_raw_weight, 3)

        print(max_raw_weight)
        print(avg_raw_weight)
        print(raw_weights)
        for src, tgt, raw_weight in edge_candidates:
            normalized_weight = round(raw_weight / max_raw_weight, 3)

            if normalized_weight >= normalized_avg_weight:
                self.graph.add_edge(src, tgt, weight=normalized_weight)

    def _calculate_edge_weight(self, a, b):
        weight = 0

        # Seniority influence
        seniority_a = a["role_enriched"].get("Seniority Level", 0)
        seniority_b = b["role_enriched"].get("Seniority Level", 0)
        if seniority_a == seniority_b:
            weight += 5
        elif seniority_a > seniority_b:
            weight += 2 * (seniority_a - seniority_b)

        # Org and sub-org alignment
        if a["role_enriched"].get("Org Unit") == b["role_enriched"].get("Org Unit"):
            weight += 5
        if a["role_enriched"].get("Suborg Unit") == b["role_enriched"].get("Suborg Unit"):
            weight += 3

        # Function type
        if a["role_enriched"].get("Function Type") == "Strategic" and b["role_enriched"].get("Function Type") == "Tactical":
            weight += 3

        # Tenure dynamics
        shared_tenure = min(a.get("tenure", 0), b.get("tenure", 0))
        tenure_diff = max(0, a.get("tenure", 0) - b.get("tenure", 0))
        weight += shared_tenure
        weight += tenure_diff * 1.5

        # Peer similarity (same seniority)
        if seniority_a == seniority_b:
            weight += 2

        # A's influence features
        weight += 5 * (a.get("num_of_connections", 0) / self.max_connections)
        weight += 3 * (a.get("industry_experience", 0) / 30)

      

        # Role tags boost
        tags_weights = {"Champion":3, "Economic Buyer":5, "Direct Owner Team":3, "Cross-Functional Reviewers":1, "Internal Influencers":2}
        weight+=tags_weights.get(a.get("tag", ""), 0)
        #print(a.get("tag", "Default"))
        #print(a)
        #weight+=tags_weights.get(b.get("role", ""), 0)
        
        # Influence score (scaled)
        influence_score = a.get("influence_score", 50)
        #weight *= (influence_score / 100)
        influence_score_a = a.get("influence_score", 50)
        influence_score_b = b.get("influence_score", 50)
        influence_diff = influence_score_a - influence_score_b
        diff = min(influence_diff, 50)  # Cap difference
        weight *= diff / 50  # Scale back to 0–1 range

        
        # Normalize
        norm_weight = 1 / (1 + math.exp(-0.2 * (weight - 10)))  # Adjust center as needed
        return norm_weight
        #return round(min(weight / 20, 1), 2)

    def get_high_influence_nodes(self, top_n=5):
        influence_scores = [(n, self.graph.out_degree(n, weight="weight")) for n in self.graph.nodes]
        return sorted(influence_scores, key=lambda x: x[1], reverse=True)[:top_n]

    def summarize_stakeholders_by_tag_markdown(self, top_n=3):

      tag_groups = defaultdict(list)

      for person in self.people:
          tags = person.get("tag", [])
          if isinstance(tags, str):  # convert single string tag to list
              tags = [tags]
          for tag in tags:
              tag_groups[tag].append(person)

      markdown = f"## 🧩 Stakeholder Summary by Role (Top {top_n})\n"
      for tag, group in tag_groups.items():
          sorted_group = sorted(group, key=lambda p: p.get("influence_score", 0), reverse=True)[:top_n]
          markdown += f"\n### 🔹 {tag}s\n"
          for p in sorted_group:
              markdown += f"- `{p['name']}` — **{p.get('default_position_title', 'N/A')}**, Influence Score: **{p.get('influence_score', 0)}**\n"
      return markdown


    # change this
    def find_shortest_path_to_decision_maker(self, from_node, dm_title="Cross-Functional Reviewers"):
        decision_makers = [n for n, d in self.graph.nodes(data=True) if d.get("tag") == dm_title ]
        paths = {}
        for dm in decision_makers:
            if self.graph.has_edge(from_node, dm):
              print("There is a direct edge from champion to dm")
            else:
              print("No direct edge")
            try:
                path = nx.shortest_path(self.graph, source=from_node, target=dm, weight="weight")
                paths[dm] = path
            except nx.NetworkXNoPath:
                continue
        return paths
    def visualize_graph(self, figsize=(10, 8)):
        pos = nx.spring_layout(self.graph)
        edge_weights = nx.get_edge_attributes(self.graph, 'weight')

        plt.figure(figsize=figsize)
        nx.draw(self.graph, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()}, font_size=8)
        plt.title(f"Stakeholder Influence Graph for Initiative: {self.initiative}")
        plt.axis('off')
        plt.show()

    def visualize_graph_interactive(self, notebook=False):
        net = Network(height="750px", width="100%", directed=True, notebook=notebook,cdn_resources='in_line')

        for node, attrs in self.graph.nodes(data=True):
            title = f"{attrs.get('default_position_title', '')}<br>Seniority: {attrs['role_enriched'].get('Seniority Level', 'N/A')}<br>Tenure: {attrs.get('tenure', 'N/A')} years"
            net.add_node(node, label=node, title=title, color='skyblue')

        for src, dst, data in self.graph.edges(data=True):
            net.add_edge(src, dst, value=data["weight"], title=f"Influence: {data['weight']:.2f}")

        net.show_buttons(filter_=['physics'])
        net.show(f"{self.initiative}_stakeholder_graph.html")
    
    def visualize_graph_plotly(self):
      pos = nx.spring_layout(self.graph, seed=42,k=2)

      edge_x = []
      edge_y = []
      edge_weights = []
      for u, v, data in self.graph.edges(data=True):
          x0, y0 = pos[u]
          x1, y1 = pos[v]
          edge_x.extend([x0, x1, None])
          edge_y.extend([y0, y1, None])
          edge_weights.append(data["weight"])

      edge_trace = go.Scatter(
          x=edge_x, y=edge_y,
          line=dict(width=1, color='#888'),
          hoverinfo='none',
          mode='lines')

      node_x = []
      node_y = []
      node_text = []
      for node, data in self.graph.nodes(data=True):
          x, y = pos[node]
          node_x.append(x)
          node_y.append(y)
          title = f"{node}<br>{data.get('default_position_title', '')}<br>Seniority: {data['role_enriched'].get('Seniority Level', 'N/A')}<br>Tenure: {data.get('tenure', 'N/A')} years"
          node_text.append(title)

      node_trace = go.Scatter(
          x=node_x, y=node_y,
          mode='markers+text',
          text=[n for n in self.graph.nodes()],
          hovertext=node_text,
          hoverinfo='text',
          marker=dict(
              showscale=True,
              colorscale='Blues',
              size=20,
              color=[self.graph.out_degree(n, weight='weight') for n in self.graph.nodes()],
              colorbar=dict(thickness=15, title='Influence Score', xanchor='left', titleside='right'),
              line_width=2))

      fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title=f"<b>Stakeholder Influence Graph</b><br>{self.initiative}",
                          titlefont_size=20,
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20,l=5,r=5,t=40),
                          xaxis=dict(showgrid=False, zeroline=False),
                          yaxis=dict(showgrid=False, zeroline=False))
                      )
      fig.show()

    # use this
    def get_consensus_paths_to_all_targets(self, tag_to_targets: dict, top_k=3, min_weight=0.2):
      """
      tag_to_targets: dict of {tag: [node_id, ...]} for each stakeholder type (e.g. Champion, Influencer, Owner, etc.)
      top_k: how many influencers per target to show
      """
      all_results_md = f"## Consensus Paths to Key Stakeholders\n"
      summary_md = "**Key paths to influence high-priority stakeholders:**\n"

      for tag, target_ids in tag_to_targets.items():
          #print(tag)
          #print(target_ids)
          if not target_ids:
              continue

          all_results_md += f"\n### Tag: {tag}\n"
          for target_id in target_ids:
              if target_id[0] not in list(self.graph.nodes):
                  continue

              target_node = self.graph.nodes[target_id[0]]
              incoming = []
              for src, tgt, data in self.graph.in_edges(target_id[0], data=True):
                  weight = data.get("weight", 0)
                  print("weight = ")
                  print(weight)
                  if weight >= min_weight:
                      incoming.append((src, weight))

              incoming_sorted = sorted(incoming, key=lambda x: x[1], reverse=True)[:top_k]
              print("len = ")
              print(len(incoming_sorted))
              all_results_md += f"\n#### {target_node.get('name', target_id)} ({target_node.get('default_position_title', 'N/A')})\n"
              all_results_md += "| Influencer | Title | Influence Weight |\n|---|---|---|\n"
              summary_md += f"- {target_node.get('name', target_id)} ({tag}):\n"

              for src_id, weight in incoming_sorted:
                  src = self.graph.nodes[src_id]
                  name = src.get("name", src_id)
                  title = src.get("default_position_title", "N/A")
                  all_results_md += f"| {name} | {title} | {round(weight, 2)} |\n"
                  summary_md += f"   - {name} ({title}), weight: {round(weight, 2)}\n"

      return all_results_md, summary_md

    # considers centraliy for champion and influence for other categories
    def get_top_influencers_by_tag(self, top_k=3):

      betweenness = nx.betweenness_centrality(self.graph)
      closeness = nx.closeness_centrality(self.graph)

      results_md = "### Top Stakeholders by Tag\n\n"
      summary_md = "**Key Influencers per Role:**\n"

      tags = set(nx.get_node_attributes(self.graph, "tag").values())
      tags.discard("Economic Buyer")
      stakeholders = {}
      for tag in tags:
          candidates = [
              node for node, data in self.graph.nodes(data=True)
              if data.get("tag") == tag
          ]

          scored = []
          for node_id in candidates:
              node = self.graph.nodes[node_id]
              influence_score = node.get("influence_score", 50)
              org_match = node["role_enriched"].get("Org Unit") or ""
              suborg_match = node["role_enriched"].get("Suborg Unit") or ""

              if tag == "Champion":
                  score = (
                      0.5 * betweenness.get(node_id, 0) +
                      0.5 * closeness.get(node_id, 0) +
                      0.3 * (influence_score / 100)
                  )
              else:
                  score = (
                      0.6 * (influence_score / 100) +
                      0.25 * (1 if org_match else 0) +
                      0.15 * (1 if suborg_match else 0)
                  )

              scored.append((node_id, score))

          if not scored:
              continue

          top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
          stakeholders[tag] = top
          results_md += f"#### {tag}\n\n"
          results_md += "| Name | Title | Score |\n|---|---|---|\n"
          summary_md += f"**{tag}**:\n"

          for node_id, score in top:
              node = self.graph.nodes[node_id]
              name = node.get("name", node_id)
              title = node.get("default_position_title", "N/A")

              results_md += f"| {name} | {title} | {round(score, 2)} |\n"
              summary_md += f"- {name} ({title}), score: {round(score, 2)}\n"

          results_md += "\n"

      return stakeholders, results_md, summary_md



    def generate_engagement_strategy(self, buyer_initiative, target_tags=["Champions", "Internal Influencer","Economic Buyer"], use_llm=True):
      strategies = []

      for node, attrs in self.graph.nodes(data=True):
          tags = attrs.get("tag", [])
          if not any(tag in tags for tag in target_tags):
              continue

          name = node
          title = attrs.get("default_position_title", "Unknown")
          org = attrs.get("role_enriched", {}).get("Org Unit", "N/A")
          suborg = attrs.get("role_enriched", {}).get("Suborg Unit", "N/A")
          function_type = attrs.get("role_enriched", {}).get("Function Type", "N/A")
          seniority = attrs.get("role_enriched", {}).get("Seniority Level", "N/A")
          tenure = attrs.get("tenure", "N/A")
          influence = attrs.get("influence_score", "N/A")
          connections = attrs.get("num_of_connections", 0)
          industry_exp = attrs.get("industry_experience", 0)
          all_tags = ", ".join(tags)

          markdown_summary = (
              f"### 👤 `{name}` — {title}\n"
              f"- **Tag**: {all_tags}\n"
              f"- **Function**: {function_type} | **Seniority**: {seniority}\n"
              f"- **Org/Suborg**: {org} / {suborg}\n"
              f"- **Tenure**: {tenure} yrs | **Influence**: {influence}\n"
              f"- **Connections**: {connections} | **Industry XP**: {industry_exp} yrs\n"
              f"- **Buyer Initiative**: *{buyer_initiative}*\n\n"
          )

          if not use_llm:
              strategies.append(markdown_summary + "➡️ Strategy: [TODO — Add manually]\n")
              continue

          # 🔥 LLM Prompt per person
          prompt = f"""
          You're a sales strategist helping an AE plan outreach to key stakeholders.

          Based on the buyer initiative: **{buyer_initiative}**

          And this stakeholder's profile:
          {markdown_summary}

          Write a 3–4 line personalized engagement strategy for how to approach this stakeholder, based on their role, org unit, influence, and tags. Be specific.
          """
          client = openai.OpenAI()


          response_engagement_strategy = client.responses.create(
          model="gpt-4o",
          #reasoning={"effort": "medium"},
          input=[
              
              {
                  "role": "user",
                  "content": prompt
              },
            ]
          )
          print(response_engagement_strategy.output_text)

          # Replace this with your LLM call, or return the prompt to use externally
          # For now we just add the prompt
          strategies.append(markdown_summary + "**Suggested Strategy (LLM prompt):**\n" + response_engagement_strategy.output_text + "\n")

      return "\n---\n".join(strategies)


    def get_stakeholders_for_engagement(self, decision_maker, max_per_tag=3):
      stakeholders = set()
      
      # 1. Get shortest paths from all nodes to decision maker
      for node in self.graph.nodes():
          if node == decision_maker:
              continue
          try:
              path = nx.shortest_path(self.graph, source=node, target=decision_maker, weight='weight')
              stakeholders.update(path)
          except nx.NetworkXNoPath:
              continue

      # 2. Collect and sort by tag + influence
      tag_groups = {"Champions": [], "Internal Influencer": [], "Economic Buyer": []}
      
      for node in stakeholders:
          attrs = self.graph.nodes[node]
          for tag in tag_groups.keys():
              if tag in attrs.get("tags", []):
                  tag_groups[tag].append((node, attrs.get("influence_score", 0)))
      
      # 3. Pick top N per tag
      filtered = set()
      for tag, nodes in tag_groups.items():
          sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:max_per_tag]
          for node, _ in sorted_nodes:
              filtered.add(node)
      
      return list(filtered)


    def find_best_champion_and_decision_maker(self):
      champions = []
      decision_makers = []

      for node, data in self.graph.nodes(data=True):
          tags = data.get("tag", [])
          influence = data.get("influence_score", 0)
          degree = self.graph.out_degree(node, weight="weight")

          if "Champions" in tags:
              champions.append((node, influence, degree))
          if any(tag in tags for tag in ["Economic Buyer", "Cross-Functional Reviewers"]):
              decision_makers.append((node, influence, degree))

      # Rank by composite: 0.7 * influence + 0.3 * degree
      champions.sort(key=lambda x: 0.7 * x[1] + 0.3 * x[2], reverse=True)
      decision_makers.sort(key=lambda x: 0.7 * x[1] + 0.3 * x[2], reverse=True)

      return {
          "Best Champion": champions[0] if champions else None,
          "Best Decision Maker": decision_makers[0] if decision_makers else None
      }

    def explain_path_to_decision_maker(self, path, llm_response=True):
      if not path or len(path) < 2:
          return "❌ No valid path to decision maker found."

      segments = []
      for i in range(len(path)):
          node = self.graph.nodes[path[i]]
          title = node.get("default_position_title", "Unknown Title")
          tags = ", ".join(node.get("tags", [])) or "No tags"
          org = node.get("role_enriched", {}).get("Org Unit", "Unknown Org")
          suborg = node.get("role_enriched", {}).get("Suborg Unit", "")
          influence_score = node.get("influence_score", "N/A")

          segments.append(f"**{i+1}. `{path[i]}` — {title}**  \n"
                          f"• Tags: {tags}  \n"
                          f"• Org: {org} / {suborg}  \n"
                          f"• Influence Score: {influence_score}\n")

          # Show edge weights if it's not the last node
          if i < len(path) - 1:
              edge = self.graph[path[i]][path[i+1]]
              segments.append(f"➡️ **Edge Influence Weight**: {edge.get('weight', 'N/A')}\n")

      if not llm_response:
          return "\n".join(segments)

      # Combine the markdown to send to LLM
      markdown_summary = "\n".join(segments)
      multithread_prompt_system = "You are an expert in multithreading strategies to target buyer accounts as an enterprise seller"
      multithread_prompt_user = f"""You're helping a seller understand the stakeholder landscape in a deal. Here's the path from a Champion to a Decision Maker:

                {markdown_summary}

                Summarize this path, explain the influence flow, and give 1 actionable next step the seller should take. Format the output in clean markdown.

                Sample response:

                ### 🧭 Path from Champion to Decision Maker

                **Champion**: Alice Rao — Director of Engineering  
                **Decision Maker**: Mark Lin — CIO, Economic Buyer  
                **Path Length**: 3 steps

                ---

                1. 👩‍💼 `Alice Rao` — Director of Engineering  
                  - Tags: Champion  
                  - Org: Engineering → Infra  
                  - Influence Score: 84

                2. 🔗 ➡️ `Sanjay Patel` — VP of Infra  
                  - Tags: Direct Owner Team  
                  - Shared Org Unit: Engineering  
                  - Weight: 0.83

                3. 🔗 ➡️ `Mark Lin` — CIO  
                  - Tags: Economic Buyer  
                  - Function: Strategic  
                  - Weight: 0.91

                ---

                💡 **Insight**: Strong chain from champion to CIO. Recommend asking Sanjay for an introduction to Mark.

                🧭 **Next Step**: Validate CIO’s priorities and map alignment to our value prop.



                """
      client = openai.OpenAI()

      response_explain_strategy = client.responses.create(
      model="gpt-4o",
      #reasoning={"effort": "medium"},
      input=[
          {
              "role": "system",
              "content": multithread_prompt_system
          },
          {
              "role": "user",
              "content": multithread_prompt_user
          },
        ]
      )
      print(response_explain_strategy.output_text)
      

    # Call your LLM here
    # Example: return openai.ChatCompletion.create(...), or however you handle LLM calls.
      return response_explain_strategy.output_text  # Replace with LLM call if needed





# give a list of people, filter duplicates and for each person, assign 

def remove_duplicates(people):
  all_people_names = {}
  all_people_deduped = []
  for person in people:
    if person["name"] in all_people_names:
      continue
    else:
      all_people_deduped.append(person)
      all_people_names[person["name"]] = True
  print(all_people_deduped)
  return all_people_deduped


def fetch_all_profiles( header,buyer, all_titles_to_search, max_pages=4,delay=1.0):
    headers = header
    all_profiles = []
    page_number = 1

    while True:
        print("running for page ")
        print(str(page_number))
        print("\n")
        if int(page_number) > int(max_pages):
          break
        params = header.copy()
        params['page_number'] = str(page_number)

        response = requests.post(
          "https://api.crustdata.com/screener/person/search",
          headers=params,
          json={
            "filters": [
              {
                "filter_type": "CURRENT_COMPANY",
                "type": "in",
                "value": [
                  buyer
                ]
              },
              {
                "filter_type": "CURRENT_TITLE",
                "type": "in",
                "value": all_titles_to_search
              },

            ],
            "page": str(page_number)
          }
        )



        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        profiles = data.get("profiles", [])
        total_display_count = int(data.get("total_display_count", 0))
        print("total Display Count")
        print(str(total_display_count))
        if not profiles:
            print(f"No more profiles returned at page {page_number}. Stopping.")
            break

        all_profiles.extend(profiles)
        print(f"Fetched {len(profiles)} profiles on page {page_number}.")

        if len(all_profiles) >= total_display_count:
            print("Fetched all profiles.")
            break

        page_number += 1
        #time.sleep(delay)  # optional delay to avoid rate limits

    return all_profiles

def get_graphs_for_initiatives(strategy_people_data_scored):
  graphs = {}
  a = strategy_people_data_scored
  all_people = {}
  for initiative in a:
    #print(initiative)
    all_people[initiative] = []
    for tag in a[initiative]:
      #print(tag)
      
      for teams in a[initiative][tag]:
        #print(teams["Team"])
        #print("\n")
        people_temp = copy.deepcopy(teams["people"])
        #people_temp = remove_duplicates(teams["people"])
        #print(people_temp
        for person in people_temp:
          person |= {"tag":tag}
          #print(person)
        all_people[initiative] += people_temp
        for person in teams["people"]:
          continue
          #print(person)

  for initiative in a:
    all_people[initiative] = remove_duplicates(copy.deepcopy(all_people[initiative]))


  for initiative in a:
    graph = StakeholderGraph(initiative, all_people[initiative])
    graphs[initiative] = graph
  return graphs


