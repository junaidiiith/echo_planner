# from echo.query_executor import aget_query_response
# from echo.query_executor import aget_query_response
import copy
import datetime
import math
import os
from collections import defaultdict

from loguru import logger
import networkx as nx
import openai
import plotly.graph_objects as go
import requests
from pydantic import BaseModel
from pyvis.network import Network
import tldextract


class StrategicInitiative(BaseModel):
    Buyer_Initiative_Or_Pain: str
    Buyer_Goals: str
    Seller_Alignment: str
    Relevance: str
    Is_Cross_Functional: bool
    IsStrategic: bool
    Timing: str  # planning, executing, evaluating)
    Buyer_Industry: str
    ExecAwareness: bool


class StrategicInitiatives(BaseModel):
    Initiatives: list[StrategicInitiative]


class Account_Plan(BaseModel):
    buyer: str
    seller: str
    Industry_Details: str
    Top_initiatives: str
    Triggers: str
    ICP_Fit: str
    Tech_stack_fit: str
    Value_Aligned: str
    User_Feedback: str
    Sources: str


class POV(BaseModel):
    Inititative: str
    Tag: str  # champion / dm/ exec buyer ect
    what_they_care_about: str
    triggers_and_motivations: str
    likely_objections: str
    emotional_drivers: str
    POV: str


def format_account_plan_markdown(plan: dict) -> str:
    def format_section(title: str, content: str) -> str:
        lines = content.strip().splitlines()
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("•") or line.startswith("- "):
                formatted_lines.append(f"- {line.lstrip('•').strip()}")
            elif line[:2].isdigit() and line[2:3] in {".", ")"}:
                formatted_lines.append(f"1. {line[3:].strip()}")
            else:
                formatted_lines.append(line)
        return f"## {title}\n" + "\n".join(formatted_lines) + "\n"

    def format_sources(sources):
        if not sources:
            return ""

        # If sources is a string, attempt to parse as list
        if isinstance(sources, str):
            try:
                import json

                sources = json.loads(sources)
                if not isinstance(sources, list):
                    sources = [sources]
            except Exception:
                sources = [sources]  # fallback to single string

        source_lines = []
        for i, src in enumerate(sources, 1):
            if isinstance(src, dict) and "title" in src and "url" in src:
                source_lines.append(f"{i}. [{src['title']}]({src['url']})")
            elif isinstance(src, str):
                source_lines.append(f"{i}. {src}")
        return "## 🔗 Sources\n" + "\n".join(source_lines)

    sections = [
        ("🏭 Industry Details", plan.get("Industry_Details", "")),
        ("🚀 Top Strategic Initiatives", plan.get("Top_initiatives", "")),
        ("⚡️ Triggers", plan.get("Triggers", "")),
        ("✅ ICP Fit", plan.get("ICP_Fit", "")),
        ("🔌 Tech Stack Fit", plan.get("Tech_stack_fit", "")),
        ("🎯 Value Alignment", plan.get("Value_Aligned", "")),
        ("📣 User Feedback", plan.get("User_Feedback", "")),
    ]

    formatted = [
        f"# 🧠 Account Plan for **{plan.get('buyer', '')}**",
        f"_Seller: {plan.get('seller', '')}_",
        "\n---\n",
    ]

    for title, content in sections:
        if content:
            formatted.append(format_section(title, content))

    if "Sources" in plan:
        formatted.append(format_sources(plan["Sources"]))

    return "\n".join(formatted).strip()


def format_stakeholder_pov(data: dict):
    def format_list(text):
        return "\n".join(
            f"- {line.strip()}" for line in text.strip().split("\n") if line.strip()
        )

    initiative = data.get("Inititative", "N/A")
    tag = data.get("Tag", "N/A").replace("_", " ")
    pov = data.get("POV", "N/A")

    care_about = format_list(data.get("what_they_care_about", ""))
    triggers = format_list(data.get("triggers_and_motivations", ""))
    emotions = format_list(data.get("emotional_drivers", ""))
    objections = format_list(data.get("likely_objections", ""))

    return f"""
## 🧠 Stakeholder Insight

**🎯 Initiative**  
{initiative}

**🏷️ Stakeholder Type**  
`{tag}`

---

### 💡 What They Care About
{care_about}

### 🚀 Triggers & Motivations
{triggers}

### ❤️ Emotional Drivers
{emotions}

### ❗ Likely Objections
{objections}

### 🗣️ Tailored POV
> {pov}
""".strip()


# Example usage
sample_input = {
    "Inititative": "Investment in Research and Development",
    "Tag": "Decision_Maker",
    "what_they_care_about": "- Rapidly onboarding new hires and up-skilling existing employees on emerging AI and R&D tools to drive platform enhancements\n- Achieving high engagement and adoption rates for both HR systems (onboarding, performance) and R&D applications with minimal support overhead\n- Success metric: measurable reduction in training time and support tickets (e.g., 30–50% faster time-to-productivity, 90%+ feature adoption)",
    "triggers_and_motivations": "- As a new VP, Tahlia needs early, visible wins to cement her credibility and demonstrate HR’s strategic value in Rippling’s R&D push\n- Avoids the risk of low adoption or stalled product launches that could reflect poorly on HR enablement and budget stewardship\n- Winning means being seen as the catalyst who accelerated both people and technology adoption—earning trust from the C-suite and R&D teams",
    "likely_objections": "- Skepticism over adding another tool on top of Rippling’s own platform or existing LMS—concerns about overlap and tool sprawl\n- Questions about integration with Rippling’s HRIS/data model, data security, compliance, and how ROI will be measured and reported\n- Worries about the effort required from HR (content creation, maintenance) to configure in-app guidance",
    "emotional_drivers": "- Fear: that as a newcomer she’ll be blamed if employees struggle with new tech or if training costs balloon\n- Pride: eager to prove herself as an innovative, people-first leader who modernizes HR processes and fuels company growth",
    "POV": "“Tahlia, Whatfix lets you launch in-app guidance for every new AI feature and HR tool without burdening your team—cut onboarding time by up to 50% and slash support tickets, so you can score quick wins that showcase HR’s impact on Rippling’s R&D success and protect your budget.”",
}


def export_graph_to_dict(G):
    tag_colors = {
        "Decision Maker": "#FF5733",
        "Economic Buyer": "#33FF57",
        "Champion": "#3357FF",
        "Influencer": "#F1C40F",
        "Blocker": "#E74C3C",
    }

    nodes = []
    for node, attrs in G.nodes(data=True):
        tag = attrs.get("tag", "Unknown")
        influence = round(attrs.get("influence_score", 0.0), 2)
        color = tag_colors.get(tag, "#CCCCCC")  # fallback color
        nodes.append(
            {
                "id": node,
                "label": node,
                "fill": color,
                "data": {"tag": tag, "influence": influence},
            }
        )

    edges = []
    for i, (source, target, attrs) in enumerate(G.edges(data=True), start=1):
        influence = round(attrs.get("weight", 0.0), 3)
        edges.append(
            {
                "id": str(i),
                "source": source,
                "target": target,
                "label": str(influence),
                "data": {"edge_influence": influence},
            }
        )

    return {"nodes": nodes, "edges": edges}


def get_yes_strategy(G, decision_maker, top_k_per_tag=1):
    tag_to_top_influencers = defaultdict(list)

    for node in G.nodes:
        if node == decision_maker or not nx.has_path(G, node, decision_maker):
            continue

        tag = G.nodes[node].get("tag", "Unknown")
        if tag in {"Decision_Maker", "Unknown"}:
            continue

        path = nx.shortest_path(G, source=node, target=decision_maker)
        influence_score = sum(
            G.get_edge_data(path[i], path[i + 1]).get("influence", 0)
            for i in range(len(path) - 1)
        )

        tag_to_top_influencers[tag].append(
            {
                "node": node,
                "name": G.nodes[node].get("label", node),
                "title": G.nodes[node].get("title", ""),
                "path": path,
                "influence": influence_score,
                "path_length": len(path),
            }
        )

    # Keep top-K per tag based on influence, then proximity
    for tag in tag_to_top_influencers:
        tag_to_top_influencers[tag] = sorted(
            tag_to_top_influencers[tag],
            key=lambda x: (-x["influence"], x["path_length"]),
        )[:top_k_per_tag]

    return tag_to_top_influencers


def format_yes_strategy_md(strategy_by_tag, decision_maker_label):
    md = f"## Who Influences Decision Maker **{decision_maker_label}**\n\n"
    md += "For each stakeholder type, target the most influential individuals who can sway the decision-maker:\n\n"

    for tag, influencers in strategy_by_tag.items():
        md += f"### {tag}\n"
        if influencers:
            for inf in influencers:
                md += f"- **{inf['name']}** ({inf['title']}): influence score **{inf['influence']}**, path: `{inf['path']}`\n"
        else:
            md += f"_No {tag.lower()} currently influencing Decision Maker._\n"
        md += "\n"

    return md


def format_buyer_initiatives_markdown(data: dict) -> str:
    def status_badge(status: str) -> str:
        return {
            "Executing": "🟢 **Executing**",
            "Planning": "🟡 **Planning**",
            "Delayed": "🔴 **Delayed**",
        }.get(status, f"📌 {status}")

    def relevance_icon(relevance: str) -> str:
        return {"High": "🔥 High", "Medium": "⚡ Medium", "Low": "🧊 Low"}.get(
            relevance, relevance
        )

    def bool_icon(value: bool) -> str:
        return "✅" if value else "❌"

    markdown = "## 🎯 Strategic Buyer Initiatives\n\n"
    markdown += "Summary of active initiatives where your solution aligns with strategic goals:\n\n---\n"

    for i, item in enumerate(data.get("Initiatives", []), 1):
        markdown += f"### {i}. 🚀 **{item.get('Buyer_Initiative_Or_Pain', 'Untitled Initiative')}**\n\n"

        markdown += f"- **Status:** {status_badge(item.get('Timing', 'Unknown'))}\n"
        markdown += f"- **Industry:** {item.get('Buyer_Industry', 'N/A')} | **Executive Awareness:** {bool_icon(item.get('ExecAwareness', False))}\n"
        markdown += f"- **Strategic:** {bool_icon(item.get('IsStrategic', False))} | **Cross-Functional:** {bool_icon(item.get('Is_Cross_Functional', False))}\n"
        markdown += (
            f"- **Relevance:** {relevance_icon(item.get('Relevance', 'Unknown'))}\n\n"
        )

        markdown += f"**🧭 Buyer Goals:**\n{item.get('Buyer_Goals', 'N/A')}\n\n"
        markdown += (
            f"**🤝 Seller Alignment:**\n{item.get('Seller_Alignment', 'N/A')}\n\n"
        )
        markdown += "---\n"

    return markdown


def get_POV(
    graph, name: str, buyer: str, seller: str, initiative: str, seller_info: str
):
    node = graph.nodes.get(name)
    social_research_prompt = f"""

  You are a top research assistant to find detailed social signals on any person in the buyer org a 
  seller is selling to. Find and summarize the content and social signals on {name} from the company {buyer}.
  Things to look for:
  Content they post about
  Companies they support or associate with 
  Tools they use right now in their role
  Their pains and goals theyve talked about publically

  Help the seller find what they post or care about, their driver and goals, their beliefs and tools and other comapnies they post or are in
  contact with. Understand what they like and dislike in the industry and role in their workplace as well.

  Get all of this from actual sources and dont guess any of this!!
  """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    # response_social = run_openai_query(social_research_prompt, use_tools=True)
    response_social = client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input=social_research_prompt,
    )
    print("Social signals for " + name + " = ")
    print(response_social.output_text)
    print("\n\n")

    account_research_system_prompt = f"""

  You are an enterprise sales AI assistant helping AEs understand how to tailor their approach to each stakeholder involved in a B2B deal. 
  Based on the input data below including social profule of the person, their experience in the company, the buyer initiative
  the tag(champion, dm) they play in the buying committee of {buyer}, generate a deep understanding 
  of the stakeholder's mindset,the product being pitched by {seller}, and how to frame a compelling, emotionally resonant Point of View (POV) message for them.
  Be specific and dont be generic. Use context given as much as possible! Dont make up facts but youre allowed to make a good hypotheiss!
  Take into account the seniority, the role and the tag in buying committee.Different roles need to be sold on
  one or a mix of different values -  Strategic value (growth, scale, revenue), Operational value (time, accuracy, workflow) or Personal value (visibility, security, career win
  You need to output the following

  what_they_care_about:
  - "<Key goals and outcomes that matter to this stakeholder>"
  - "<Their perceived success metric within this initiative>"

triggers_and_motivations:
  - "<Why this initiative matters to them personally or politically>"
  - "<Risks they might be trying to avoid>"
  - "<What winning means for them>"
  - "What kind of value - Strategic value (growth, scale, revenue), Operational value (time, accuracy, workflow) or Personal value (visibility, security, career win)"

likely_objections:
  - "<What would make them skeptical about a tool like this>"
  - "<What questions or blockers they might raise internally>"

emotional_drivers:
  - "<Fear or concern they don't want to admit>"
  - "<Pride or ambition they want to fulfill>"

recommended_pov:
  - "<1-2 sentence POV that frames the seller’s message in their language, tying personal gain to initiative success and avoiding risk>"
  """
    account_research_user_prompt = f"""
  here are your inputs:
  initiative details: {initiative}
  
stakeholder:
  name: {node.get("name", "")}
  title: {node.get("default_position_title", "")}
  org_unit: {node.get("role_enriched", {}).get("Org_Unit", "")}
  tenure: {node.get("tenure", 0)}
  seniority (1-7): {node.get("seniority", 0)}
  relevance_to_initiative: {node.get("reason", "")}
  social_signals: {response_social.output_text}
  tag: {node.get("tag", "")}

Seller Product Details:
{seller_info}
  
  """

    response_account_research = client.responses.parse(
        model="o4-mini",
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": account_research_system_prompt},
            {"role": "user", "content": account_research_user_prompt},
        ],
        text_format=POV,
    )
    return response_account_research.output_parsed


def account_plan_extract_data(account_research_data: str, buyer, seller):
    account_research_system_prompt = f"""
  You are a strategic account executive selling {seller} products to {buyer}. You need to extract signals from the data provided below on {buyer} and create a detailed account plan for the buyer {buyer}.
  You need to extract the following data points from the data provided below:
  1. Top initiatives and goals of the buyer this year
  2. Industry Details - (what is the industry of the buyer, what are the trends in the industry, what are the challenges in the industry, etc.)
  2. Triggers for the buyer - (funding, new product launches, hiring trends, leadership changes, etc.)
  3. ICP Fit - (right type of company for {seller}, correct initiatives to solve for {buyer}, are they feeling the pains the seller can solve for , etc.) - Low, Meidium, High with a reasoning.
  4. Tech stack  - (is the buyer's tech stack aligned with the sellers product, are they using similar products, what gaps do w) - Low, Medium, High with a reasoning.
  5. Value Alignment - (low,medium, high along with reasoning) - List how the seller can solve for each of the buyers initiatives above and how well positioned are the seller to solve it. keep this detailed.
  6. User Feedback - (what are the users saying about the product, what are the reviews, what are the customers saying about the product, etc.)
  7. Sources - All the links and sources for above
  
  """
    account_research_user_prompt = f"""
  here is the data on {buyer} and {seller} - {account_research_data}
  """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    response_account_research = client.responses.parse(
        model="o4-mini",
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": account_research_system_prompt},
            {"role": "user", "content": account_research_user_prompt},
        ],
        text_format=Account_Plan,
    )
    return response_account_research.output_parsed


def create_value_prop_pydantic(buyer, seller, buyer_initiatives, seller_info):
    value_align_prompt_system = f"""You are a strategic sales executive at {seller}\n
  You are trying to find the initiatives of the buyer {buyer} and align them to your product."""

    value_align_prompt_user = f"""The top initiatives of the buyer and the details of your product are given. Of all initiatives, find the most relevant ones
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
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    response_value_prop = client.responses.parse(
        model="o4-mini",
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": value_align_prompt_system},
            {"role": "user", "content": value_align_prompt_user},
        ],
        text_format=StrategicInitiatives,
    )

    return response_value_prop.output_parsed


def create_value_prop(
    buyer: str, seller: str, buyer_initiatives: str, seller_info: str
):
    value_align_prompt_system = f"""You are a strategic sales executive at {seller}\n
  You are trying to find the initiatives of the buyer {buyer} and align them to your product."""

    value_align_prompt_user = f"""The top initiatives of the buyer and the details of your product are given. Of all initiatives, find the most relevant ones
  the sellers product can help solve and achieve. rate the relevance and problem solution fit as well. if all seem important, its okay to mark as relevant but make sure you assess properly.
  Output format in json:
  Buyer Initiative or pain
  Why the seller can help? - be specific and dont force fit.
  Relevance of seller to solving the pain or initiative
  "Is_Cross_Functional" :true / false based if its a broad initiative like AI adoption that will span multiple teams. Initiatives like better CSAT or onboarding is not cross functional as its owned entirely by one org unit.
  "IsStrategic":true or false based on if the initiative is strategic or more tactical and operational.
  "Timing":  planning, executing, evaluating along with reasoning of why based on maturity of initiative from signals like hiring now, talks about maturity, implementation of it going on, exploring vendors and partners along with reasoning of why.
  "Buyer_Industry":Industry of buyer - healthtech, pharma, Saas etc
  "ExecAwareness":true / false based on if execs are aware of it or not and need to be brought into.

  Sample output:
  {{
    "Strategic Initiatives": [
      {{
        "Buyer Initiative or pain": "International Expansion (hiring 100+ engineers in Bengaluru, scaling globally)",
        "What is the buyer trying to achieve":"Product Expansion, Faster Ship speed",
        "Why the seller can help?": "Whatfix’s in-app guidance and onboarding flows accelerate time‑to‑productivity for new hires across geographies. Localized self‑help modules, task lists, and smart tips ensure consistent training and process adherence, reducing reliance on instructor‑led sessions and enabling Rippling to scale its workforce efficiently in India and beyond.",
        "Relevance of seller to solving the pain or initiative": "High"
        "Is_Cross_Functional" :true / false based on if initiative is huge and cross fucntional and not limited to a single org unit
        "IsStrategic":true or false based on if the initiative is strategic or more tactical and operational.
  "Timing":  planning, executing, evaluating along with reasoning of why based on maturity of initiative from signals like hiring now, talks about maturity, implementation of it going on, exploring vendors and partners along with reasoning of why.
        "Buyer_Industry":Industry of buyer - healthtech, pharma, Saas etc
        "ExecAwareness":true / false based on if execs are aware of it or not and need to be brought into.
      }},
      {{
        "Buyer Initiative or pain": "Investment in R&D (building new products, integrating AI, enhancing existing platform)",
        "What is the buyer trying to achieve":"Competitor differentiation",
        "Why the seller can help?": "Whatfix streamlines internal adoption of newly developed tools and features by embedding contextual walkthroughs and in‑app prompts. This ensures Rippling’s engineers and early adopters can instantly learn and validate new product capabilities, accelerating feedback loops and reducing friction in beta testing and rollout phases.",
        "Relevance of seller to solving the pain or initiative": "Medium"
        "Is_Cross_Functional" :true / false based on if initiative is huge and cross fucntional and not limited to a single org unit
        "IsStrategic":true or false based on if the initiative is strategic or more tactical and operational.
  "Timing":  planning, executing, evaluating along with reasoning of why based on maturity of initiative from signals like hiring now, talks about maturity, implementation of it going on, exploring vendors and partners along with reasoning of why.
        "Buyer_Industry":Industry of buyer - healthtech, pharma, Saas etc
        "ExecAwareness":true / false based on if execs are aware of it or not and need to be brought into.
      }},
      {{
        "Buyer Initiative or pain": "Enhancing IT Security and Automation (data privacy, compliance, identity & device management, reducing manual tasks)",
        "What is the buyer trying to achieve":"Security, Lesser support tickets",
        "Why the seller can help?": "Whatfix embeds real‑time guidance and compliance checks directly within Rippling’s IT and security workflows. By guiding users through standardized processes, enforcing policy steps via task lists, and capturing process analytics, Whatfix helps minimize human error, automate repetitive tasks, and strengthen overall security posture.",
        "Relevance of seller to solving the pain or initiative": "High"
        "Is_Cross_Functional" :true / false based on if initiative is huge and cross fucntional and not limited to a single org unit
        "IsStrategic":true or false based on if the initiative is strategic or more tactical and operational.
  "Timing":  planning, executing, evaluating along with reasoning of why based on maturity of initiative from signals like hiring now, talks about maturity, implementation of it going on, exploring vendors and partners along with reasoning of why.
        "Buyer_Industry":Industry of buyer - healthtech, pharma, Saas etc
        "ExecAwareness":true / false based on if execs are aware of it or not and need to be brought into.
      }}
    ]
  }}

  Here is the initiative on buyer
  {buyer_initiatives}

  Here is the sellers details and pains they solve for their customers
  {seller_info}

  Output only the json and nothing else. Dont make anything up and answer from context provided.
  """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    response_value_prop = client.responses.parse(
        model="o4-mini",
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": value_align_prompt_system},
            {"role": "user", "content": value_align_prompt_user},
        ],
        text_format=StrategicInitiatives,
    )

    return response_value_prop.output_parsed


class Team(BaseModel):
    team: str
    Relevance: str


class Decision_Maker(BaseModel):
    teams: list[Team]


class Economic_Buyer(BaseModel):
    teams: list[Team]


class Champions(BaseModel):
    teams: list[Team]


class Influencers(BaseModel):
    teams: list[Team]


class Blockers(BaseModel):
    teams: list[Team]


class BuyerCommittee(BaseModel):
    Buyer_Initiative_Or_Pain: str
    Seller_Alignment: str
    Relevance: str
    Decision_Maker: Decision_Maker
    Economic_Buyer: Economic_Buyer
    Blockers: Blockers
    Influencers: Influencers
    Champions: Champions


class BuyerCommittees(BaseModel):
    Initiatives: list[BuyerCommittee]


def find_teams(
    buyer: str, seller: str, response_value_prop: StrategicInitiatives, seller_info: str
):
    best_fit_team_system_prompt = f"""

  You are a sales executive at {seller} who is an expert at mapping the buying committees for different buyer initiatives.
  You have the top relevant initiatives of buyer {buyer} and info on how the seller can solve these. For each initiative thats relevant (dont pick poor relevant ones),
  Ypu need to predict the possible buyer committee responsible for the initiative.
  This is how to work through:
  For each initiative:
  1. Identify teams split by tag where each team belongs to one of these

  "Accounting", "Administrative", "Arts and Design", "Business Development", "Community and Social Services", "Consulting", "Education", "Engineering", "Entrepreneurship", "Finance", "Healthcare Services", "Human Resources", "Information Technology", "Legal", "Marketing", "Media and Communication", "Military and Protective Services", "Operations", "Product Management", "Program and Project Management", "Purchasing", "Quality Assurance", "Real Estate", "Research", "Sales", "Customer Success and Support"




    - Decision Maker - Based on the initiative, find which team owns the initiative.

    - Economic Buyer: Based on the initiative, find which team is the cost center. If the initiative is too strategic and
      involves multiple teams, finance will be the cost center for the initiative.

    - Blockers :This involves every team thats risk averse and can be badly impacted
      by the initiative and sellers tool. Think IT for high integration tools,
      compliance if the product is highly regulatory in nature and industry.
      Make sure that this isnt the same as the team that directly uses the tool.

    - Influencers: This involves both direct teams that are consulted by the decision maker and have a say in the initiative
    as well as cross functional teams that would have a say in the decision making process for the initiative.
      Say RevOps and enablement for sales tools as well. Dont be too broad.

    - Champions: The team that directly feels the pain and needs it solved.

  2. For each team:
    justify why they would play the role in the buying committee.

  3. Teams can be both decision makers and champions for example. There shouldnt be too much overlap though.

  Sample output:
  {{
    "Strategic Initiatives": [
      {{
        "Buyer Initiative or pain": "International Expansion (hiring 100+ engineers in Bengaluru, scaling globally)",
        "Why the seller can help?": "Whatfix’s in-app guidance and onboarding flows accelerate time‑to‑productivity for new hires across geographies. Localized self‑help modules, task lists, and smart tips ensure consistent training and process adherence, reducing reliance on instructor‑led sessions and enabling Rippling to scale its workforce efficiently in India and beyond.",
        "Relevance of seller to solving the pain or initiative": "High",
        {{
    "Decision Maker": [
      {{"team": "Sales", "Relevance":"why its relevant"}}
    ],
    "Economic Buyer": [
      {{"team": "Finance", "Relevance":"why its relevant"}}
    ],
    "Champions": [
      {{"team": "IT", "Relevance":"why its relevant"}},
      {{"team": "Legal", "Relevance":"why its relevant"}},
      {{"team": "Procurement", "Relevance":"why its relevant"}}
    ],
    "Influencers": [
      {{"team": "Customer Success", "Relevance":"why its relevant"}},
      {{"team": "RevOps", "Relevance":"why its relevant"}}
    ],
    "Blockers": [
      {{"team": "Sales", "Relevance":"why its relevant"}},
      {{"team": "Enablement", "Relevance":"why its relevant"}}
    ]
  }}
      }},
      {{
        "Buyer Initiative or pain": "Investment in R&D (building new products, integrating AI, enhancing existing platform)",
        "Why the seller can help?": "Whatfix streamlines internal adoption of newly developed tools and features by embedding contextual walkthroughs and in‑app prompts. This ensures Rippling’s engineers and early adopters can instantly learn and validate new product capabilities, accelerating feedback loops and reducing friction in beta testing and rollout phases.",
        "Relevance of seller to solving the pain or initiative": "Medium",
        {{
    "Decision Maker": [
      {{"team": "Sales","Relevance":"why its relevant"}}
    ],
    "Economic Buyer": [
      {{"team": "Finance", "Relevance":"why its relevant"}}
    ],
    "Champions": [
      {{"team": "IT", "Relevance":"why its relevant"}},
      {{"team": "Legal", "Relevance":"why its relevant"}},
      {{"team": "Procurement", "Relevance":"why its relevant"}}
    ],
    "Influencers": [
      {{"team": "Customer Success", "Relevance":"why its relevant"}},
      {{"team": "RevOps", "Relevance":"why its relevant"}}
    ],
    "Blockers": [
      {{"team": "Sales", "Relevance":"why its relevant"}},
      {{"team": "Enablement","Relevance":"why its relevant"}}
    ]
  }}
      }},
    ]
  }}

  Make sure again the teams is one of the ones mentioned ONLY!
  """

    best_fit_team_user_prompt = f"""
  here are the initiaitves identified for the buyer and how well the seller {seller} fits to solve them -        
    {str(response_value_prop.model_dump())}
  Output just the json and nothing else.

  Here's details on the sellers' product and pains they solve for - {seller_info}
  """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    response_team_guess_reasoning = client.responses.parse(
        model="o4-mini",
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": best_fit_team_system_prompt},
            {"role": "user", "content": best_fit_team_user_prompt},
        ],
        text_format=BuyerCommittees,
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
    return response_team_guess_reasoning.output_parsed


def get_people_data(buyer, team_data: dict):
    people_data = {}

    for initiative in team_data["Initiatives"]:
        initiative_people_data = []
        decision_makers = initiative["Decision_Maker"]["teams"]
        economic_buyers = initiative["Economic_Buyer"]["teams"]
        blockers = initiative["Blockers"]["teams"]
        influencers = initiative["Influencers"]["teams"]
        champions = initiative["Champions"]["teams"]

        # do this instead of below

        initiative["Decision_Maker"]["people"] = run_crustdata_query(
            buyer, "Decision Maker", [team["team"] for team in decision_makers]
        )["profiles"]
        initiative["Economic_Buyer"]["people"] = run_crustdata_query(
            buyer, "Economic Buyer", [team["team"] for team in decision_makers]
        )["profiles"]
        initiative["Blockers"]["people"] = run_crustdata_query(
            buyer, "Blockers", [team["team"] for team in decision_makers]
        )["profiles"]
        initiative["Influencers"]["people"] = run_crustdata_query(
            buyer, "Influencers", [team["team"] for team in decision_makers]
        )["profiles"]
        initiative["Champions"]["people"] = run_crustdata_query(
            buyer, "Champions", [team["team"] for team in decision_makers]
        )["profiles"]

    return team_data
    # return people_data


def run_crustdata_query(
    company,
    tag,
    departments=[],
    tenure=["3 to 5 years", "6 to 10 years", "More than 10 years"],
):
    seniority_map = {
        "Decision Maker": ["CXO", "Vice President"],
        "Economic Buyer": ["Vice President"],
        "Champions": ["Experienced Manager", "Director"],
        "Blockers": ["Directors", "Vice President"],
        "Influencers": ["Experienced Manager", "Senior"],
    }
    if tag == "Influencers":
        response = requests.post(
            "https://api.crustdata.com/screener/person/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Token 8582455305237735a32d0be5b74dda9b22dc9857",
            },
            json={
                "filters": [
                    {
                        "filter_type": "CURRENT_COMPANY",
                        "type": "in",
                        "value": [company],
                    },
                    {"filter_type": "FUNCTION", "type": "in", "value": departments},
                    {
                        "filter_type": "SENIORITY_LEVEL",
                        "type": "in",
                        "value": seniority_map[tag],
                    },
                    {
                        "filter_type": "YEARS_AT_CURRENT_COMPANY",
                        "type": "in",
                        "value": tenure,
                    },
                ],
                "page": 1,
            },
        )
    else:
        print(company)
        print(departments)
        print(seniority_map[tag])
        response = requests.post(
            "https://api.crustdata.com/screener/person/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Token 8582455305237735a32d0be5b74dda9b22dc9857",
            },
            json={
                "filters": [
                    {
                        "filter_type": "CURRENT_COMPANY",
                        "type": "in",
                        "value": [company],
                    },
                    {"filter_type": "FUNCTION", "type": "in", "value": departments},
                    {
                        "filter_type": "SENIORITY_LEVEL",
                        "type": "in",
                        "value": seniority_map[tag],
                    },
                ],
                "page": 1,
            },
        )
    try:
        response.raise_for_status()
        data = response.json()
        print(data)
        return data
    except Exception as e:
        logger.error(
            f"Error in run_crustdata_query: {e}, company: {company}, tag: {tag}, departments: {departments}, tenure: {tenure}"
        )
        return {
            "profiles": [],
            "total": 0,
        }
    # use tenure for influencers only


# then use direct people enrichment here


class Person_Enrichment(BaseModel):
    Org_Unit: str
    SubOrg_Unit: str
    Seniority_Level: int
    Function_Type: str


def get_current_tenure(employer_data):
    tenure = 0.0
    for experience in employer_data:
        if not experience.get("start_date"):
            continue
        if not experience.get("end_date"):
            start_date = datetime.datetime.fromisoformat(experience["start_date"])
            end_date = datetime.datetime.fromisoformat(
                datetime.datetime.now().isoformat()
            )
            delta = end_date - start_date
            tenure += int(delta.days)
    return tenure // 365


def get_industry_experience(employer_data):
    experience_days = 0
    for experience in employer_data:
        logger.info(experience)
        if not experience.get("start_date"):
            continue
        start_date = datetime.datetime.fromisoformat(experience["start_date"])
        end_date = datetime.datetime.fromisoformat(
            experience.get("end_date") or datetime.datetime.now().isoformat()
        )
        # Compute the difference
        delta = end_date - start_date
        experience_days += int(delta.days)
        # Get the number of days, years, etc.
        # print("Difference in days:", delta.days)

    return experience_days // 365


def return_dummy_people():
    return [
        {
            "name": "Gricelda V.",
            "location": "Netherlands",
            "linkedin_profile_url": "https://www.linkedin.com/in/ACwAAAMAopIBMOK1jg6smiZdVZbMTZTZgTGAQZE",
            "linkedin_profile_urn": "ACwAAAMAopIBMOK1jg6smiZdVZbMTZTZgTGAQZE",
            "default_position_title": "Non Executive Director",
            "default_position_company_linkedin_id": "96821337",
            "default_position_is_decision_maker": True,
            "flagship_profile_url": "https://www.linkedin.com/in/gricelda-v-242a5314",
            "profile_picture_url": None,
            "headline": "Global Head HR, Recruitment & Systems Operations (EMEA, LATAM & APAC) @ Scale-ups, Start-ups,SaaS platforms",
            "summary": "Lawyer with 15+ years experience managing and leading International Global HR Operations & Payroll Teams in  APAC, EMEA, Americas (including LATAM & Canada). \n- Specialization in leading HR Operations Strategy, Company Transformations Globally, including USA.\n- Global leader in Compensation & Benefits, Talent Acquisition/Recruitment, Change Management, Payroll processes in 25+ countries worldwide \n-SaaS platforms, EOR HR Internalization Strategy, Entities' setup, creating SOP’s, SOW’s, employee relationship management\n- Labor Relations management across responsible Regions\n- Experience implementing various HRM Systems and employee engagement tools \n- Experience in various industries: High Tech, Start-ups, Scale-ups, SaaS, AI, Oil & Gas, PEO/EOR \n ",
            "num_of_connections": 1781,
            "related_colleague_company_id": 96821337,
            "skills": [
                "Acquisitions",
                "Acquisition Integration",
                "Strategic advice to engineers to build HR SaaS",
                "System Development",
                "Performance Management",
                "Executive Development",
                "Internal Investigations",
                "Employment Tribunal",
                "Dispute Resolution",
                "Scale up",
                "Software as a Service (SaaS)",
                "Artificial Intelligence (AI)",
                "senior management",
                "director",
                "Executive Management",
                "recruitment",
                "Leadership",
                "Project Management",
                "HR Management",
                "Human Resources Information Systems (HRIS)",
                "Employment Law Compliance",
                "Change Management",
                "Works Council",
                "European Works Councils",
                "Software Testing",
                "Pension",
                "HR Consulting",
                "Group Restructuring and Reorganization",
                "Total Rewards Strategies",
                "Strategic Recruitment Planning",
                "Employee Handbooks",
                "Employee Engagement",
                "Employee Learning & Development",
                "Career Development",
                "Employee Benefits Design",
                "Training",
                "International Recruitment",
                "Processes Development",
                "Operations Management",
                "Payroll Management",
                "International Law",
                "Arbo",
                "HR Project Management",
                "Succession Planning",
                "Strategic Planning",
                "HR Strategy",
                "Legal Compliance",
                "Legal Advice",
                "Compensation & Benefits",
                "Labor and Employment Law",
                "Onboarding and offboarding processes",
                "Strategic Human Resource Planning",
                "Team Leadership",
                "Senior manager",
                "HR Operations",
                "Employee Benefits",
                "Joint Ventures",
                "Mergers & Acquisitions (M&A)",
                "New Entity Setup",
                "Professional Employer Organization (PEO)",
                "Employer Of Record",
                "Start-ups",
                "HR Transformation",
                "Organizational Development",
                "Personal Development",
                "Talent Acquisition",
                "Job Description Creation",
                "International Exposure",
                "English-Spanish",
                "Compliance",
                "HR Policy",
                "Human Resources",
                "Implementation of HRM systems",
                "Planning Budgeting & Forecasting",
                "30% ruling",
            ],
            "employer": [
                {
                    "title": "Non Executive Director",
                    "company_name": "Globalize HR",
                    "company_linkedin_id": "96821337",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQEZIQp9OoWpVQ/company-logo_400_400/company-logo_400_400/0/1686136989258?e=1752105600&v=beta&t=cbMIbwE4vrypfBGDBd5qxpRRsSdU0G1iln9NGi43Wjg",
                    "start_date": "2023-03-01T00:00:00",
                    "end_date": None,
                    "position_id": 2196886893,
                    "description": None,
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": " Global EOR Operations  (HR, SaaS Analysis)",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2023-06-01T00:00:00",
                    "end_date": None,
                    "position_id": 2196889670,
                    "description": "Global Operations Strategy, Legal, HR and SaaS platform analysis/compliance ",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": " Head of People Operations  (Short Project to Launch USA Ops)",
                    "company_name": "All Options",
                    "company_linkedin_id": "53969",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C4D0BAQGGFMKZA5a4hQ/company-logo_400_400/company-logo_400_400/0/1631307271803?e=1752105600&v=beta&t=dlZ74-4RiZXGchgmE8Qszw_OGheR0LPbPnvIlaEXXBA",
                    "start_date": "2023-03-01T00:00:00",
                    "end_date": "2023-05-01T00:00:00",
                    "position_id": 2139440255,
                    "description": "• Led a project to setup, build and rollout HR infrastructure for the launch of US Operations in Austin, TX for All Options.\n• Developed and implemented processes to ensure smooth operations and compliance with local regulations.\n• Collaborated with cross-functional teams to streamline onboarding and training processes for new employees.",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Global Sr. HR Manager - APAC, EMEA, Americas (LATAM & Canada)",
                    "company_name": "AngioDynamics",
                    "company_linkedin_id": "29698",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQFIYlL15S4B5w/company-logo_400_400/company-logo_400_400/0/1680291922892/angiodynamics_logo?e=1752105600&v=beta&t=WCH3hE_rr2605xIUOVlOUm11bTU3epC0CFoyzEeWl0o",
                    "start_date": "2021-12-01T00:00:00",
                    "end_date": "2023-03-01T00:00:00",
                    "position_id": 1877958506,
                    "description": "Member of International Leadership Team Leading Global HR Operations OUS: EMEA, APAC, Americas (LATAM & Canada) at AngioDynamics International ",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Global Head HR Operations ( APAC, EMEA, Americas (LATAM & Canada)",
                    "company_name": "Velocity Global, LLC",
                    "company_linkedin_id": "3681130",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQFPph4bEZgeQA/company-logo_400_400/company-logo_400_400/0/1686802877723/velocity_global_llc_logo?e=1752105600&v=beta&t=XwKM88GuM90ezG4TX_Frs0tnTlBLJnRnYBxuswuDdis",
                    "start_date": "2020-06-01T00:00:00",
                    "end_date": "2021-11-01T00:00:00",
                    "position_id": 1626754056,
                    "description": "• Led global HR operations and payroll teams in APAC, EMEA, and Americas, focusing on process transformation and organizational alignment.\n• Advised ELT & Regional MDs on global and regional HR strategies, statutory labor compliance, M&A, and labor union engagement.\n• Managed labor relationships with unions worldwide, ensuring compliance with labor contracts and practices.",
                    "location": "Amsterdam, North Holland, Netherlands",
                    "rich_media": [],
                },
                {
                    "title": "Global Head- HR Operations (EMEA, APAC, LATAM)",
                    "company_name": "Gazprom International",
                    "company_linkedin_id": "818385",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQEdbx38VQe0sg/company-logo_400_400/company-logo_400_400/0/1631311744746?e=1752105600&v=beta&t=QnJp3qjkNb9uakxS_G1v2v3Hy20DdFmcMenChX6Dc10",
                    "start_date": "2013-07-01T00:00:00",
                    "end_date": "2020-05-01T00:00:00",
                    "position_id": 497895660,
                    "description": "• Global Managing role leading HR advisory & strategy for Gazprom Group Companies, including Global Payroll Management, Recruitment, Change Management, Compensation & Benefits strategies.\n• Spearheaded Talent Acquisition campaigns and participated in company-wide reorganization and M&A activities in EMEA.\n• Member of Global International Leadership Team for Wintershall Noordzee and Gazprom Group.",
                    "location": "Amsterdam Area, Netherlands",
                    "rich_media": [],
                },
                {
                    "title": "Senior International Recruitment & HR Consultant @International Desk",
                    "company_name": "Martin Ward Anderson ",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2008-07-01T00:00:00",
                    "end_date": "2013-06-01T00:00:00",
                    "position_id": 987127792,
                    "description": "Senior Recruitment/HR Consultant @ Martin Ward Anderson (A Randstad Company) performing high-level Recruitment (Finance and Tax) and HR consultancy for International Companies in the Netherlands (Retail, Energy, Oil & Gas and IT industries). On the job training, coaching and leading junior consultants @ International level.",
                    "location": "Amsterdam Area, Netherlands",
                    "rich_media": [],
                },
                {
                    "title": "International HR & Recruitment Manager(New Markets)",
                    "company_name": "Hong Kong (APAC), London and USA",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2003-01-01T00:00:00",
                    "end_date": "2006-01-01T00:00:00",
                    "position_id": 1316384006,
                    "description": "As the International HR & Recruitment Project Director, I spearheaded the setup and rollout of HR and Recruitment infrastructure in new markets, ensuring legal compliance and efficient personnel management. I collaborated with cross-functional teams to streamline processes and develop effective recruitment strategies.",
                    "location": None,
                    "rich_media": [],
                },
            ],
            "education_background": [
                {
                    "degree_name": "Master of Laws - LLM",
                    "institute_name": "Utrecht University",
                    "field_of_study": "Public International Law - Iuris Publiciti Internationalis",
                    "start_date": None,
                    "end_date": None,
                    "institute_linkedin_id": "166740",
                    "institute_linkedin_url": "https://www.linkedin.com/school/166740/",
                    "institute_logo_url": "https://media.licdn.com/dms/image/v2/C510BAQG6HfmPTdJKXw/company-logo_400_400/company-logo_400_400/0/1631328481154?e=1752105600&v=beta&t=O3_VBlpSeCwBMlWDSUVir1ucUv-GtGivWqhEFJybVO8",
                },
                {
                    "degree_name": "Bachelor of Science - BSc",
                    "institute_name": "Political Science and Government",
                    "field_of_study": "",
                    "start_date": None,
                    "end_date": None,
                    "institute_linkedin_id": None,
                    "institute_linkedin_url": None,
                    "institute_logo_url": None,
                },
            ],
            "emails": [],
            "websites": [],
            "twitter_handle": None,
            "languages": ["English", "Spanish", "Dutch"],
            "pronoun": None,
            "query_person_linkedin_urn": "ACwAAAMAopIBMOK1jg6smiZdVZbMTZTZgTGAQZE",
            "linkedin_slug_or_urns": [
                "gricelda-v-242a5314",
                "ACwAAAMAopIBMOK1jg6smiZdVZbMTZTZgTGAQZE",
            ],
            "current_title": " Global EOR Operations  (HR, SaaS Analysis)",
        },
        {
            "name": "Tahlia Spiegel",
            "location": "San Francisco, California, United States",
            "linkedin_profile_url": "https://www.linkedin.com/in/ACwAAAE_V1YBnB_dvg8sEe0tJz4l4LcWFuBrdNc",
            "linkedin_profile_urn": "ACwAAAE_V1YBnB_dvg8sEe0tJz4l4LcWFuBrdNc",
            "default_position_title": "VP, Human Resources",
            "default_position_company_linkedin_id": "17988315",
            "default_position_is_decision_maker": False,
            "flagship_profile_url": "https://www.linkedin.com/in/tahlia-spiegel-3860137",
            "profile_picture_url": "https://media.licdn.com/dms/image/v2/D5603AQEWAG943kG-dQ/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1707703384350?e=1752105600&v=beta&t=p1inV8S3N293E-wQBZ-xdjYUKSspROhl2zYSEInL1Zs",
            "headline": "VP, Human Resources at Rippling",
            "summary": "Human Resources Leader with proven ability to develop teams & grow scaling tech companies via effective HR programs & strategic business partnership.",
            "num_of_connections": 3854,
            "related_colleague_company_id": 17988315,
            "skills": [
                "Human Resources",
                "Program Management",
                "Performance Management",
                "Programming",
                "Executive Search",
                "Recruiting",
                "HR Policies",
                "Onboarding",
                "Employee Benefits Design",
                "E-Learning",
                "Benefits Administration",
                "Human Resources Information Systems (HRIS)",
                "Leadership",
                "Management",
                "Sourcing",
                "People Management",
                "People Development",
                "Communication",
                "Employee Relations",
                "Employee Training",
                "Offboarding",
                "Employee Learning & Development",
                "Employee Wellness",
                "Employee Handbooks",
                "Organizational Learning",
                "Leave Administration",
            ],
            "employer": [
                {
                    "title": "VP, Human Resources",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2024-10-01T00:00:00",
                    "end_date": None,
                    "position_id": 2506840841,
                    "description": "Global HR Business Partners, HR Programs, and Workplace",
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "Senior Director, Human Resources",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2023-08-01T00:00:00",
                    "end_date": "2024-10-01T00:00:00",
                    "position_id": 2241483602,
                    "description": None,
                    "location": "San Francisco, California, United States",
                    "rich_media": [],
                },
                {
                    "title": "Director, Human Resources",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2022-07-01T00:00:00",
                    "end_date": "2023-08-01T00:00:00",
                    "position_id": 2007805813,
                    "description": None,
                    "location": "Los Angeles Metropolitan Area",
                    "rich_media": [],
                },
                {
                    "title": "VP, People",
                    "company_name": "PlayVS",
                    "company_linkedin_id": "18600748",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQH01JTeXKt04g/company-logo_400_400/company-logo_400_400/0/1658090880915/playversus_logo?e=1752105600&v=beta&t=5FOJ4pUORIYW2HTQzWRhdDuCC2mAja11yhX3L2JyzH4",
                    "start_date": "2020-06-01T00:00:00",
                    "end_date": "2022-07-01T00:00:00",
                    "position_id": 1627354317,
                    "description": "HR, People Operations & Programs, Total Rewards, & Recruiting",
                    "location": "Los Angeles, California, United States",
                    "rich_media": [],
                },
                {
                    "title": "Human Resources",
                    "company_name": "Snap Inc.",
                    "company_linkedin_id": "15191764",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQGS67pnekd_qQ/company-logo_400_400/company-logo_400_400/0/1706902360485/snap_inc_co_logo?e=1752105600&v=beta&t=C5QsdEs7jN28kcoyWLmIknl46lRhhlS4-PkDwtbm2Q0",
                    "start_date": "2018-05-01T00:00:00",
                    "end_date": "2020-06-01T00:00:00",
                    "position_id": 1410807610,
                    "description": "Product & Engineering",
                    "location": "Los Angeles, California",
                    "rich_media": [],
                },
                {
                    "title": "Recruiting",
                    "company_name": "Snap Inc.",
                    "company_linkedin_id": "15191764",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQGS67pnekd_qQ/company-logo_400_400/company-logo_400_400/0/1706902360485/snap_inc_co_logo?e=1752105600&v=beta&t=C5QsdEs7jN28kcoyWLmIknl46lRhhlS4-PkDwtbm2Q0",
                    "start_date": "2017-01-01T00:00:00",
                    "end_date": "2018-05-01T00:00:00",
                    "position_id": 1211555541,
                    "description": "Go-to-market Teams",
                    "location": "Greater New York City Area",
                    "rich_media": [],
                },
                {
                    "title": "Senior Recruiting Partner",
                    "company_name": "SoundCloud",
                    "company_linkedin_id": "200200",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4D0BAQEB1c304FdRcQ/company-logo_400_400/company-logo_400_400/0/1719255466674/soundcloud_logo?e=1752105600&v=beta&t=7p2y1UqMcF_FuXxq6TwvEnXEa39kw0jBcRn6sALNB8w",
                    "start_date": "2016-08-01T00:00:00",
                    "end_date": "2016-12-01T00:00:00",
                    "position_id": 843695215,
                    "description": "Sales & Engineering",
                    "location": "Greater New York City Area",
                    "rich_media": [],
                },
                {
                    "title": "Head of Talent & People",
                    "company_name": "Reonomy",
                    "company_linkedin_id": "792049",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQHrnq7zwA4aXw/company-logo_400_400/company-logo_400_400/0/1730561543028/reonomy_logo?e=1752105600&v=beta&t=sANNat3fCG_ynarumDVIxDtQvgmZs8xiR5SWMTy1x4k",
                    "start_date": "2015-01-01T00:00:00",
                    "end_date": "2016-08-01T00:00:00",
                    "position_id": 634387470,
                    "description": "HR, People Ops, & Recruiting",
                    "location": "Greater New York City Area",
                    "rich_media": [],
                },
                {
                    "title": "Director of Recruitment",
                    "company_name": "Betts Recruiting",
                    "company_linkedin_id": "626840",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQEG4R7AfNmwzw/company-logo_400_400/company-logo_400_400/0/1630616481523/betts_logo?e=1752105600&v=beta&t=VMl5IqoGtAVw1b2JfZVNjahz7g1qQDx9BuHa7cXhRy8",
                    "start_date": "2014-05-01T00:00:00",
                    "end_date": "2015-01-01T00:00:00",
                    "position_id": 550413588,
                    "description": None,
                    "location": "Greater New York City Area",
                    "rich_media": [],
                },
                {
                    "title": "Executive Recruiter",
                    "company_name": "ADVIZA",
                    "company_linkedin_id": "537041",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGPmWm4exNH3g/company-logo_400_400/company-logo_400_400/0/1661991906527/adviza_logo?e=1752105600&v=beta&t=vRZGcVLwlOwi0MyTJ30gk8oLhG2uj_TJevsp5Sftxes",
                    "start_date": "2013-01-01T00:00:00",
                    "end_date": "2014-05-01T00:00:00",
                    "position_id": 376040417,
                    "description": None,
                    "location": "Sydney, Australia",
                    "rich_media": [],
                },
                {
                    "title": "6 Month Sabbatical",
                    "company_name": "UK, Europe, North & Central America",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2012-07-01T00:00:00",
                    "end_date": "2012-12-01T00:00:00",
                    "position_id": 369856283,
                    "description": None,
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Business Development",
                    "company_name": "pureprofile",
                    "company_linkedin_id": "48119",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQHT4iJPSHeCgQ/company-logo_400_400/company-logo_400_400/0/1631308149737?e=1752105600&v=beta&t=jpmQCgERYDP1Tz6hIpqdL2Ewdqmio0PbwnX4lypqgDI",
                    "start_date": "2008-01-01T00:00:00",
                    "end_date": "2012-06-01T00:00:00",
                    "position_id": 138950060,
                    "description": None,
                    "location": "Sydney, Australia",
                    "rich_media": [],
                },
            ],
            "education_background": [
                {
                    "degree_name": "Bachelor of Arts (Communication - Public Relations & Organizational Communication)",
                    "institute_name": "Charles Sturt University",
                    "field_of_study": "",
                    "start_date": None,
                    "end_date": None,
                    "institute_linkedin_id": "14243",
                    "institute_linkedin_url": "https://www.linkedin.com/school/14243/",
                    "institute_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQHoSGy-Lu56Hw/company-logo_400_400/company-logo_400_400/0/1700439378529/charles_sturt_university_logo?e=1752105600&v=beta&t=-o7x2NDT2S0_hYdNftXjVFeynXjSLtgzl6fBv5V-Y_c",
                },
                {
                    "degree_name": "Higher School Certificate",
                    "institute_name": "Pymble Ladies'\u200b College",
                    "field_of_study": "",
                    "start_date": None,
                    "end_date": None,
                    "institute_linkedin_id": "2562469",
                    "institute_linkedin_url": "https://www.linkedin.com/school/2562469/",
                    "institute_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQEVQhk9oF46PQ/company-logo_400_400/company-logo_400_400/0/1630641513042/pymble_ladies_college_logo?e=1752105600&v=beta&t=GGmU0u31edobLjec0zedqZ6WfzIVLnYpoY6m3Ei5Zi8",
                },
            ],
            "emails": [],
            "websites": [],
            "twitter_handle": None,
            "languages": [],
            "pronoun": None,
            "query_person_linkedin_urn": "ACwAAAE_V1YBnB_dvg8sEe0tJz4l4LcWFuBrdNc",
            "linkedin_slug_or_urns": [
                "tahlia-spiegel-3860137",
                "ACwAAAE_V1YBnB_dvg8sEe0tJz4l4LcWFuBrdNc",
            ],
            "current_title": "VP, Human Resources",
        },
        {
            "name": "Lizzie Jaeger, PHR",
            "location": "San Francisco, California, United States",
            "linkedin_profile_url": "https://www.linkedin.com/in/ACwAAAW2ZoEBj_-RaElUNWliI0sZrxn1TFfjuXs",
            "linkedin_profile_urn": "ACwAAAW2ZoEBj_-RaElUNWliI0sZrxn1TFfjuXs",
            "default_position_title": "Director, HR Business Partnering (Business)",
            "default_position_company_linkedin_id": "17988315",
            "default_position_is_decision_maker": True,
            "flagship_profile_url": "https://www.linkedin.com/in/lizziejaeger",
            "profile_picture_url": "https://media.licdn.com/dms/image/v2/C4E03AQHcDIB2hGrhQA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1531437647597?e=1752105600&v=beta&t=DWDSm_KvFx21tFo6X78l-VqLFeuTOecwnRYWnheufv4",
            "headline": "HR Leader @ Rippling - We're hiring!",
            "summary": None,
            "num_of_connections": 2002,
            "related_colleague_company_id": 17988315,
            "skills": [
                "Product Management",
                "Product Development",
                "Product Marketing",
                "Marketing Strategy",
                "Agile Methodologies",
                "Agile Project Management",
                "Start-ups",
                "Data Analysis",
                "Data Analytics",
                "UX",
                "UI",
                "Recruiting",
                "Human Resources",
                "Employee Relations",
                "Customer Service",
                "Event Planning",
                "Payroll",
                "Employee Training",
                "Training",
                "Workday",
                "ADP Payroll",
                "Sourcing",
                "Talent Management",
                "Event Management",
            ],
            "employer": [
                {
                    "title": "Director, HR Business Partnering (Business)",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2025-04-01T00:00:00",
                    "end_date": None,
                    "position_id": 2627208213,
                    "description": None,
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "Senior Manager, Human Resources Business Partner",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2024-02-01T00:00:00",
                    "end_date": "2025-04-01T00:00:00",
                    "position_id": 2337890137,
                    "description": None,
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "Senior Human Resources Business Partner",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2023-10-01T00:00:00",
                    "end_date": "2024-01-01T00:00:00",
                    "position_id": 2337034738,
                    "description": None,
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "Human Resources Business Partner",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2022-01-01T00:00:00",
                    "end_date": "2023-09-01T00:00:00",
                    "position_id": 1913190726,
                    "description": None,
                    "location": "San Francisco, California, United States",
                    "rich_media": [],
                },
                {
                    "title": "Lead Product Manager",
                    "company_name": "Cultivate",
                    "company_linkedin_id": "18101003",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQHL2KotNRIDTA/company-logo_400_400/company-logo_400_400/0/1664842759201/trycultivate_logo?e=1752105600&v=beta&t=iR3ClZcqqFsNaQRQT3yZ32GhJs1X-mUXSSkK4lwhYgM",
                    "start_date": "2020-05-01T00:00:00",
                    "end_date": "2022-01-01T00:00:00",
                    "position_id": 1618990923,
                    "description": None,
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Senior Product Manager",
                    "company_name": "Reflektive",
                    "company_linkedin_id": "6390927",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQFY3ZLvwcspZg/company-logo_400_400/company-logo_400_400/0/1630661509660/reflektive_logo?e=1752105600&v=beta&t=No5JFL6om2q20sAkFT7svB0ejd5Y2wIkTqP7fEIJTcw",
                    "start_date": "2019-09-01T00:00:00",
                    "end_date": "2020-04-01T00:00:00",
                    "position_id": 1528972815,
                    "description": None,
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Product Manager",
                    "company_name": "Reflektive",
                    "company_linkedin_id": "6390927",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQFY3ZLvwcspZg/company-logo_400_400/company-logo_400_400/0/1630661509660/reflektive_logo?e=1752105600&v=beta&t=No5JFL6om2q20sAkFT7svB0ejd5Y2wIkTqP7fEIJTcw",
                    "start_date": "2018-03-01T00:00:00",
                    "end_date": "2019-09-01T00:00:00",
                    "position_id": 1271571038,
                    "description": None,
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Product Manager",
                    "company_name": "Sipree, Inc.",
                    "company_linkedin_id": "2245302",
                    "company_logo_url": None,
                    "start_date": "2017-10-01T00:00:00",
                    "end_date": "2018-03-01T00:00:00",
                    "position_id": 1528973807,
                    "description": None,
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Product Manager",
                    "company_name": "Zenefits",
                    "company_linkedin_id": "2997680",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGefoLIAXO9Zg/company-logo_400_400/company-logo_400_400/0/1688260427614?e=1752105600&v=beta&t=iPvyUvtp3EgLmw_1UK8qhujkNaaqGeLgWmgoAcZ85Q4",
                    "start_date": "2016-03-01T00:00:00",
                    "end_date": "2017-10-01T00:00:00",
                    "position_id": 789504811,
                    "description": "•Product Manager on the Zenefits Payroll team which offers Zenefits Payroll, Vacation & Time Off Tracking, Time & Attendance, and Third Party Payroll Integrations\n\n•Full ownership of the Time & Attendance and Time Off products serving over 180,000 users, bringing in over 3.5 million in ARR\n\n•Full ownership of the Zenefits Company Pay Schedule product - an effort to move to a unified data model - with over 50,000 users\n\n•SWAT team Product Manager, assisting additional teams to stabilize, driving process and a triage system for interrupt live sites weighing severity, users impacted and engineering effort required to determine priority - focusing on root cause oriented fixes\n\n•Manage the product lifecycle, from both development and marketing perspective\n\n•Recognized for shipping iterative versions of features with quick user facing deliverables without sacrificing quality\n\n•Analyze and join current, former, and prospective user data to drive roadmap prioritization that aligns with company initiatives \n\n•Collaborate with UX designers, copywriters, and tech leads to reimagine current product offerings\n\n•Lead an effort for compliance - proactively seeking and developing a legal audit of products at Zenefits, comparing the actual product behavior with legal recommended product behavior\n\n•Acting as Product Marketing Manager for Time & Attendance product\n\n•Lead 200% growth of the Time & Attendance product in past 6 months while decreasing support cost by 40%\n•Developed and delivered original webinar content for our users, receiving record high engagement 4x the Zenefits average - initiated a new source of lead gen across the entire company\n\n•Regularly host break out sessions at user conferences (Roadshows, Z2, Z22U) - speaking to 50+ prospective and current users\n\n•Mentor product specialists interested in career growth\n\n•Recognized as a SuperZeneWoman - an award given to 5 influential women at Zenefits, determined by both leaders and peers\n",
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "Senior Client HR Business Partner (HRBP)",
                    "company_name": "Zenefits",
                    "company_linkedin_id": "2997680",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGefoLIAXO9Zg/company-logo_400_400/company-logo_400_400/0/1688260427614?e=1752105600&v=beta&t=iPvyUvtp3EgLmw_1UK8qhujkNaaqGeLgWmgoAcZ85Q4",
                    "start_date": "2015-03-01T00:00:00",
                    "end_date": "2016-03-01T00:00:00",
                    "position_id": 1802259884,
                    "description": "Provided HR consulting for leaders of our large client base, while leveraging software issues and client requests to collaborate with the product and engineering teams to improve the product.",
                    "location": "San Francisco, California, United States",
                    "rich_media": [],
                },
                {
                    "title": "Human Resources Manager",
                    "company_name": "Four Seasons Hotels and Resorts",
                    "company_linkedin_id": "163883",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQEAgE4l2Z40Ig/company-logo_400_400/company-logo_400_400/0/1685134371955/four_seasons_hotels_and_resorts_logo?e=1752105600&v=beta&t=yt7bmKi0djNxMKpSZ9zKC-DVMU0SEPFPIUqjcZjgWSM",
                    "start_date": "2012-02-01T00:00:00",
                    "end_date": "2015-02-01T00:00:00",
                    "position_id": 507285005,
                    "description": "•\tNominated for 2014 Manager of the Quarter\n•\tResponsible for unemployment claims, FMLA and medical leaves, benefits, workers’ compensation\nHandled all employee grievances and concerns, at all levels of the organization through fair and consistent practice\n•\tDirectly managed two employees, an HR intern and an HR coordinator\n•\tCreated new recruitment procedures including formal pre-screening, what to expect during your interview email correspondence, and automatic emails to all applicants.\n•\tSelected to assist the Four Seasons Orlando property on task force during their opening and mass hiring and orientation of 400+ new employees.\n•\tSelected to assist the Four Seasons Vail property on task force to cover the Senior Assistant Director’s leave\n•\tSelected to be a Bluewater Ambassador, an Innovation Ambassador for Four Seasons\n•\tCo-chair of the Community Outreach Committee\n•\tSuccessfully managed 2014 Employee Engagement Survey and achieved a 99% participation rate\n•\tManage all employee relations including monthly celebrations, quarterly service award recognition, and annual employee party\n•\tResponsible for all hourly recruitment, 70+ hires annually\n•\tCreated and maintained relationships with local universities and culinary art institutions\n•\tFluent in ADP Enterprise, Timesaver, eTime and Workday systems",
                    "location": "Greater Chicago Area",
                    "rich_media": [],
                },
                {
                    "title": "Front Office Supervisor",
                    "company_name": "Hilton Gaslamp Quarter",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2011-09-01T00:00:00",
                    "end_date": "2012-02-01T00:00:00",
                    "position_id": 218270902,
                    "description": "•\tDirectly supervised 11 guest service agents, 1 concierge, and 7 bellmen at a 283-room hotel which runs 91% occupancy year round\n•\tPerformed Manager on Duty (MOD) shifts \n•\tRecognized twice consecutively as the “Blue Energy Story of the Month,” which is an award given to an employee recognizing their customer service to hotel guests and/or hotel employees\n•\tDoubled the number of Hilton Honors Enrollments per month through non monetary incentives\n•\tCreated a quantitative tracking system for each agent to record guest satisfaction surveys and positive comments",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Guest Service Agent",
                    "company_name": "Loews Coronado Bay Resort",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2009-11-01T00:00:00",
                    "end_date": "2010-12-01T00:00:00",
                    "position_id": 164783924,
                    "description": "Recognized as Team Member of the Month, May 2010",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Location Manager",
                    "company_name": "American Thoracic Society\t\t\t\tMay 2009",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2008-05-01T00:00:00",
                    "end_date": "2008-05-01T00:00:00",
                    "position_id": 164783925,
                    "description": "Communicated with three other managers to ensure a 14,000 person international conference ran smoothly\nImproved ability to multi-task in a fast paced environment",
                    "location": None,
                    "rich_media": [],
                },
            ],
            "education_background": [
                {
                    "degree_name": "Bachelors of Science",
                    "institute_name": "San Diego State University-California State University",
                    "field_of_study": "Hospitality and Tourism Management",
                    "start_date": "2008-01-01T00:00:00",
                    "end_date": "2012-01-01T00:00:00",
                    "institute_linkedin_id": "6206",
                    "institute_linkedin_url": "https://www.linkedin.com/school/6206/",
                    "institute_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQFcwgUDxDXz3Q/company-logo_400_400/company-logo_400_400/0/1666719462484/san_diego_state_university_logo?e=1752105600&v=beta&t=FA-GeXq_DO8OeWNHBY4gfX4kNiGQofyT6erFddAMOkk",
                },
                {
                    "degree_name": "Study Abroad",
                    "institute_name": "National University of Ireland, Galway",
                    "field_of_study": "History, Psychology, and Irish Studies",
                    "start_date": "2011-01-01T00:00:00",
                    "end_date": "2011-01-01T00:00:00",
                    "institute_linkedin_id": "7899",
                    "institute_linkedin_url": "https://www.linkedin.com/school/7899/",
                    "institute_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQGs-MjKBicdRg/company-logo_400_400/company-logo_400_400/0/1688394521630/universityofgalway_logo?e=1752105600&v=beta&t=DjNFLx1Kn8luyRfspx9Bsx95ZWeug1AWdp6lpT1loJQ",
                },
                {
                    "degree_name": "High School",
                    "institute_name": "Casa Grande High School",
                    "field_of_study": "Honors",
                    "start_date": "2004-01-01T00:00:00",
                    "end_date": "2008-01-01T00:00:00",
                    "institute_linkedin_id": None,
                    "institute_linkedin_url": None,
                    "institute_logo_url": None,
                },
            ],
            "emails": [],
            "websites": [],
            "twitter_handle": None,
            "languages": ["English"],
            "pronoun": "She/Her",
            "query_person_linkedin_urn": "ACwAAAW2ZoEBj_-RaElUNWliI0sZrxn1TFfjuXs",
            "linkedin_slug_or_urns": [
                "lizziejaeger",
                "ACwAAAW2ZoEBj_-RaElUNWliI0sZrxn1TFfjuXs",
            ],
            "current_title": "Director, HR Business Partnering (Business)",
        },
        {
            "name": "Mike Leary",
            "location": "Albany, New York Metropolitan Area",
            "linkedin_profile_url": "https://www.linkedin.com/in/ACwAAAAuOuYBQub0LuK-cyoCRMZ47eV5EYmQNJ8",
            "linkedin_profile_urn": "ACwAAAAuOuYBQub0LuK-cyoCRMZ47eV5EYmQNJ8",
            "default_position_title": "VP, Global Talent Acquisition",
            "default_position_company_linkedin_id": "17988315",
            "default_position_is_decision_maker": False,
            "flagship_profile_url": "https://www.linkedin.com/in/mleary",
            "profile_picture_url": "https://media.licdn.com/dms/image/v2/D4E03AQHUzVuBr3PhJw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1693964070263?e=1752105600&v=beta&t=7mDx4g4gRfb4dexaGOj31jWzhIuO-FKwHntw1hPzTcI",
            "headline": "TA leader for Rippling, father, gardener, wannabe chef, bowling fanatic, cancer survivor ",
            "summary": "Experienced HR executive with expertise building scalable, global recruiting engines.\n\nCurrent: \nLead global recruiting for Rippling\n\nPrior:\nLed global talent acquisition for GLOBALFOUNDRIES, one of the world’s leading semiconductor manufacturers, and the only one with a truly global footprint. Previously at GF I led HR Ops, and drove a transformation to a centralized HR Shared Services model, and I led the HR Reporting and Analytics as well. We went public in 2021 and have over 15K employees in 15+ locations globally.\n\nPrior experience :\nAnaplan: Rebuilt and led TA for Anaplan from pre-IPO, with 350 ee's, through an IPO and beyond, and over 1200 ee's.\n\nNetSuite / Oracle:\nGlobal VP for the leading provider of cloud-based financials / ERP and omnichannel commerce software suites. Lead a recruiting team of 100+ globally to support a hiring demand of over 2K hires per year. \n\nZenefits:\nBuilt and led recruiting for Zenefits, the fastest growing SaaS company in history at the time. Grew the recruiting team from 5 to 55 recruiters, as hiring increased to a peak of 1800 hires in 2015. Built recruiting processes and operations from scratch. \n\nSAP:\nLed the global TA function for this 75K employee company. Led a recruiting team of 300+ globally that yielded 15K hires annually. Drove a major org and budget overhaul that resulted in a centralized, efficient, scalable, operationally sound recruiting model. Yielded large budget savings while increasing production per recruiter and quality of hire.",
            "num_of_connections": 13963,
            "related_colleague_company_id": 17988315,
            "skills": [
                "Recruiting",
                "SaaS",
                "Talent Acquisition",
                "HCM",
                "Enterprise Software",
                "Salesforce.com",
                "Applicant Tracking Systems",
                "Talent Management",
                "Technical Recruiting",
                "Cloud Computing",
                "Solution Selling",
                "Internet Recruiting",
                "Sourcing",
                "Executive Search",
                "Account Management",
                "Sales Process",
                "CRM",
                "Sandwiches",
                "Global Management",
                "International Recruitment",
                "Hugs",
                "Strategy",
                "Lead Generation",
                "Sales",
                "Business Development",
                "Management",
                "Pre-sales",
                "global HCM",
                "Global Talent Acquisition",
                "Team Leadership",
                "Start-ups",
                "Analytics",
                "Team Management",
                "Temporary Placement",
                "Software Industry",
                "Strategic Partnerships",
                "Networking",
                "Consulting",
                "Business Alliances",
                "Complex Sales",
                "Vendor Management",
                "Go-to-market Strategy",
                "Outsourcing",
                "Hiring",
                "Executive Management",
                "Sales Enablement",
                "SAP",
                "Demand Generation",
                "SuccessFactors",
                "Customer Relationship Management (CRM)",
            ],
            "employer": [
                {
                    "title": "VP, Global Talent Acquisition",
                    "company_name": "Rippling",
                    "company_linkedin_id": "17988315",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQGTg3igNET25Q/company-logo_400_400/company-logo_400_400/0/1654722845874/rippling_logo?e=1752105600&v=beta&t=qGutQiQoqSe9I_ICPUWqkMLG1M0CHhdOWBPSPS_3b88",
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": None,
                    "position_id": 2100844847,
                    "description": None,
                    "location": "NY / SF",
                    "rich_media": [],
                },
                {
                    "title": "VP, Talent Acquisition, HR Operations",
                    "company_name": "GLOBALFOUNDRIES",
                    "company_linkedin_id": "241309",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQEzX3Ke-SvR5A/company-logo_400_400/company-logo_400_400/0/1697741489495/globalfoundries_logo?e=1752105600&v=beta&t=Ms-VvZ-pGgaBL4Y5uNFGhGfcTrbYqDMgdyp3avXqzRo",
                    "start_date": "2019-10-01T00:00:00",
                    "end_date": "2022-12-01T00:00:00",
                    "position_id": 1533534214,
                    "description": "Led Global Talent Acquisition during a time of unprecedented demand and growth in the semiconductor industry, and during GF’s journey from pre to post IPO. Our TA team had 70+ recruiters across the US, Germany, Singapore and Bangalore. Also led a transformation of HR Operations, shifting all global ops work into a centralized hub in Bangalore. ",
                    "location": "Malta, NY",
                    "rich_media": [],
                },
                {
                    "title": "Global Head of Talent Acquisition",
                    "company_name": "Anaplan",
                    "company_linkedin_id": "658814",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGYt40mQR_WYw/company-logo_400_400/B56ZUXZoi4GsAk-/0/1739854350849/anaplan_logo?e=1752105600&v=beta&t=ufmRANwJxjVNpVXXda1Zs2BjMtkOsYtvMg8VzthfqF0",
                    "start_date": "2017-06-01T00:00:00",
                    "end_date": "2019-10-01T00:00:00",
                    "position_id": 1022112619,
                    "description": "\nGlobal Head of Talent Acquisition at Anaplan\n\nBuilt and led the Global Talent Acquisition team for Anaplan during a time of massive growth and change, going from pre-IPO, sub-500 employees to a highly successful IPO and over 1600+ employees globally. ",
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "VP Global Talent Acquisition",
                    "company_name": "NetSuite",
                    "company_linkedin_id": "6137",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQEsZ3b3bX6srg/company-logo_400_400/company-logo_400_400/0/1706795717115/netsuite_logo?e=1752105600&v=beta&t=1CqvIVCRN3WK_YS0WKNYIlSRySNQM01voePI8H_gMdE",
                    "start_date": "2016-05-01T00:00:00",
                    "end_date": "2017-06-01T00:00:00",
                    "position_id": 809120370,
                    "description": "Led Global TA for all of Netsuite prior and through the acquisition into Oracle. \n\n•\tResponsible for a global team of 110 across EMEA, APJ and Americas\n•\tCreated a completely revised recruiting org model and strategy in order to centralize efforts, improve efficiencies, delivery and quality, and decrease spend. Net result: sourcing strategy and org overhaul, a shift to low cost locations and centralization, improved branding and social, and stronger resourcing and higher capacity and yield per recruiter across TA. \n•\tRebuilt a close partnership and synergy with FP&A and the LOB Operations teams, which previously did not exist. Established cadence of synching and reporting that led to improved reliability and accuracy of forecasting and hiring planning. \n•\tOver 2K hires in 2016. 35% increase in hires/quarter/recruiter YOY.\n•\tLed NetSuite TA through acquisition of NetSuite to Oracle which was announced during 3rd month at NetSuite.\n",
                    "location": "New York",
                    "rich_media": [],
                },
                {
                    "title": "VP Talent",
                    "company_name": "Zenefits",
                    "company_linkedin_id": "2997680",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGefoLIAXO9Zg/company-logo_400_400/company-logo_400_400/0/1688260427614?e=1752105600&v=beta&t=iPvyUvtp3EgLmw_1UK8qhujkNaaqGeLgWmgoAcZ85Q4",
                    "start_date": "2014-11-01T00:00:00",
                    "end_date": "2016-03-01T00:00:00",
                    "position_id": 604609976,
                    "description": "Led recruiting for what was, at one time, the fastest growing Saas company in history. \n•\tEstablished a consolidated recruiting org for the first time in company history. Grew team from 5 to 56 at peak.\n•\tDesigned company’s first ever hiring plan, working w Finance/FP&A, business leaders, CEO and COO.\n•\tDrove numerous strategic initiatives, including interview training, succession planning, internal and external benchmarking for recruiting targets, recruiter pitch certification, ATS setup and reporting, company recruiting video, facilities planning, equity burn analysis and adjustment. \n•\t1859 hires in 2015, having entered the year with 300 total employees, company-wide. ",
                    "location": "San Francisco Bay Area",
                    "rich_media": [],
                },
                {
                    "title": "Head of Global Recruiting, SAP",
                    "company_name": "Vice President and Global Lead, Talent Acquisition, SAP",
                    "company_linkedin_id": None,
                    "company_logo_url": None,
                    "start_date": "2013-11-01T00:00:00",
                    "end_date": "2014-11-01T00:00:00",
                    "position_id": 483238318,
                    "description": 'Led Talent Acquisition globally for SAP. Obsessively focused on candidate quality, hiring manager satisfaction, candidate experience, and enabling our world class recruiting org of 300 plus team members. \n\nOver 15K hires globally in 2014, a 110% increase year over year. 69 hires per FTE, vs 50 in 2013 and 46 in 2012. \n\nLed the integration and merging of three TA orgs (SAP, SuccessFactors and Ariba), with three different systems and processes, into one unified global team. \n\nRebuilt and centralized the global sourcing org, lowering cost and increasing production. \n\nBuilt "License to Recruit" program: rolled out an annual interview training and pitch certification project to “license” managers and recruiters to interview properly and effectively, enabling better hiring decisions. \n\nRebuilt and reestablished a strong presence on social. Built new sourcing channels and increased career site traffic by 4x. Established first in-house video production team within SAP TA. \n\nCareer site redesign – mobile friendly, simple and bold, more engaging dynamic content.',
                    "location": "Newtown Square, PA",
                    "rich_media": [],
                },
                {
                    "title": "VP, WW Cloud Recruiting, SAP",
                    "company_name": "SuccessFactors, an SAP Company",
                    "company_linkedin_id": "166185",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQG5P2DKlvtkZQ/company-logo_400_400/company-logo_400_400/0/1719839665866/sapsuccessfactors_logo?e=1752105600&v=beta&t=jGoe5qaPwfXAAqoRJh2UcGH6fjequ3uVWctox1MkmfQ",
                    "start_date": "2010-10-01T00:00:00",
                    "end_date": "2013-11-01T00:00:00",
                    "position_id": 147873225,
                    "description": "Global VP of Recruiting for all of the Cloud for SAP.\n\nSuccessFactors is the leading provider of cloud-based Business Execution Software, and delivers business alignment, team execution, people performance, and learning management solutions to organizations of all sizes across more than 60 industries. With approximately 15 million subscription seats globally, we strive to delight our customers by delivering innovative solutions, content and analytics, process expertise and best practices insights from serving our broad and diverse customer base. Today, we have more than 3,500 customers in more than 168 countries using our application suite in 35 languages.",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Global Director of Recruiting, Sales, PS and Marketing",
                    "company_name": "SuccessFactors",
                    "company_linkedin_id": "166185",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D4E0BAQG5P2DKlvtkZQ/company-logo_400_400/company-logo_400_400/0/1719839665866/sapsuccessfactors_logo?e=1752105600&v=beta&t=jGoe5qaPwfXAAqoRJh2UcGH6fjequ3uVWctox1MkmfQ",
                    "start_date": "2007-08-01T00:00:00",
                    "end_date": "2010-10-01T00:00:00",
                    "position_id": 21189334,
                    "description": "Drive recruiting globally for sales, presales, and marketing.",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Manager, Enterprise Sales and Executive Sales Recruiting",
                    "company_name": "salesforce.com",
                    "company_linkedin_id": "3185",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/C560BAQHZ9xYomLW7zg/company-logo_400_400/company-logo_400_400/0/1630658255326/salesforce_logo?e=1752105600&v=beta&t=v_pKVYyy0mI3KqLlmz5ssCRqvmjlHr8WNQ2ZaE_omPY",
                    "start_date": "2005-05-01T00:00:00",
                    "end_date": "2007-08-01T00:00:00",
                    "position_id": 4147176,
                    "description": "May 2005-August 2007\nSalesforce.com, San Francisco, CA \nA global provider of on-demand customer relationship management (CRM) services.\n\nDrive recruiting for sales, PS, presales, and other functions as needed, nationwide (US).",
                    "location": None,
                    "rich_media": [],
                },
                {
                    "title": "Senior Staff Recruiter",
                    "company_name": "Symantec formerly VERITAS Software",
                    "company_linkedin_id": "1231",
                    "company_logo_url": "https://media.licdn.com/dms/image/v2/D560BAQGwqTRMsfuUsg/company-logo_400_400/B56ZWpgIAqGUAc-/0/1742305527529/symantec_logo?e=1752105600&v=beta&t=MnBDsDTIewVH8fwSqVMlorrH5FtupQpErbmTw1LSsNE",
                    "start_date": "2003-04-01T00:00:00",
                    "end_date": "2005-05-01T00:00:00",
                    "position_id": 4615030,
                    "description": "August 2003-May 2005 \nVERITAS Software \nLeading provider of products and services for data protection, storage and server management, high availability and application performance management. \n\nRecruiting, Field Sales, PS, nationwide (US)",
                    "location": None,
                    "rich_media": [],
                },
            ],
            "education_background": [
                {
                    "degree_name": "BS",
                    "institute_name": "Fairfield University",
                    "field_of_study": "Business",
                    "start_date": None,
                    "end_date": None,
                    "institute_linkedin_id": "16708",
                    "institute_linkedin_url": "https://www.linkedin.com/school/16708/",
                    "institute_logo_url": "https://media.licdn.com/dms/image/v2/C4E0BAQE3wYIz3Q372A/company-logo_400_400/company-logo_400_400/0/1644977665248/fairfield_university_logo?e=1752105600&v=beta&t=4yhugyS0jTnPkrZ4Dd1zNijCi1QEYryPiaMDw47uYHE",
                },
                {
                    "degree_name": None,
                    "institute_name": "Wheatley",
                    "field_of_study": "",
                    "start_date": None,
                    "end_date": None,
                    "institute_linkedin_id": None,
                    "institute_linkedin_url": None,
                    "institute_logo_url": None,
                },
            ],
            "emails": [],
            "websites": [],
            "twitter_handle": "mikelearylive",
            "languages": [],
            "pronoun": "He/Him",
            "query_person_linkedin_urn": "ACwAAAAuOuYBQub0LuK-cyoCRMZ47eV5EYmQNJ8",
            "linkedin_slug_or_urns": [
                "ACwAAAAuOuYBQub0LuK-cyoCRMZ47eV5EYmQNJ8",
                "mleary",
            ],
            "current_title": "VP, Global Talent Acquisition",
        },
    ]


# extract relevant fields only from the person data dict provided
# add more fields here if needed
def extract_relevant_person_data(person_data, target_company):
    print("PERson data = \n")
    print(person_data)
    person_data_relevant = {}
    fields = [
        "name",
        "default_position_title",
        "num_of_connections",
    ]
    person_data_relevant = {k: person_data[k] for k in fields}
    person_data_relevant["industry_experience"] = get_industry_experience(
        person_data["employer"]
    )
    person_data_relevant["tenure"] = get_current_tenure(person_data["employer"])
    # call llm here only ?

    enrich_profile_llm_system = """
  You are an expert data analyst. Given an api response data on various employees in a company,
  you need to extract data from the response per employee from the json provided.

  Given the person's title and headline, infer their org unit (e.g., Sales, Marketing, RevOps, Enablement, Executive), suborganization unit as well (say Engineering at Azure Cloud vs Engineering at Bing Ads or Human Resources Hiring or Human resources Talent Recruiting)
  and seniority level on a 1–7 scale (1 = junior IC, 7 = C-level) and Function type (strategic, tactical, IC).

  Here's what you need to extract finally and output as a json
  Final Output format -
  {
  "Org Unit" - infer from title
  "Suborg Unit" - Say sales enablement , sales ops instead of just sales (infer from title)
  "Seniority Level" (1–7)
  "Function Type" (Strategic, Tactical, IC)
  }

  OUtput only json with the 4 fields and nothing else.
  """

    enrich_profile_llm_user = f"""
  here is the title - {person_data["default_position_title"]} and here is the headline - {person_data["headline"]}

  """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    person_role_response = client.responses.parse(
        model="gpt-4o",
        # reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": enrich_profile_llm_system},
            {"role": "user", "content": enrich_profile_llm_user},
        ],
        text_format=Person_Enrichment,
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
        person_data_relevant["role_enriched"] = (
            person_role_response.output_parsed.model_dump()
        )  # json.loads(person_role_response.output_text.split("json")[1].strip("```"))
        return person_data_relevant
    except:
        return person_data_relevant


def enrich_people_data(buyer, people_strategy_data: dict):
    keys = ["Decision_Maker", "Economic_Buyer", "Champions", "Influencers", "Blockers"]
    for initiative in people_strategy_data["Initiatives"]:
        for tag in keys:
            people_enriched = []
            for person in initiative[tag]["people"]:
                person_data_relevant = extract_relevant_person_data(person, buyer)
                if person_data_relevant:
                    people_enriched.append(person_data_relevant)
            initiative[tag]["people"] = people_enriched
    return people_strategy_data


class Influence(BaseModel):
    influence_score: float
    reason: str


def get_influence_score(strategy_people_data):
    strategy_people_data_temp = {}
    keys = ["Decision_Maker", "Economic_Buyer", "Champions", "Influencers", "Blockers"]
    for initiative in strategy_people_data["Initiatives"]:
        for tag in keys:
            people_enriched = []
            for person in initiative[tag]["people"]:
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
            "Reason": "Explain the influence and the reasons for high or low influence to an account executive"
          }

          """

                    influence_score_prompt_user = f"""
          here is persona data - {str(person)} and here is the buyer company's initiative - {str(initiative["Buyer_Initiative_Or_Pain"])}
          """
                    api_key = os.getenv("OPENAI_API_KEY")
                    client = openai.OpenAI(api_key=api_key)
                    node_influence = client.responses.parse(
                        model="gpt-4o",
                        # reasoning={"effort": "medium"},
                        input=[
                            {
                                "role": "system",
                                "content": influence_score_prompt_system,
                            },
                            {"role": "user", "content": influence_score_prompt_user},
                        ],
                        text_format=Influence,
                    )
                    print(node_influence.output_parsed)
                    print("\n\n\n\n")

                    person_enriched = person | node_influence.output_parsed.model_dump()
                    print(node_influence)
                except Exception as e:
                    print(e)
                    print("error")
                    person_enriched = person | {
                        "influence_score": 50,
                        "Reason": "Default Score",
                    }
                people_enriched.append(person_enriched)

            initiative[tag]["people"] = people_enriched
    return strategy_people_data


class StakeholderGraph:
    def __init__(self, initiative_name, people_data):
        self.initiative = initiative_name
        self.graph = nx.DiGraph()
        self.people = people_data
        self.max_connections = max(p.get("num_of_connections", 1) for p in people_data)
        self._build_graph()
        # self.edge_threshold = edge_threshold

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
        same_org = a["role_enriched"].get("Org Unit") == b["role_enriched"].get(
            "Org Unit"
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
                if a.get("tag", "") == "Economic_Buyer":
                    continue  # DMs are sinks only

                if a["name"] == b["name"]:
                    continue  # Skip self

                if a["influence_score"] <= b["influence_score"]:
                    continue
                if not self._should_create_edge(a, b):
                    continue
                if i != j:  # and a["name"] != b["name"]:
                    weight = self._calculate_edge_weight(a, b)
                    raw_weights.append(weight)
                    edge_candidates.append((a["name"], b["name"], weight))
        max_raw_weight = max(raw_weights) if raw_weights else 1
        avg_raw_weight = 1.0 * (sum(raw_weights) / (len(raw_weights)))
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
        if a["role_enriched"].get("Suborg Unit") == b["role_enriched"].get(
            "Suborg Unit"
        ):
            weight += 3

        # Function type
        if (
            a["role_enriched"].get("Function Type") == "Strategic"
            and b["role_enriched"].get("Function Type") == "Tactical"
        ):
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
        tags_weights = {
            "Champions": 3,
            "Economic_Buyer": 4,
            "Decision_Maker": 5,
            "Blockers": 2,
            "Influencers": 2,
        }
        weight += tags_weights.get(a.get("tag", ""), 0)
        # print(a.get("tag", "Default"))
        # print(a)
        # weight+=tags_weights.get(b.get("role", ""), 0)

        # Influence score (scaled)
        influence_score = a.get("influence_score", 50)
        # weight *= (influence_score / 100)
        influence_score_a = a.get("influence_score", 50)
        influence_score_b = b.get("influence_score", 50)
        influence_diff = influence_score_a - influence_score_b
        diff = min(influence_diff, 50)  # Cap difference
        weight *= diff / 50  # Scale back to 0–1 range

        # Normalize
        norm_weight = 1 / (
            1 + math.exp(-0.2 * (weight - 10))
        )  # Adjust center as needed
        return norm_weight
        # return round(min(weight / 20, 1), 2)


def find_economic_buyer(G):
    economic_buyers = [
        (node, data.get("influence_score", 0))
        for node, data in G.nodes(data=True)
        if data.get("tag") == "Economic_Buyer"
    ]

    if not economic_buyers:
        return None

    # Sort by influence score descending
    sorted_buyers = sorted(economic_buyers, key=lambda x: x[1], reverse=True)

    top_node = sorted_buyers[0][0]
    return (top_node, G.nodes[top_node].get("influence_score", 0))


def get_all_stakeholders_across_roles(G):
    blockers = find_top_blockers(G, top_n=3)
    champions = rank_champions(G, top_n=3)
    influencers = find_top_influencers(G, top_n=3)
    decision_maker = find_best_decision_maker(G)
    economic_buyer = find_economic_buyer(G)
    return blockers, champions, influencers, decision_maker, economic_buyer


def generate_stakeholder_summary(G):
    def get_node_data(n):
        data = G.nodes[n]
        return {
            "Name": data.get("name", n).strip("\n"),
            "Title": data.get("default_position_title", "").strip("\n"),
            "Dept": data.get("role_enriched", {}).get("Org_Unit", ""),
            "Score": data.get("influence_score", ""),
            "Notes": data.get("reason", "").strip("\n"),
            "Industry Experience": data.get("industry_experience", 0),
            "Tenure": data.get("tenure", 0),
        }

    blockers, champions, influencers, decision_maker, economic_buyer = (
        get_all_stakeholders_across_roles(G)
    )
    rows = []
    for c in champions:
        d = get_node_data(c[0])
        rows.append(
            (
                "Champion",
                d["Name"],
                d["Title"],
                d["Dept"],
                d["Score"],
                d["Notes"],
                d["Industry Experience"],
                d["Tenure"],
            )
        )

    for i in influencers:
        d = get_node_data(i[0])
        rows.append(
            (
                "Influencer",
                d["Name"],
                d["Title"],
                d["Dept"],
                d["Score"],
                d["Notes"],
                d["Industry Experience"],
                d["Tenure"],
            )
        )

    for b in blockers:
        d = get_node_data(b[0])
        rows.append(
            (
                "Blocker",
                d["Name"],
                d["Title"],
                d["Dept"],
                d["Score"],
                d["Notes"],
                d["Industry Experience"],
                d["Tenure"],
            )
        )

    if decision_maker:
        d = get_node_data(decision_maker[0])
        rows.append(
            (
                "Decision Maker",
                d["Name"],
                d["Title"],
                d["Dept"],
                d["Score"],
                d["Notes"],
                d["Industry Experience"],
                d["Tenure"],
            )
        )
    if economic_buyer:
        d = get_node_data(economic_buyer[0])
        rows.append(
            (
                "Economic Buyer",
                d["Name"],
                d["Title"],
                d["Dept"],
                d["Score"],
                d["Notes"],
                d["Industry Experience"],
                d["Tenure"],
            )
        )

    markdown = "| Role | Name | Title | Dept/Function | Influence Score | Notes | Industry Experience | Tenure |\n"
    markdown += "|------|------|-------|----------------|------------------|-------|----------------------|--------|\n"
    for r in rows:
        markdown += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} Years | {r[7]} Years|\n"

    return markdown


def find_top_blockers(G, top_n=3):
    betweenness = nx.betweenness_centrality(G)
    blocker_scores = []

    for n, data in G.nodes(data=True):
        if data.get("tag") != "Blockers":
            continue

        block_score = data.get("blocking_score", 0)
        indeg = G.in_degree(n)
        btw = betweenness.get(n, 0)

        score = 0.5 * block_score + 0.3 * indeg + 0.2 * btw

        blocker_scores.append((n, score))

    # Sort and return top N
    top_blockers = sorted(blocker_scores, key=lambda x: x[1], reverse=True)
    if len(top_blockers) > 0:
        return top_blockers[:top_n]
    else:
        return []


def find_top_influencers(G, top_n=3):
    pageranks = nx.pagerank(G)
    influencer_scores = []

    for n, data in G.nodes(data=True):
        if data.get("tag") != "Influencers":
            continue

        influence = data.get("influence_score", 0)
        pr = pageranks.get(n, 0)
        outdeg = G.out_degree(n)

        score = 0.5 * influence + 0.3 * pr + 0.2 * outdeg

        influencer_scores.append((n, score))

    # Sort and return top N
    top_influencers = sorted(influencer_scores, key=lambda x: x[1], reverse=True)
    if len(top_influencers) > 0:
        return top_influencers[:top_n]
    else:
        return []


def find_best_decision_maker(G):
    pageranks = nx.pagerank(G)
    candidates = []

    for n, data in G.nodes(data=True):
        if data.get("tag") != "Decision_Maker":
            continue

        influence = data.get("influence_score", 0)
        indegree = G.in_degree(n)
        pr = pageranks.get(n, 0)

        # Adjusted weighted score
        score = 0.5 * influence + 0.3 * indegree + 0.2 * pr

        candidates.append((n, score))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])


def rank_champions(G, top_n=3):
    dms = [n for n, d in G.nodes(data=True) if d.get("tag") == "Decision_Maker"]
    champion_scores = []

    for n, data in G.nodes(data=True):
        if data.get("tag") != "Champions":
            continue

        visibility = 1  # data.get("visibility_score", 0)
        influence = data.get("influence_score", 0)

        # Proximity to decision maker (shortest path)
        proximity = 0
        if dms:
            try:
                min_dist = min(
                    [
                        nx.shortest_path_length(G, source=n, target=dm)
                        for dm in dms
                        if nx.has_path(G, n, dm)
                    ]
                )
                proximity = 1 / (min_dist + 1e-6)  # avoid div by zero
            except:  # noqa: E722
                proximity = 0

        # Reachability score (fraction of nodes this node can reach)
        reachability = len(nx.descendants(G, n)) / max((G.number_of_nodes() - 1), 1)

        # Final weighted score
        score = (
            0.35 * visibility + 0.25 * influence + 0.2 * proximity + 0.2 * reachability
        )

        champion_scores.append((n, score))

    # Sort champions by score descending
    ranked = list(sorted(champion_scores, key=lambda x: x[1], reverse=True))
    logger.debug(ranked)
    if len(ranked) > 0:
        return ranked[:top_n]
    else:
        return []

    # ends here --------------------

    def get_high_influence_nodes(self, top_n=5):
        influence_scores = [
            (n, self.graph.out_degree(n, weight="weight")) for n in self.graph.nodes
        ]
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
            sorted_group = sorted(
                group, key=lambda p: p.get("influence_score", 0), reverse=True
            )[:top_n]
            markdown += f"\n### 🔹 {tag}s\n"
            for p in sorted_group:
                markdown += f"- `{p['name']}` — **{p.get('default_position_title', 'N/A')}**, Influence Score: **{p.get('influence_score', 0)}**\n"
        return markdown

    # change this
    def find_shortest_path_to_decision_maker(
        self, from_node, dm_title="Cross-Functional Reviewers"
    ):
        decision_makers = [
            n for n, d in self.graph.nodes(data=True) if d.get("tag") == dm_title
        ]
        paths = {}
        for dm in decision_makers:
            if self.graph.has_edge(from_node, dm):
                print("There is a direct edge from champion to dm")
            else:
                print("No direct edge")
            try:
                path = nx.shortest_path(
                    self.graph, source=from_node, target=dm, weight="weight"
                )
                paths[dm] = path
            except nx.NetworkXNoPath:
                continue
        return paths

    def visualize_graph(self, figsize=(10, 8)):
        pos = nx.spring_layout(self.graph)
        edge_weights = nx.get_edge_attributes(self.graph, "weight")

        plt.figure(figsize=figsize)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=1500,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            edge_color="gray",
        )
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()},
            font_size=8,
        )
        plt.title(f"Stakeholder Influence Graph for Initiative: {self.initiative}")
        plt.axis("off")
        plt.show()

    def visualize_graph_interactive(self, notebook=False):
        net = Network(
            height="750px",
            width="100%",
            directed=True,
            notebook=notebook,
            cdn_resources="in_line",
        )

        for node, attrs in self.graph.nodes(data=True):
            title = f"{attrs.get('default_position_title', '')}<br>Seniority: {attrs['role_enriched'].get('Seniority Level', 'N/A')}<br>Tenure: {attrs.get('tenure', 'N/A')} years"
            net.add_node(node, label=node, title=title, color="skyblue")

        for src, dst, data in self.graph.edges(data=True):
            net.add_edge(
                src, dst, value=data["weight"], title=f"Influence: {data['weight']:.2f}"
            )

        net.show_buttons(filter_=["physics"])
        net.show(f"{self.initiative}_stakeholder_graph.html")

    def visualize_graph_plotly(self):
        pos = nx.spring_layout(self.graph, seed=42, k=2)

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
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

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
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[n for n in self.graph.nodes()],
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="Blues",
                size=20,
                color=[
                    self.graph.out_degree(n, weight="weight")
                    for n in self.graph.nodes()
                ],
                colorbar=dict(
                    thickness=15,
                    title="Influence Score",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"<b>Stakeholder Influence Graph</b><br>{self.initiative}",
                titlefont_size=20,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            ),
        )
        fig.show()

    # use this
    def get_consensus_paths_to_all_targets(
        self, tag_to_targets: dict, top_k=3, min_weight=0.2
    ):
        """
        tag_to_targets: dict of {tag: [node_id, ...]} for each stakeholder type (e.g. Champion, Influencer, Owner, etc.)
        top_k: how many influencers per target to show
        """
        all_results_md = "## Consensus Paths to Key Stakeholders\n"
        summary_md = "**Key paths to influence high-priority stakeholders:**\n"

        for tag, target_ids in tag_to_targets.items():
            # print(tag)
            # print(target_ids)
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

                incoming_sorted = sorted(incoming, key=lambda x: x[1], reverse=True)[
                    :top_k
                ]
                print("len = ")
                print(len(incoming_sorted))
                all_results_md += f"\n#### {target_node.get('name', target_id)} ({target_node.get('default_position_title', 'N/A')})\n"
                all_results_md += (
                    "| Influencer | Title | Influence Weight |\n|---|---|---|\n"
                )
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
                node
                for node, data in self.graph.nodes(data=True)
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
                        0.5 * betweenness.get(node_id, 0)
                        + 0.5 * closeness.get(node_id, 0)
                        + 0.3 * (influence_score / 100)
                    )
                else:
                    score = (
                        0.6 * (influence_score / 100)
                        + 0.25 * (1 if org_match else 0)
                        + 0.15 * (1 if suborg_match else 0)
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

    def generate_engagement_strategy(
        self,
        buyer_initiative,
        target_tags=["Champions", "Internal Influencer", "Economic Buyer"],
        use_llm=True,
    ):
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
                strategies.append(
                    markdown_summary + "➡️ Strategy: [TODO — Add manually]\n"
                )
                continue

            # 🔥 LLM Prompt per person
            prompt = f"""
          You're a sales strategist helping an AE plan outreach to key stakeholders.

          Based on the buyer initiative: **{buyer_initiative}**

          And this stakeholder's profile:
          {markdown_summary}

          Write a 3–4 line personalized engagement strategy for how to approach this stakeholder, based on their role, org unit, influence, and tags. Be specific.
          """
            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)
            response_engagement_strategy = client.responses.create(
                model="gpt-4o",
                # reasoning={"effort": "medium"},
                input=[
                    {"role": "user", "content": prompt},
                ],
            )
            print(response_engagement_strategy.output_text)

            # Replace this with your LLM call, or return the prompt to use externally
            # For now we just add the prompt
            strategies.append(
                markdown_summary
                + "**Suggested Strategy (LLM prompt):**\n"
                + response_engagement_strategy.output_text
                + "\n"
            )

        return "\n---\n".join(strategies)

    def get_stakeholders_for_engagement(self, decision_maker, max_per_tag=3):
        stakeholders = set()

        # 1. Get shortest paths from all nodes to decision maker
        for node in self.graph.nodes():
            if node == decision_maker:
                continue
            try:
                path = nx.shortest_path(
                    self.graph, source=node, target=decision_maker, weight="weight"
                )
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
            if any(
                tag in tags for tag in ["Economic Buyer", "Cross-Functional Reviewers"]
            ):
                decision_makers.append((node, influence, degree))

        # Rank by composite: 0.7 * influence + 0.3 * degree
        champions.sort(key=lambda x: 0.7 * x[1] + 0.3 * x[2], reverse=True)
        decision_makers.sort(key=lambda x: 0.7 * x[1] + 0.3 * x[2], reverse=True)

        return {
            "Best Champion": champions[0] if champions else None,
            "Best Decision Maker": decision_makers[0] if decision_makers else None,
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

            segments.append(
                f"**{i + 1}. `{path[i]}` — {title}**  \n"
                f"• Tags: {tags}  \n"
                f"• Org: {org} / {suborg}  \n"
                f"• Influence Score: {influence_score}\n"
            )

            # Show edge weights if it's not the last node
            if i < len(path) - 1:
                edge = self.graph[path[i]][path[i + 1]]
                segments.append(
                    f"➡️ **Edge Influence Weight**: {edge.get('weight', 'N/A')}\n"
                )

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
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)
        response_explain_strategy = client.responses.create(
            model="gpt-4o",
            # reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": multithread_prompt_system},
                {"role": "user", "content": multithread_prompt_user},
            ],
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
    # print(all_people_deduped)
    return all_people_deduped


def get_graphs_for_initiatives(strategy_people_data_scored):
    graphs = {}
    a = strategy_people_data_scored
    all_people = {}
    for initiatives in a["Initiatives"]:
        initiative_name = initiatives["Buyer_Initiative_Or_Pain"]
        all_people[initiative_name] = []
        keys = [
            "Decision_Maker",
            "Economic_Buyer",
            "Champions",
            "Influencers",
            "Blockers",
        ]

        for tag in keys:
            # print(tag)
            # print(initiatives[tag])
            people_temp = copy.deepcopy(initiatives[tag]["people"])
            for person in people_temp:
                person |= {"tag": tag}
            all_people[initiative_name] += people_temp
        all_people[initiative_name] = remove_duplicates(
            copy.deepcopy(all_people[initiative_name])
        )
        print("People data\n\n")
        print(all_people[initiative_name])
        graph = StakeholderGraph(initiative_name, all_people[initiative_name])
        graphs[initiative_name] = graph
    return graphs


def get_company_from_url(url):
    extracted = tldextract.extract(url)
    return extracted.domain if extracted.domain else None
