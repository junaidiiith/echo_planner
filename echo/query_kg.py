import os
import enum
from itertools import product
import pickle
from typing import List
import networkx as nx
import numpy as np

from echo.utils import db_storage_path
from echo.echo_agent import get_crew


from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class UserType(enum.Enum):
    DECISION_MAKER = "Decision_Maker"
    ECONOMIC_BUYER = "Economic Buyer"
    CHAMPIONS = "Champions"
    INFLUENCERS = "Influencers"
    BLOCKER = "Blocker"


class BestPathInputs(BaseModel):
    """Input schema to get the best path between two types of nodes in the org graph."""

    buyer: str = Field(..., description="Name of the buyer.")
    seller: str = Field(..., description="Name of the seller.")
    initiative: str = Field(
        ..., description="Initiative for which the seller is focusing on the buyer"
    )
    source: str = Field(
        default=UserType.CHAMPIONS.value,
        description="Source node type. Must be one of Decision_Maker, Economic Buyer, Champions, Influencers, or Blocker.",
    )
    target: str = Field(
        default=UserType.DECISION_MAKER.value,
        description="Target node type. Must be one of Decision_Maker, Economic Buyer, Champions, Influencers, or Blocker."
    )


class BestPathExtractor(BaseTool):
    name: str = "Best Path Extractor"
    description: str = (
        "Extract the best path between two types of nodes from the organization graph of a company. "
        "The path represents the people that need to be convinced or won over to get the deal done. "
        "Guidelines: "
        "This tool is used to extract the best path between two types of nodes in the organization graph. "
        "Whenever there is a need to find a path between two types of nodes, this tool should be used. "
        "The source and target node types can be specified by the user. "
        "If the source is not provided by the user, it defaults to Champions. "
        "If the target is not provided by the user, it defaults to Decision_Maker. "
        "The source and target node types must be one of Decision_Maker, Economic Buyer, Champions, Influencers, or Blocker. "
        
        "\nArgs: \n"
        "  - buyer: Name of the buyer. \n"
        "  - seller: Name of the seller. \n"
        "  - initiative: Initiative for which the seller is focusing on the buyer. \n"
    )
    args_schema: Type[BaseModel] = BestPathInputs

    def _run(
        self,
        seller: str,
        buyer: str,
        initiative: str,
        source: str = UserType.CHAMPIONS.value,
        target: str = UserType.DECISION_MAKER.value,
    ) -> str:
        graph = get_org_graph(seller, buyer, initiative)
        source_nodes = get_nodes_by_tag(graph, source)
        target_nodes = get_nodes_by_tag(graph, target)
        paths = list()
        for source, target in product(source_nodes, target_nodes):
            # print(f"Finding path from {source} to {target}")

            path = find_shortest_path(graph, source, target)
            paths.append(path)
        paths = sorted_paths_by_cost(graph, paths)
        paths_str = print_path(graph, paths[0]) if paths else None
        # print(f"Best path from {source} to {target}: {paths_str}")
        if paths_str is None:
            return f"No path found from {source} to {target}."
        return paths_str


class BestNodeTypeInputs(BaseModel):
    """Input schema to get the best nodes of a specific type in the org graph."""

    buyer: str = Field(..., description="Name of the buyer.")
    seller: str = Field(..., description="Name of the seller.")
    initiative: str = Field(
        ..., description="Initiative for which the seller is focusing on the buyer"
    )
    node_types: List[UserType] = Field(
        ..., description="List of node types to find best nodes for."
    )
    top_n: int = Field(default=-1, description="Number of best nodes to return.")


class BestNodeTypesExtractor(BaseTool):
    name: str = "Best Node Types Extractor"
    description: str = (
        "Extract the best nodes of a specific type from the organization graph of a company."
        "Guidelines: "
        "This tool is used to extract the best nodes of a specific type in the organization graph. "
        "Whenever there is a need to find the best nodes of a specific type, this tool should be used. "
        "The node types can be specified by the user. "
        "If the node types are not provided by the user, it defaults to Decision_Maker, Economic Buyer, Champions, Influencers, or Blocker. "
        
        "\nArgs: \n"
        "  - buyer: Name of the buyer. \n"
        "  - seller: Name of the seller. \n"
        "  - initiative: Initiative for which the seller is focusing on the buyer. \n"
    )
    args_schema: Type[BaseModel] = BestNodeTypeInputs

    def _run(
        self,
        seller: str,
        buyer: str,
        initiative: str,
        node_types: List[UserType],
        top_n: int = -1,
    ) -> str:
        graph = get_org_graph(seller, buyer, initiative)
        node_types = [node_type.value for node_type in node_types]
        best_nodes = get_best_nodes(graph, node_types, top_n=top_n)
        best_nodes = [
            f"{node['data']['tag']}:{node['name']} ({node['data']['default_position_title']})"
            for node in best_nodes
        ]
        return "\n".join(best_nodes)


def print_path(g, p):
    path = [p[0] + f"({g.nodes[p[0]]['tag']})"]
    for node in p[1:]:
        path += [node + f"({g.nodes[node]['tag']})"]

    return " -> ".join(path)


def get_nodes_by_tag(graph, tag):
    """
    Get nodes in the graph that have a specific tag.
    """
    return [node for node, data in graph.nodes(data=True) if data.get("tag") == tag]


def add_economic_buyer_tag(g):
    tag_dict = dict()

    u_g = g.to_undirected()
    for u in u_g.nodes():
        tag = u_g.nodes(data="tag")[u]
        # print(u, tag)
        if tag not in tag_dict:
            tag_dict[tag] = set()
        tag_dict[tag].add(u)

    # print(tag_dict.keys())

    champions = tag_dict.get(UserType.CHAMPIONS.value, set())
    for champion in champions:
        for tag, nodes in tag_dict.items():
            if tag != UserType.CHAMPIONS.value and len(nodes) > 5:
                for node in list(nodes):
                    # print(f"Source node: {champion}, Target node: {node}")
                    pl = nx.shortest_path_length(u_g, source=champion, target=node)
                    # print(pl)
                    if pl > 2:
                        u_g.nodes[node]["tag"] = UserType.ECONOMIC_BUYER.value
                        # print(f"Node {node} tagged as 'Economic Buyer'")
    return u_g


def compute_node_costs(G, type_penalty=100.0, influence_weight=1.0):
    """
    Annotate each node in G with a 'node_cost' =
      type_penalty (if it's a Blocker)
      + influence_weight * (1 - normalized_influence_score).
    """
    # collect all influence scores
    infs = [data.get("influence_score", 0.0) for _, data in G.nodes(data=True)]
    min_inf, max_inf = min(infs), max(infs)

    for n, data in G.nodes(data=True):
        inf = data.get("influence_score", 0.0)
        # normalize to [0,1], guard against zero‐range
        if max_inf > min_inf:
            norm = (inf - min_inf) / (max_inf - min_inf)
        else:
            norm = 1.0
        influence_cost = (1.0 - norm) * influence_weight

        # heavy penalty for reviewer nodes
        type_cost = (
            type_penalty
            if data.get("node_type") == UserType.BLOCKER.value
            else 0.0
        )

        data["node_cost"] = influence_cost + type_cost


def find_shortest_path(G, source, target, type_penalty=100.0, influence_weight=1.0):
    """
    Returns the path from source to target that minimizes:
      sum(inv_weight of edges) + sum(node_cost of intermediate+target nodes).
    """
    # first compute per-node costs
    compute_node_costs(G, type_penalty=type_penalty, influence_weight=influence_weight)

    # define an edge‐weight function that adds the target‐node cost
    def edge_weight(u, v, data):
        inv_w = data.get("inv_weight", 1.0)
        node_c = G.nodes[v].get("node_cost", 0.0)
        return inv_w + node_c

    # use Dijkstra’s algorithm over that weight
    return nx.dijkstra_path(G, source, target, weight=edge_weight)


def make_edge_weight(G):
    """
    Returns a function f(u, v, data) = inv_weight + node_cost(v)
    """

    def edge_weight(u, v, data):
        inv_w = data.get("inv_weight", 1.0)
        node_c = G.nodes[v].get("node_cost", 0.0)
        return inv_w + node_c

    return edge_weight


def path_cost(G, path, edge_weight):
    """
    Sum up edge_weight(u,v,data) over each consecutive pair (u,v) in path.
    """
    total = 0.0
    for u, v in zip(path, path[1:]):
        data = G.get_edge_data(u, v, default={})
        total += edge_weight(u, v, data)

    # print(f"Path cost for {path}: {total}")
    return total


def sorted_paths_by_cost(G, paths, type_penalty=100.0, influence_weight=1.0):
    """
    Compute node_costs, build edge_weight fn, then sort 'paths' by total cost.
    """
    compute_node_costs(G, type_penalty=type_penalty, influence_weight=influence_weight)
    ew = make_edge_weight(G)
    return sorted(paths, key=lambda p: path_cost(G, p, ew))


def best_path(
    graph,
    src_type=UserType.CHAMPIONS.value,
    tgt_type=UserType.ECONOMIC_BUYER.value,
):
    source_nodes = get_nodes_by_tag(graph, src_type)
    target_nodes = get_nodes_by_tag(graph, tgt_type)
    print(f"Source nodes: {source_nodes}")
    print(f"Target nodes: {target_nodes}")
    paths = list()
    for source, target in product(source_nodes, target_nodes):
        path = find_shortest_path(graph, source, target)
        paths.append(path)
    paths = sorted_paths_by_cost(graph, paths)
    return print_path(graph, paths[0]) if paths else None


def get_best_nodes(graph: nx.Graph, tags: List[str], top_n=-1):
    """
    Get champions data from the graph.
    """
    for tag in tags:
        tag_elements = get_nodes_by_tag(graph, tag)
        tag_data = []
        for tag_element in tag_elements:
            data = graph.nodes[tag_element]
            neighbors = list(graph.neighbors(tag_element))
            betweeness = nx.betweenness_centrality(graph)[tag_element]
            closeness = nx.closeness_centrality(graph)[tag_element]
            influence = data.get("influence_score", 0.0)
            tag_data.append(
                {
                    "name": tag_element,
                    "data": data,
                    "neighbors": neighbors,
                    "betweeness": betweeness,
                    "closeness": closeness,
                    "score": np.mean([betweeness, closeness, influence]),
                }
            )
    best_tag_elements = sorted(tag_data, key=lambda x: x["score"], reverse=True)
    if top_n > 0:
        best_tag_elements = best_tag_elements[:top_n]
    else:
        best_tag_elements = [best_tag_elements[0]]
    return best_tag_elements


def get_org_graph(seller: str, buyer: str, initiative: str):
    """
    Get the org graph for the given org.
    """

    print(f"Getting org graph for {seller} and {buyer}")

    graph_storage_path = db_storage_path() / "graphs"
    os.makedirs(graph_storage_path, exist_ok=True)

    graph_file_path = graph_storage_path / f"{seller.lower()}_{buyer.lower()}.pkl"
    if os.path.exists(graph_file_path):
        with open(graph_file_path, "rb") as f:
            all_graphs = pickle.load(f)
        print(f"Loaded {len(all_graphs)} graphs from {graph_file_path}")
        return all_graphs[
            "Investment in R&D (building new products, integrating AI, enhancing existing platform)"
        ][initiative].graph.to_undirected()
    else:
        raise FileNotFoundError(f"Graph storage path {graph_file_path} does not exist.")


def ask_kg(seller: str, buyer: str, initiative: str, query: str) -> str:
    agents = {
        "OrgGraphDataExtractionExpert": dict(
            role="Organizational Graph Data Extraction Expert",
            goal=(
                "You are an expert in extracting relevant data from the organization graph of a company."
            ),
            backstory=(
                "You have access to the organization graph of '{buyer}' that is a buyer of a seller, i.e., '{seller}' for a specific initiative - '{initiative}' and can extract relevant data from it. "
                "The organization graph is a directed graph where nodes represent people and edges represent relationships between them. "
                "The graph is annotated with various attributes such as influence score, node type, and tags. "
                "You can use this information to extract relevant data for the user. "
                "You can also use this information to extract relevant data for the user. "
            ),
            tools=[
                BestPathExtractor(),
                BestNodeTypesExtractor(),
            ],
        )
    }

    tasks = {
        "graph_querying_task": dict(
            name="Organizational Graph Data Extraction",
            description=(
                "You can extract relevant data from the organization graph of '{buyer}' that is a buyer of a seller, i.e., '{seller}'  for a specific initiative - '{initiative}' using a natural language query."
                "Provide the answer to the below query -\n"
                "{query}"
                "If you infer that the query will be better answered by using the tools, then use the tools to get the answer."
                "Extract the relevant data from the query to pass as input to the tools."
                "While using the tools, in case of literal values, infer the closest match from the query and use it as input to the tools. "
            ),
            expected_output=(
                "The output should be a markdown formatted string with the relevant data extracted from the organization graph."
            ),
            agent="OrgGraphDataExtractionExpert",
        )
    }

    crew = get_crew(agent_templates=agents, task_templates=tasks)
    response = crew.kickoff(
        inputs={
            "seller": seller,
            "buyer": buyer,
            "initiative": initiative,
            "query": query,
        }
    )
    return response.raw


def save_org_graph(graph, seller: str, buyer: str, initiative: str):
    """
    Save the org graph to a file.
    """
    graph_storage_path = db_storage_path() / "graphs"
    os.makedirs(graph_storage_path, exist_ok=True)
    graph_file_path = graph_storage_path / f"{seller.lower()}_{buyer.lower()}.pkl"
    if os.path.exists(graph_file_path):
        with open(graph_file_path, "rb") as f:
            all_graphs = pickle.load(f)
    else:
        all_graphs = {}
    all_graphs[initiative] = graph
    
    with open(graph_file_path, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"Saved {len(graph)} graphs to {graph_file_path}")
