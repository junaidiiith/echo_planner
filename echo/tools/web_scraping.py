from crewai import Agent, Task
from echo.echo_agent import EchoAgent
from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import concurrent.futures
from typing import List
from echo.settings import CHUNK_OVERLAP, CHUNK_SIZE, MAX_TEXT_TOKENS
from echo.llm_utils import get_response

from echo.utils import (
    get_crew_llm,
    format_response,
    get_num_tokens,
    get_text_upto_tokens,
)


class Link(BaseModel):
    text: str = Field(
        ...,
        title="Text of the navigation link",
        description="The text of the navigation link",
    )
    url: str = Field(..., title="URL of the navigation link")


class ExtractSchema(BaseModel):
    nav_links: List[Link] = Field(
        ...,
        title="Navigation links on the website from Link Object",
        description="List of navigation links on the website from Link Object",
    )


MAX_CONCURRENT_REQUESTS = 10
MAX_POTENTIAL_LINKS = 100

DATA_EXTRACTION_SYS_PROMPT = """
You are an expert in sales such that you can extract out the content from a website of a company that would be relevant to a potential client of that company. 
"""

DATA_EXTRACTION_PROMPT = """
You are provided with a webpage link and the extracted content of the webpage.

You need to analyze the webpage content and extract information that would be most relevant and persuasive to a potential client. 
Focus on key offerings, unique selling points, pricing (if available), case studies, testimonials, competitive advantages, and any value propositions that differentiate this company from its competitors. 
Remove any fluff, internal jargon, or non-client-relevant details. The final output should be clear, concise, and sales-oriented.
You should provide a title of the page that is representative of the content extracted and then provide a summary of the content extracted from the webpage.
THE SUMMARY SHOULD BE UNDER 1000 WORDS.
---

Here is the content extracted from the webpage - 

Webpage Link: {webpage}
Extracted Content: {content}
"""


DATA_SUMMARIZATION_SYS_PROMPT = """
You are an expert in sales such that you can summarize the content extracted from a website of a company that would be relevant to a potential client of that company.
"""

DATA_SUMMARIZATION_PROMPT = """
You are provided with the extracted contents from some webpages.
You need to summarize the extracted content that would be most relevant and persuasive to a potential client.
Focus on key offerings, unique selling points, pricing (if available), case studies, testimonials, competitive advantages, and any value propositions that differentiate this company from its competitors.
Remove any fluff, internal jargon, or non-client-relevant details. The final output should be clear, concise, and sales-oriented.

THE SUMMARY MUST COVER ALL THE KEY POINTS FROM THE EXTRACTED CONTENT.

Below are the extracted contents from the webpages -
{content}
"""


def extract_text_from_url(url, timeout=30):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(
            url, headers=headers, timeout=timeout
        )  # Set timeout to 30 seconds
    except requests.exceptions.Timeout:
        print("Request timed out after 30 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            print(f"Warning: Content type is not HTML: {content_type}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    else:
        print(f"Failed to fetch page, status code: {response.status_code}")

    return None


async def extract_nav_links(url, num_links=15):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Start with the header if it exists; otherwise, use the body.
    container = soup.find("header") or soup.body

    # Find all anchor tags within the container.
    all_links = container.find_all("a", href=True)

    # Filter links by ensuring they have non-empty text (you might add further heuristics)
    links = [
        (link.get_text(strip=True), link["href"])
        for link in all_links
        if link.get_text(strip=True)
    ]
    content = get_text_upto_tokens(soup.get_text(separator="\n", strip=True), 4000)

    async def get_nav_links(num_links=num_links):
        agent = Agent(
            role="Website Crawling Expert",
            goal="Filter out the relevant links that might be relevant for an Account Executive to understand the company's offerings and services",
            backstory="You are an expert in extracting or filter out all the important links from a given website that might be relevant for an Account Executive to understand the company's offerings and services.",
            llm=get_crew_llm(),
        )
        task = Task(
            name="Extracting Navigation Links",
            description=(
                "Given below the {website} landing page content and a list navigation links from {website}, filter out the top {num_links} most relevant links that might be relevant for an Account Executive to understand the company's offerings and services."
                "A link is relevant if it can provide information about the company's products, services, pricing, case studies, testimonials, competitive advantages, and any value propositions that differentiate this company from its competitors."
                "The final output should have atmost {num_links} links."
                "---Website Content---\n"
                "{content}"
                "---End of Content---"
                "---Extracted Navigation Links---\n"
                "{links}"
                "---End of Navigation Links---"
                "Based on this information, filter only the relevant links that might be relevant for an Account Executive to understand the company's offerings and services."
            ),
            expected_output=(
                "The response should conform to the provided schema."
                "You need to extract the following information in the following pydantic structure -\n"
                "{pydantic_structure}\n"
            ),
            output_pydantic=ExtractSchema,
            agent=agent,
        )

        crew = EchoAgent(agents=[agent], tasks=[task])
        inputs = {
            "website": url,
            "content": content,
            "links": links_str,
            "num_links": num_links,
        }

        response = await crew.kickoff_async(inputs=inputs)
        return format_response(response)

    all_links: List[ExtractSchema] = list()
    for i in tqdm(range(0, len(links), num_links), desc="Extracting Navigation Links"):
        links_str = "\n".join(
            [f"{text}: {href}" for text, href in links[i : i + num_links]]
        )
        navbar_links = await get_nav_links()
        all_links.append(navbar_links)

    all_links = list(
        {link["url"]: link for al in all_links for link in al["nav_links"]}.values()
    )[:MAX_POTENTIAL_LINKS]
    links_str = "\n".join([f"{link['text']}: {link['url']}" for link in all_links])

    print("Getting final links")
    navbar_links = await get_nav_links()
    final_links = [link["url"] for link in navbar_links["nav_links"]]

    print("Final Links:", final_links)
    return [
        link if link.startswith("http") else f"{url}{link}"
        for link in final_links[:num_links]
    ]


def extract_data_from_webpage(
    content: str, system_prompt: str = DATA_EXTRACTION_SYS_PROMPT
):
    response = get_response(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ]
    )
    return response


async def extract_data_from_website(url: str):
    navbar_links = await extract_nav_links(url)
    extracted_results = extract_data_from_links(navbar_links)

    website_content = "\n\n".join(
        [f"Link: {r['link']}\nData: {r['data']}" for r in extracted_results]
    )

    if get_num_tokens(website_content) > MAX_TEXT_TOKENS:
        website_content = get_text_upto_tokens(website_content, MAX_TEXT_TOKENS)

    return website_content


def extract_data_from_links(
    links: List[str],
    user_prompt: str = DATA_EXTRACTION_PROMPT,
    system_prompt: str = DATA_EXTRACTION_SYS_PROMPT,
):
    extracted_results = []

    def process_link(link: str):
        print("Processing link:", link)
        try:
            # Construct the full URL if necessary.
            # Extract the text from the URL (assumes extract_text_from_url is defined elsewhere)
            extracted_text = extract_text_from_url(link)
            if not extracted_text:
                print(f"Failed to extract text from {link}")
                return None
            # Use the extracted text to get client-focused data
            extraction_prompt = user_prompt.format(webpage=link, content=extracted_text)
            extracted_data = extract_data_from_webpage(
                extraction_prompt, system_prompt=system_prompt
            )
            print(f"Extracted data from {link}")
            return {"link": link, "data": extracted_data}
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            return None

    # Use ThreadPoolExecutor to process links concurrently.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS
    ) as executor:
        # Submit all tasks and store futures in a dictionary.
        futures = {executor.submit(process_link, link): link for link in set(links)}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Summarizing Data From Links",
        ):
            result = future.result()
            if result is not None:
                extracted_results.append(result)

    # Filter out None results
    extracted_results = [result for result in extracted_results if result is not None]
    return extracted_results


def summarize_website_content(
    content: str,
    summarization_prompt: str = DATA_SUMMARIZATION_PROMPT,
    system_prompt: str = DATA_SUMMARIZATION_SYS_PROMPT,
):
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_text(content)

    summaries = [
        get_response(
            [
                {"role": "user", "content": system_prompt},
                {"role": "user", "content": summarization_prompt.format(content=doc)},
            ]
        )
        for doc in docs
    ]

    return "\n\n".join(summaries)
