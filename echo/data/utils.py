import os
from typing import Dict, List
import pandas as pd


def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_buyer_insights_categories_df():
    return pd.read_csv(f"{get_current_dir()}/.csvs/Buyer_Insights_Categories.csv")


def get_seller_relevant_link_categories():
    link_extraction_df = pd.read_csv(f"{get_current_dir()}/.csvs/Link_Extraction.csv")
    categories = {r["Section"]: dict(r) for _, r in link_extraction_df.iterrows()}
    return categories


def get_category_prompts():
    categories = get_seller_relevant_link_categories()
    category_prompts = {
        category: (
            f"Task: {r['Prompt']}\nPurpose: {r['Purpose']}\nExamples: {r['Examples']}\n"
        )
        for category, r in categories.items()
    }
    return category_prompts


def generate_enums():
    def create_enums(enums_data: Dict[str, List[str]], filename: str = 'echo/data/index_enums.py'):
        """
        Creates a Python file defining an Enum class.

        Parameters:
        - name (str): The name of the Enum class.
        - values (list[str]): List of enum member names.
        - filename (str): Optional filename. If None, uses name + ".py"
        """
        import string
        def process_name_to_enum_name(name: str):
            ## remove special characters and spaces
            ## convert to uppercase
            ## replace spaces with underscores
            
            name = name.translate(str.maketrans('', '', string.punctuation))
            name = name.replace(' ', '_')
            name = name.upper()
            return name
            
        enum_code = "from enum import Enum\n\n"
        for name, values in enums_data.items():
            enum_code += f"\n\nclass {name}(Enum):\n"
            for val in values:
                enum_code += f"    {process_name_to_enum_name(val)} = '{val}'\n"

        with open(filename, "w") as f:
            f.write(enum_code)
        print(f"Enums created in {filename}")

    if not os.path.exists(f"{get_current_dir()}/.csvs/Buyer_Insights_Categories.csv"):
        print("Buyer Insights Categories CSV file not found.")
        return
    if not os.path.exists(f"{get_current_dir()}/.csvs/Link_Extraction.csv"):
        print("Link Extraction CSV file not found.")
        return
    if os.path.exists('echo/data/index_enums.py'):
        print("Enums file already exists.")
        return
    if not os.path.exists('echo/data'):
        os.makedirs('echo/data')
    # Example usage:
    seller_categories = list(get_category_prompts().keys()) + ['Landing Page']
    buyer_categories = [r['Category'] for _, r in get_buyer_insights_categories_df().iterrows()] + ['Landing Page']
    create_enums({
        "SellerIndexQueryTypes": seller_categories,
        "BuyerIndexQueryTypes": buyer_categories,
    })
    print("Enums created")

generate_enums()