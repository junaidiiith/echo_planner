import os
import pandas as pd


def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_buyer_insights_categories_df():
    return pd.read_csv(f'{get_current_dir()}/.csvs/Buyer_Insights_Categories.csv')


def get_relevant_link_categories():
    link_extraction_df = pd.read_csv(f'{get_current_dir()}/.csvs/Link_Extraction.csv')
    categories = {
        r['Section']: dict(r)
        for _, r in link_extraction_df.iterrows()
    }
    return categories


def get_category_prompts():
    categories = get_relevant_link_categories()
    category_prompts = {
    category: (
            f"Task: {r['Prompt']}\n"
            f"Purpose: {r['Purpose']}\n"
            f"Examples: {r['Examples']}\n"
        )
        for category, r in categories.items()
    }
    return category_prompts
