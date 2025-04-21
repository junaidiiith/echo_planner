from enum import Enum



class SellerIndexQueryTypes(Enum):
    HERO_HEADLINE__SUBHEAD = 'Hero Headline & Subhead'
    FEATURES__CAPABILITIES = 'Features & Capabilities'
    PERSONAFOCUSED_PAGES = 'Persona-Focused Pages'
    USE_CASE_PAGES = 'Use Case Pages'
    CUSTOMER_LOGOS = 'Customer Logos'
    CASE_STUDIES__OUTCOMES = 'Case Studies / Outcomes'
    BLOG__THOUGHT_LEADERSHIP = 'Blog & Thought Leadership'
    CTA__CONVERSION_COPY = 'CTA / Conversion Copy'
    VIDEO__WEBINAR_CONTENT = 'Video & Webinar Content'


class BuyerIndexQueryTypes(Enum):
    STRATEGIC_INITIATIVES = 'Strategic Initiatives'
    LEADERSHIP_MOVEMENTS = 'Leadership Movements'
    JOB_LISTINGS = 'Job Listings'
    EXISTING_TOOLS__TECH_STACK = 'Existing Tools & Tech Stack'
    SOCIAL__COMMUNITY_SIGNALS = 'Social & Community Signals'
    CONTENT_SHARED_BY_BUYER_ORG = 'Content Shared by Buyer Org'
    CUSTOMER_REVIEWS = 'Customer Reviews'
