DISCOVERY_ANALYSIS_PROMPT = """
Given a the discovery sales call transcript between {buyer} and {seller}, you need to identify the following key elements about the {buyer}'s data for the {stakeholder} of {buyer} present in the call -\n"
1. {buyer} 's Pain Points - The pain points expressed by the buyer during the call.\n"
2. {buyer} 's Objections - The objections raised by the buyer during the call.\n"
3. {buyer} 's Time Lines - The time lines mentioned by the buyer during the call.\n"
4. {buyer} 's Success Indicators - The success indicators mentioend by the buyer during the call.\n"
5. {buyer} 's Budget Constraints - The budget constraints expressed by the buyer during the call.\n"
6. {buyer} 's Decision Committee - The members of the decision mentioned by the buyer during the call.\n"
7. {buyer} 's Competition - The competitors mentioned by the buyer during the call.\n"
If there is no information about any of the above points, you should provide response as - No information found. "
The goal is to provide insights to the sales team for improving future calls and addressing the buyer's needs effectively and also to prepare for the discovery call."
The discovery call transcript:\n{discovery_transcript}\n"
"""

DEMO_BUSINESS_CASE_CONTENT_CHAMPION = """
You are an elite enterprise-sales deal architect.
Goal: create a ONE-PAGE Business Case champions can forward internally. This need not be grounded in real information but is a demo business case.
Feel free to estimate information based on the buyer and seller supplied below.
Stay ≤ 320 *body* words (tables & headings excluded).

VOICE & STYLE RULES
• ≥ 30 % of the words must come verbatim from “buyer_phrases” or any other buyer-supplied text.
– If the buyer-word ratio < 30 %, regenerate with more buyer wording.

• Use neutral, data-driven language; avoid seller adjectives (innovative, cutting-edge, etc.).

• Embed hyperlinks **only** when provided; label them explicitly (e.g. “SOC-2 Certificate”).

• No mention of AI, GPT, or internal tools.

• If any required placeholder is missing, output “<<MISSING_FIELD: X>>” and finish.


OUTPUT  (markdown)

SAMPLE BUSINESS CASE FORMAT

# {exec_priority} – avoid {neg_outcome} by {critical_event}

*Developed by*: **{champion_name}**, with **{key_players}**

---

**Headline**

Because **{buyer_phrases[0]}** is reshaping {industry}, we should **{one_liner}** by **{investment.timeline}**.

Doing so avoids **{neg_outcome}** and unlocks **{positive_outcome_phrase}**.

---

### The Problem Statement

Every **{problem.frequency}**, **{problem.reach}** experience **{problem.pain_description}**, costing **{problem.cost}**

(*source – {problem.baseline_source}*). If unaddressed by **{critical_event}**, **{problem.gets_worse_if}**.

---

### Recommended Approach

- *Pain ➜ Capability ➜ Proof*
1. **{problem.pain_description}** ➜ **{seller.differentiators[0]}** ➜ [{seller.proof_links[0].label}](https://www.notion.so/abitnikunj/%7Bseller.proof_links%5B0%5D.url%7D)
2. **{next_pain}** ➜ **{seller.differentiators[1]}** ➜ [{seller.proof_links[1].label}](https://www.notion.so/abitnikunj/%7Bseller.proof_links%5B1%5D.url%7D)
3. Sandbox rollout & multilingual support ensure adoption across {problem.reach}.
• Success factors: **{investment.people_req}**, SSO integration, store-level champions.

---

### Target Outcomes

| Metric | Current | Target by {critical_event} | Δ |
| --- | --- | --- | --- |
| {outcomes[0].metric} | {outcomes[0].current} | {outcomes[0].target} | {{Δ1}} |
| {outcomes[1].metric} | {outcomes[1].current} | {outcomes[1].target} | {{Δ2}} |
| {outcomes[2]?.metric} | {outcomes[2]?.current} | {outcomes[2]?.target} | {{Δ3}} |

*(Δ = Target − Current when both are numeric; leave blank otherwise.)*

---

### Required Investment

- **Budget**: {investment.budget_note} – {investment.per_unit_cost}; payback in < {investment.roi_payback_months} mo.

• **People & Time**: {investment.people_req}. Kickoff {investment.timeline}.

• **Next Step**: approve charter this week to meet **{critical_event}**.

---

### Evidence & Compliance

{# for each evidence item}[*{label}*]

"""