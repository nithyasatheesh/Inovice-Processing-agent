"""
Every prompt sent to Ollama, in one place (edit here, re-run, compare).
Design rule used in all demos: Python retrieves and calculates; the LLM
explains, summarises and drafts. The LLM is never asked to compute.
"""

SYSTEM_BANK = (
    "You are an assistant for NovaBank staff. Use ONLY the data provided. Never invent figures, "
    "transactions or policy. Quote amounts exactly as given (Indian rupee format). If information "
    "is missing, say so. You support professionals; you do not make final decisions."
)

# Warm-up - slide 8
COMPLAINT_GENAI = """A customer wrote: "{complaint}"
Context (from bank systems): {context}
Write: 1) a one-line complaint summary, 2) the likely root cause, 3) customer impact,
4) recommended action for the agent, 5) a short, empathetic draft reply to the customer (do not promise a refund
before validation)."""

# Demo 1 - slides 16, 17, 30
CUSTOMER_BRIEFING = """You are a banking customer intelligence assistant preparing a relationship manager for a meeting.
Analyze this customer's financial situation. Identify financial pressure points, unresolved service issues,
and suggested actions before the relationship manager's meeting.
Use this structure: CUSTOMER BRIEFING; Customer Profile; Financial Profile; Operational Issues;
Meeting Priorities (numbered); Relationship Opportunity.
Use only the facts below.

Customer data (JSON): {facts}"""

# Demo 2 - slide 20 (prompt kept from the slide)
INVESTIGATION = """You are a banking transaction investigation assistant.
Analyze the transaction flags and customer history. Do not declare fraud.
Identify: 1) Suspicious indicators 2) Supporting evidence 3) Customer impact
4) Additional information required 5) Recommended investigation steps.
Finish with one line "Recommended next action: ...". Use only the facts below.

Input context from Python (JSON): {facts}"""

# Demo 3 - slide 23 (prompt kept from the slide)
FINANCIAL_PROFILE = """Analyze the aggregated financial data. Prepare a financial profile for an authorized banking analyst.
Include: income stability, spending behavior, debt obligations, cash-flow health, financial risk indicators,
and questions requiring clarification. Use headings: Financial Overview, Financial Health (one word:
Strong / Moderate / Weak), Risk Indicators, Clarifications Required, Suggested Next Steps.
Use only these calculated figures.

Aggregated data (JSON): {facts}"""

# Demo 4 - slide 25
POLICY_ANSWER = """Answer the operations question using ONLY the policy extracts below.
Structure: Applicable Policy, Procedure (numbered), Required Evidence, Escalation, Source References.
Cite every point as [Policy name - Section x.y]. If the extracts do not contain the answer, reply exactly:
"{refusal}"

Question: {question}

Policy extracts:
{context}"""

# Demo 5 - slides 27, 29
DAILY_OPS = """You are preparing the end-of-day operations report for the operations manager.
Using ONLY these calculated metrics, write: "Daily Operations Summary" (4-6 bullets) and
"Recommended Priorities" (numbered, most urgent first) with one reason each.

Metrics (JSON): {facts}"""

CASE_SUMMARY = """Summarise this operations case for a supervisor in 4 bullets: issue, what has been verified,
what is pending, recommended next step (cite the relevant policy if given). Use only the case data.

Case (JSON): {case}
Relevant policy extract: {policy}"""

OPS_INSIGHTS = """Explain the top operational issues of the last 30 days in 3-4 bullets for the operations manager,
using ONLY these counts. Suggest one improvement per issue.

Counts (JSON): {facts}"""

COPILOT_ROUTER = """Classify this operations request into one tool:
policy_qa (process / policy / procedure questions), case_summary (summarise a specific case),
ops_insights (trends, top issues, volumes), daily_report (today's operations report).
Request: "{q}"
Return JSON only: {{"tool": "<one of the tools>", "case_id": "<case id if mentioned, else null>"}}"""

# Final challenge - slide 30
CHALLENGE_REPORT = """You are a banking operations analyst's assistant. Using ONLY the evidence below, write a report
with these six numbered headings, answering each:
1. What is the customer's financial profile?
2. What unusual transactions exist?
3. What operational issues are pending?
4. What policy guidance is relevant? (cite sections)
5. What should the analyst investigate next?
6. What should NOT be decided automatically by AI?

Evidence (JSON): {facts}"""
