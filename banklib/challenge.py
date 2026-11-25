"""
Final challenge (slide 30) - "Your AI Banking Assistant": one report for C101 that
combines all five demos and answers six questions. Python gathers the evidence;
Ollama writes the report; humans decide.
"""
import json

from . import customer, data, investigation, llm, openbanking, operations, policy_rag, prompts
from .utils import inr

NOT_AUTOMATED = [
    "Declaring a transaction fraudulent or blocking/closing the account permanently",
    "Approving or rejecting a refund or chargeback",
    "Approving, declining or repricing credit (e.g., new loans, limit changes)",
    "Changing customer master data or bank/beneficiary details",
    "Sending final communication to the customer without human review",
]


def gather(customer_id="C101"):
    s, _ = customer.customer_360(customer_id)                        # Demo 1
    st, tx = investigation.analytics(customer_id=customer_id)        # Demo 2
    flags = investigation.flag_transactions(tx, st)
    fi, consent_log = openbanking.fetch_fi_data(data.consent(customer_id))   # Demo 3
    m, ex, _ = openbanking.aggregate(fi)
    prof = openbanking.profile_facts(m, ex)
    idx = policy_rag.PolicyIndex()                                     # Demo 4
    pol = {}
    for q in ["duplicate EMI debit refund procedure evidence", "unauthorised international card transaction investigation"]:
        hits = idx.search(q, k=2)
        pol[q] = [f"{h['policy']} - Section {h['section']} ({h['heading']}): {h['text'][:300]}" for h in hits]
    cases = data.cases()                                               # Demo 5
    open_cases = cases[(cases.Customer_ID == customer_id) & (cases.Status != "Closed")]
    evidence = {
        "1_financial_profile": {"crm": {k: s[k] for k in ("customer", "segment", "declared_monthly_income",
                                                          "total_fixed_obligations", "emi_to_income_pct")},
                                "open_banking": prof},
        "2_unusual_transactions": flags[flags.Investigate == "Yes"].assign(Amount=lambda d: d.Amount.map(inr))
                                  [["Transaction", "Time", "Amount", "Merchant", "Country", "Flags", "Risk_Level"]].to_dict("records"),
        "3_pending_operational_issues": {"open_complaints": s["open_complaints"],
                                         "evidence_in_transactions": s["possible_duplicate_debits"],
                                         "open_cases": open_cases[["Case_ID", "Type", "Sub_Type", "Priority", "Status"]].to_dict("records")},
        "4_policy_guidance": pol,
        "5_investigate_next_inputs": "derive from items 2 and 3",
        "6_not_for_automation": NOT_AUTOMATED,
        "data_access": consent_log.to_dict("records"),
    }
    return evidence


def report_prompt(evidence):
    return prompts.CHALLENGE_REPORT.format(facts=json.dumps(evidence, default=str, indent=1))


def report(evidence):
    txt = llm.ask(prompts.CHALLENGE_REPORT.format(facts=json.dumps(evidence, default=str, indent=1)),
                  system=prompts.SYSTEM_BANK)
    if txt:
        return txt, False
    e = evidence
    p = e["1_financial_profile"]["open_banking"]
    ms = p["monthly_summary"]
    lines = ["1. Financial profile",
             f"- Declared income {e['1_financial_profile']['crm']['declared_monthly_income']} (CRM); Open Banking shows "
             f"{len(p['income_streams'])} income streams totalling {ms['total_income']} in {p['latest_month']}.",
             f"- Outflow {ms['total_outflow']} -> net {ms['net_available_cash']}; card utilisation "
             f"{p['credit_utilisation_pct']:.0f}%; health {p['rule_based_health']}.",
             "2. Unusual transactions"]
    lines += [f"- {t['Transaction']} {t['Amount']} {t['Merchant']} ({t['Country']}, {t['Time']}): {t['Flags']} -> {t['Risk_Level']}"
              for t in e["2_unusual_transactions"]]
    lines += ["3. Pending operational issues"] + [f"- {c}" for c in e["3_pending_operational_issues"]["open_complaints"]]
    lines += [f"- Open case {c['Case_ID']}: {c['Sub_Type']} ({c['Priority']}, {c['Status']})"
              for c in e["3_pending_operational_issues"]["open_cases"]]
    lines += ["4. Relevant policy guidance"] + [f"- {x.split(':')[0]}" for v in e["4_policy_guidance"].values() for x in v]
    lines += ["5. Investigate next",
              "- Validate the duplicate EMI in logs and refund the duplicate amount per Refunds Policy 4.2.",
              "- Verify device/IP/3DS for TX1004 and TX1005; confirm TX1003 with the customer.",
              "- Clarify whether the second income stream is recurring before any credit discussion.",
              "6. Not to be decided automatically by AI"] + [f"- {x}" for x in e["6_not_for_automation"]]
    return "\n".join(lines), True
