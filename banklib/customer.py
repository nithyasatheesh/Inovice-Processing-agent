"""
Warm-up (slide 8): one complaint handled three ways - rule, ML, GenAI.
Demo 1 (slides 16-17, 30): Customer Intelligence Assistant - RM briefing in minutes.
"""
import json
import math
import re
from collections import Counter

import pandas as pd

from . import data, llm, prompts
from .utils import inr

# ============================================================ warm-up: three approaches
COMPLAINT = ("I was charged twice for my credit card payment of Rs.12,500. The amount is debited from my "
             "account, but my card balance is not updated.")

TRAINING = [  # tiny labelled history for the "ML" approach
    ("charged twice on my card", "Card", "High"), ("card payment not reflected in card balance", "Card", "High"),
    ("credit card bill paid but still showing due", "Card", "Medium"), ("annual fee charged on card", "Card", "Low"),
    ("upi payment failed but money debited", "Payments", "High"), ("neft transfer delayed", "Payments", "Medium"),
    ("loan emi deducted twice", "Loans", "High"), ("loan statement not received", "Loans", "Low"),
    ("branch staff were rude", "Service", "Low"), ("app login not working", "Digital", "Medium"),
]


def rule_based(text):
    t = text.lower()
    if "card" in t:
        return "Category = Card -> route to Card Team"
    if "loan" in t or "emi" in t:
        return "Category = Loans -> route to Loan Servicing"
    return "Category = General -> route to Service Desk"


def ml_predict(text):
    """Multinomial naive Bayes in a few lines - learns from past labelled complaints."""
    tok = lambda s: re.findall(r"[a-z]+", s.lower())  # noqa: E731
    out = {}
    for target in (1, 2):
        labels = Counter(r[target] for r in TRAINING)
        words = {lab: Counter(w for r in TRAINING if r[target] == lab for w in tok(r[0])) for lab in labels}
        vocab = {w for c in words.values() for w in c}
        scores = {}
        for lab in labels:
            total = sum(words[lab].values())
            s = math.log(labels[lab] / len(TRAINING))
            for w in tok(text):
                s += math.log((words[lab][w] + 1) / (total + len(vocab)))
            scores[lab] = s
        out["category" if target == 1 else "urgency"] = max(scores, key=scores.get)
    return out


def genai_prompt(text):
    return prompts.COMPLAINT_GENAI.format(complaint=text, context=json.dumps(_GENAI_CONTEXT))


_GENAI_CONTEXT = {"transactions": [{"date": "2026-11-12", "description": "Card bill payment XX4417", "amount": "\u20b912,500"},
                                   {"date": "2026-11-12", "description": "Card bill payment XX4417", "amount": "\u20b912,500"}],
                  "card_ledger": "only one payment of Rs.12,500 posted to the card",
                  "previous_interactions": "customer called on 13-Nov; told to wait 24 hours",
                  "policy": "Refunds & Chargebacks Policy 4.2 - verify duplicate, confirm with customer, refund duplicate amount"}


def briefing_prompt(summary):
    return prompts.CUSTOMER_BRIEFING.format(facts=json.dumps(summary, indent=1))


def genai_response(text):
    context = {"transactions": [{"date": "2026-11-12", "description": "Card bill payment XX4417", "amount": inr(12500)},
                                {"date": "2026-11-12", "description": "Card bill payment XX4417", "amount": inr(12500)}],
               "card_ledger": "only one payment of Rs.12,500 posted to the card",
               "previous_interactions": "customer called on 13-Nov; told to wait 24 hours",
               "policy": "Refunds & Chargebacks Policy 4.2 - verify duplicate, confirm with customer, refund duplicate amount"}
    txt = llm.ask(prompts.COMPLAINT_GENAI.format(complaint=text, context=json.dumps(context)), system=prompts.SYSTEM_BANK)
    if txt:
        return txt, False
    return ("1) Summary: customer charged twice for a Rs.12,500 card payment; card balance shows only one.\n"
            "2) Root cause: duplicate debit - payment processed twice from the account.\n"
            f"3) Impact: {inr(12500)} blocked from the customer's account; card limit unaffected.\n"
            "4) Action: validate the duplicate in transaction logs and initiate a refund per Policy 4.2.\n"
            "5) Draft reply: 'We're sorry for the trouble. We can see two debits of Rs.12,500 on 12-Nov and are "
            "verifying them now. We'll update you within 3 business days.'"), True


# ============================================================ Demo 1 - customer intelligence
def customer_360(customer_id="C101", profiles=None, transactions=None, complaints=None, products=None):
    """Steps 1-4: load data, calculate summary, retrieve complaints, combine. Pass DataFrames to use uploaded data."""
    p = (data.profiles() if profiles is None else profiles).set_index("Customer_ID").loc[customer_id]
    tx = data.transactions() if transactions is None else transactions
    tx = tx[tx.Customer_ID == customer_id]
    comp = data.complaints() if complaints is None else complaints
    comp = comp[comp.Customer_ID == customer_id]
    prod = data.products() if products is None else products
    prod = prod[prod.Customer_ID == customer_id]

    credits = tx[tx.Type == "Credit"].Amount.sum()
    debits = tx[tx.Type == "Debit"]
    by_cat = debits.groupby("Category").Amount.sum()
    dups = debits[debits.duplicated(["Date", "Category", "Amount", "Narration"], keep="first")]
    obligations = {c: float(debits[(debits.Category == c)].drop(dups.index, errors="ignore").Amount.sum())
                   for c in ["Rent", "EMI", "Credit Card", "Utilities"]}
    total_oblig = sum(obligations.values())
    summary = {
        "customer": p.Name, "customer_id": customer_id, "segment": p.Segment, "occupation": p.Occupation,
        "declared_monthly_income": inr(p.Monthly_Income), "credits_this_month": inr(credits),
        "fixed_obligations": {k: inr(v) for k, v in obligations.items()},
        "total_fixed_obligations": inr(total_oblig),
        "surplus_after_obligations": inr(p.Monthly_Income - total_oblig),
        "emi_to_income_pct": round(float(obligations["EMI"] / p.Monthly_Income * 100), 1),
        "obligations_to_income_pct": round(float(total_oblig / p.Monthly_Income * 100), 1),
        "possible_duplicate_debits": [f"{r.Date:%d-%b} {r.Narration} {inr(r.Amount)}" for r in dups.itertuples()],
        "open_complaints": [f"{r.Complaint} ({r.Priority}, open since {r.Date:%d-%b})" for r in comp[comp.Status == "Open"].itertuples()],
        "closed_complaints_12m": int((comp.Status == "Closed").sum()),
        "products": [f"{r.Product} {r.Account_Ref}: {r.Details}".strip(": ") for r in prod.itertuples()],
    }
    return summary, {"profile": p, "transactions": tx, "complaints": comp, "products": prod, "by_category": by_cat}


def briefing(summary):
    """Step 5: ask Ollama to prepare the customer briefing."""
    txt = llm.ask(prompts.CUSTOMER_BRIEFING.format(facts=json.dumps(summary, indent=1)), system=prompts.SYSTEM_BANK)
    if txt:
        return txt, False
    s = summary
    lines = [f"CUSTOMER BRIEFING - {s['customer']} ({s['customer_id']}, {s['segment']})",
             "Customer Profile:",
             f"- {s['occupation']}; declared monthly income {s['declared_monthly_income']}.",
             "Financial Profile:",
             f"- Fixed obligations {s['total_fixed_obligations']} ({s['obligations_to_income_pct']}% of income); "
             f"surplus {s['surplus_after_obligations']}.",
             f"- EMI is {s['emi_to_income_pct']}% of income; credit card payment is a large monthly outflow.",
             "Operational Issues:"]
    lines += [f"- {c}" for c in s["open_complaints"]]
    lines += [f"- Evidence in transactions: {d}" for d in s["possible_duplicate_debits"]]
    lines += ["Meeting Priorities:", "1. Address the duplicate EMI issue (evidence attached).",
              "2. Provide status of the card refund.", "3. Review customer satisfaction.",
              "Relationship Opportunity:", "- Focus on service resolution before discussing additional products."]
    return "\n".join(lines), True
