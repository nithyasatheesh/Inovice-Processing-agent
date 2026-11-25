"""
Demo 3 - Open Banking Financial Intelligence (slides 10-15, 21-23).
Consent artefact -> authorised fetch from providers (data minimisation) ->
Python aggregation & cash-flow analysis -> Ollama financial profile -> analyst review.

This is a MOCK account-aggregator flow (JSON files stand in for provider APIs) to
show the controls: no consent / revoked / out-of-scope data => no data reaches the model.
"""
import json
from datetime import date

import pandas as pd

from . import config, data, llm, prompts
from .utils import inr

FI_OF_PROVIDER_FILE = {"NOVABANK": "DEPOSIT", "BANK_B": "DEPOSIT", "BANK_C_CARD": "CREDIT_CARD",
                       "BANK_C_LOAN": "LOAN", "BANK_C_INVEST": "MUTUAL_FUNDS"}


class ConsentError(Exception):
    pass


def validate_consent(consent, purpose_required=None, today=None):
    today = today or config.TODAY
    if consent.get("status") != "ACTIVE":
        raise ConsentError(f"Consent {consent.get('consent_id')} is {consent.get('status')} - no data can be fetched.")
    if date.fromisoformat(consent["consent_expiry"]) < today:
        raise ConsentError("Consent has expired.")
    if purpose_required and purpose_required.lower() not in consent["purpose"].lower():
        raise ConsentError(f"Requested purpose '{purpose_required}' is not covered by the consent.")
    return True


def fetch_fi_data(consent, fip=None):
    """Return only the FI types and date range the customer consented to (data minimisation). Logs every decision.
    fip: {provider_key: payload} - defaults to the sample provider files."""
    validate_consent(consent)
    log, allowed = [], {}
    lo, hi = consent["data_range"]["from"], consent["data_range"]["to"]
    for key, payload in (data.fip_files() if fip is None else fip).items():
        fi = FI_OF_PROVIDER_FILE.get(key, payload.get("fi_type"))
        if fi not in consent["fi_types"]:
            log.append({"provider": payload["fip_name"], "fi_type": fi, "decision": "NOT FETCHED - outside consent scope"})
            continue
        p = dict(payload)
        if "transactions" in p:
            p["transactions"] = [t for t in p["transactions"] if lo <= t["date"] <= hi]
        log.append({"provider": payload["fip_name"], "fi_type": fi, "decision": "Fetched under consent " + consent["consent_id"]})
        allowed[key] = p
    return allowed, pd.DataFrame(log)


def aggregate(fi):
    """Monthly view across providers (Python calculates everything)."""
    rows = []
    for key, p in fi.items():
        for t in p.get("transactions", []):
            rows.append({"month": t["date"][:7], "provider": p["fip_name"], "tag": t["tag"], "amount": t["amount"]})
    tx = pd.DataFrame(rows)
    piv = tx.pivot_table(index="month", columns="tag", values="amount", aggfunc="sum", fill_value=0)
    m = pd.DataFrame({
        "income": piv.get("INCOME", 0),
        "expenses": piv.get("EXPENSE", 0),
        "emi": piv.get("EMI", 0),
        "credit_card_payment": piv.get("CARD_PAYMENT", 0),
    })
    m["total_outflow"] = m.expenses + m.emi + m.credit_card_payment
    m["net_available_cash"] = m.income - m.total_outflow
    m["savings_rate_pct"] = (m.net_available_cash / m.income * 100).round(1)
    m["debt_to_income_pct"] = ((m.emi + m.credit_card_payment) / m.income * 100).round(1)
    income_streams = tx[tx.tag == "INCOME"].groupby("provider").amount.agg(["mean", "std", "count"]).round(0)
    card = fi.get("BANK_C_CARD", {}).get("statements", [])
    loan = fi.get("BANK_C_LOAN", {}).get("schedule", [])
    extras = {
        "income_streams": {k: {"avg": float(v["mean"]), "std": float(v["std"]), "months": int(v["count"])}
                           for k, v in income_streams.iterrows()},
        "income_volatility_pct": round(float(m.income.std() / m.income.mean() * 100), 1),
        "credit_utilisation_pct": round(card[-1]["statement_balance"] / card[-1]["credit_limit"] * 100, 1) if card else None,
        "credit_limit": card[-1]["credit_limit"] if card else None,
        "loan_outstanding": loan[-1]["outstanding_after"] if loan else None,
        "expense_trend": m.expenses.astype(float).to_dict(),
    }
    return m.reset_index(), extras, tx


def indicators(m, extras):
    last = m.iloc[-1]
    risk = []
    if last.net_available_cash < 0:
        risk.append(f"Cash shortfall of {inr(-last.net_available_cash)} in {last.month}")
    if extras["credit_utilisation_pct"] and extras["credit_utilisation_pct"] >= 70:
        risk.append(f"High credit card utilisation ({extras['credit_utilisation_pct']:.0f}%)")
    if last.debt_to_income_pct >= 35:
        risk.append(f"High debt burden (EMI + card = {last.debt_to_income_pct:.1f}% of income)")
    if extras["income_volatility_pct"] >= 3:
        risk.append(f"Some income variability ({extras['income_volatility_pct']}%) - second stream varies month to month")
    # transparent rating: Weak if the shortfall exceeds 10% of income, Moderate if any risk flag, else Strong
    if last.net_available_cash < -0.10 * last.income:
        health = "Weak"
    elif risk:
        health = "Moderate"
    else:
        health = "Strong"
    return risk, health


def profile_facts(m, extras):
    last = m.iloc[-1]
    risk, health = indicators(m, extras)
    return {
        "latest_month": last.month,
        "monthly_summary": {"total_income": inr(last.income), "expenses": inr(last.expenses), "emi": inr(last.emi),
                            "credit_card_payment": inr(last.credit_card_payment),
                            "total_outflow": inr(last.total_outflow), "net_available_cash": inr(last.net_available_cash)},
        "income_streams": {k: inr(v["avg"]) + " avg" for k, v in extras["income_streams"].items()},
        "income_volatility_pct": extras["income_volatility_pct"],
        "savings_rate_pct": float(last.savings_rate_pct),
        "debt_to_income_pct": float(last.debt_to_income_pct),
        "emi_to_income_pct": round(float(last.emi / last.income * 100), 1),
        "credit_utilisation_pct": extras["credit_utilisation_pct"],
        "loan_outstanding": inr(extras["loan_outstanding"]),
        "rule_based_health": health, "rule_based_risks": risk,
        "three_month_net_cash": {r.month: inr(r.net_available_cash) for r in m.itertuples()},
    }


def profile_prompt(facts):
    return prompts.FINANCIAL_PROFILE.format(facts=json.dumps(facts, indent=1, default=str))


def financial_profile(facts, use_llm=True):
    txt = llm.ask(profile_prompt(facts), system=prompts.SYSTEM_BANK) if use_llm else None
    if txt:
        return txt, False
    f = facts
    s = f["monthly_summary"]
    lines = ["Financial Overview:",
             f"- {len(f['income_streams'])} income streams identified ({', '.join(f'{k}: {v}' for k, v in f['income_streams'].items())}); "
             f"total income {s['total_income']} in {f['latest_month']}.",
             f"- High recurring obligations: EMI {s['emi']} and credit card payment {s['credit_card_payment']}.",
             f"- Net available cash {s['net_available_cash']} (total outflow {s['total_outflow']}).",
             f"Financial Health: {f['rule_based_health']}",
             "Risk Indicators:"] + [f"- {r}" for r in f["rule_based_risks"]] + [
             "Clarifications Required:", "- Is the second income recurring?",
             "- Are credit card payments revolving or fully settled?", "- Are any expenses temporary?",
             "Suggested Next Steps: discuss reducing card utilisation; review repayment burden before any new credit."]
    return "\n".join(lines), True
