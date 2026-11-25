"""
Demo 5 - Intelligent Banking Operations Copilot (slide 27) and
Daily Operations Report (slides 28-29).
Excel cases -> Python metrics -> Ollama summary & priorities.
The copilot routes a request to: policy Q&A (Demo 4 RAG), case summary,
operational insights, or the daily report.
"""
import json
import re

import pandas as pd

from . import config, data, llm, policy_rag, prompts


def metrics(cases=None, now=None):
    cases = data.cases() if cases is None else cases
    now = pd.Timestamp(now or config.NOW)
    op = cases[cases.Status != "Closed"]
    breached = op[op.SLA_Due < now]
    cat = cases.Type.value_counts()
    last7 = cases[cases.Created >= now - pd.Timedelta(days=7)]
    prev7 = cases[(cases.Created < now - pd.Timedelta(days=7)) & (cases.Created >= now - pd.Timedelta(days=14))]
    trend = {t: {"last_7_days": int((last7.Type == t).sum()), "previous_7_days": int((prev7.Type == t).sum())}
             for t in ["Payment", "Complaint", "Fraud", "Refund"]}
    return {
        "as_of": f"{now:%d-%b-%Y %H:%M}",
        "total_cases": int(len(cases)), "open_cases": int(len(op)), "sla_breaches": int(len(breached)),
        "critical_cases": int((op.Priority == "Critical").sum()),
        "cases_by_category": {k: f"{v} ({v / len(cases):.0%})" for k, v in cat.items()},
        "open_by_category": op.Type.value_counts().to_dict(),
        "pending_refunds": int(((op.Type == "Refund")).sum()),
        "weekly_trend": trend,
        "breached_cases": breached[["Case_ID", "Type", "Priority", "Assigned_Team"]].to_dict("records"),
        "critical_open": op[op.Priority == "Critical"][["Case_ID", "Type", "Sub_Type"]].to_dict("records"),
    }


def daily_prompt(m):
    return prompts.DAILY_OPS.format(facts=json.dumps(m, default=str))


def daily_report(m):
    txt = llm.ask(prompts.DAILY_OPS.format(facts=json.dumps(m, default=str)), system=prompts.SYSTEM_BANK)
    if txt:
        return txt, False
    fr = m["weekly_trend"]["Fraud"]
    top_cat = max(m["cases_by_category"], key=lambda k: int(m["cases_by_category"][k].split()[0]))
    lines = [f"Daily Operations Summary ({m['as_of']})",
             f"- {m['open_cases']} open cases out of {m['total_cases']}.",
             f"- {m['sla_breaches']} SLA breaches.",
             f"- {m['critical_cases']} critical cases open.",
             f"- Fraud-related cases {'increased' if fr['last_7_days'] > fr['previous_7_days'] else 'did not increase'} "
             f"({fr['previous_7_days']} -> {fr['last_7_days']} week on week).",
             f"- {top_cat} cases represent the highest volume ({m['cases_by_category'][top_cat]}).",
             "Recommended Priorities:",
             "1. Critical fraud cases - customer funds at risk.",
             "2. SLA breaches - regulatory and customer-experience impact.",
             "3. Payment disputes - highest volume.",
             f"4. Pending refund cases ({m['pending_refunds']} open)."]
    return "\n".join(lines), True


def case_summary(case_id, cases=None, index=None):
    cases = data.cases() if cases is None else cases
    row = cases[cases.Case_ID.str.upper() == case_id.upper()]
    if row.empty:
        return f"Case {case_id} not found.", False
    r = row.iloc[0]
    case = {k: str(v) for k, v in r.items()}
    hits = (index or policy_rag.PolicyIndex()).search(f"{r.Type} {r.Sub_Type} {r.Notes}", k=1)
    pol = f"[{hits[0]['policy']} - Section {hits[0]['section']}] {hits[0]['text']}" if hits else "none"
    txt = llm.ask(prompts.CASE_SUMMARY.format(case=json.dumps(case), policy=pol), system=prompts.SYSTEM_BANK)
    if txt:
        return txt, False
    breached = r.Status != "Closed" and r.SLA_Due < pd.Timestamp(config.NOW)
    return (f"- Issue: {r.Type} / {r.Sub_Type} ({r.Priority}), status {r.Status}"
            f"{', SLA BREACHED' if breached else ''}; team {r.Assigned_Team}.\n"
            f"- Notes: {r.Notes or 'none recorded'}\n"
            f"- Relevant policy: {pol[:160]}...\n- Next step: follow the cited procedure and update the case."), True


def insights(cases=None, days=30):
    cases = data.cases() if cases is None else cases
    now = pd.Timestamp(config.NOW)
    recent = cases[cases.Created >= now - pd.Timedelta(days=days)]
    top = recent.groupby(["Type", "Sub_Type"]).size().sort_values(ascending=False).head(5)
    facts = {"period_days": days, "top_issues": [{"type": t, "issue": s, "cases": int(n)} for (t, s), n in top.items()]}
    txt = llm.ask(prompts.OPS_INSIGHTS.format(facts=json.dumps(facts)), system=prompts.SYSTEM_BANK)
    if txt:
        return txt, False, top
    return ("Top operational issues (last 30 days):\n" +
            "\n".join(f"- {t} / {s}: {n} cases" for (t, s), n in top.items())), True, top


def route(q):
    raw = llm.ask_json(prompts.COPILOT_ROUTER.format(q=q))
    cid = re.search(r"\bCS\d{3}\b", q.upper())
    if raw and raw.get("tool") in ("policy_qa", "case_summary", "ops_insights", "daily_report"):
        return raw["tool"], (raw.get("case_id") or (cid.group(0) if cid else None)), "ollama"
    t = q.lower()
    if cid or "summar" in t and "case" in t:
        return "case_summary", cid.group(0) if cid else "CS101", "keywords (fallback)"
    if any(w in t for w in ["top", "trend", "last 30", "volume", "issues"]):
        return "ops_insights", None, "keywords (fallback)"
    if "daily" in t or "today" in t or "report" in t:
        return "daily_report", None, "keywords (fallback)"
    return "policy_qa", None, "keywords (fallback)"


def copilot(q, cases=None, index=None):
    tool, cid, how = route(q)
    if tool == "policy_qa":
        r = policy_rag.answer(q, index)
        return {"tool": tool, "routed_by": how, "answer": r["answer"], "fallback": r["fallback"],
                "sources": [f"{h['policy']} {h['section']}" for h in r["sources"]]}
    if tool == "case_summary":
        txt, fb = case_summary(cid or "CS101", cases, index)
        return {"tool": tool, "routed_by": how, "answer": txt, "fallback": fb, "sources": [cid or "CS101"]}
    if tool == "ops_insights":
        txt, fb, _ = insights(cases)
        return {"tool": tool, "routed_by": how, "answer": txt, "fallback": fb, "sources": ["operations_cases.xlsx"]}
    txt, fb = daily_report(metrics(cases))
    return {"tool": tool, "routed_by": how, "answer": txt, "fallback": fb, "sources": ["operations_cases.xlsx"]}
