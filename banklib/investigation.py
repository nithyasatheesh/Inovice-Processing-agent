"""
Demo 2 - Intelligent / complex transaction investigation (slides 18-20, 31).
Transaction data -> Python analytics -> rule-based flags -> anomaly indicators ->
GenAI investigation report (Ollama) -> analyst review.  The AI never declares fraud.
"""
import json

import pandas as pd

from . import config, data, llm, prompts
from .utils import inr


def analytics(tx=None, customer_id="C101"):
    """Python calculates (slide 19): average, max, duplicates, out-of-range, category spend, frequency."""
    tx = data.card_transactions() if tx is None else tx
    tx = tx[tx.Customer_ID == customer_id]
    hist = tx[tx.Status == "History"]
    stats = {
        "history_days": int((hist.Timestamp.max() - hist.Timestamp.min()).days + 1),
        "history_count": int(len(hist)),
        "average_transaction": float(hist.Amount.mean()),
        "maximum_transaction": float(hist.Amount.max()),
        "usual_range": [round(float(hist.Amount.quantile(0.05))), round(float(hist.Amount.quantile(0.95)))],
        "usual_countries": sorted(hist.Country.unique().tolist()),
        "usual_hours": [int(hist.Timestamp.dt.hour.min()), int(hist.Timestamp.dt.hour.max())],
        "spend_by_category": hist.groupby("Category").Amount.sum().astype(float).to_dict(),
    }
    return stats, tx


def flag_transactions(tx, stats):
    """Rule-based investigation: duplicate, high value, foreign, odd time, new device."""
    avg = stats["average_transaction"]
    disputed = tx[tx.Status == "Disputed"].sort_values("Timestamp").reset_index(drop=True)
    rows = []
    for i, r in disputed.iterrows():
        flags, findings = [], []
        mult = r.Amount / avg
        findings.append(f"Amount {inr(r.Amount)} ({mult:.1f}x customer avg {inr(avg)})" if mult >= 2
                        else f"Amount {inr(r.Amount)} (within normal limits; avg {inr(avg)})")
        prev = disputed[(disputed.index < i) & (disputed.Merchant == r.Merchant) & (disputed.Amount == r.Amount)]
        prev = prev[(r.Timestamp - prev.Timestamp).dt.total_seconds() <= config.DUPLICATE_WINDOW_MIN * 60]
        if not prev.empty:
            gap = int((r.Timestamp - prev.iloc[-1].Timestamp).total_seconds() // 60)
            flags.append("Duplicate"); findings.append(f"Same merchant & amount {gap} minute(s) after {prev.iloc[-1].Transaction_ID}")
        if mult >= config.HIGH_VALUE_MULTIPLE:
            flags.append("High value")
        if r.Country != config.HOME_COUNTRY:
            flags.append("International"); findings.append(f"Country: {r.Country}")
        if config.ODD_HOURS[0] <= r.Timestamp.hour < config.ODD_HOURS[1]:
            flags.append("Odd hour"); findings.append(f"Time: {r.Timestamp:%H:%M} (unusual)")
        if r.Device == "New device":
            flags.append("New device")
        # risk scoring - transparent
        if "High value" in flags and ({"International", "Odd hour"} & set(flags)):
            risk = "High"
        elif flags:
            risk = "Medium"
        else:
            risk = "Low"
        rows.append({"Transaction": r.Transaction_ID, "Time": f"{r.Timestamp:%d-%b %H:%M}", "Amount": r.Amount,
                     "Merchant": r.Merchant, "Country": r.Country, "Multiple_of_avg": round(mult, 1),
                     "Flags": ", ".join(flags) or "-", "Key_Findings": "; ".join(findings), "Risk_Level": risk,
                     "Investigate": "Yes" if risk != "Low" else "No"})
    return pd.DataFrame(rows)


def reason_text(row):
    """Short per-transaction explanation (rule text; the LLM writes the full report)."""
    f = row["Flags"]
    if row["Risk_Level"] == "Low":
        return "Within normal spending pattern; no anomaly detected individually."
    if "Duplicate" in f:
        return "Possible duplicate transaction (same merchant, amount and near-simultaneous time)."
    return "High-value international transaction at an unusual hour; significant deviation from usual spend."


def report_prompt(flags, stats, complaint="I noticed several unusual transactions in my account..."):
    facts = {"customer_complaint": complaint,
             "customer_average_spend": inr(stats["average_transaction"]),
             "usual_countries": stats["usual_countries"], "usual_hours": stats["usual_hours"],
             "transactions": flags.assign(Amount=flags.Amount.map(inr)).to_dict("records")}
    return prompts.INVESTIGATION.format(facts=json.dumps(facts, indent=1, default=str))


def report(flags, stats, complaint="I noticed several unusual transactions in my account...", use_llm=True):
    to_check = flags[flags.Investigate == "Yes"]
    txt = llm.ask(report_prompt(flags, stats, complaint), system=prompts.SYSTEM_BANK) if use_llm else None
    if txt:
        return txt, False
    lines = [f"Investigation Summary: investigate {len(to_check)} transactions - {', '.join(to_check.Transaction)}.",
             "Suspicious indicators:"]
    lines += [f"- {r.Transaction}: {r.Key_Findings}" for r in to_check.itertuples()]
    lines += [f"Supporting evidence: transaction log (amount, merchant, country, time), customer average spend "
              f"{inr(stats['average_transaction'])}, rule flags (duplicate, high value, international, odd hour, new device).",
              f"Customer impact: {inr(to_check.Amount.sum())} under dispute.",
              "Additional information required: travel status, device/IP details, OTP/3D Secure logs, merchant history.",
              "Recommended investigation steps:",
              "1. Confirm with the customer whether the repeat online-store payment was authorised.",
              "2. Verify card/device, IP or login details for the international transactions.",
              "3. Check merchant records and authorisation logs.",
              "4. If unauthorised, initiate the chargeback/refund process as per policy.",
              "Recommended next action: escalate for manual review (no fraud decision made by AI)."]
    return "\n".join(lines), True
