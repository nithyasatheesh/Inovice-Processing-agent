"""
NovaBank interactive Streamlit apps - one per demo, same design as the finance kit.
Every dataset: Sample data or Upload my file (CSV / Excel / JSON / PDF as relevant).
"""
import json

import pandas as pd
import streamlit as st

from .. import (challenge, config, customer, data, investigation, openbanking, operations,
                policy_rag, prompts)
from ..utils import inr
from . import kit

SYS = prompts.SYSTEM_BANK
RISK_BG = {"High": "background-color:#fde2e2", "Critical": "background-color:#f9c9c9",
           "Medium": "background-color:#fff4cc", "Low": "background-color:#e3f4e8"}


# ============================================================ warm-up (slide 8)
def render_warmup(embedded=False):
    kit.setup("GenAI vs Automation vs AI", "One customer complaint, three approaches: rules route, ML predicts, "
              "GenAI understands the context and drafts a full response.", "\U0001F9E9", embedded)
    text = st.text_area("Customer complaint", customer.COMPLAINT, height=90)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kit.card_html("Traditional automation", "IF / THEN rules", "Outcome: task routing only"),
                    unsafe_allow_html=True)
        kit.banner(customer.rule_based(text), "info")
    with c2:
        st.markdown(kit.card_html("Machine learning", "Learns from past cases", "Outcome: prediction & classification"),
                    unsafe_allow_html=True)
        p = customer.ml_predict(text)
        kit.banner(f"Category: <b>{p['category']}</b> &middot; Urgency: <b>{p['urgency']}</b>", "info")
    with c3:
        st.markdown(kit.card_html("Generative AI", "Understands & responds", "Outcome: full contextual response", "navy"),
                    unsafe_allow_html=True)
    kit.ai_stream("✨ Generate the GenAI response (Ollama)", "warm", customer.genai_prompt(text), SYS,
                  lambda: customer.genai_response(text))
    kit.takeaway("Rules route | AI predicts | GenAI understands, analyses and takes it forward - with human review.")


# ============================================================ Demo 1 - customer intelligence
def render_customer(embedded=False):
    kit.setup("Customer Intelligence Assistant", "A relationship manager has 5 minutes before a meeting. Python combines "
              "profile, transactions, complaints and products &rarr; Ollama prepares the briefing.", "\U0001F464", embedded)
    with st.expander("\U0001F4C2  Data sources (sample data or upload your own)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            prof, _ = kit.dataset("Customer profile", "prof", data.profiles,
                                  ["Customer_ID", "Name", "Occupation", "Monthly_Income", "Segment"])
            comp, _ = kit.dataset("Complaints", "comp", data.complaints,
                                  ["Complaint_ID", "Customer_ID", "Date", "Complaint", "Status", "Priority"])
        with c2:
            tx, _ = kit.dataset("Transactions", "tx", data.transactions,
                                ["Customer_ID", "Date", "Category", "Amount", "Type", "Narration"])
            prod, _ = kit.dataset("Products", "prod", data.products, ["Customer_ID", "Product", "Account_Ref", "Details"],
                                  text_cols=["Details"])
    for df, col in ((tx, "Date"), (comp, "Date")):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce").fillna(0)
    prof["Monthly_Income"] = pd.to_numeric(prof["Monthly_Income"], errors="coerce").fillna(0)
    cid = st.selectbox("Customer", list(prof.Customer_ID),
                       format_func=lambda c: f"{c} - {prof.set_index('Customer_ID').loc[c, 'Name']}")
    try:
        summary, parts = customer.customer_360(cid, prof, tx, comp, prod)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build the customer view: {e}")
        return
    kit.stepper(["Load data", "Financial summary", "Complaints", "Combine", "Ollama briefing"],
                5 if "ai_brief_" + cid in st.session_state else 4)
    kit.cards([("Customer", summary["customer"], f"{summary['segment']} &middot; {summary['occupation']}", "navy"),
               ("Declared income", summary["declared_monthly_income"], "per month"),
               ("Fixed obligations", summary["total_fixed_obligations"], f"{summary['obligations_to_income_pct']}% of income",
                "warn" if summary["obligations_to_income_pct"] > 60 else ""),
               ("Open complaints", len(summary["open_complaints"]), "", "bad" if summary["open_complaints"] else "good")])
    if summary["possible_duplicate_debits"]:
        kit.banner("&#9888; Python found a possible duplicate debit: " + "; ".join(summary["possible_duplicate_debits"])
                   + " - the complaint is backed by evidence.", "bad")
    kit.sec(1, "Customer 360")
    t1, t2, t3 = st.tabs(["Transactions", "Complaints", "Products"])
    t1.dataframe(parts["transactions"].assign(Date=parts["transactions"].Date.dt.date), hide_index=True)
    t2.dataframe(kit.status_style(parts["complaints"].assign(Date=parts["complaints"].Date.dt.date), "Status",
                                  good=("Closed",), bad=("Open",)), hide_index=True)
    t3.dataframe(parts["products"], hide_index=True)
    kit.sec(2, "Customer briefing for the relationship manager")
    kit.ai_stream("✨ Prepare customer briefing (Ollama)", "brief_" + cid, customer.briefing_prompt(summary), SYS,
                  lambda: customer.briefing(summary))
    with st.expander("Facts sent to the model (calculated by Python)"):
        st.json(summary)
    kit.takeaway("Meeting preparation from ~20 minutes to ~2 minutes, with every fact traceable to the source data.")


# ============================================================ Demo 2 - investigation
def render_investigation(embedded=False):
    kit.setup("Intelligent Transaction Investigation", "Python analytics + rule-based flags + GenAI reasoning. "
              "The AI explains the risk and next steps - it does not declare fraud.", "\U0001F50D", embedded)
    tx, _ = kit.dataset("Card / UPI transactions", "card", data.card_transactions,
                        ["Transaction_ID", "Customer_ID", "Timestamp", "Amount", "Merchant", "Country", "Status"],
                        help_text="Status = History (baseline) or Disputed (to investigate). Optional: Category, Device, Channel")
    tx["Timestamp"] = pd.to_datetime(tx["Timestamp"], errors="coerce")
    tx["Amount"] = pd.to_numeric(tx["Amount"], errors="coerce").fillna(0)
    for c, d in (("Device", "Known device"), ("Category", ""), ("Channel", "")):
        if c not in tx.columns:
            tx[c] = d
    c1, c2, c3 = st.columns(3)
    cid = c1.selectbox("Customer", sorted(tx.Customer_ID.unique()))
    config.HIGH_VALUE_MULTIPLE = c2.slider("High value = amount ≥ N x customer average", 2.0, 25.0, 5.0, 0.5)
    config.DUPLICATE_WINDOW_MIN = c3.slider("Duplicate window (minutes)", 1, 60, 10)
    stats, ctx = investigation.analytics(tx, cid)
    if stats["history_count"] == 0 or ctx[ctx.Status == "Disputed"].empty:
        st.warning("Need both 'History' and 'Disputed' rows for this customer.")
        return
    flags = investigation.flag_transactions(ctx, stats)
    flags["Reason"] = flags.apply(investigation.reason_text, axis=1)
    n_inv = int((flags.Investigate == "Yes").sum())
    kit.stepper(["Transaction data", "Python analytics", "Rule flags", "Anomaly indicators", "GenAI report",
                 "Analyst review"], 4, failed=None)
    kit.sec(1, "Customer baseline (Python)")
    kit.cards([("Average transaction", inr(stats["average_transaction"]), f"{stats['history_count']} transactions"),
               ("Maximum in history", inr(stats["maximum_transaction"])),
               ("Usual countries", ", ".join(stats["usual_countries"])),
               ("To investigate", f"{n_inv} of {len(flags)}", inr(flags[flags.Investigate == 'Yes'].Amount.sum()),
                "bad" if n_inv else "good")])
    kit.sec(2, "Investigation results")
    kit.checklist([(f"{r.Transaction} &middot; {inr(r.Amount)} &middot; {r.Merchant} ({r.Country}, {r.Time})",
                    f"<b>{r.Risk_Level} risk</b> - {r.Reason}<br><span style='color:#6b7a99'>{r.Key_Findings}</span>",
                    {"High": "fail", "Medium": "warn"}.get(r.Risk_Level, "pass")) for r in flags.itertuples()])
    with st.expander("Results table"):
        st.dataframe(flags.style.map(lambda v: RISK_BG.get(v, ""), subset=["Risk_Level"]), hide_index=True)
    kit.sec(3, "GenAI investigation summary")
    kit.ai_stream("✨ Write investigation report (Ollama)", "inv_" + cid, investigation.report_prompt(flags, stats), SYS,
                  lambda: investigation.report(flags, stats, use_llm=False))
    kit.download({"Investigation": flags}, "investigation.xlsx")
    kit.takeaway("Combine Python analytics, rule-based checks and GenAI reasoning to explain risk and take focused "
                 "next steps. A flag is an indicator, not a verdict.")


# ============================================================ Demo 3 - Open Banking
def render_openbanking(embedded=False):
    kit.setup("Open Banking Financial Intelligence", "Consent &rarr; authorised fetch from each bank &rarr; Python "
              "aggregates the cash flow &rarr; Ollama writes the financial profile for the analyst.", "\U0001F310", embedded)
    with st.container(border=True):
        st.markdown("**1. Customer consent**")
        a, b = st.columns([1, 2])
        up = a.file_uploader("Upload a consent JSON (optional)", type="json", key="consent_up")
        consent = data.consent("C101")
        if up is not None:
            try:
                consent = json.loads(up.getvalue())
            except Exception as e:  # noqa: BLE001
                st.error(f"Invalid consent JSON: {e}")
        status = a.radio("Consent status", ["ACTIVE", "REVOKED"], horizontal=True,
                         index=0 if consent.get("status") == "ACTIVE" else 1)
        types = a.multiselect("Data types the customer consented to", ["DEPOSIT", "CREDIT_CARD", "LOAN", "MUTUAL_FUNDS"],
                              [t for t in consent.get("fi_types", []) if t in ["DEPOSIT", "CREDIT_CARD", "LOAN", "MUTUAL_FUNDS"]])
        consent = dict(consent, status=status, fi_types=types)
        b.json(consent, expanded=False)
    with st.container(border=True):
        st.markdown("**2. Provider data (Open Banking API responses)**")
        mode = st.radio("Source", ["Sample providers (NovaBank, Bank B, Bank C)", "Upload provider JSON files"],
                        horizontal=True, label_visibility="collapsed")
        fip = None
        if mode.startswith("Upload"):
            ups = st.file_uploader("One JSON per account, same format as data/open_banking/fip_*.json",
                                   type="json", accept_multiple_files=True)
            st.download_button("Download a sample provider file (template)",
                               (config.OPEN_BANKING / "fip_novabank.json").read_bytes(), file_name="fip_template.json")
            if ups:
                fip = {}
                for u in ups:
                    try:
                        fip[u.name.rsplit(".", 1)[0].upper()] = json.loads(u.getvalue())
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{u.name}: {e}")
    try:
        fi, log = openbanking.fetch_fi_data(consent, fip)
    except openbanking.ConsentError as e:
        kit.stepper(["Consent", "Authorised fetch", "Aggregate", "Cash-flow analysis", "AI profile"], 0, failed=1)
        kit.banner(f"&#128683; No data fetched - {e}", "bad")
        kit.takeaway("Without valid consent, no customer data reaches the model.")
        return
    st.dataframe(kit.status_style(log, "decision", good=("Fetched",), bad=("NOT",)), hide_index=True)
    deposits = [k for k, v in fi.items() if v.get("transactions")]
    if not deposits:
        kit.banner("No deposit-account transactions in scope - add DEPOSIT to the consent.", "bad")
        return
    monthly, extras, _ = openbanking.aggregate(fi)
    last = monthly.iloc[-1]
    facts = openbanking.profile_facts(monthly, extras)
    kit.stepper(["Consent", "Authorised fetch", "Aggregate", "Cash-flow analysis", "AI profile"],
                5 if "ai_ob" in st.session_state else 4)
    kit.sec(3, f"Cash-flow analysis - {last.month}")
    kit.cards([("Total income", inr(last.income), f"{len(extras['income_streams'])} income streams", "good"),
               ("Expenses", inr(last.expenses)), ("EMI", inr(last.emi)), ("Card payment", inr(last.credit_card_payment))])
    kit.cards([("Total outflow", inr(last.total_outflow)),
               ("Net available cash", inr(last.net_available_cash), f"savings rate {last.savings_rate_pct}%",
                "bad" if last.net_available_cash < 0 else "good"),
               ("Credit utilisation", f"{extras['credit_utilisation_pct']:.0f}%" if extras["credit_utilisation_pct"] else "-",
                "", "warn"),
               ("Financial health", facts["rule_based_health"], "rule-based", "navy")])
    c1, c2 = st.columns(2)
    with c1:
        kit.chart(pd.DataFrame({"item": ["Income", "Expenses", "EMI", "Card payment", "Net cash"],
                                "amount": [last.income, last.expenses, last.emi, last.credit_card_payment,
                                           last.net_available_cash]}), "item", "amount")
    with c2:
        kit.chart(monthly, "month", ["income", "total_outflow", "net_available_cash"], kind="line")
    if facts["rule_based_risks"]:
        kit.checklist([(r.split("(")[0].strip(), r, "warn") for r in facts["rule_based_risks"]])
    kit.sec(4, "AI-generated financial profile (for an authorised analyst)")
    kit.ai_stream("✨ Generate financial profile (Ollama)", "ob", openbanking.profile_prompt(facts), SYS,
                  lambda: openbanking.financial_profile(facts, use_llm=False))
    kit.download({"Monthly": monthly, "Access_Log": log}, "open_banking_profile.xlsx")
    kit.takeaway("Open Banking gives a 360-degree view across banks - and data reaches the model only through an "
                 "authorised, consented, minimised workflow.")


# ============================================================ Demo 4 - policy assistant
def _policy_index():
    with st.container(border=True):
        a, b = st.columns([3, 2])
        a.markdown("**Policy library**  \n<span style='color:#6b7a99;font-size:.85rem'>Text-based PDFs; numbered sections "
                   "(e.g. '4.2 Duplicate Transactions') give the best citations</span>", unsafe_allow_html=True)
        mode = b.radio("Source", ["Sample policies", "Upload policy PDFs"], horizontal=True, label_visibility="collapsed",
                       key="pol_src")
        files = None
        if mode.startswith("Upload"):
            ups = st.file_uploader("Upload policy PDFs", type="pdf", accept_multiple_files=True, key="pol_up")
            if ups:
                files = [(u.name, u.getvalue()) for u in ups]
        key = tuple(n for n, _ in files) if files else ("sample",)
        if st.session_state.get("pol_key") != key:
            st.session_state["pol_key"] = key
            st.session_state["pol_idx"] = policy_rag.PolicyIndex(policy_rag.load_sections(files=files))
        idx = st.session_state["pol_idx"]
        if not idx.chunks:
            st.error("No text found in the uploaded PDFs - using the sample policies.")
            idx = policy_rag.PolicyIndex()
        st.caption(f"{len(idx.chunks)} sections from {len({c['policy'] for c in idx.chunks})} policies")
        with st.expander("Browse sections"):
            st.dataframe(pd.DataFrame(idx.chunks)[["policy", "section", "heading", "text"]], hide_index=True)
    return idx


def render_policy(embedded=False):
    kit.setup("Policy-Based Refund Assistant", "Grounded AI: Python retrieves the relevant policy sections &rarr; "
              "Ollama answers only from them, with citations - or says it cannot determine the answer.",
              "\U0001F4DC", embedded)
    idx = _policy_index()
    ex = ["A customer has been charged twice for the same transaction. What should the operations team do? "
          "What evidence should be collected?", "How long does a customer have to dispute a card transaction?",
          "What is the interest rate on a home loan?"]
    pick = st.selectbox("Example questions", ex)
    q = st.text_area("Question", pick, height=80)
    prompt, hits = policy_rag.answer_prompt(q, idx)
    kit.stepper(["Policy PDFs", "Retrieve sections", "Ground the prompt", "Ollama answer", "Citations"],
                2 if prompt else 1, failed=None if prompt else 3)
    kit.sec(1, "Retrieved policy sections")
    if hits:
        kit.checklist([(f"{h['policy']} &middot; Section {h['section']}", f"{h['heading']} &middot; relevance {h['score']}",
                        "pass" if h["score"] >= config.RAG_MIN_SCORE else "warn") for h in hits])
    kit.sec(2, "Answer")
    if prompt is None:
        kit.banner(f"&#128683; {config.REFUSAL}", "bad")
        st.caption(f"Critical control: no policy section scored above {config.RAG_MIN_SCORE} - the model is not asked at all.")
    else:
        kit.ai_stream("✨ Answer from policy (Ollama)", "pol_" + str(abs(hash(q))), prompt, SYS,
                      lambda: (policy_rag.answer(q, idx)["answer"], True))
    kit.takeaway("Grounded AI keeps answers accurate, policy-compliant, explainable and auditable.")


# ============================================================ Demo 5a - daily operations report
CASE_COLS = ["Case_ID", "Type", "Priority", "Status", "Created", "SLA_Due"]


def _cases():
    cases, _ = kit.dataset("Operations cases", "cases", data.cases, CASE_COLS,
                           help_text="Optional columns: Sub_Type, Assigned_Team, Customer_ID, Notes", text_cols=["Notes"])
    for c in ("Created", "SLA_Due", "Closed_On"):
        if c in cases.columns:
            cases[c] = pd.to_datetime(cases[c], errors="coerce")
    for c, d in (("Sub_Type", ""), ("Assigned_Team", ""), ("Customer_ID", ""), ("Notes", "")):
        if c not in cases.columns:
            cases[c] = d
    return cases


def render_dailyops(embedded=False):
    kit.setup("Daily Operations Report", "45 minutes of manual work &rarr; Python calculates the metrics from the case "
              "file &rarr; Ollama writes the summary and priorities.", "\U0001F4CB", embedded)
    cases = _cases()
    now = st.date_input("Report date", config.NOW.date())
    m = operations.metrics(cases, pd.Timestamp(now) + pd.Timedelta(hours=18))
    kit.stepper(["Load cases", "Python metrics", "Trends", "Ollama summary", "Priorities"],
                5 if "ai_daily" in st.session_state else 3)
    kit.cards([("Total cases", m["total_cases"], "", "navy"), ("Open cases", m["open_cases"], "", "warn"),
               ("SLA breaches", m["sla_breaches"], "", "bad" if m["sla_breaches"] else "good"),
               ("Critical cases", m["critical_cases"], "open", "bad" if m["critical_cases"] else "good")])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        vc = cases.Type.value_counts().reset_index()
        vc.columns = ["type", "cases"]
        kit.chart(vc, "type", "cases")
        tr = pd.DataFrame([{"type": t, "last 7 days": v["last_7_days"], "previous 7 days": v["previous_7_days"]}
                           for t, v in m["weekly_trend"].items()])
        kit.chart(tr, "type", ["previous 7 days", "last 7 days"], height=220)
    with c2:
        st.markdown("**SLA breaches**")
        st.dataframe(pd.DataFrame(m["breached_cases"]), hide_index=True)
        st.markdown("**Critical open cases**")
        st.dataframe(pd.DataFrame(m["critical_open"]), hide_index=True)
    kit.sec(1, "Daily operations summary & recommended priorities")
    kit.ai_stream("✨ Generate daily report (Ollama)", "daily", operations.daily_prompt(m), SYS,
                  lambda: _daily_fallback(m))
    kit.download({"Cases": cases, "SLA_Breaches": pd.DataFrame(m["breached_cases"])}, "daily_operations.xlsx")
    kit.takeaway("Python owns the numbers; GenAI turns them into a report the manager can act on.")


def _daily_fallback(m):
    from .. import llm
    host = llm.config.OLLAMA_HOST
    llm.config.OLLAMA_HOST = "http://127.0.0.1:9"   # force the rule-based text
    try:
        return operations.daily_report(m)
    finally:
        llm.config.OLLAMA_HOST = host


# ============================================================ Demo 5b - copilot
def render_copilot(embedded=False):
    kit.setup("Banking Operations Copilot", "One assistant, four skills: policy Q&A (grounded), case summaries, "
              "operational insights and the daily report.", "\U0001F9D1‍\U0001F4BC", embedded)
    with st.expander("\U0001F4C2  Data sources (sample or upload)"):
        cases = _cases()
        idx = _policy_index()
    ex = ["What is the process for a duplicate transaction refund?", "Summarize this customer dispute case CS101.",
          "Show the top operational issues in the last 30 days.", "Give me today's operations report."]
    cols = st.columns(len(ex))
    clicked = next((q for c, q in zip(cols, ex) if c.button(q, key=f"cp_{q}")), None)
    if "cp_chat" not in st.session_state:
        st.session_state.cp_chat = []
    for role, msg, meta in st.session_state.cp_chat:
        with st.chat_message(role):
            st.markdown(msg.replace("\n", "  \n"))
            if meta:
                st.markdown(f'<span class="pill {"rule" if meta["fallback"] else "ai"}">skill: {meta["tool"]} &middot; '
                            f'sources: {", ".join(meta["sources"])}</span>', unsafe_allow_html=True)
    q = st.chat_input("Ask the operations copilot...") or clicked
    if q:
        st.session_state.cp_chat.append(("user", q, None))
        with st.spinner("Working..."):
            r = operations.copilot(q, cases, idx)
        st.session_state.cp_chat.append(("assistant", r["answer"], r))
        st.rerun()
    if st.session_state.cp_chat and st.button("Clear chat"):
        st.session_state.cp_chat = []
        st.rerun()
    kit.takeaway("A policy-grounded copilot helps operations teams work faster, more accurately and more compliantly.")


# ============================================================ Final challenge
def render_challenge(embedded=False):
    kit.setup("Final Challenge - Your AI Banking Assistant", "One report for customer C101 combining transactions, "
              "complaints, Open Banking data, policies and operational cases.", "\U0001F3C6", embedded)
    qs = ["Financial profile", "Unusual transactions", "Pending operational issues", "Relevant policy guidance",
          "Investigate next", "Not for AI to decide"]
    if st.button("▶  Gather evidence from Demos 1-5", type="primary"):
        with st.spinner("Running all five demos..."):
            st.session_state["ch_ev"] = challenge.gather("C101")
        st.session_state.pop("ai_challenge", None)
    ev = st.session_state.get("ch_ev")
    kit.stepper(qs, 5 if ev else 0)
    if not ev:
        st.info("Press **Gather evidence** - Python collects the facts; then Ollama writes the six-part report.")
        return
    p = ev["1_financial_profile"]["open_banking"]
    kit.cards([("Income (Open Banking)", p["monthly_summary"]["total_income"], "", "good"),
               ("Net cash", p["monthly_summary"]["net_available_cash"], "", "bad"),
               ("Unusual transactions", len(ev["2_unusual_transactions"]), "to investigate", "warn"),
               ("Open issues", len(ev["3_pending_operational_issues"]["open_complaints"]) +
                len(ev["3_pending_operational_issues"]["open_cases"]), "complaints + cases", "bad")])
    with st.expander("Evidence gathered by Python"):
        st.json(ev)
    kit.sec(1, "AI Banking Assistant report")
    kit.ai_stream("✨ Write the six-part report (Ollama)", "challenge", challenge.report_prompt(ev), SYS,
                  lambda: _challenge_fallback(ev))
    if "ai_challenge" in st.session_state:
        st.download_button("⬇  Download report (.md)", st.session_state["ai_challenge"][0],
                           file_name="AI_banking_assistant_report_C101.md")
    kit.takeaway("GenAI does not replace banking expertise. It reduces the time spent collecting, analysing, "
                 "summarising and explaining - so professionals can focus on judgement and decisions.")


def _challenge_fallback(ev):
    from .. import llm
    host = llm.config.OLLAMA_HOST
    llm.config.OLLAMA_HOST = "http://127.0.0.1:9"
    try:
        return challenge.report(ev)
    finally:
        llm.config.OLLAMA_HOST = host
