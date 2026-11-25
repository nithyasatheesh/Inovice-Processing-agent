"""
Shared look & feel for the interactive demo apps (same design as Demo 1):
hero banner, workflow stepper, numbered sections, KPI cards, check lists,
and dataset inputs that accept the sample data OR an uploaded CSV/Excel file.
"""
import io

import pandas as pd
import streamlit as st

from .. import config, llm

CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#0f2a5f 0%,#1f4fbf 60%,#2f80ed 100%); color:#fff;
       padding:22px 28px; border-radius:16px; margin-bottom:14px;}
.hero h1 {color:#fff; font-size:1.9rem; margin:0 0 4px 0;}
.hero p {color:#dbe7ff; margin:0; font-size:0.98rem;}
.stepper {display:flex; gap:6px; margin:6px 0 18px 0; flex-wrap:wrap;}
.step {flex:1; min-width:105px; text-align:center; padding:9px 6px; border-radius:10px; font-size:0.82rem;
       background:#eef2f9; color:#6b7a99; border:1px solid #dde4f0;}
.step b {display:block; font-size:1.05rem;}
.step.done {background:#e7f6ec; color:#17693a; border-color:#a8dcb9;}
.step.fail {background:#fdecec; color:#a12626; border-color:#f3b4b4;}
.kcard {background:#fff; border:1px solid #e3e8f2; border-radius:12px; padding:12px 14px; height:100%;
        box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom:8px;}
.kcard .lbl {font-size:0.74rem; color:#6b7a99; text-transform:uppercase; letter-spacing:.04em;}
.kcard .val {font-size:1.35rem; font-weight:700; color:#14213d; margin-top:2px; word-break:break-word;}
.kcard .sub {font-size:0.82rem; color:#6b7a99; margin-top:2px;}
.kcard.navy {background:#0f2a5f; border-color:#0f2a5f;} .kcard.navy .lbl,.kcard.navy .sub {color:#b9c8ea;}
.kcard.navy .val {color:#fff;}
.kcard.good {border-left:5px solid #2e9e5b;} .kcard.bad {border-left:5px solid #d64545;}
.kcard.warn {border-left:5px solid #e0a100;}
.chk {display:flex; gap:10px; align-items:flex-start; padding:9px 12px; border-radius:10px; margin-bottom:6px;
      border:1px solid #e3e8f2; background:#fff;}
.chk .nm {font-weight:600; color:#14213d; min-width:190px;}
.chk .dt {color:#44506b; font-size:0.92rem;}
.chk.pass {border-left:5px solid #2e9e5b;} .chk.fail {border-left:5px solid #d64545; background:#fffafa;}
.chk.warn {border-left:5px solid #e0a100; background:#fffdf5;}
.banner {padding:14px 18px; border-radius:12px; font-weight:600; font-size:1.02rem; margin:8px 0 12px 0;}
.banner.ok {background:#e7f6ec; color:#17693a; border:1px solid #a8dcb9;}
.banner.bad {background:#fdecec; color:#a12626; border:1px solid #f3b4b4;}
.banner.info {background:#eef4ff; color:#1f3a93; border:1px solid #c9d8fb;}
.pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;}
.pill.ai {background:#e8efff; color:#1f4fbf;} .pill.rule {background:#fff4dd; color:#8a5a00;}
.sec {font-size:1.15rem; font-weight:700; color:#14213d; margin:20px 0 8px 0;}
.sec span {display:inline-block; background:#1f4fbf; color:#fff; border-radius:50%; width:26px; height:26px;
           text-align:center; line-height:26px; font-size:0.85rem; margin-right:8px;}
.aibox {background:#f7f9ff; border:1px solid #d9e2f7; border-radius:12px; padding:14px 18px; color:#14213d;}
.takeaway {background:#edf8f1; border:1px solid #bfe5cc; color:#17693a; border-radius:12px; padding:12px 16px;
           margin-top:18px;}
</style>
"""


# ---------------------------------------------------------------- page frame
def setup(title, subtitle, icon="\U0001F4CA", embedded=False, steps=None):
    if not embedded:
        try:
            st.set_page_config(page_title=title, page_icon=icon, layout="wide")
        except Exception:  # noqa: BLE001
            pass
        with st.sidebar:
            st.markdown("### Settings")
            llm.auto_select()
            config.OLLAMA_MODEL = st.text_input("Ollama model", value=config.OLLAMA_MODEL)
            ok, msg, _ = llm.status()
            (st.success if ok else st.warning)(msg)
            st.caption("Sample data is synthetic. You can upload your own files on the page.")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def stepper(steps, done=0, failed=None):
    html = '<div class="stepper">'
    for i, s in enumerate(steps, 1):
        cls = "fail" if failed == i else ("done" if i <= done else "")
        icon = "&#10007;" if cls == "fail" else ("&#10003;" if cls == "done" else str(i))
        html += f'<div class="step {cls}"><b>{icon}</b>{s}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def stepper_status(items):
    """items: list of (label, state) with state in done / fail / todo - one box per item."""
    html = '<div class="stepper">'
    for i, (s, state) in enumerate(items, 1):
        cls = {"done": "done", "fail": "fail"}.get(state, "")
        icon = "&#10007;" if cls == "fail" else ("&#10003;" if cls == "done" else str(i))
        html += f'<div class="step {cls}"><b>{icon}</b>{s}</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


PALETTE = ["#1f4fbf", "#8fb3ff", "#2e9e5b", "#e0a100", "#d64545"]


def chart(df, x, ys, kind="bar", horizontal=False, height=300):
    """Altair chart that keeps the row order of the data (no alphabetical sorting)."""
    import altair as alt
    ys = [ys] if isinstance(ys, str) else list(ys)
    d = df[[x] + ys].copy()
    d[x] = d[x].astype(str)
    order = list(dict.fromkeys(d[x]))
    long = d.melt(x, ys, var_name="series", value_name="value")
    enc_x = alt.X(f"{x}:N", sort=order, title=None)
    enc_v = alt.Y("value:Q", title=None)
    if horizontal:
        enc_x, enc_v = alt.Y(f"{x}:N", sort=order, title=None), alt.X("value:Q", title=None)
    color = alt.Color("series:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(orient="bottom", title=None))
    base = alt.Chart(long)
    if kind == "line":
        c = base.mark_line(point=True, strokeWidth=3).encode(x=enc_x, y=enc_v, color=color)
    else:
        offset = {"yOffset": "series:N"} if horizontal else {"xOffset": "series:N"}
        c = base.mark_bar(cornerRadius=4).encode(x=enc_x, y=enc_v, color=color, **(offset if len(ys) > 1 else {}))
    try:
        st.altair_chart(c.properties(height=height), width="stretch")
    except TypeError:   # older Streamlit
        st.altair_chart(c.properties(height=height), use_container_width=True)


def sec(n, title):
    st.markdown(f'<div class="sec"><span>{n}</span>{title}</div>', unsafe_allow_html=True)


def card_html(label, value, sub="", tone=""):
    v = "&mdash;" if value in (None, "", "None") else value
    return (f'<div class="kcard {tone}"><div class="lbl">{label}</div><div class="val">{v}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>")


def cards(items, per_row=4):
    """items: list of (label, value, sub, tone)."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, it in zip(cols, items[i:i + per_row]):
            it = tuple(it) + ("",) * (4 - len(it))
            col.markdown(card_html(*it[:4]), unsafe_allow_html=True)


def checklist(rows):
    """rows: list of (name, detail, status) with status in pass/fail/warn."""
    icon = {"pass": "&#9989;", "fail": "&#10060;", "warn": "&#9888;&#65039;"}
    html = "".join(f'<div class="chk {s}"><div>{icon.get(s, "")}</div><div class="nm">{n}</div>'
                   f'<div class="dt">{d}</div></div>' for n, d, s in rows)
    st.markdown(html, unsafe_allow_html=True)


def banner(text, kind="ok"):
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def ai_output(text, fallback):
    pill = ('<span class="pill rule">Rule-based text (Ollama offline)</span>' if fallback
            else f'<span class="pill ai">&#9889; Written by Ollama - {config.OLLAMA_MODEL}</span>')
    st.markdown(pill, unsafe_allow_html=True)
    st.markdown(f'<div class="aibox">{_md(text)}</div>', unsafe_allow_html=True)


def _md(text):
    import html
    out = html.escape(text).replace("\n", "<br>")
    return out.replace("**", "")


def ai_button(label, key, fn):
    """Button that runs an Ollama call once and keeps the result on screen."""
    if st.button(label, key=f"btn_{key}", type="primary"):
        with st.spinner("Ollama is writing..."):
            st.session_state[f"ai_{key}"] = fn()
    if f"ai_{key}" in st.session_state:
        ai_output(*st.session_state[f"ai_{key}"])


def ai_stream(label, key, prompt, system, fallback_fn, max_tokens=None):
    """Button -> Ollama answer streams onto the page word by word. Falls back to fallback_fn() -> (text, fb)."""
    if st.button(label, key=f"btn_{key}", type="primary"):
        st.markdown(f'<span class="pill ai">&#9889; Ollama is writing - {config.OLLAMA_MODEL}</span>', unsafe_allow_html=True)
        import time as _t
        t0 = _t.time()
        txt = st.write_stream(llm.stream(prompt, system=system, max_tokens=max_tokens)) or ""
        if isinstance(txt, list):
            txt = "".join(map(str, txt))
        if txt.strip():
            st.caption(f"Written in {_t.time() - t0:.0f} s")
            st.session_state[f"ai_{key}"] = (txt, False)
        else:
            st.session_state[f"ai_{key}"] = fallback_fn()
            ai_output(*st.session_state[f"ai_{key}"])
        return
    if f"ai_{key}" in st.session_state:
        ai_output(*st.session_state[f"ai_{key}"])


def pdf_preview(pdf_bytes, dpi=110):
    import pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    blocks = [b[:4] for b in page.get_text("blocks")]
    clip = None
    if blocks:
        clip = pymupdf.Rect(max(0, min(b[0] for b in blocks) - 20), max(0, min(b[1] for b in blocks) - 20),
                            min(page.rect.x1, max(b[2] for b in blocks) + 20), min(page.rect.y1, max(b[3] for b in blocks) + 20))
    return page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")


def takeaway(text):
    st.markdown(f'<div class="takeaway"><b>Key takeaway:</b> {text}</div>', unsafe_allow_html=True)


def status_style(df, col, good=("Validated", "Matched", "Completed", "Approve", "Favourable", "Low"),
                 bad=("Exception", "Unmatched", "Pending", "Review", "Unfavourable", "High", "Critical")):
    def f(v):
        s = str(v)
        if any(s.startswith(g) for g in good):
            return "color:#17693a;font-weight:600"
        if any(s.startswith(b) for b in bad):
            return "color:#a12626;font-weight:600"
        if s.startswith(("Ambiguous", "Medium", "In progress")):
            return "color:#8a5a00;font-weight:600"
        return ""
    return df.style.map(f, subset=[col])


def excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for n, d in sheets.items():
            d.to_excel(xw, sheet_name=n[:31], index=False)
    return buf.getvalue()


def download(sheets, filename, label="⬇  Download results (Excel)"):
    st.download_button(label, excel_bytes(sheets), file_name=filename, type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------- dataset input
def _read(upload, sheet=None):
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    xl = pd.ExcelFile(upload)
    use = sheet if sheet in xl.sheet_names else xl.sheet_names[0]
    return pd.read_excel(xl, sheet_name=use)


def dataset(label, key, sample_loader, required, dates=(), text_cols=(), help_text="", sheet=None):
    """
    Let the user use the sample dataset or upload a CSV/Excel with the same columns.
    Returns (DataFrame, source_label). Uploaded files are checked for required columns.
    """
    sample = sample_loader()
    with st.container(border=True):
        top = st.columns([3, 2])
        top[0].markdown(f"**{label}**" + (f"  \n<span style='color:#6b7a99;font-size:.85rem'>{help_text}</span>"
                                           if help_text else ""), unsafe_allow_html=True)
        mode = top[1].radio("Source", ["Sample data", "Upload my file"], key=f"src_{key}", horizontal=True,
                            label_visibility="collapsed")
        df, src = sample, "sample"
        if mode == "Upload my file":
            up = st.file_uploader(f"Upload {label} (CSV or Excel)", type=["csv", "xlsx", "xls"], key=f"up_{key}")
            st.download_button("Download a template with the required columns", sample.head(5).to_csv(index=False),
                               file_name=f"template_{key}.csv", key=f"tpl_{key}")
            if up is None:
                st.caption("No file uploaded yet - using the sample data.")
            else:
                try:
                    raw = _read(up, sheet)
                    raw.columns = [str(c).strip() for c in raw.columns]
                    missing = [c for c in required if c not in raw.columns]
                    if missing:
                        st.error(f"Missing column(s): {', '.join(missing)}. Required: {', '.join(required)}. "
                                 "Download the template above. Using the sample data for now.")
                    else:
                        df, src = raw, up.name
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read the file: {e}. Using the sample data.")
        df = df.copy()
        for c in dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=False).dt.date
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str).replace({"nan": ""})
        with st.expander(f"Preview - {len(df):,} rows ({'sample data' if src == 'sample' else src})"):
            st.dataframe(df.head(200), hide_index=True)
    return df, src
