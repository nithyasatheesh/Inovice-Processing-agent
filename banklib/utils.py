"""Formatting, printing and Excel helpers shared by all demos."""
from pathlib import Path
import pandas as pd


def inr(x, decimals=0, symbol=True):
    """Indian digit grouping: 118000 -> 'Rs.1,18,000' style (uses the rupee sign)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    neg = x < 0
    x = abs(float(x))
    whole = int(round(x, decimals)) if decimals == 0 else int(x)
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    if decimals:
        s += f"{x - int(x):.{decimals}f}"[1:]
    return ("-" if neg else "") + ("₹" if symbol else "") + s


def lakh(x, signed=False):
    return (f"{x:+.2f}" if signed else f"{x:.2f}") + " lakh"


# When a demo runs inside the Streamlit "Demo Runner" page, the runner sets this
# to an object with step()/table()/info() so output appears in the browser.
_SINK = None


def step(title):
    """Print a step banner that mirrors the workflow boxes on the slide."""
    if _SINK is not None:
        _SINK.step(title)
        return
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def show(df, max_rows=30):
    """Print a DataFrame neatly in the terminal / VS Code interactive window / Streamlit."""
    if _SINK is not None:
        _SINK.table(df.head(max_rows))
        return
    try:
        from IPython import get_ipython  # noqa: F401
        from IPython.display import display
        if get_ipython() is not None:
            display(df.head(max_rows))
            return
    except Exception:  # noqa: BLE001
        pass
    with pd.option_context("display.max_columns", 30, "display.width", 200,
                           "display.max_colwidth", 60):
        print(df.head(max_rows).to_string(index=False))


def save_excel(sheets: dict, path: Path):
    """Write {sheet_name: DataFrame} to a formatted Excel workbook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            from openpyxl.styles import Font, PatternFill, Alignment
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F3A93")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            ws.freeze_panes = "A2"
    return path


def fallback_note(used_fallback: bool):
    if used_fallback and _SINK is not None:
        _SINK.info("Ollama not reachable - showing the rule-based fallback text. Start Ollama to see the live LLM output.")
        return
    if used_fallback:
        print("  [Ollama not reachable - showing the rule-based fallback text. "
              "Start Ollama to see the live LLM output.]")
