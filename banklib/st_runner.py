"""
Show a demo script in the browser (Streamlit), step by step.

Used two ways:
  * app page "Demo Runner"            -> render_demo(path, ...)
  * `streamlit run demos/demo01_....py` -> each demo calls browser_mode(__file__)
    at the top; under Streamlit it renders itself here and stops, so the output
    appears in the browser instead of the terminal. Under plain Python it does nothing.
"""
import contextlib
import io
import re
import traceback
from pathlib import Path

from . import utils


def running_in_streamlit():
    try:
        import streamlit as st
        return st.runtime.exists()
    except Exception:  # noqa: BLE001
        return False


def split_cells(src):
    cells, title, buf = [], "Start", []
    for line in src.splitlines():
        if line.startswith("# %%"):
            if buf:
                cells.append((title, "\n".join(buf)))
            title, buf = line[4:].strip() or "Step", []
        else:
            buf.append(line)
    if buf:
        cells.append((title, "\n".join(buf)))
    return cells


class _Sink(io.TextIOBase):
    """Receives print() output and step()/show() calls and renders them in Streamlit."""

    def __init__(self, st):
        self.st, self.buf = st, []

    def write(self, s):
        self.buf.append(s)
        return len(s)

    def flush(self):
        text = "".join(self.buf).strip("\n")
        self.buf = []
        if text.strip():
            try:
                self.st.code(text, language=None, wrap_lines=True)
            except TypeError:
                self.st.code(text, language=None)

    def step(self, title):
        self.flush()
        self.st.markdown(f"**{title.strip()}**")

    def table(self, df):
        self.flush()
        self.st.dataframe(df, hide_index=True)

    def info(self, msg):
        self.flush()
        self.st.info(msg)


def render_demo(path, show_code=False):
    """Execute every '# %%' cell of a demo script and show its output on the page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import streamlit as st

    path = Path(path)
    ns = {"__file__": str(path), "__name__": "__demo__"}
    sink = _Sink(st)
    utils._SINK = sink
    try:
        for title, code in split_cells(path.read_text(encoding="utf-8")):
            if title.startswith("[markdown]"):
                st.markdown("\n".join(re.sub(r"^# ?", "", l) for l in code.splitlines()))
                continue
            if not code.strip():
                continue
            with st.container(border=True):
                st.caption(title)
                if show_code:
                    st.code(code.strip(), language="python")
                with st.spinner(f"Running: {title}"):
                    try:
                        with contextlib.redirect_stdout(sink):
                            exec(compile(code, str(path), "exec"), ns)  # noqa: S102
                    except SystemExit:
                        pass
                    except Exception:  # noqa: BLE001
                        sink.flush()
                        st.error("This step failed:")
                        st.code(traceback.format_exc(), language=None)
                        return False
                    sink.flush()
                for num in plt.get_fignums():
                    st.pyplot(plt.figure(num))
                plt.close("all")
        st.success("Demo finished. Excel outputs are saved in the outputs/ folder.")
        return True
    finally:
        utils._SINK = None


# demos with a dedicated interactive UI:  name -> (module, function)
APPS = {
    "warmup": ("banklib.ui.bank_apps", "render_warmup"),
    "customer": ("banklib.ui.bank_apps", "render_customer"),
    "investigation": ("banklib.ui.bank_apps", "render_investigation"),
    "openbanking": ("banklib.ui.bank_apps", "render_openbanking"),
    "policy": ("banklib.ui.bank_apps", "render_policy"),
    "dailyops": ("banklib.ui.bank_apps", "render_dailyops"),
    "copilot": ("banklib.ui.bank_apps", "render_copilot"),
    "challenge": ("banklib.ui.bank_apps", "render_challenge"),
}


def browser_mode(file, app=None):
    """Call at the top of a demo. Under `streamlit run` it renders the demo in the browser and stops.
    app=<name> opens that demo's interactive UI (upload -> results) instead of the step-by-step view."""
    if utils._SINK is not None or not running_in_streamlit():
        return                      # plain Python / VS Code / already inside the runner
    import importlib
    import streamlit as st
    if app in APPS:
        mod, fn = APPS[app]
        getattr(importlib.import_module(mod), fn)()
        st.stop()
    name = Path(file).stem
    try:
        st.set_page_config(page_title=name.replace("_", " "), layout="wide")
    except Exception:  # noqa: BLE001
        pass
    st.caption(f"Running demos/{Path(file).name} in the browser - tip: `python run_app.py` opens all demos.")
    show_code = st.toggle("Show code", value=False)
    render_demo(file, show_code=show_code)
    st.stop()
