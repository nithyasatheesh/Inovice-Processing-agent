# %% [markdown]
# # Demo 1 - Customer Intelligence Assistant
# A relationship manager has 5 minutes before a meeting. Data is spread across profile, transactions,
# complaints and products. Flow: 1 Load > 2 Calculate summary > 3 Retrieve complaints > 4 Combine > 5 Ask Ollama.
# Goal: meeting preparation from ~20 minutes to ~2 minutes.

# %% Setup
import sys
from pathlib import Path
_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "banklib").is_dir())
sys.path.insert(0, str(ROOT))
from banklib.st_runner import browser_mode; browser_mode(globals().get("__file__", ""), app="customer")  # `streamlit run` -> interactive app

import json, time
from banklib import customer, data, llm, prompts
from banklib.utils import step, show, fallback_note

CUSTOMER_ID = "C101"   # try C102
t0 = time.time()

# %% STEP 1 - Load customer data (CSV files)
profile, tx, comp, prod = data.profiles(), data.transactions(), data.complaints(), data.products()
step("STEP 1 - customer_profile.csv")
show(profile)

# %% STEP 2-4 - Calculate financial summary, retrieve complaints, combine
summary, parts = customer.customer_360(CUSTOMER_ID)
step("Transactions this month")
show(parts["transactions"])
step("Open complaints")
show(parts["complaints"])
step("STEP 4 - Combined facts sent to the model (calculated by Python)")
print(json.dumps(summary, indent=2))

# %% STEP 5 - Ask Ollama to prepare the customer briefing
step("STEP 5 - CUSTOMER BRIEFING (Ollama)")
text, fb = customer.briefing(summary)
fallback_note(fb)
print(text)
print(f"\nPrepared in {time.time() - t0:.0f} seconds.")

# %% The slide's code, using the official `ollama` Python package (optional)
try:
    import ollama
    from banklib import config
    response = ollama.chat(model=config.OLLAMA_MODEL,
                           messages=[{"role": "user", "content": "Explain Open Banking to a banking operations analyst in 3 lines."}])
    print(response["message"]["content"])
except Exception as e:  # package missing or Ollama not running
    print("(ollama package / server not available:", type(e).__name__, ")")

# %% [markdown]
# ## Try it yourself
# - Edit CUSTOMER_BRIEFING in banklib/prompts.py: ask for a "cross-sell opportunity". Does the model respect
#   "focus on service resolution first"? Should it?
# - Notice Python found the duplicate EMI in the transactions - the complaint is backed by evidence.
