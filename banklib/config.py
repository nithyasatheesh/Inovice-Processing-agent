"""Central settings for the NovaBank GenAI & Open Banking demo kit."""
import os
from pathlib import Path
from datetime import date, datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POLICIES = DATA / "policies"
OPEN_BANKING = DATA / "open_banking"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Ollama
# Deck setup: ollama pull qwen2.5:7b  (16 GB RAM). On 8 GB laptops use llama3.2.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "240"))
ALLOW_FALLBACK = os.getenv("ALLOW_FALLBACK", "1") == "1"

BANK = "NovaBank"
TODAY = date(2026, 11, 20)                 # fixed "today" so every run matches the slides
NOW = datetime(2026, 11, 20, 18, 0)

# ---------------------------------------------------------------- rules
HIGH_VALUE_MULTIPLE = 5.0        # Demo 2: amount >= 5x customer average = high value
DUPLICATE_WINDOW_MIN = 10        # Demo 2: same merchant + amount within 10 minutes
ODD_HOURS = (0, 5)               # Demo 2: 00:00-04:59 = unusual time
HOME_COUNTRY = "India"
RAG_TOP_K = 3                    # Demo 4: policy sections retrieved
RAG_MIN_SCORE = 2.0              # Demo 4: below this, answer "cannot determine"
REFUSAL = "I cannot determine the answer from the available policy."
