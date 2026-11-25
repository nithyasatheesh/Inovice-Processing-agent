"""Loaders for the synthetic NovaBank data pack."""
import json

import pandas as pd

from . import config

D = config.DATA


def profiles():
    return pd.read_csv(D / "customer_profile.csv")


def transactions():
    return pd.read_csv(D / "transactions.csv", parse_dates=["Date"])


def complaints():
    return pd.read_csv(D / "complaints.csv", parse_dates=["Date"])


def products():
    return pd.read_csv(D / "products.csv").fillna("")


def card_transactions():
    return pd.read_csv(D / "card_transactions.csv", parse_dates=["Timestamp"])


def consent(customer_id="C101"):
    return json.loads((config.OPEN_BANKING / f"consent_{customer_id}.json").read_text())


def fip_files():
    return {p.stem.replace("fip_", "").upper(): json.loads(p.read_text())
            for p in sorted(config.OPEN_BANKING.glob("fip_*.json"))}


def cases():
    df = pd.read_excel(D / "operations_cases.xlsx", sheet_name="Cases")
    for c in ("Created", "SLA_Due", "Closed_On"):
        df[c] = pd.to_datetime(df[c])
    df["Notes"] = df["Notes"].fillna("")
    return df
