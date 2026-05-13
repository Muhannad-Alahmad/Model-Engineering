"""
evaluate_model.py
-----------------
Load the saved spam filter and classify new messages from the command line
or from a CSV file.  Returns label, spam probability, and the action that
would be taken under each risk level.

Usage:
    python scripts/evaluate_model.py --text "Free entry! Win cash!"
    python scripts/evaluate_model.py --file path/to/messages.csv --column message
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "model" / "spam_filter.joblib"

RISK_THRESHOLDS = {
    "Strict (low-risk)": 0.30,
    "Balanced (medium-risk)": 0.50,
    "Permissive (high-risk)": 0.70,
}


def decide(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "send to review folder"
    return "display immediately"


def predict_batch(messages: list[str]) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    probs = model.predict_proba(messages)[:, 1]
    out = pd.DataFrame({"message": messages, "spam_probability": probs})
    out["prediction_default"] = pd.Series((probs >= 0.5).astype(int)).map({0: "ham", 1: "spam"}).values
    for level, thr in RISK_THRESHOLDS.items():
        out[f"action [{level}, t={thr}]"] = [decide(p, thr) for p in probs]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spam classifier inference")
    parser.add_argument("--text", help="single message to classify")
    parser.add_argument("--file", help="CSV file with messages")
    parser.add_argument("--column", default="message",
                        help="column name in the CSV (default: message)")
    args = parser.parse_args()

    if args.text:
        messages = [args.text]
    elif args.file:
        df = pd.read_csv(args.file)
        messages = df[args.column].astype(str).tolist()
    else:
        parser.error("Provide --text or --file")

    result = predict_batch(messages)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
