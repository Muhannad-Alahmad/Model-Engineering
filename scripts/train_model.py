"""
train_model.py
--------------
End-to-end training script for the SMS spam filter case study.

Runs the full CRISP-DM workflow:
  1. Load the SMSSpamCollection dataset
  2. Quick data quality checks (duplicates, nulls, length stats)
  3. Light text cleaning (lowercase, strip URLs/numbers/punct)
  4. Train/test split (stratified, random_state=42)
  5. Build three candidate pipelines: MNB, Logistic Regression, Linear SVM
  6. 5-fold cross-validation on the training set
  7. Final fit + evaluation on the held-out test set
  8. Save the selected model and the fitted vectorizer
  9. Save evaluation tables (csv) and figures (png) used for analysis and reporting

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "SMSSpamCollection.csv"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "model"

for d in (PROC_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Academic plotting style
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)
GREY_DARK = "#0072B2"   # primary (Okabe-Ito blue)
GREY_MID = "#D55E00"    # secondary (Okabe-Ito vermillion)
GREY_LIGHT = "#56B4E9"  # tertiary fill (Okabe-Ito sky blue)
ACCENT_GREEN = "#009E73"  # third series colour (Okabe-Ito bluish green)


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )
    return df


# ----------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
NUM_RE = re.compile(r"\b\d+\b")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light, reversible cleaning suitable for short SMS messages."""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    # keep letters and the two special tokens
    text = re.sub(r"[^a-z\s]", " ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


# ----------------------------------------------------------------------------
# 3. EDA figures
# ----------------------------------------------------------------------------
def fig_class_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    bars = ax.bar(counts.index, counts.values, color=[GREY_DARK, GREY_MID])
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 30,
            f"{val} ({val / len(df):.1%})",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Number of messages")
    ax.set_title("Class distribution: ham vs. spam")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.savefig(FIG_DIR / "fig01_class_distribution.png")
    plt.close(fig)


def fig_length_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bins = np.linspace(0, 350, 36)
    ax.hist(
        df.loc[df.label == "ham", "length"],
        bins=bins,
        color=GREY_LIGHT,
        edgecolor=GREY_DARK,
        label="ham",
        alpha=0.9,
    )
    ax.hist(
        df.loc[df.label == "spam", "length"],
        bins=bins,
        color=GREY_DARK,
        edgecolor="black",
        label="spam",
        alpha=0.75,
    )
    ax.set_xlabel("Message length (characters)")
    ax.set_ylabel("Frequency")
    ax.set_title("Message length distribution by class")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "fig02_length_distribution.png")
    plt.close(fig)


def fig_avg_length(df: pd.DataFrame) -> None:
    means = df.groupby("label")["length"].mean()
    medians = df.groupby("label")["length"].median()
    x = np.arange(len(means))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x - width / 2, means.values, width, color=GREY_DARK, label="mean")
    ax.bar(x + width / 2, medians.values, width, color=GREY_MID, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Characters")
    ax.set_title("Average and median message length by class")
    for i, (m, md) in enumerate(zip(means.values, medians.values)):
        ax.text(i - width / 2, m + 2, f"{m:.0f}", ha="center", fontsize=9)
        ax.text(i + width / 2, md + 2, f"{md:.0f}", ha="center", fontsize=9)
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "fig03_avg_length.png")
    plt.close(fig)


def fig_top_words(df_clean: pd.DataFrame) -> None:
    """Top words per class using a simple stopword-free CountVectorizer."""
    vec = CountVectorizer(stop_words="english", min_df=3)
    X = vec.fit_transform(df_clean["clean"])
    vocab = np.array(vec.get_feature_names_out())

    rows = []
    for cls in ["ham", "spam"]:
        mask = (df_clean["label"] == cls).values
        sums = np.asarray(X[mask].sum(axis=0)).ravel()
        top_idx = np.argsort(sums)[-15:][::-1]
        for w, c in zip(vocab[top_idx], sums[top_idx]):
            rows.append({"class": cls, "word": w, "count": int(c)})
    top_df = pd.DataFrame(rows)
    top_df.to_csv(TAB_DIR / "top_words.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, cls, color in zip(axes, ["ham", "spam"], [GREY_MID, GREY_DARK]):
        sub = top_df[top_df["class"] == cls].iloc[::-1]
        ax.barh(sub["word"], sub["count"], color=color)
        ax.set_title(f"Top 15 tokens – {cls}")
        ax.set_xlabel("Token frequency")
    fig.suptitle("Most frequent tokens by class (English stopwords removed)")
    fig.savefig(FIG_DIR / "fig04_top_words.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Modeling
# ----------------------------------------------------------------------------
def build_pipelines() -> dict[str, Pipeline]:
    """Return the three candidate pipelines."""
    return {
        "MNB_count": Pipeline(
            [
                ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
                ("clf", MultinomialNB()),
            ]
        ),
        "LogReg_tfidf": Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=4.0,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "LinearSVM_tfidf": Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(C=1.0, class_weight="balanced"),
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def cv_scores(pipelines: dict[str, Pipeline], X, y) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in pipelines.items():
        accs = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        f1s = cross_val_score(pipe, X, y, cv=skf, scoring="f1", n_jobs=-1)
        precs = cross_val_score(pipe, X, y, cv=skf, scoring="precision", n_jobs=-1)
        recs = cross_val_score(pipe, X, y, cv=skf, scoring="recall", n_jobs=-1)
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": accs.mean(),
                "cv_accuracy_std": accs.std(),
                "cv_precision_mean": precs.mean(),
                "cv_recall_mean": recs.mean(),
                "cv_f1_mean": f1s.mean(),
                "cv_f1_std": f1s.std(),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)


def evaluate_on_test(name: str, pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


# ----------------------------------------------------------------------------
# 5. Evaluation figures
# ----------------------------------------------------------------------------
def fig_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix – {model_name}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG_DIR / "fig05_confusion_matrix.png")
    plt.close(fig)


def fig_pr_curve(y_test, y_proba, model_name: str) -> None:
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.plot(rec, prec, color=GREY_DARK, linewidth=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve – {model_name}")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    fig.savefig(FIG_DIR / "fig06_pr_curve.png")
    plt.close(fig)


def fig_threshold_table_chart(thresh_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(thresh_df["threshold"], thresh_df["precision"], "-o",
            color=GREY_DARK, label="Precision (spam)")
    ax.plot(thresh_df["threshold"], thresh_df["recall"], "-s",
            color=GREY_MID, label="Recall (spam)")
    ax.plot(thresh_df["threshold"], thresh_df["f1"], "--^",
            color=ACCENT_GREEN, label="F1 (spam)")
    ax.set_xlabel("Spam probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("Effect of decision threshold on spam classification")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG_DIR / "fig07_threshold_curve.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Threshold / risk-level analysis
# ----------------------------------------------------------------------------
def threshold_analysis(y_test, y_proba) -> pd.DataFrame:
    rows = []
    for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        y_hat = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_hat)
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "threshold": t,
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "precision": precision_score(y_test, y_hat, zero_division=0),
                "recall": recall_score(y_test, y_hat, zero_division=0),
                "f1": f1_score(y_test, y_hat, zero_division=0),
                "accuracy": accuracy_score(y_test, y_hat),
            }
        )
    return pd.DataFrame(rows)


def risk_levels(threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Map three operational risk levels to thresholds."""
    mapping = {
        "Strict (low-risk)": 0.30,
        "Balanced (medium-risk)": 0.50,
        "Permissive (high-risk)": 0.70,
    }
    rows = []
    for label, t in mapping.items():
        sub = threshold_df[threshold_df["threshold"] == t].iloc[0].to_dict()
        sub["risk_level"] = label
        rows.append(sub)
    cols = ["risk_level", "threshold", "TP", "FP", "FN", "TN",
            "precision", "recall", "f1", "accuracy"]
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------
def main() -> None:
    print("[1/7] Loading data ...")
    df = load_data()
    df["length"] = df["message"].str.len()

    dup_before = int(df.duplicated().sum())
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    df_dedup["clean"] = df_dedup["message"].apply(clean_text)

    # Save processed
    df_dedup.to_csv(PROC_DIR / "sms_clean.csv", index=False)

    # Dataset overview table
    overview = pd.DataFrame(
        {
            "metric": [
                "rows (raw)",
                "rows (deduplicated)",
                "duplicates removed",
                "missing values",
                "ham count",
                "spam count",
                "spam share (%)",
                "mean message length (chars)",
                "median message length (chars)",
                "max message length (chars)",
            ],
            "value": [
                len(df),
                len(df_dedup),
                dup_before,
                int(df.isnull().sum().sum()),
                int((df_dedup.label == "ham").sum()),
                int((df_dedup.label == "spam").sum()),
                round(100 * (df_dedup.label == "spam").mean(), 2),
                round(df_dedup["length"].mean(), 1),
                int(df_dedup["length"].median()),
                int(df_dedup["length"].max()),
            ],
        }
    )
    overview.to_csv(TAB_DIR / "dataset_overview.csv", index=False)
    print(overview.to_string(index=False))

    print("[2/7] EDA figures ...")
    fig_class_distribution(df_dedup)
    fig_length_distribution(df_dedup)
    fig_avg_length(df_dedup)
    fig_top_words(df_dedup)

    print("[3/7] Train/test split ...")
    y = (df_dedup["label"] == "spam").astype(int).values
    X = df_dedup["clean"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("[4/7] Cross-validating candidate models ...")
    pipelines = build_pipelines()
    cv_df = cv_scores(pipelines, X_train, y_train)
    cv_df.round(4).to_csv(TAB_DIR / "cv_results.csv", index=False)
    print(cv_df.round(4).to_string(index=False))

    print("[5/7] Fitting and evaluating on test set ...")
    test_rows = []
    fitted: dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        test_rows.append(evaluate_on_test(name, pipe, X_test, y_test))
    test_df = pd.DataFrame(test_rows).sort_values("f1", ascending=False)
    test_df.round(4).to_csv(TAB_DIR / "test_results.csv", index=False)
    print(test_df.round(4).to_string(index=False))

    # Select best by F1 on test set, tiebreaker = recall (catching spam matters)
    best_name = test_df.sort_values(["f1", "recall"], ascending=False).iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"[6/7] Selected model: {best_name}")

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    fig_confusion_matrix(cm, best_name)
    fig_pr_curve(y_test, y_proba, best_name)

    # Classification report
    cls_report = classification_report(
        y_test, y_pred, target_names=["ham", "spam"], digits=4
    )
    (TAB_DIR / "classification_report.txt").write_text(cls_report)
    print(cls_report)

    # Threshold / risk-level analysis
    thr_df = threshold_analysis(y_test, y_proba)
    thr_df.round(4).to_csv(TAB_DIR / "threshold_table.csv", index=False)
    fig_threshold_table_chart(thr_df)
    rl_df = risk_levels(thr_df)
    rl_df.round(4).to_csv(TAB_DIR / "risk_levels.csv", index=False)
    print(rl_df.round(4).to_string(index=False))

    # Error analysis (sample misclassifications)
    test_df_err = pd.DataFrame(
        {
            "message": X_test,
            "actual": np.where(y_test == 1, "spam", "ham"),
            "predicted": np.where(y_pred == 1, "spam", "ham"),
            "spam_prob": y_proba,
        }
    )
    fp = test_df_err[(test_df_err.actual == "ham") & (test_df_err.predicted == "spam")]
    fn = test_df_err[(test_df_err.actual == "spam") & (test_df_err.predicted == "ham")]
    fp.sort_values("spam_prob", ascending=False).head(10).to_csv(
        TAB_DIR / "false_positives.csv", index=False
    )
    fn.sort_values("spam_prob").head(10).to_csv(
        TAB_DIR / "false_negatives.csv", index=False
    )

    print("[7/7] Saving model and summary ...")
    joblib.dump(best_pipe, MODEL_DIR / "spam_filter.joblib")

    summary = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "selected_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_metrics": test_df.set_index("model").loc[best_name].to_dict(),
        "confusion_matrix": cm.tolist(),
        "risk_levels": rl_df.to_dict(orient="records"),
    }
    (MODEL_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done. Outputs written under outputs/")


if __name__ == "__main__":
    main()
