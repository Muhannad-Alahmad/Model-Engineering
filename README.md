# Spam Filter Case Study

End-to-end binary classifier that flags SMS messages as **ham** or **spam**, with
adjustable spam-risk levels (strict / balanced / permissive).

The deliverables are:

- **Notebook:** `notebooks/01_spam_filter_model_engineering.ipynb` runs the full
  CRISP-DM workflow end-to-end.
- **Scripts:** `scripts/train_model.py` regenerates every figure, table and the
  serialised model; `scripts/evaluate_model.py` performs inference on new
  messages.
- **Outputs:** `outputs/figures/` (Figures 1-7), `outputs/tables/` (CSV exports
  and classification report), `outputs/model/spam_filter.joblib` and `summary.json`.

## Headline numbers

| Metric (test set, n=1,034) | Selected model — Linear SVM (TF-IDF) |
|---|---|
| Accuracy | 0.9884 |
| Precision (spam) | 0.9837 |
| Recall (spam) | 0.9237 |
| F1 (spam) | 0.9528 |
| ROC-AUC | 0.9974 |

Multinomial Naive Bayes and Logistic Regression were also evaluated and
trail Linear SVM by less than one percentage point; full results in
`outputs/tables/test_results.csv`.

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train + evaluate + persist artefacts
python scripts/train_model.py

# Classify a new message
python scripts/evaluate_model.py --text "Free entry! Click http://bit.ly/xyz"

# Or run the full notebook narrative
jupyter notebook notebooks/01_spam_filter_model_engineering.ipynb
```

All randomness is seeded (`random_state=42`).

## Project structure

```
Model-Engineering
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── SMSSpamCollection.csv
│   └── processed/
│       └── sms_clean.csv
├── notebooks/
│   └── 01_spam_filter_model_engineering.ipynb
├── scripts/
│   ├── train_model.py
│   └── evaluate_model.py
└── outputs/
    ├── figures/
    │   ├── fig01_class_distribution.png
    │   ├── fig02_length_distribution.png
    │   ├── fig03_avg_length.png
    │   ├── fig04_top_words.png
    │   ├── fig05_confusion_matrix.png
    │   ├── fig06_pr_curve.png
    │   └── fig07_threshold_curve.png
    ├── tables/
    └── model/
        ├── spam_filter.joblib
        └── summary.json
```

## Dataset

Almeida, T. A., & Hidalgo, J. M. G. (2011). *SMS Spam Collection* [Dataset].
UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CC84>

5,572 English SMS messages tagged `ham` / `spam`. After removing 403 exact
duplicates, the working set has 5,169 messages (12.6 % spam).

## Notes for the reviewer

- Reproducibility: `scripts/train_model.py` is the single entry point; the
  notebook reuses the same logic so that figures and numbers in the notebook
  and the saved CSVs match exactly.
- The selected model is calibrated (Platt scaling) so that the spam
  probability is meaningful as a threshold for the risk levels.
- Error analysis outputs are available in
  `outputs/tables/false_positives.csv` and `outputs/tables/false_negatives.csv`.

## License
The code in this repository is released under the MIT License.
The SMS Spam Collection dataset is provided by the UCI Machine Learning Repository and is cited separately.