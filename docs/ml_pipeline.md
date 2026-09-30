# Machine Learning & NLP Pipeline

## 1. Overview
The machine learning pipeline is designed with strict production engineering standards:
- **Reproducibility**: All random seeds pinned to `42`.
- **Zero Data Leakage**: Stratified splits performed prior to any feature extraction; TF-IDF fit strictly on training samples only.
- **Multi-Task Decoupling**: Separate models for Category Classification and Urgency Prediction, recognizing that domain categories and priority signals rely on distinct linguistic features.
- **Calibrated Probabilities**: Raw tree ensemble probabilities are calibrated using 5-fold cross-validated Platt scaling (sigmoid) on training data.
- **Explainability**: Linguistic feature attribution extracted from calibrated fold estimators weighted by document TF-IDF vectors.
- **Closed-Loop Active Learning**: Continuous feedback loop allowing human reviewer corrections to retrain candidate models with automated test-set validation before promotion.

---

## 2. Text Preprocessing Pipeline
Customer tickets contain unstructured noise, email headers, boilerplate signatures, and URLs. The preprocessing routine (`clean_text`) applies:

1. **Case Normalization**: Lowercase all characters.
2. **Entity Removal**:
   - Email addresses stripped via regex `r'\S+@\S+'`.
   - Web URLs stripped via regex `r'http\S+|www\.\S+'`.
   - HTML markup stripped via regex `r'<[^>]+>'`.
3. **Punctuation & Character Filtering**:
   - Keep alphanumeric tokens, basic sentence boundaries (`.`, `,`, `!`, `?`, `-`).
   - Strip standalone numeric tokens while retaining alphanumeric identifiers.
4. **Keyword Preservation**:
   - Preserves high-signal urgency and domain keywords: `urgent`, `refund`, `failed`, `blocked`, `payment`, `error`, `outage`, `crash`.
5. **Whitespace Canonicalization**: Multi-spaces and linebreaks collapsed to single spaces.

---

## 3. Train / Validation / Test Splitting Strategy

```
Total Clean Tickets: 23,747
  │
  ├── 70% Train (16,622 tickets)  ──> Feature Extraction, Model Training & 5-Fold Calibration
  │
  ├── 15% Val   (3,562 tickets)   ──> Model Selection & Operating Threshold Tuning
  │
  └── 15% Test  (3,562 tickets)   ──> Held-out Final Benchmark & Reliability Evaluation (Touched ONCE)
```

### Data Leakage Safeguards
- Verified pairwise text overlap:
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- No target-derived features or temporal metadata included in inference features.
- Calibration parameters fitted strictly on the training partition via cross-validation; test set preserved untouched.

---

## 4. Feature Engineering Strategies

### Strategy A: N-Gram TF-IDF Bag-of-Words
- **Vocabulary Size**: 10,000 maximum features (Category), 8,000 maximum features (Urgency).
- **N-Gram Range**: Unigrams and Bigrams `(1, 2)`.
- **Sublinear Term Frequency**: Enabled ($1 + \log(\text{TF})$) to discount recurring repetitive words.
- **Frequency Cutoffs**: `min_df=2` (filters typo noise), `max_df=0.95` (filters corpus-wide stopwords).

### Strategy B: Dense Sentence Transformer Embeddings
- **Model**: `all-MiniLM-L6-v2` (~80MB, 384-dimensional dense vectors).
- **Inference Latency**: Batch encoding at ~66 texts/second on CPU (~243 seconds total for training corpus).

---

## 5. Probability Calibration Methodology

Raw Random Forest `predict_proba()` computes the proportion of trees voting for a class. In multi-class settings with class imbalance, this is known to produce overconfident or poorly calibrated probability distributions.

1. **Platt Scaling (Sigmoid)**: Fits a logistic transformation on out-of-fold predictions using 5-fold cross-validation (`cv=5`) directly on training data.
2. **Why Not Isotonic Regression?**:
   - Isotonic regression fits a non-parametric, monotonic step function.
   - With 10 classes and severe support disparities (e.g., `General Inquiry` with 238 training examples vs. `Technical Support` with 4,800), isotonic regression overfits and creates piecewise flat probability artifacts.
   - Platt sigmoid calibration applies a smooth parametric sigmoid regularized across folds.
3. **Artifact Compression**:
   - Deep Random Forest ensembles exceed 150MB uncompressed, violating GitHub's 100MB file limit.
   - We apply `joblib.dump(..., compress=3)`, reducing `category_model.joblib` to 75.9MB and `urgency_model.joblib` to 62.8MB with zero precision loss.

---

## 6. Model Benchmarking (Validation Set)

### Category Classification (10 Classes)

| Model Architecture | Features | Val Accuracy | Macro F1 | Weighted F1 | Train Latency | Inference Latency |
|---|---|---|---|---|---|---|
| **Logistic Regression** | TF-IDF (10k) | 44.13% | 0.4307 | 0.4433 | 3.12s | 3.0 ms |
| **Multinomial Naive Bayes** | TF-IDF (10k) | 41.58% | 0.3251 | 0.3891 | 0.05s | 3.0 ms |
| **Linear SVM (Calibrated)** | TF-IDF (10k) | 54.01% | 0.5274 | 0.5318 | 8.87s | 14.5 ms |
| **Random Forest (Selected)** | TF-IDF (10k) | **54.66%** | **0.5672** | **0.5512** | 5.55s | 89.1 ms |
| **Logistic Regression** | Embeddings (384d) | 29.62% | 0.2805 | 0.3059 | 35.06s | 2.0 ms |

### Urgency Classification (3 Classes: High, Medium, Low)

| Model Architecture | Features | Val Accuracy | Macro F1 | Weighted F1 | Train Latency |
|---|---|---|---|---|---|
| **Logistic Regression** | TF-IDF (8k) | 53.93% | 0.5303 | 0.5427 | 1.09s |
| **Multinomial Naive Bayes** | TF-IDF (8k) | 49.58% | 0.4345 | 0.4749 | 0.04s |
| **Linear SVM (Calibrated)** | TF-IDF (8k) | 56.71% | 0.5151 | 0.5503 | 3.82s |
| **Random Forest (Selected)** | TF-IDF (8k) | **68.95%** | **0.6761** | **0.6871** | 15.78s |
| **Logistic Regression** | Embeddings (384d) | 43.77% | 0.4305 | 0.4410 | 2.50s |

---

## 7. Active Learning & Human Feedback Retraining Pipeline

The system establishes a closed-loop human-in-the-loop retraining mechanism:

1. **Extraction & Validation (`scripts/build_feedback_dataset.py`)**:
   - Queries `human_reviews` table from PostgreSQL/SQLite.
   - Validates that corrected categories and urgencies match canonical label sets.
   - Strips blank/corrupted records and writes clean feedback samples to `data/processed/human_feedback.csv`.
2. **Candidate Retraining & Gating (`scripts/retrain_from_feedback.py`)**:
   - Augments base training data with verified human corrections.
   - Retrains the candidate category and urgency pipelines with 5-fold sigmoid calibration.
   - Evaluates the candidate against the baseline model on held-out test data.
   - Candidate models are written to `ml/models/candidates/` and are **only promoted** to production if explicitly approved via the `--promote` CLI flag.
