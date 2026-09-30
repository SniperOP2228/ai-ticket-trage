# Machine Learning & NLP Pipeline

## 1. Overview
The machine learning pipeline is designed with strict production engineering standards:
- **Reproducibility**: All random seeds pinned to `42`.
- **Zero Data Leakage**: Stratified splits performed prior to any feature extraction; TF-IDF fit strictly on training samples only.
- **Multi-Task Decoupling**: Separate models for Category Classification and Urgency Prediction, recognizing that domain categories and priority signals rely on distinct linguistic features.
- **Calibrated Scoring**: Threshold-based routing rather than blind probability acceptance.

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

## 3. Train / Validation / Test Splitting Strategy

```
Total Clean Tickets: 23,747
  │
  ├── 70% Train (16,622 tickets)  ──> Model Training & Feature Extraction
  │
  ├── 15% Val   (3,562 tickets)   ──> Hyperparameter Tuning & Model Selection
  │
  └── 15% Test  (3,562 tickets)   ──> Held-out Final Benchmark (Touched ONCE)
```

### Data Leakage Inspection
- Verified pairwise text overlap:
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- No target-derived features or temporal metadata included in inference features.

## 4. Feature Engineering Strategies

### Strategy A: N-Gram TF-IDF Bag-of-Words
- **Vocabulary Size**: 10,000 maximum features.
- **N-Gram Range**: Unigrams and Bigrams `(1, 2)`.
- **Sublinear Term Frequency**: Enabled ($1 + \log(\text{TF})$) to discount recurring repetitive words.
- **Frequency Cutoffs**: `min_df=2` (filters typo noise), `max_df=0.95` (filters corpus-wide stopwords).

### Strategy B: Dense Sentence Transformer Embeddings
- **Model**: `all-MiniLM-L6-v2` (~80MB, 384-dimensional dense vectors).
- **Inference Latency**: Batch encoding at ~66 texts/second on CPU (~243 seconds total for training corpus).

## 5. Model Benchmarking & Comparison

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

## 6. Engineering Analysis: TF-IDF vs. Dense Embeddings
In this domain-specific technical support dataset:
1. **Vocabulary Specificity**: High-signal terminology (`500 internal server error`, `refund duplicate`, `malwarebytes`, `failed payout`) is sparse and discriminative. N-gram TF-IDF cleanly isolates these terms with high feature weights.
2. **Dense Vector Dispersion**: General-purpose MiniLM embeddings map general conversational text effectively but lose sensitivity to rare domain-specific technical acronyms without task-specific fine-tuning.
3. **Operational Feasibility**: TF-IDF transforms in **1.5 seconds** vs. **243 seconds** for transformer embedding on CPU, requiring 10x less memory footprint for cloud deployment.
