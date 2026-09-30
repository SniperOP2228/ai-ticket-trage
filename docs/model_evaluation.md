# Model Evaluation & In-Depth Error Analysis

## 1. Held-out Test Set Benchmarks
The held-out test set ($N = 3,562$ tickets) was evaluated **only once** following final model selection on the validation set.

### Category Classification (Selected: Random Forest, 10 Classes)
- **Test Accuracy**: **56.88%** (2,026 / 3,562 correct)
- **Macro Average F1**: **0.5999**
- **Weighted Average F1**: **0.5731**

#### Detailed Per-Class Classification Report

| Support Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Billing and Payments** | **0.89** | 0.72 | **0.80** | 363 |
| **Customer Service** | 0.38 | 0.55 | 0.45 | 536 |
| **General Inquiry** | 0.84 | 0.51 | 0.63 | 51 |
| **Human Resources** | 0.72 | 0.52 | 0.61 | 69 |
| **IT Support** | 0.60 | 0.46 | 0.52 | 425 |
| **Product Support** | 0.57 | 0.45 | 0.51 | 664 |
| **Returns and Exchanges** | 0.73 | 0.62 | 0.67 | 176 |
| **Sales and Pre-Sales** | 0.56 | 0.65 | 0.60 | 108 |
| **Service Outages and Maintenance** | 0.57 | 0.68 | 0.62 | 141 |
| **Technical Support** | 0.56 | 0.62 | 0.59 | 1,029 |
| **Overall Macro Avg** | **0.64** | **0.58** | **0.60** | **3,562** |

### Urgency Classification (Selected: Random Forest, 3 Classes)
- **Test Accuracy**: **69.62%** (2,480 / 3,562 correct)
- **Macro Average F1**: **0.6822**
- **Weighted Average F1**: **0.6908**

| Urgency Level | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **High** | 0.70 | 0.71 | 0.70 | 1,356 |
| **Medium** | 0.67 | 0.77 | 0.72 | 1,492 |
| **Low** | 0.77 | 0.53 | 0.63 | 714 |

---

## 2. In-Depth Error Analysis

### Top Confusion Pairs (Actual $\rightarrow$ Predicted)
Error inspection revealed distinct linguistic ambiguity clusters:

1. **Technical Support $\rightarrow$ Customer Service (192 errors)**: Tickets phrased politely with customer service phrasing but describing subtle software glitches.
2. **Product Support $\rightarrow$ Technical Support (183 errors)**: Inquiries about product compatibility mistaken for technical bugs.
3. **Customer Service $\rightarrow$ Technical Support (112 errors)**: Account login difficulties described using technical words like "authentication timeout".
4. **Product Support $\rightarrow$ Customer Service (107 errors)**: Questions on product feature availability.
5. **IT Support $\rightarrow$ Technical Support (91 errors)**: Internal hardware/system issues confused with external product technical issues.

### Impact of Ticket Text Length on Accuracy

| Text Length Tier | Word Count | Sample Count ($N$) | Category Accuracy | Average Model Confidence |
|---|---|---|---|---|
| **Very Short** | $\le 10$ words | 62 | 56.5% | 0.197 |
| **Short** | 11 - 25 words | 484 | 52.1% | 0.207 |
| **Medium** | 26 - 50 words | 914 | 55.9% | 0.238 |
| **Long (Optimal)** | 51 - 100 words | 1,830 | **60.4%** | **0.265** |
| **Very Long** | $> 100$ words | 272 | 45.2% | 0.241 |

*Key Takeaway*: Medium-to-long tickets provide sufficient context for TF-IDF n-grams to match discriminative patterns. Extremely verbose tickets ($>100$ words) often mention multiple issues across departments, driving confusion.

### Confidence Calibration vs. Empirical Accuracy

| Confidence Bucket | Sample Count ($N$) | Empirical Accuracy | Routing Decision |
|---|---|---|---|
| **$< 0.50$** | 3,264 | 53.1% | Human Review Required |
| **$0.50 - 0.60$** | 57 | 91.2% | Human Review Required |
| **$0.60 - 0.70$** | 49 | 100.0% | Human Review Required |
| **$0.70 - 0.80$** | 80 | 100.0% | Human Review Required |
| **$0.80 - 0.90$** | 76 | **100.0%** | **Automatically Routed** |
| **$0.90 - 1.00$** | 36 | **100.0%** | **Automatically Routed** |

*Verification*: When confidence reaches or exceeds 0.80, empirical accuracy on the held-out test set is **100%**! The confidence threshold reliably filters out ambiguous predictions.
