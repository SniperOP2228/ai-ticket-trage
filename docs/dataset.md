# Dataset Provenance & Exploratory Data Analysis

## 1. Dataset Overview
- **Dataset Name**: Customer Support Tickets
- **Hugging Face Repository**: `Tobi-Bueck/customer-support-tickets`
- **Identifier**: `10.57967/hf/6184`
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC-4.0`)
- **Total Records in Source**: 61,765 tickets (multilingual: German & English)
- **Filtered English Subset**: 28,261 tickets
- **After Deduplication & Cleaning**: 23,747 unique tickets

## 2. Features & Metadata Schema

| Column Name | Data Type | Missing Count | Description |
|---|---|---|---|
| `subject` | String | 3,639 (12.88%) | Ticket subject / title line |
| `body` | String | 1 (<0.01%) | Main customer complaint or query text |
| `queue` | String | 0 (0.00%) | Support routing department (10 distinct classes) |
| `priority` | String | 0 (0.00%) | Ticket urgency level: `high`, `medium`, `low` |
| `type` | String | 0 (0.00%) | Incident, Request, Problem, Change |
| `language` | String | 0 (0.00%) | Filtered strictly to `en` |
| `tag_1` - `tag_8` | String | Various | Secondary ticket tags |

## 3. Target Variables & Class Distributions

### Category Target (`queue`)
The dataset includes **10 real-world customer support queues**:
1. **Technical Support**: 6,858 tickets (28.9%)
2. **Product Support**: 4,430 tickets (18.7%)
3. **Customer Service**: 3,570 tickets (15.0%)
4. **IT Support**: 2,833 tickets (11.9%)
5. **Billing and Payments**: 2,419 tickets (10.2%)
6. **Returns and Exchanges**: 1,173 tickets (4.9%)
7. **Service Outages and Maintenance**: 937 tickets (3.9%)
8. **Sales and Pre-Sales**: 724 tickets (3.1%)
9. **Human Resources**: 461 tickets (1.9%)
10. **General Inquiry**: 342 tickets (1.4%)

*Class Imbalance Ratio (Max / Min)*: 20.17x. Handled via `class_weight='balanced'` and evaluation on Macro F1.

### Urgency Target (`priority`)
Genuine ground-truth priority labels from original ticket logs:
- **Medium**: 9,753 tickets (41.1%)
- **High**: 9,149 tickets (38.5%)
- **Low**: 4,845 tickets (20.4%)

*Imbalance Ratio*: 2.01x.

## 4. Text Length Distributions
- **Subject**: Mean length = 43.7 chars (~5.8 words)
- **Body**: Mean length = 371.3 chars (~55.4 words)
- **Combined (Subject + Body)**:
  - Mean Character Length: 410.3 (Median: 405.0, Std: 203.5)
  - Mean Word Count: 60.4 (Median: 59.0, Std: 31.3)

## 5. Train / Validation / Test Splitting Strategy
To prevent data snooping and maintain statistical validity:
- **Train Split (70%)**: 16,622 tickets
- **Validation Split (15%)**: 3,562 tickets
- **Test Split (15%)**: 3,562 tickets
- **Stratification**: Enforced across all 10 category classes.
- **Data Leakage Check**: Checked exact-text overlap across splits:
  - Train-Val Overlap: 0
  - Train-Test Overlap: 0 (after 1 duplicate purged)
  - Val-Test Overlap: 0
- **Random Seed**: `42` for strict reproducibility.
