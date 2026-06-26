# ChurnSense — Phase 1
### Multi-Class Churn Prediction with Cost-Sensitive Learning

---

## The problem with standard churn models

Most churn prediction projects answer a binary question: **will this customer leave?**  
ChurnSense asks a better one: **how urgently will they leave, and what does it cost to get it wrong?**

A customer about to churn this month needs a different response than one who might leave in six months. A model that treats both the same is leaving money on the table.

---

## What this project does

| | Standard approach | ChurnSense |
|---|---|---|
| Prediction | Binary Yes / No | 4 risk tiers |
| Model | Random Forest | XGBoost |
| Class imbalance | Basic class weights | SMOTE oversampling |
| Objective | Minimise error rate | Minimise business cost |
| Explainability | None | SHAP per-customer |
| Output | Churn probability | Urgency tier + intervention action |

---

## The 4 risk tiers

| Class | Churn Score | Meaning | Default action |
|---|---|---|---|
| **Loyal** | 0–25 | Low risk, long-term customer | Loyalty rewards |
| **At-Risk** | 26–50 | Early warning signs | Re-engagement email |
| **Critical** | 51–75 | High risk, still reachable | Personal call + offer |
| **Will-Churn** | 76–100 | Imminent churner | Urgent: immediate high-value offer |

---

## Business cost matrix

Standard models optimise for accuracy. ChurnSense optimises for **revenue protection**.

```
                  Pred: Loyal   At-Risk   Critical   Will-Churn
Actual: Loyal         $0         $10        $25          $60
Actual: At-Risk       $30         $0        $10          $40
Actual: Critical      $80        $25         $0          $20
Actual: Will-Churn   $200        $80        $40           $0
```

These costs are passed as `sample_weight` to XGBoost, so the model learns
that missing a Will-Churn customer costs $200 — not the same as mislabelling a Loyal one.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ChurnSense.git
cd ChurnSense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data file
# Copy Telco_customer_churn.xlsx into the data/ folder

# 4. Launch the notebook
jupyter notebook ChurnSense_Phase1.ipynb
```

---

## Project structure

```
ChurnSense/
├── ChurnSense_Phase1.ipynb    ← main notebook (37 cells, 11 sections)
├── data/
│   └── Telco_customer_churn.xlsx
├── outputs/
│   ├── model_phase1.pkl       ← saved XGBoost model
│   └── plots/                 ← all 17 saved charts
├── requirements.txt
└── README.md
```

---

## Notebook sections

1. **Business context** — why 4-class beats binary
2. **Setup & imports**
3. **Load & audit** — shape, churn rate, missing values
4. **EDA** (8 charts) — contract type, tenure, charges, services, CLTV, payment method
5. **K-Means segmentation** — elbow method, 3 named customer segments
6. **4-class target engineering** — Loyal / At-Risk / Critical / Will-Churn from Churn Score
7. **Feature engineering** — charge-per-tenure ratio, revenue share, service count
8. **Business cost matrix** — sample weights that reflect actual business loss
9. **XGBoost training** — multi-class, cost-sensitive, SMOTE-balanced
10. **Evaluation** — per-class F1, confusion matrix, feature importance, SHAP
11. **Business recommendations** — executive summary, intervention table, ROI estimate

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 33 columns.  
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
or in the IBM Cognos Analytics sample dataset library.

Key columns used:
- `Churn Score` (0–100) — basis for the 4-class target label
- `CLTV` — Customer Lifetime Value
- `Contract`, `Tenure Months`, `Monthly Charges` — top churn drivers
- `Churn Label` / `Churn Value` — used only for EDA validation, dropped before training

---

## Phase 2 (coming next week)

- **30/60/90-day temporal predictions** — not just *who* churns but *when*
- **Risk escalation trajectories** — "this customer will go At-Risk → Critical in 60 days"
- **Budget-constrained ROI optimizer** — given $X, contact these N customers first
- **Streamlit dashboard** — upload any CSV, get live predictions + SHAP charts
- **Live deployment link** — Streamlit Cloud

---

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `imbalanced-learn` · `shap` · `matplotlib` · `seaborn`

---

*Built as part of Coding Blocks School of Technology SIP 2026*
