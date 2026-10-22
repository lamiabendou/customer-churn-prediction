# Customer Churn Prediction | Machine Learning

**Portfolio Data Science project — Python · Pandas · Scikit-learn**

## Objective
Predict whether a telecommunications customer will cancel their subscription (*churn*), using customer account, service, and billing information. Compare interpretable and tree-based classifiers and evaluate the trade-off between identifying churners and false alarms.

## Dataset
**Telco Customer Churn**, publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

During the original exploratory work: **7,043 customers**, **21 columns**, **1,869 churners (26.54%)**. Eleven blank `TotalCharges` values were identified and are handled by median imputation in the training pipeline.

## Exploratory insights
- Month-to-month contracts: **42.71%** churn rate; one-year contracts: **11.27%**; two-year contracts: **2.83%**.
- Customers who churned tended to have shorter tenure and higher monthly charges.
- These are **associations**, not causal conclusions.

## Modeling approach
1. Clean `TotalCharges` and exclude the customer identifier.
2. Stratified **60% train / 20% validation / 20% test** split.
3. Impute missing numeric values, scale numeric features, and one-hot encode categories **inside a pipeline** (avoids train/test preprocessing leakage).
4. Train **Logistic Regression** and **Random Forest**.
5. Select the model and decision threshold using **validation F1**.
6. Evaluate the chosen model once on the untouched test set. Export metrics, plots, and a fitted pipeline.

## Previously observed results (earlier Colab experiment)
The following values come from an **earlier 80/20 train/test experiment with threshold 0.50**, not from the reproducible 60/20/20 script. They should not be confused with results from a new run.

| Model | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | **0.8419** |
| Random Forest | 0.7814 | 0.6170 | 0.4652 | 0.5305 | 0.8217 |

**Business perspective:** Lowering the classification threshold can identify more churners, but also creates more false positives. The best threshold depends on the cost of missing a customer versus the cost of retention outreach.

## Run locally

```bash

## Run locally

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the Telco Customer Churn dataset from Kaggle and place the CSV file inside the `data/` directory.

Run the training pipeline:

```bash
python src/train_model.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The script generates:

- `reports/metrics.json` — final test metrics
- `reports/validation_thresholds.csv` — validation results
- `reports/confusion_matrix.png` — confusion matrix
- `reports/roc_curve.png` — ROC curve
- `reports/threshold_analysis.png` — threshold analysis
- `models/churn_pipeline.joblib` — trained model

Run these commands from the repository root.


## Repository structure

```text

customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   └── train_model.py
└── reports/
    ├── README.md
    ├── churn_distribution.png
    ├── churn_by_contract.png
    ├── churn_rate_by_contract.png
    ├── churn_by_payment_method.png
    ├── tenure_by_churn.png
    ├── monthly_charges_by_churn.png
    ├── predicted_probabilities_by_churn.png
    └── roc_curve.png


```

## Skills demonstrated
Exploratory data analysis, preprocessing, classification, model comparison, stratified splits, validation-based threshold selection, confusion matrices, precision/recall/F1, ROC-AUC, and reproducible Python workflows.

## Limitations
- This dataset is a learning benchmark, not live telecom production data.
- No prospective validation, drift monitoring, or fairness audit is included.
- No claim is made that the models have been deployed in a business setting.

## Author
**Lamia Bendou** · [LinkedIn](https://www.linkedin.com/in/lamia-bendou-4195843b0) · [GitHub](https://github.com/lamiabendou)
