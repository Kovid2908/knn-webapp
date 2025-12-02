# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Loan Approval Dashboard")


# -----------------------------------
# Utilities: load model and data
# -----------------------------------
@st.cache_resource
def load_model(path="loan_knn_model.pkl"):
    return joblib.load(path)


@st.cache_data
def load_dataset(path="loan_data.csv"):
    df = pd.read_csv(path)
    return df


model = load_model()
df = load_dataset()

# Columns used by the model (should match training)
FEATURE_COLUMNS = [c for c in df.columns if c != "Loan_Status"]


# -----------------------------------
# Sidebar: controls and dataset
# -----------------------------------
st.sidebar.title("Controls")
show_data = st.sidebar.checkbox("Show dataset (first 50 rows)")
show_metrics = st.sidebar.checkbox("Show model metrics & diagnostics")
show_importance = st.sidebar.checkbox("Show feature importance")
threshold = st.sidebar.slider("Decision threshold (probability for approval)", 0.0, 1.0, 0.5, 0.01)

st.title("Loan Approval Prediction — Interactive Dashboard")
st.write(
    "This dashboard displays model behavior (k-NN) and allows interactive prediction and diagnostics."
)

if show_data:
    st.subheader("Dataset Sample")
    st.dataframe(df.head(50))


# -----------------------------------
# Prepare train/test and metrics
# -----------------------------------
@st.cache_data
def prepare_metrics(df):
    X = df[FEATURE_COLUMNS]
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Predictions and probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_default),
        "precision": precision_score(y_test, y_pred_default, zero_division=0),
        "recall": recall_score(y_test, y_pred_default, zero_division=0),
        "f1": f1_score(y_test, y_pred_default, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_default)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Precision-recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred_default": y_pred_default,
        "metrics": metrics,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "prec": prec,
        "rec": rec,
    }


analysis = prepare_metrics(df)

if show_metrics:
    st.subheader("Model Evaluation on Hold-out Test Set")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{analysis['metrics']['accuracy']:.3f}")
    col2.metric("Precision", f"{analysis['metrics']['precision']:.3f}")
    col3.metric("Recall", f"{analysis['metrics']['recall']:.3f}")
    col4.metric("F1 Score", f"{analysis['metrics']['f1']:.3f}")

    # Confusion matrix
    fig_cm, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(analysis['cm'], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # ROC curve
    fig_roc, ax = plt.subplots(figsize=(5, 3))
    ax.plot(analysis['fpr'], analysis['tpr'], label=f"AUC = {analysis['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig_roc)

    # Precision-Recall
    fig_pr, ax = plt.subplots(figsize=(5, 3))
    ax.plot(analysis['rec'], analysis['prec'])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig_pr)


# -----------------------------------
# Feature importance (permutation)
# -----------------------------------
if show_importance:
    st.subheader("Feature Importance (Permutation on Test Set)")

    @st.cache_data
    def compute_permutation(X_test, y_test, _model):
        result = permutation_importance(_model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
        return result

    perm = compute_permutation(analysis['X_test'], analysis['y_test'], model)
    imp_df = pd.DataFrame({
        "feature": analysis['X_test'].columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False)

    fig_imp, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="importance_mean", y="feature", data=imp_df, ax=ax, palette="viridis")
    ax.set_xlabel("Permutation Importance (mean)")
    ax.set_ylabel("")
    st.pyplot(fig_imp)


# -----------------------------------
# Interactive prediction panel
# -----------------------------------
st.header("Interactive Prediction")
st.write("Enter applicant details and get a probability + decision based on the threshold.")

col1, col2 = st.columns(2)
with col1:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, step=1000, value=30000)
    coapplicant_income = st.number_input("Co-applicant Income (monthly)", min_value=0, step=1000, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, step=10, value=200)
    loan_term = st.selectbox("Loan Amount Term (months)", options=[120, 180, 240, 300, 360], index=4)
    credit_history = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
    employment_type = st.selectbox("Employment Type", options=[1, 0], format_func=lambda x: "Salaried (1)" if x == 1 else "Self-employed (0)")

with col2:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=10, value=0)
    assets_value = st.number_input("Total Assets Value (₹)", min_value=0, step=50000, value=200000)
    savings = st.number_input("Total Savings (₹)", min_value=0, step=5000, value=50000)
    education = st.selectbox("Education", options=[1, 0], format_func=lambda x: "Graduate (1)" if x == 1 else "Not Graduate (0)")
    marital_status = st.selectbox("Marital Status", options=[1, 0], format_func=lambda x: "Married (1)" if x == 1 else "Single (0)")
    residential_area = st.selectbox("Residential Area", options=[0, 1, 2], format_func=lambda x: {0: "Urban", 1: "Semi-Urban", 2: "Rural"}[x])

if st.button("Predict"):
    input_data = [[
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        employment_type,
        age,
        dependents,
        existing_loans,
        assets_value,
        savings,
        education,
        marital_status,
        residential_area,
    ]] 

    input_df = pd.DataFrame(input_data, columns=FEATURE_COLUMNS)

    proba = model.predict_proba(input_df)[0, 1]
    predicted = int(proba >= threshold)

    st.subheader("Prediction Result")
    st.write(f"Probability of Approval: {proba:.3f}")
    if predicted == 1:
        st.success(f"✅ Loan Approved (threshold = {threshold})")
    else:
        st.error(f"❌ Loan Rejected (threshold = {threshold})")

    st.subheader("Input Data")
    st.write(input_df)

    # Show nearest neighbors from training set (helps explain k-NN decision)
    try:
        knn = model.named_steps.get('knn', None)
        scaler = model.named_steps.get('scaler', None)
        if knn is not None and scaler is not None:
            # Scale the training set and input the same way
            X_train = analysis['X_train']
            y_train = analysis['y_train']
            X_train_scaled = scaler.transform(X_train)
            input_scaled = scaler.transform(input_df)
            distances, indices = knn.kneighbors(input_scaled, n_neighbors=min(knn.n_neighbors, len(X_train)))
            nn_df = X_train.reset_index(drop=True).loc[indices[0]]
            nn_df = nn_df.copy()
            nn_df['Distance'] = distances[0]
            nn_df['Loan_Status'] = y_train.reset_index(drop=True).loc[indices[0]].values
            st.subheader("Nearest Neighbors (from training set)")
            st.write(nn_df)
    except Exception as e:
        st.write("Could not compute nearest neighbors:", e)

    # Show which features pushed the probability up/down via simple difference from median
    st.subheader("Quick Local Explanation")
    medians = analysis['X_train'].median()
    diff = input_df.iloc[0] - medians
    explain_df = pd.DataFrame({'feature': diff.index, 'diff_from_median': diff.values})
    explain_df = explain_df.sort_values(by='diff_from_median', key=lambda s: s.abs(), ascending=False).head(8)
    st.table(explain_df.set_index('feature'))


st.info("Tip: retrain the model with `train_model.py` to change behavior (k value etc.).")