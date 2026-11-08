import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Spambase Classifier Hub", layout="wide")

# Resolve paths from this file's location so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("Email Spam Classification & Model Evaluation Suite")
st.markdown(
    "Upload the held-out `test_data.csv` from the UCI Spambase dataset and select any pre-trained "
    "classifier from the sidebar to evaluate performance without retraining."
)

# Sidebar Controls
st.sidebar.header("Evaluation Controls")
uploaded_csv = st.sidebar.file_uploader("1. Upload Test Dataset (test_data.csv)", type=["csv"])

# Convenience button to load bundled test_data.csv directly if present in the repo
use_default_test = st.sidebar.checkbox("Or use bundled test_data.csv from repository", value=False)

model_registry = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree Classifier": "model/Decision_Tree.pkl",
    "K-Nearest Neighbor (KNN)": "model/KNN.pkl",
    "Naive Bayes (Gaussian)": "model/Naive_Bayes.pkl",
    "Support Vector Machine (SVM)": "model/Support_Vector_Machine.pkl"
}

chosen_model_name = st.sidebar.selectbox("2. Select Pre-Trained Model", list(model_registry.keys()))


@st.cache_resource
def load_pipeline(path):
    """Load a saved pipeline once and reuse it across reruns (no retraining)."""
    return joblib.load(path)


default_test_path = os.path.join(BASE_DIR, "test_data.csv")

df_test = None
if uploaded_csv is not None:
    df_test = pd.read_csv(uploaded_csv)
elif use_default_test and os.path.exists(default_test_path):
    df_test = pd.read_csv(default_test_path)

if df_test is not None:
    df_test.columns = df_test.columns.str.strip()

    st.subheader(f"Uploaded Test Dataset Preview ({df_test.shape[0]} rows, {df_test.shape[1]} columns)")
    st.dataframe(df_test.head(5), use_container_width=True)

    # The target column is always `spam` (1 = spam, 0 = non-spam)
    target_col = "spam"
    if target_col not in df_test.columns:
        st.error("The uploaded CSV must contain the target column `spam` along with the 57 feature "
                 "columns. Please upload the `test_data.csv` from the GitHub repository.")
        st.stop()

    X_eval = df_test.drop(columns=[target_col])
    y_eval = df_test[target_col]

    # Load pre-trained pipeline (StandardScaler + Classifier) without retraining
    model_path = os.path.join(BASE_DIR, model_registry[chosen_model_name])
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_registry[chosen_model_name]}` not found. "
                 "Please make sure the `model/` folder is in the repository.")
        st.stop()

    pipeline_model = load_pipeline(model_path)

    # Verify uploaded CSV columns match the trained Spambase features
    expected_cols = list(pipeline_model.feature_names_in_)
    missing_cols = set(expected_cols) - set(X_eval.columns)

    if missing_cols:
        st.error(
            "The uploaded CSV is missing required Spambase feature column(s): "
            + ", ".join(sorted(missing_cols))
            + ". Please upload the `test_data.csv` from the GitHub repository (or tick "
            "'Or use bundled test_data.csv from repository' in the sidebar)."
        )
        st.stop()

    # Align column order and predict
    X_eval = X_eval[expected_cols]
    y_pred = pipeline_model.predict(X_eval)
    y_prob = pipeline_model.predict_proba(X_eval)[:, 1]

    # AUC needs both classes present in the uploaded data
    if y_eval.nunique() > 1:
        auc_text = f"{roc_auc_score(y_eval, y_prob):.4f}"
    else:
        auc_text = "n/a"

    # Display the 6 Mandatory Evaluation Metrics
    st.subheader(f"Performance Metrics — {chosen_model_name}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{accuracy_score(y_eval, y_pred):.4f}")
    m2.metric("AUC Score", auc_text)
    m3.metric("Precision", f"{precision_score(y_eval, y_pred, zero_division=0):.4f}")
    m4.metric("Recall", f"{recall_score(y_eval, y_pred, zero_division=0):.4f}")
    m5.metric("F1 Score", f"{f1_score(y_eval, y_pred, zero_division=0):.4f}")
    m6.metric("MCC Score", f"{matthews_corrcoef(y_eval, y_pred):.4f}")

    st.divider()
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=["Non-Spam (0)", "Spam (1)"],
            yticklabels=["Non-Spam (0)", "Spam (1)"], ax=ax
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        st.pyplot(fig)

    with right_col:
        st.subheader("Classification Report")
        report = classification_report(
            y_eval, y_pred, labels=[0, 1],
            target_names=["Non-Spam (0)", "Spam (1)"],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
else:
    st.info(
        "Awaiting CSV upload: Please upload `test_data.csv` from the GitHub repository using the sidebar "
        "(or check the box in the sidebar to load the bundled `test_data.csv`)."
    )