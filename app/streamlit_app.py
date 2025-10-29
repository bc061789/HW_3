import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from io import BytesIO

# -------- helpers for export --------
def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def df_to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return buf

def text_to_txt_bytes(txt):
    buf = BytesIO(txt.encode("utf-8"))
    buf.seek(0)
    return buf
# -----------------------------------

st.set_page_config(page_title="Phishing/Spam Numeric Classifier", layout="wide")
st.title("Phishing/Spam Classifier (Numeric Features)")

# ---------------- data load ----------------
st.sidebar.header("Data Source")
use_default = st.sidebar.checkbox("Use repo dataset: datasets/phishing_dataset.csv", value=True)
uploaded = st.sidebar.file_uploader("or Upload CSV", type=["csv"])

def load_repo_data():
    try:
        return pd.read_csv("datasets/phishing_dataset.csv", header=None, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.warning(f"Failed to load repo dataset: {e}")
        return None

df = None
if use_default:
    df = load_repo_data()
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded, header=None, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Read error: {e}")

if df is None:
    st.info("Please enable 'Use repo dataset' or upload a CSV.")
    st.stop()

st.subheader("Preview")
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns (treated as **headerless**, last column = label)")
st.dataframe(df.head(20))

# --------------- columns ----------------
st.sidebar.header("Column Settings")
headerless = st.sidebar.checkbox("No header (headerless)", value=True)
if not headerless:
    df.columns = [str(c) for c in df.columns]
else:
    df.columns = list(range(df.shape[1]))

label_col = st.sidebar.selectbox("Label column", options=list(df.columns), index=len(df.columns)-1)
feature_cols = [c for c in df.columns if c != label_col]

# ------------- numeric + labels ----------
for c in feature_cols + [label_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

y_raw = df[label_col].values
unique_labels = np.unique(y_raw)
y = y_raw.copy()
pos_label_name, neg_label_name = "Positive", "Negative"
if set(unique_labels.tolist()) == {-1, 1}:
    y = (y_raw == 1).astype(int)
    pos_label_name, neg_label_name = "1 (Phishing)", "-1 (Legit)"
else:
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    if y_min != y_max:
        y = ((y_raw - y_min) / (y_max - y_min) >= 0.5).astype(int)

X = df[feature_cols].values

# ------------- training setup ------------
st.markdown("### Train / Eval Settings")
test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.number_input("Random State", value=42, step=1)
model_name = st.selectbox("Model", ["LogisticRegression", "RandomForest"])
do_standardize = st.checkbox("Standardize (StandardScaler)", value=(model_name=="LogisticRegression"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

if do_standardize:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    scaler = None

if model_name == "LogisticRegression":
    clf = LogisticRegression(class_weight="balanced", C=2.0, max_iter=200, solver="liblinear")
else:
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=random_state, n_jobs=-1)

clf.fit(X_train, y_train)

# ------------- probabilities -------------
if hasattr(clf, "predict_proba"):
    y_proba = clf.predict_proba(X_test)[:, 1]
else:
    s = clf.decision_function(X_test)
    y_proba = (s - s.min()) / (s.max() - s.min() + 1e-9)

# ------------- metrics & plots -----------
st.markdown("### Metrics & Plots")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
y_pred = (y_proba >= threshold).astype(int)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Classification Report**")
    report_txt = classification_report(y_test, y_pred,
                                       target_names=[neg_label_name, pos_label_name],
                                       digits=3)
    st.text(report_txt)
    st.download_button("下載 Classification Report (.txt)",
                       data=text_to_txt_bytes(report_txt),
                       file_name="classification_report.txt", mime="text/plain")

with c2:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([neg_label_name, pos_label_name])
    ax.set_yticklabels([neg_label_name, pos_label_name])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

    cm_df = pd.DataFrame(cm, index=[neg_label_name, pos_label_name],
                         columns=[neg_label_name, pos_label_name])
    st.download_button("下載 Confusion Matrix 圖 (PNG)",
                       data=fig_to_png_bytes(fig), file_name="confusion_matrix.png",
                       mime="image/png")
    st.download_button("下載 Confusion Matrix 數據 (CSV)",
                       data=df_to_csv_bytes(cm_df.reset_index().rename(columns={"index": "true"})),
                       file_name="confusion_matrix.csv", mime="text/csv")

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
st.markdown(f"**ROC AUC:** {roc_auc:.3f}")
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
ax2.plot([0, 1], [0, 1], linestyle="--")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right")
st.pyplot(fig2)

roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
st.download_button("下載 ROC 圖 (PNG)",
                   data=fig_to_png_bytes(fig2), file_name="roc_curve.png",
                   mime="image/png")
st.download_button("下載 ROC 數據 (CSV)",
                   data=df_to_csv_bytes(roc_df), file_name="roc_curve.csv",
                   mime="text/csv")

# PR
prec, rec, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(rec, prec)
st.markdown(f"**PR AUC:** {pr_auc:.3f}")
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(rec, prec, label=f"PR curve (AUC = {pr_auc:.3f})")
ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
ax3.legend(loc="lower left")
st.pyplot(fig3)

pr_df = pd.DataFrame({"recall": rec, "precision": prec})
st.download_button("下載 PR 圖 (PNG)",
                   data=fig_to_png_bytes(fig3), file_name="pr_curve.png",
                   mime="image/png")
st.download_button("下載 PR 數據 (CSV)",
                   data=df_to_csv_bytes(pr_df), file_name="pr_curve.csv",
                   mime="text/csv")

# ------------- feature importance / coef -------------
st.markdown("### Feature Importance / Coefficients")

if model_name == "RandomForest":
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.bar(range(len(idx)), importances[idx])
    ax4.set_xticks(range(len(idx)))
    ax4.set_xticklabels([str(feature_cols[i]) for i in idx], rotation=45, ha="right")
    ax4.set_ylabel("Importance")
    st.pyplot(fig4)

    fi_df = pd.DataFrame({
        "feature": [str(feature_cols[i]) for i in idx],
        "importance": importances[idx]
    })
    st.download_button("下載 Feature Importance 圖 (PNG)",
                       data=fig_to_png_bytes(fig4), file_name="feature_importance.png",
                       mime="image/png")
    st.download_button("下載 Feature Importance 數據 (CSV)",
                       data=df_to_csv_bytes(fi_df), file_name="feature_importance.csv",
                       mime="text/csv")

elif model_name == "LogisticRegression":
    coefs = clf.coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:20]

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.bar(range(len(idx)), coefs[idx])
    ax5.set_xticks(range(len(idx)))
    ax5.set_xticklabels([str(feature_cols[i]) for i in idx], rotation=45, ha="right")
    ax5.set_ylabel("Coefficient")
    st.pyplot(fig5)

    coef_df = pd.DataFrame({
        "feature": [str(feature_cols[i]) for i in idx],
        "coefficient": coefs[idx]
    })
    st.download_button("下載 Coefficients 圖 (PNG)",
                       data=fig_to_png_bytes(fig5), file_name="coefficients.png",
                       mime="image/png")
    st.download_button("下載 Coefficients 數據 (CSV)",
                       data=df_to_csv_bytes(coef_df), file_name="coefficients.csv",
                       mime="text/csv")

# ------------- live inference ----------------
st.markdown("### Live Inference")
use_row = st.checkbox("Use a row from test set", value=True)
if use_row:
    row_idx = st.slider("Row index (in test set)", 0, X_test.shape[0]-1, 0)
    sample = X_test[row_idx].reshape(1, -1)
else:
    cols = st.columns(3)
    inputs = []
    for i, c in enumerate(feature_cols):
        with cols[i % 3]:
            val = st.slider(f"Feature {str(c)}", -1, 1, 0, 1)
            inputs.append(val)
    sample = np.array(inputs, dtype=float).reshape(1, -1)
    if do_standardize and scaler is not None:
        sample = scaler.transform(sample)

proba = clf.predict_proba(sample)[:, 1][0] if hasattr(clf, "predict_proba") else 0.0
pred = int(proba >= threshold)
st.write(f"**Spam/Phish probability**: {proba:.3f}  |  **Predicted label**: {pred}")

st.success("Ready for deployment on Streamlit Cloud: set main file to app/streamlit_app.py")
