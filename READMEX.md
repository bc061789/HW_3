# Phishing/Spam Numeric Classifier (Streamlit)

This repo contains a Streamlit app that trains and evaluates a binary classifier on a numeric phishing/spam dataset.

## Project Structure
```
app/streamlit_app.py     # Main Streamlit app
datasets/phishing_dataset.csv  # Default dataset (headerless; last column = label)
reports/visualizations/  # (Optional) Saved figures directory
requirements.txt         # Python dependencies
```

## Local Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate
# macOS / Linux
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

streamlit run app/streamlit_app.py
# if the default port is busy:
# streamlit run app/streamlit_app.py --server.port 8502
```

## Dataset Notes
- Treated as **headerless** CSV by default.
- The **last column** is used as the label.
- If labels are {-1, 1}, they are mapped to {0, 1}.

## Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app with:
   - **Repository**: your-user/your-repo
   - **Branch**: main
   - **Main file**: `app/streamlit_app.py`
3. Ensure the dataset exists in `datasets/phishing_dataset.csv`, or upload a CSV in the app sidebar.
