# Spam Classification Project

A machine learning project for spam email/SMS classification with a Streamlit web interface.

## Project Structure
```
├── data/               # Data directory
├── models/            # Saved model files
├── src/               # Source code
│   ├── app.py         # Streamlit web application
│   ├── data_processing.py  # Data preprocessing utilities
│   ├── model.py       # Spam classifier implementation
│   └── train.py       # Model training script
├── tests/             # Test files
├── .github/           # GitHub Actions workflows
├── environment.yml    # Conda environment file
└── requirements.txt   # Python dependencies
```

## Setup Instructions

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate spam-classification-app
```

2. Or install with pip:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python src/train.py
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

## Development

- Follow PEP 8 style guide
- Write tests for new features
- Update requirements.txt when adding dependencies
- Run tests before committing:
```bash
pytest tests/
```