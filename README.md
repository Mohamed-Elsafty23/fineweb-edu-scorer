# FineWeb-Edu: Educational Content Scorer

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive demo of educational content scoring using Teacher-Student knowledge distillation. Based on ["The FineWeb Datasets"](https://arxiv.org/abs/2406.17557) paper.

## What is This?

A web app that scores educational content using two models:

1. **Teacher Model** (Llama-3.3-70B): Slow but accurate LLM that scores content 0-5
2. **Student Model** (Ridge Regressor): Fast classifier that learns from the Teacher

The Student model learns to predict scores quickly, making it useful for filtering large amounts of web content.

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Mohamed-Elsafty23/fineweb-edu-scorer
cd fineweb-edu-scorer
pip install -r requirements.txt
```

### 2. Configure API Key

Create `.streamlit/secrets.toml`:

```toml
[api]
OPENAI_API_KEY = "your-api-key-here"
OPENAI_BASE_URL = "https://api.aimlapi.com/v1"
```

### 3. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## How It Works

The pipeline has three phases:

1. **Extraction**: Clean text from web pages (removes ads, navigation, etc.)
2. **Teacher Scoring**: LLM evaluates content and gives a score 0-5
3. **Student Prediction**: Fast classifier predicts the score

```
URL/Text → Extract → Teacher Score → Student Prediction → Decision
```

Scores ≥ 3 are considered educational and kept.

## Training the Student Model

If you want to retrain the model:

```bash
# Generate training data
python scripts/create_training_data.py

# Train the model
python scripts/train_student.py

# Test everything
python scripts/test_pipeline.py
```

## Results

- **Training**: 150 samples from FineWeb-Edu dataset
- **Test Set**: 30 samples (20% split)
- **F1 Score**: 96.6%
- **Accuracy**: 96.7%

The Student model matches the Teacher's decisions 96.6% of the time but runs much faster (< 1s vs 5-10s).

## Project Structure

```
fineweb-edu-scorer/
├── app.py                      # Main Streamlit app
├── config.py                   # Configuration
├── src/                        # Source code
│   ├── extractors/            # Web text extraction
│   ├── annotators/            # Teacher LLM scoring
│   ├── classifiers/           # Student model
│   └── utils/                 # API client
├── scripts/                    # Training scripts
├── models/                     # Trained models
└── data/                       # Training data
```

## Deployment

To deploy on Streamlit Cloud:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create a new app and select your repository
4. Add your API credentials in the secrets section
5. Deploy!

## Configuration

Edit `config.py` or `.streamlit/secrets.toml` to change:
- Models (Teacher LLM, Embedding model)
- Educational threshold (default: 3)
- API endpoints

## License

MIT License

## Acknowledgments

- FineWeb paper: [Penedo et al., 2024](https://arxiv.org/abs/2406.17557)
- HuggingFace FineWeb-Edu dataset
- Trafilatura for web extraction
- Streamlit for the UI

---

**Course**: Machine Learning Seminar - LLM | Leuphana University | Winter 2025/2026
