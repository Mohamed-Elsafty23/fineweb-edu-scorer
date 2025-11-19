# 📚 FineWeb-Edu: Educational Content Scorer

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Interactive implementation of the FineWeb-Edu educational content scoring methodology using Teacher-Student knowledge distillation.**

Replicating the methodology from ["The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"](https://arxiv.org/abs/2406.17557)

---

## 🎯 What is This?

This project demonstrates **Knowledge Distillation** for filtering educational web content:

1. **Teacher Model** (Llama-3.3-70B): Evaluates content quality and assigns 0-5 scores
2. **Student Model** (Ridge Regressor): Learns from the Teacher to make fast predictions
3. **Interactive UI** (Streamlit): Real-time demo with URL and text input

### Key Features

✅ **Three-Phase Pipeline**:
- Phase 1: Text Extraction (using Trafilatura)
- Phase 2: Teacher Evaluation (LLM scoring)
- Phase 3: Student Prediction (fast classifier)

✅ **Quick Examples**: Pre-loaded URLs and texts for instant testing

✅ **Educational Demo**: Perfect for class presentations

✅ **Production-Ready**: Deployable to Streamlit Cloud

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fineweb-edu-scorer.git
cd fineweb-edu-scorer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials

Copy the example secrets file and add your API key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your actual API key:

```toml
[api]
OPENAI_API_KEY = "your-api-key-here"
OPENAI_BASE_URL = "https://api.aimlapi.com/v1"
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
fineweb-edu-scorer/
├── app.py                      # Streamlit web interface
├── config.py                   # Configuration with secrets handling
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── extractors/
│   │   └── web_extractor.py    # Web scraping with Trafilatura
│   ├── annotators/
│   │   └── teacher_annotator.py # Teacher LLM scoring
│   ├── classifiers/
│   │   └── student_classifier.py # Student model training/prediction
│   └── utils/
│       └── api_client.py       # API client for LLM and embeddings
│
├── scripts/
│   ├── create_training_data.py # Generate training dataset
│   ├── train_student.py        # Train the Student model
│   └── test_pipeline.py        # End-to-end testing
│
├── models/
│   └── student_classifier.pkl  # Trained Student model
│
├── data/
│   └── training_data.csv       # 150 annotated samples
│
├── .streamlit/
│   ├── secrets.toml.example    # Template for API credentials
│   └── secrets.toml            # Your actual secrets (gitignored)
│
└── docs/
    ├── QUICKSTART.md           # Quick setup guide
    ├── PRESENTATION.md         # Presentation script
    ├── INSTRUCTOR_GUIDE.md     # Comprehensive project explanation
    ├── PROJECT_SUMMARY.md      # Overview and results
    └── DEPLOY.md               # Streamlit Cloud deployment guide
```

---

## 🎓 Methodology

### Knowledge Distillation Process

1. **Data Collection**: Load 150 samples from HuggingFace `fineweb-edu` dataset

2. **Teacher Annotation**: 
   - Use Llama-3.3-70B with exact prompt from Appendix F.1 of the FineWeb paper
   - Get educational scores (0-5) and reasoning

3. **Student Training**:
   - Generate embeddings using `multilingual-e5-large-instruct`
   - Train Ridge Regressor on (embeddings, scores) pairs
   - Achieve 96.6% F1 score on test set

4. **Deployment**:
   - Fast Student model for real-time predictions
   - Compare Student vs Teacher for transparency

### Architecture

```
User Input (URL/Text)
      ↓
[Web Extractor] → Clean text (Trafilatura)
      ↓
[Teacher LLM] → Educational score + reasoning (Llama-3.3-70B)
      ↓
[Student Model] → Fast prediction (Ridge + Embeddings)
      ↓
Decision: Keep (≥3) or Discard (<3)
```

---

## 🛠️ Development

### Retrain the Student Model

If you want to retrain with different data:

```bash
# 1. Generate new training data (modify NUM_SAMPLES in script)
python scripts/create_training_data.py

# 2. Train the Student model
python scripts/train_student.py

# 3. Test the pipeline
python scripts/test_pipeline.py
```

### Run Tests

```bash
python scripts/test_pipeline.py
```

Expected output:
- Processing 5 diverse URLs
- Teacher and Student predictions
- Performance metrics

---

## 🌐 Deployment to Streamlit Cloud

### Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
3. Your API credentials

### Steps

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/fineweb-edu-scorer.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file to `app.py`

3. **Configure Secrets**:
   - In Streamlit Cloud settings, add your secrets:
   ```toml
   [api]
   OPENAI_API_KEY = "your-actual-key"
   OPENAI_BASE_URL = "https://chat-ai.academiccloud.de/v1"
   
   [models]
   LLM_MODEL = "llama-3.3-70b-instruct"
   EMBEDDING_MODEL = "multilingual-e5-large-instruct"
   ```

4. **Deploy!** Your app will be live at `https://your-app-name.streamlit.app`

📖 **Detailed Instructions**: See [DEPLOY.md](DEPLOY.md)

---

## 📊 Results

### Student Model Performance

- **Training Samples**: 150 from FineWeb-Edu dataset
- **Test Set**: 30 samples (20% split)
- **F1 Score**: 96.6%
- **Accuracy**: 96.7%

### Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
| 2     | 2     | 1.3%       |
| 3     | 44    | 29.3%      |
| 4     | 79    | 52.7%      |
| 5     | 25    | 16.7%      |

### Key Insights

- Student model successfully learns from Teacher
- Fast inference (< 1s vs 5-10s for LLM)
- High agreement with Teacher (96.6% F1)
- Perfect for real-time filtering at scale

---

## 🎥 Demo Features

### Input Methods

1. **URL Input**: Paste any webpage URL
   - Automatic text extraction with Trafilatura
   - Removes ads, navigation, boilerplate

2. **Direct Text**: Paste text content directly
   - Skip extraction, go straight to scoring

### Quick Examples

**For URLs**:
- 📚 Wikipedia Article (High educational value)
- 📖 Python Tutorial (Medium educational value)
- 📰 News Website (Low educational value)

**For Text**:
- 📚 Educational Text (Pure learning content)
- 📊 Mixed Content (Educational + promotional)
- 🛍️ Promotional Text (Spam/non-educational)

### Three-Phase Visualization

1. **Phase 1**: See raw HTML vs. clean extracted text
2. **Phase 2**: View Teacher's score, reasoning, and highlighted keywords
3. **Phase 3**: Compare Student's prediction with Teacher's score

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Quick setup guide (5 minutes) |
| [PRESENTATION.md](PRESENTATION.md) | Presentation script for class demo |
| [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md) | Comprehensive project explanation |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Overview and technical details |
| [DEPLOY.md](DEPLOY.md) | Streamlit Cloud deployment guide |

---

## 🔧 Configuration

### API Models

- **Teacher LLM**: `llama-3.3-70b-instruct` (via OpenAI-compatible API)
- **Embedding Model**: `multilingual-e5-large-instruct`

### Parameters

```python
EDUCATIONAL_THRESHOLD = 3  # Score ≥ 3 = Educational
MAX_TOKENS = 2000          # Max tokens per text
TEMPERATURE = 0.3          # Lower for consistent scoring
```

### Customization

Edit `config.py` or `.streamlit/secrets.toml` to:
- Change models
- Adjust threshold
- Modify API endpoints

---

## 🤝 Contributing

This is an academic project for the Machine Learning Seminar at Leuphana University.

### Potential Improvements

- [ ] Add more embedding models for comparison
- [ ] Support multiple languages
- [ ] Add visualization for model attention
- [ ] Implement batch processing for URLs
- [ ] Add caching for faster repeated queries

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **FineWeb Paper**: [Penedo et al., 2024](https://arxiv.org/abs/2406.17557)
- **HuggingFace**: FineWeb-Edu dataset
- **Trafilatura**: Web text extraction
- **Streamlit**: Interactive UI framework
- **OpenAI-compatible API**: LLM and embedding models

---

## 📧 Contact

For questions about this project, please:
1. Open an issue on GitHub
2. Check the documentation in the `docs/` folder
3. Review the [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md) for detailed explanations

---

## 🎓 Academic Context

**Course**: Machine Learning Seminar - LLM
**Institution**: Leuphana University
**Semester**: Winter 2025/2026
**Paper**: "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"

---

## ⭐ Star This Repo!

If you find this project useful for learning about knowledge distillation or educational content filtering, please give it a star! ⭐

---

**Built with ❤️ for education and open knowledge**
