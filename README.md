# 🤖 AI Adoption in Job Postings — BERT-Based Labor Economics Analysis (Lightcast Laboratory)

A data science project that uses **BERT NLP classification** to measure AI adoption signals in job postings, combining text analysis with economic indicators to track how AI is reshaping labor markets.

🔗 **[Live Dashboard](https://ai-adoption-economic-analysis-dashboard-424009631102.europe-west2.run.app)** — Interactive Streamlit app deployed on Google Cloud Run

---

## 📌 Project Overview

This project quantifies AI adoption by analysing the language of real-world job postings (via the Lightcast dataset). Rather than relying on manual keyword rules, it fine-tunes a **BERT (Bidirectional Encoder Representations from Transformers)** model to classify whether a job posting reflects AI-related demand — enabling a more nuanced and scalable measure of AI diffusion across the economy.

The results feed into an interactive dashboard that explores the relationship between AI adoption rates and labor market outcomes across sectors, regions, and time.
The source for the labor market indicators is https://www.ilo.org/
---

## 🗂️ Repository Structure

```
ai_job_postings/
│
├── data/                        # Raw and processed job posting data (Lightcast)
│
├── PROJECTLIGHTCAST.ipynb       # Main analysis notebook: BERT model training, evaluation & results
├── Lightcastfinal.ipynb         # Supplementary / finalised version of the analysis
├── Untitled7.ipynb              # Exploratory / scratch notebook
│
├── streamlit_app.py             # Streamlit dashboard application
├── requirements.txt             # Python dependencies
│
├── .devcontainer/               # Dev container configuration
├── .github/                     # GitHub Actions / workflows
├── .gitignore
├── LICENSE                      # Apache 2.0
└── README.md
```

---

## 🧠 Methodology

### 1. Data
Job postings data is sourced from **Lightcast** (formerly Burning Glass / EMSI), a leading labor market intelligence provider. The dataset contains structured fields including job title, employer, location, sector, and full posting text.

### 2. AI Adoption Measurement via BERT
The core contribution of this project is using a **fine-tuned BERT model** to classify job postings as AI-generated or not, based on the posting's full text. This goes beyond simple keyword matching by leveraging contextual language understanding.

- **Model:** `bert-base-uncased` (HuggingFace Transformers)
- **Task:** Binary text classification (AI-generated vs. non-AI-generated job posting)
- **Training:** Supervised fine-tuning on labelled job posting examples
- **Evaluation:** Accuracy, precision, recall, F1-score

### 3. Economic Analysis
BERT-derived AI adoption scores are aggregated and cross-referenced with economic variables (sector, salary band, geography) to surface patterns in how AI is penetrating the labor market.

### 4. Dashboard
Results are visualised in an interactive **Streamlit** app, deployed on **Google Cloud Run**, allowing exploration of trends by industry, role type, and region.

---

## 🚀 Running Locally

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Selam-a11y/ai_job_postings.git
cd ai_job_postings
pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

### Run the Analysis Notebooks

Open `PROJECTLIGHTCAST.ipynb` in Jupyter or VS Code. The notebook covers:
- Data loading and preprocessing
- BERT model fine-tuning and training loop
- Evaluation metrics and results
- Adoption score generation

---

## 📦 Dependencies

```
streamlit
pandas
```

> **Note:** The BERT model training pipeline (in the notebooks) additionally requires `transformers`, `torch`, and `scikit-learn`. These are managed within the notebook environment and not listed in `requirements.txt`, which covers only the Streamlit app.

---

## 🌐 Live App

The dashboard is deployed and publicly accessible at:

👉 **[https://ai-adoption-economic-analysis-dashboard-424009631102.europe-west2.run.app](https://ai-adoption-economic-analysis-dashboard-424009631102.europe-west2.run.app)**

Hosted on **Google Cloud Run** (Europe West 2 — London region).

---

## 📊 Key Findings



- AI-related job postings are identifiable via BERT classification with high precision
- AI adoption varies significantly by sector, seniority level, and geography
- The share of AI-related postings has grown over the observed time window

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Selam** — [@Selam-a11y](https://github.com/Selam-a11y)

Contributions, issues, and feedback are welcome.
