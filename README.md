# AnswerEvaluationEngine-Uptoskills

# Objective
##### A semantic assessment engine that evaluates student answers using NLP-based concept extraction and decomposed scoring. Provides REST API and interactive web dashboard for answer evaluation with detailed feedback.

## 🎯 Features

- **Semantic Answer Evaluation** — AI-powered assessment using NLP concept extraction
- **Multi-Dimensional Scoring** — Rates answers on accuracy, completeness, clarity, and relevance
- **Concept-Level Granularity** — Breaks down scoring by individual concepts
- **Coverage Analysis** — Identifies covered, missing, and extra concepts
- **Explainable Feedback** — Human-readable explanations with strengths and recommendations
- **Batch Processing** — Evaluate multiple answers efficiently
- **Interactive Dashboard** — Web-based UI for easy answer evaluation
- **REST API** — 6 endpoints for seamless integration
- **Configurable Weights** — Customize scoring dimensions per use case

## 🛠️ Technologies

**Backend:**
- Flask — Web framework and REST API
- Sentence-Transformers — Semantic similarity & embeddings
- NLTK — Natural language processing
- Scikit-learn — Machine learning utilities
- NumPy — Numerical computations

**Frontend:**
- HTML/CSS/JavaScript — Interactive web dashboard

## Run Locally
- pip install -r requirements.txt
- app.py
