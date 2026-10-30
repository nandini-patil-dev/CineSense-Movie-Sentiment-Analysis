
# 🎬 CineSense – AI Movie Review Sentiment Analysis

## 📌 Project Overview
CineSense is a machine learning-based application that analyzes movie reviews and predicts whether the review expresses positive or negative sentiment.

It uses Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression.

## 🚀 Features
- Movie review sentiment prediction
- Positive and negative classification
- Prediction confidence score
- Sentiment probability visualization
- Interactive Streamlit web interface
- Data preprocessing and model evaluation

## 🛠️ Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- NLP
- TF-IDF
- Logistic Regression
- Streamlit
- Matplotlib and Seaborn
- Joblib

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 90.69% |
| ROC-AUC | 0.9682 |
| Negative F1-score | 0.91 |
| Positive F1-score | 0.91 |

## 📂 Project Structure
```text
movie-sentiment-analysis/
├── data/
├── models/
├── reports/
├── explore_data.py
├── preprocess.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd movie-sentiment-analysis
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate environment (Windows)
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the dataset
Download the IMDb 50K Movie Reviews dataset from:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place `IMDB Dataset.csv` inside the `data/` folder.

### 6. Train the model (optional)
```bash
python train_model.py
```

### 7. Run the application
```bash
streamlit run app.py
```

## 📈 Model
Algorithm: Logistic Regression

Feature Extraction: TF-IDF

Dataset: IMDb 50K Movie Reviews

Test Accuracy: 90.69%

## 👩‍💻 Author
Nandini Patil

B.Tech – Computer Science and Business Systems
