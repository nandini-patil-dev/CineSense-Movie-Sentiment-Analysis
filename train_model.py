
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

from preprocess import load_data

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load cleaned dataset
df = load_data()

X = df["review"]
y = df["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC Score:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot(cmap="Blues")
plt.title("Sentiment Analysis Confusion Matrix")
plt.savefig("reports/confusion_matrix.png")
plt.close()

# Save metrics
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "ROC-AUC"],
    "Score": [accuracy, roc_auc]
})

metrics.to_csv("reports/metrics.csv", index=False)

# Save trained model
joblib.dump(model, "models/sentiment_pipeline.joblib")

print("\nModel saved successfully!")
print("Location: models/sentiment_pipeline.joblib")
