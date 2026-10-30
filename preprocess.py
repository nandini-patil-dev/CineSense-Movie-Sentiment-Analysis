

import re
import pandas as pd

def clean_text(text):
    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Keep letters, digits, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data(path="data/IMDB Dataset.csv"):
    # Read CSV file
    df = pd.read_csv(path)

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
        .str.replace("'", '', regex=False)
        .str.replace(";", '', regex=False)
        .str.lower()
    )

    print("Cleaned columns:", df.columns.tolist())

    # Keep required columns
    df = df[["review", "sentiment"]].copy()

    # Remove missing and duplicate records
    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df = df.drop_duplicates(subset=["review"])

    # Clean the review text
    df["review"] = df["review"].apply(clean_text)

    # Remove empty reviews
    df = df[df["review"].str.len() > 0]

    # Encode labels as numbers
    df["sentiment"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print("Clean dataset shape:", df.shape)
    print(df.head())
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts())
