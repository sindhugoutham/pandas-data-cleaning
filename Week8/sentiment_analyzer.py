
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/reviews.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review", "sentiment"])
    df["clean_review"] = df["review"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["sentiment"],
        test_size=0.25, random_state=42, stratify=df["sentiment"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n=== SENTIMENT ANALYZER ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.2f}\n")
    print(classification_report(y_test, pred, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=["negative", "positive"])
    print("Confusion Matrix (rows=actual, columns=predicted):")
    print(cm)

    errors = pd.DataFrame({"review": X_test, "actual": y_test, "predicted": pred})
    errors = errors[errors["actual"] != errors["predicted"]]
    print("\n=== ERROR ANALYSIS ===")
    if len(errors) == 0:
        print("No errors on this small demo test split.")
    else:
        print(errors.to_string(index=False))

    print("\n=== TRY YOUR OWN REVIEW ===")
    while True:
        text = input("Enter a review (or type 'quit'): ")
        if text.lower() == "quit":
            break
        print("Predicted sentiment:", model.predict([clean_text(text)])[0])

if __name__ == "__main__":
    main()
