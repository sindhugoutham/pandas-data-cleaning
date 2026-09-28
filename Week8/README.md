# Sentiment Analyzer for Product Reviews

## Project
A machine-learning based sentiment analyzer that classifies product reviews as **positive** or **negative**.

## Workflow
1. Load product reviews from CSV.
2. Clean text by lowercasing, removing URLs/special characters, and normalizing spaces.
3. Convert text into TF-IDF features.
4. Train a Logistic Regression classifier.
5. Evaluate using accuracy, classification report, and confusion matrix.
6. Perform error analysis on misclassified reviews.
7. Test the model with a custom review.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression

## Run
```bash
pip install -r requirements.txt
python sentiment_analyzer.py
```

## Example
Input:
`The battery is excellent and the phone is very easy to use.`

Output:
`Predicted sentiment: positive`

## Error Analysis
Misclassifications can happen because reviews may contain mixed opinions, sarcasm, very short text, unfamiliar words, or context that TF-IDF cannot fully understand.

## Limitations
- The included dataset is a small demonstration dataset.
- The model supports only positive/negative sentiment.
- It does not reliably understand sarcasm or complex context.
- Real-world performance depends strongly on the size and quality of the training data.

## Future Scope
Use a larger real-world review dataset, add neutral/mixed sentiment, compare SVM with Logistic Regression, and deploy the model through a web interface.
