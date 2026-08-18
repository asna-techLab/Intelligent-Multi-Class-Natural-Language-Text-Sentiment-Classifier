import joblib
from preprocess import clean_text

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        print(f"Predicted sentiment: {predict_sentiment(text)}")