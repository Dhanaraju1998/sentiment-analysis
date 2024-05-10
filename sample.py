from flask import Flask, render_template
import pandas as pd
from textblob import TextBlob
import spacy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

app = Flask(__name__)

def analyze_sentiment(text):
    if pd.notnull(text):
        sentiment = TextBlob(text)
        sentiment_category = "positive" if sentiment.polarity > 0.2 else (
            "negative" if sentiment.polarity < -0.2 else "neutral"
        )
        return sentiment_category, sentiment.polarity
    else:
        return "empty", 0

file_path = "input-review-data1.csv"
reviews_df = pd.read_csv(file_path)

sentiments = [analyze_sentiment(review)[0] for review in reviews_df['reviews_text']]

X_train, X_test, y_train, y_test = train_test_split(reviews_df['reviews_text'], sentiments, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear'))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
classification_report_data = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

nlp = spacy.load("en_core_web_sm")
def process_review(review):
    doc = nlp(review)
    relevant_info = {
        "Quality": [],
        "Effect on skin": [],
        "Texture": [],
        "Performance": [],
        "Anti aging": [],
        "Appearance": [],
        "Brightness": [],
        "Value": []
    }
    for token in doc:
        if token.pos_ == "ADJ":
            for aspect in relevant_info:
                if token.head.text.lower() == aspect.lower():
                    relevant_info[aspect].append(token.text.lower())
    return relevant_info

quality_keywords = ["good quality", "well made", "effective", "long lasting"]
performance_keywords = ["hydrated", "wrinkles", "fine lines", "soft", "smooth"]
cost_keywords = ["expensive", "cheap", "price", "affordable", "value"]
positive_sentiment_keywords = ["love", "great", "recommend"]
neutral_sentiment_keywords = ["average", "okay", "fine"]

aspect_adjectives = {
    "Quality": [],
    "Effect on skin": [],
    "Texture": [],
    "Performance": [],
    "Anti aging": [],
    "Appearance": [],
    "Brightness": [],
    "Value": []
}
for review_text in reviews_df['reviews_text']:
    review_info = process_review(review_text)
    for aspect, adjectives in review_info.items():
        aspect_adjectives[aspect].extend(adjectives)

summary = "Customers generally appreciate the "
for aspect, adjectives in aspect_adjectives.items():
    if adjectives:
        summary += f"{aspect.lower()} ({', '.join(adjectives)}), "
summary = summary.rstrip(", ") + " of the product. Opinions vary on its value and overall performance."

sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
text_counts = {"quality": 0, "performance": 0}
cost_counts = {"expensive": 0, "cheap": 0, "price": 0}

for review in reviews_df['reviews_text']:
    sentiment_category, polarity = analyze_sentiment(review)
    sentiment_counts[sentiment_category] += 1

    if sentiment_category in ["positive", "neutral"]:
        for word in review.lower().split():
            if word in quality_keywords:
                text_counts["quality"] += 1
            if word in performance_keywords:
                text_counts["performance"] += 1
            if word in positive_sentiment_keywords:
                pass
            if word in cost_keywords:
                cost_counts[word] += 1

overall_sentiment = {}
for sentiment, count in sentiment_counts.items():
    sentiment_percentage = (count / len(reviews_df['reviews_text'])) * 100
    overall_sentiment[sentiment] = "{:.2f}".format(sentiment_percentage)

@app.route('/')
def index():
    template_data = {
        'summary': summary,
        'classification_report': classification_report_data,
        'overall_sentiment': overall_sentiment
    }
    return render_template('index.html', **template_data)

if __name__ == '__main__':
    app.run(debug=True, host=""0.0.0.0")
