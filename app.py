from flask import Flask, request, render_template,jsonify
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import warnings
warnings.filterwarnings ("ignore")
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from ast import literal_eval
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, make_scorer, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD


multilabel_binarizer1= joblib.load("Movie_genre_prediction/modelfiles/multilabel_binarizer1.pkl")
tfidf_vectorizer = joblib.load("Movie_genre_prediction/modelfiles/tfidf_vectorizer.pkl")
bow_vectorizer = joblib.load("Movie_genre_prediction/modelfiles/bow_vectorizer.pkl")
clf_tfidf = joblib.load("Movie_genre_prediction/modelfiles/clf_tfidf.pkl")
clf_bow = joblib.load("Movie_genre_prediction/modelfiles/clf_bow.pkl")

def filter_synopsis(synopsis):
    synopsis = re.sub("\'","", synopsis)
    synopsis = re.sub("[^a-zA-Z]"," ", synopsis)
    synopsis = " ".join(synopsis.split())
    synopsis = synopsis.lower ()
    return synopsis

eng_stopwords = set(stopwords.words("english"))
def stopwords_filtering(plot):
    filtered_text= [x for x in plot.split() if not x in eng_stopwords]
    return " ".join(filtered_text)

def lemmatization_plot(plot):
    lemmatizer = WordNetLemmatizer()
    filtered_text = [lemmatizer.lemmatize (x) for x in plot.split()]
    return " ".join(filtered_text)

def predicting(synopsis):
    synopsis = filter_synopsis(synopsis)
    synopsis = stopwords_filtering(synopsis)
    synopsis = lemmatization_plot(synopsis)
    tfidf_text = tfidf_vectorizer.transform([synopsis])
    bow_text = bow_vectorizer.transform([synopsis])
    out1=clf_bow.predict(bow_text)
    out2=clf_tfidf.predict(tfidf_text)
    bow_genre = multilabel_binarizer1.inverse_transform(out1)
    tfidf_genre = multilabel_binarizer1.inverse_transform(out2)
    resultmain= bow_genre+tfidf_genre
    resultmain= list(set(sum(resultmain,())))
    prediction = "Genres are {}".format((resultmain))
    return prediction


app = Flask(__name__,template_folder="template")


@app.route("/")
def helloWorld():
    return render_template("index.html")

@app.route('/extract', methods=['POST'])
def extract():
    if request.method == "POST":
        data = request.get_json()
        movie_name = data.get("movie_name")

        # Construct the Google search URL
        search_url = "https://www.google.com/search?q={}+synopsis".format(movie_name)

        # Send an HTTP GET request to Google
        response = requests.get(search_url)

        # Parse the HTML content of the search results
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        match = re.search(r'Film synopsis(.{1,1000})', text)

        # Call the predicting function within the route context
        prediction = predicting(text)

        # Return the prediction as JSON
        return jsonify(result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
