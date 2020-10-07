from flask import Flask, request, jsonify
import json
import codecs
from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import gensim
from string import punctuation
app = Flask(__name__)

@app.route('/')
def main():
    return "<h1>Welcome to my check similarity server</h1>"

if __name__ == "__main__":
    app.run()