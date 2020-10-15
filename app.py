from flask import Flask, request, jsonify
import json
import codecs
from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import gensim
from string import punctuation
app = Flask(__name__)
app.config["DEBUG"] = True
STOP_WORD= []
with codecs.open("vietnamese-stopwords-dash.txt", encoding='utf-8') as f:
    for line in f:
        STOP_WORD += [line.strip()]
    STOP_WORD += list(punctuation)
def filter_doc(docs):
    filtered_sentence= ''
    tokenize = ViTokenizer.tokenize(u""+docs)
    tokenize_vi = tokenize.split(' ')
    filtered_sentence= ''
    for w in tokenize_vi: 
        if w not in STOP_WORD and w not in STOP_WORD: 
            filtered_sentence += w + ' '
    return filtered_sentence

@app.route('/check_similarity', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return 'Hello world'
    elif request.method == 'POST':
        if(any(i == None for i in request.json['from_db'])): return "0"
        db = request.json['from_db']
        data_check = request.json['data_check']
        arr_filter = []
        data_filter = filter_doc(data_check)
        for item in db:
            arr_filter += [filter_doc(item)]
        doc_vectors = TfidfVectorizer().fit_transform([data_filter] + arr_filter)
        cosine_similarities = cosine_similarity(doc_vectors[0:1], doc_vectors).flatten()
        document_scores = [item.item() for item in cosine_similarities[1:]]
        print(str(document_scores) +"\t")
        if(any(i >= 0.85 for i in document_scores)):
            return "1"
        else: 
            return "0"
@app.route('/')
def main():
    return "<h1>Welcome to my check similarity server</h1>"

if __name__ == "__main__":
    app.run()