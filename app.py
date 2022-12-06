from flask import Flask, jsonify, request
import os
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from platinum_challenge_dataset_cleaner import clean_texts

import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

with open ('total_data', 'rb') as fp:
    total_data = pickle.load(fp)

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__)) 
UPLOAD_FOLDER = join(basedir, 'static')
print(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda: 'API Documentation for Deep Learning'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Deep Learning (teks)')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": '/flasgger_static',
    "swagger_ui": True,
    "specs_route": '/docs/'
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

def cleansing(sent):
    strings = sent.lower()
    strings = re.sub(r'[^a-zA-Z0-9]', ' ', strings)
    return strings

file = open("lstm_model/resources/x_pad_sequences.pickle", "rb")
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model("lstm_model/model/model.h5")

file = open("rnn_model/resources/x_pad_sequences.pickle", "rb")
feature_file_from_rnn = pickle.load(file)
file.close()

model_file_from_rnn = load_model("rnn_model/model/model.h5")

@swag_from("docs/lstm.yml", methods=['POST'])
@app.route("/lstm", methods=['POST'])
def lstm():
    tokenizer.fit_on_texts(total_data)
    original_text = request.form.get('text')
    # text = clean_texts(original_text)
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    print(np.argmax(prediction[0]))
    print(prediction)
    json_response = {
        'status_code': 200,
        'description': 'Results of LSTM model',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/lstmCSV.yml", methods=['POST'])
@app.route("/lstmCSV", methods=['POST'])
def lstmCSV():
    tokenizer.fit_on_texts(total_data)
    original_file = request.files['file']
    filename = secure_filename(original_file.filename)
    filepath = 'static/' + filename
    original_file.save(filepath)
    
    df = pd.read_csv(filepath, header=0)
    
    sentiment_results = []
    for text in df.iloc[:, 0]:
        original_text = text
        text = [cleansing(original_text)]
    
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        sentiment_results.append(get_sentiment)
    
    json_response = {
        'status_code': 200,
        'description': 'Results of LSTM model',
        'data': {
            'text': original_text,
            'sentiment': sentiment_results
        }
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/rnn.yml", methods=['POST'])
@app.route("/rnn", methods=['POST'])
def rnn():
    tokenizer.fit_on_texts(total_data)
    original_text = request.form.get('text')
    # text = clean_texts(original_text)
    text = [cleansing(original_text)]
    print(text)
    feature = tokenizer.texts_to_sequences(text)
    print("tokenizer", feature)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
    print("pad seq", feature)
    
    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    print(np.argmax(prediction[0]))
    print(prediction)
    json_response = {
        'status_code': 200,
        'description': 'Results of RNN model',
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        }
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/rnnCSV.yml", methods=['POST'])
@app.route("/rnnCSV", methods=['POST'])
def rnnCSV():
    tokenizer.fit_on_texts(total_data)
    original_file = request.files['file']
    filename = secure_filename(original_file.filename)
    filepath = 'static/' + filename
    original_file.save(filepath)
    
    df = pd.read_csv(filepath, header=0)
    
    sentiment_results = []
    for text in df.iloc[:, 0]:
        original_text = text
        text = [cleansing(original_text)]
    
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])
        
        prediction = model_file_from_rnn.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        sentiment_results.append(get_sentiment)
    
    json_response = {
        'status_code': 200,
        'description': 'Results of RNN model',
        'data': {
            'text': original_text,
            'sentiment': sentiment_results
        }
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()
    