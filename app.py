import re
import nltk
import keras
import pandas as pd
from flask import Flask, render_template

import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
from flask import Flask, request, jsonify

# Preprocessing functions
def remove_mentions(input_text):
    return re.sub(r'@\w+', '', input_text)

def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)

# Load and preprocess data
df = pd.read_csv('Tweets.csv')
df = df.reindex(np.random.permutation(df.index))
df = df[['text', 'airline_sentiment']]
df.text = df.text.apply(remove_stopwords).apply(remove_mentions)

# Tokenization and one-hot encoding
NB_WORDS = 10000
tk = Tokenizer(num_words=NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n', lower=True, char_level=False, split=' ')
tk.fit_on_texts(df.text)
X_oh = tk.texts_to_matrix(df.text, mode='count')

# Label encoding and one-hot encoding of labels
le = LabelEncoder()
le.fit(df.airline_sentiment)
classes = list(le.classes_)

# Model definition
reduce_dropout_model = Sequential()
reduce_dropout_model.add(Dense(16, activation='relu', input_shape=(NB_WORDS,)))
reduce_dropout_model.add(Dropout(0.5))
reduce_dropout_model.add(Dense(len(classes), activation='softmax'))

# Model training
reduce_dropout_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
reduce_dropout_model.fit(X_oh, to_categorical(le.transform(df.airline_sentiment), num_classes=len(classes)), epochs=20, batch_size=512, verbose=0)

# Save the trained model
joblib.dump(tk, 'tokenizer.pkl')
joblib.dump(le, 'label_encoder.pkl')
reduce_dropout_model.save('sentiment_analysis_model.h5')

# Flask application
app = Flask(__name__)

# Load the saved model
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model = keras.models.load_model('sentiment_analysis_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

# Define the route for sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()  # Get the JSON data from the request
    input_text = data['text']  # Extract the text to be analyzed from the JSON data

    # Preprocess input text
    input_text = remove_stopwords(remove_mentions(input_text))
    input_oh = tokenizer.texts_to_matrix([input_text], mode='count')

    # Predict sentiment
    sentiment_label = label_encoder.inverse_transform([np.argmax(model.predict(input_oh))])[0]
    return jsonify({'sentiment': sentiment_label})

# Run the Flask application
if __name__ == '__main__':
    app.run()