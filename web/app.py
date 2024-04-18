from flask import Flask, render_template, request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)


# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
    
# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)



port_stem = PorterStemmer()

def preprocess(text):
    # Preprocess the input text
    processed_text = re.sub('[^a-zA-Z]',' ',text)
    processed_text = processed_text.lower()
    processed_text = processed_text.split()
    processed_text = [port_stem.stem(word) for word in processed_text if not word in stopwords.words('english')]
    processed_text = ' '.join(processed_text)
    return processed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        # Preprocess the input text
        processed_text = preprocess(news_text)
        # Vectorize the processed text
        vectorized_text = vectorizer.transform([processed_text])
        # Make prediction
        prediction = model.predict(vectorized_text)
        result = 'Real' if prediction == 0 else 'Fake'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
