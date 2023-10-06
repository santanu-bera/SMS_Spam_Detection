import pickle
import string
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

app=Flask(__name__)

# import ridge and standard scaler pickle
tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
model = pickle.load(open('models/model.pkl','rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



## Route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Message=(request.form.get('Message'))
        transformed_sms = transform_text(Message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            result="Spam"
        else:
            result="Not Spam"

        return render_template('home.html',result=result)

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")