from transformers import pipeline
from flask import Flask, render_template, request, redirect
app = Flask(__name__)

sentiment_pipeline = pipeline("sentiment-analysis")
#data = ["I love you", "I hate you"]

@app.route('/mostrar', methods= ["post"])
def mostrar():
    text = sentiment_pipeline(request.form['text'])
    print(text)
    return render_template('mostrar.html', text=text)


@app.route('/')
def inicio():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")