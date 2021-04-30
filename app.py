from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
from dogPredictor import make_prediction
import os


app = Flask(__name__)
CORS(app, support_credentials = True)

@app.route('/',methods=['GET','POST'])
def index():
  if request.method == 'POST':
    f = request.files['dogImg']
    filename = f.filename
    f.save(f.filename)
    pred = make_prediction(filename)
    os.remove(filename)
    return render_template('index.html',prediction = pred[0])
  else:
    return render_template('index.html',prediction = '')


if __name__ == '__main__':
  app.run(debug=True)