from flask import Flask, render_template, request, redirect
from dogPredictor import make_prediction


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
  print(request.method)
  if request.method == 'POST':
    # print(request.form['dogImg'])
    f = request.files['dogImg']
    filename = f.filename
    f.save(f.filename)
    # print(img)
    print("****************************")
    pred = make_prediction(filename)
    # return render_template('index.html',prediction = pred)
    # pred = 'hahahahah'
    return render_template('index.html', prediction = pred[0])
  else:
    return render_template('index.html',prediction = 'Hello from GET')
  

if __name__ == '__main__':
  app.run(debug=True)