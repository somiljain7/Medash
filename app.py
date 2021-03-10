import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from svm_func import train_svm, test_svm, predict_svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time


app = Flask(__name__)
model1 = pickle.load(open('pickle/parkinson/model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('dashboard.htm')

@app.route('/parkinson-home')
def index1():
    return render_template('parkinson/parkinson.html')


@app.route('/parkinson-predict', methods=['POST','GET'])
def predict1():
    int_features = [[float(x) for x in request.form.values()]]
    final = np.array(int_features)

    prediction = model1.predict(final)
    output = prediction[0]
    if output==0:
        s= 'Negative'
    elif output==1:
        s = 'Positive'
    proba = model1.predict_proba(final)

    prob1 = proba[0][1]*100
    if prob1>int(70):
       a="High"
    elif int(30)<prob1<=int(70):
       a="Medium"
    else:
       a="Low"
    return render_template('parkinson/result.html',
                               pred='Test Result : {}'.format(s)
                           ,pred1='Percentage of risk  : {:.2f}%'.format(prob1)
                           ,pred2='Risk Level : {}'.format(a))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    final = np.array([list(data.values())])
    prediction = model1.predict_proba(final)
    output = prediction[0]
    return jsonify(output)
@app.route("/heartdisease-home")
def index2():
    return render_template('heartdisease/home.html')


@app.route('/heartdisease-result', methods=['POST', 'GET'])
def result2():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    cp = int(request.form['cp'])
    fbs = float(request.form['fbs'])
    x = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang]).reshape(1, -1)

    scaler_path = os.path.join(os.path.dirname(__file__), 'pickle/models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    model_path = os.path.join(os.path.dirname(__file__), 'pickle/models/rfc.sav')
    clf = joblib.load(model_path)

    y = clf.predict(x)
    print(y)

    if y == 0:
        return render_template('heartdisease/nodisease.html')

    else:
        return render_template('heartdisease/heartdisease.htm', stage=int(y))
@app.route('/breast-cancer-home')
def hello_method():
  return render_template('breast-cancer/home.html')

@app.route('/breast-cancer-predict', methods=['POST']) 
def login_user():

  if(request.form['space']=='None'):
    data = []
    string = 'value'
    for i in range(1,31):
      data.append(float(request.form['value'+str(i)]))

    for i in range(30):
      print(data[i])

  else:
    string = request.form['space']
    data = string.split()
    print(data)
    print("Type:", type(data))
    print("Length:", len(data))
    for i in range(30):
      print(data[i])
    data = [float(x.strip()) for x in data]

    for i in range(30):
      print(data[i])

  data_np = np.asarray(data, dtype = float)
  data_np = data_np.reshape(1,-1)
  out, acc, t = predict_svm(clf, data_np)

  if(out==1):
    output = 'Malignant'
  else:
    output = 'Benign'

  acc_x = acc[0][0]
  acc_y = acc[0][1]
  if(acc_x>acc_y):
    acc = acc_x
  else:
    acc=acc_y
  return render_template('breast-cancer/result.html', output=output, accuracy=round(acc*100,3), time=t)

@app.route('/profile')
def display():
  return render_template('breast-cancer/profile.html')

@app.route('/fad/')
def gain():
  return render_template('breast-cancer/connect.html')


@app.route('/my_form_post', methods=["GET",'POST'])
def my_form_post():
  print(request.form)
  if request.method=="POST":
    resultss=request.form
    file_name='yes.csv'
    from csv import writer
    def append_list_as_row(file_name, list_of_elem):
      # Open file in append mode
      with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
      
    
    print(resultss)
    lsv=[]
    for key,value in enumerate(resultss.items()):
      lsv.append(value[1])
    print(lsv,'red')



if __name__ == "__main__":
    app.run(debug=True)