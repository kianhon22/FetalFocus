# @app.route('/', methods = ['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile= request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)

from flask import Flask, render_template, request
import numpy as np
from sklearn.svm import SVC
from SGA import X_test, X_train, y_train, train_value

classifier = SVC(kernel='rbf', C=85)

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('firstfunction.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    data14 = request.form['n']
    data15 = request.form['o']
    data16 = request.form['p']

    arr = np.array([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16]).reshape(1, -1)
    classifier.fit(X_train, y_train)
    train_value(arr)
    pred = classifier.predict(arr)
   

    return render_template('after.html', data = pred)

if __name__ == "__main__":
    app.run(debug=True)