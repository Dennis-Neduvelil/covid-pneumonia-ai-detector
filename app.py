from flask import Flask, render_template, jsonify, request, make_response, redirect

# import tensorflow as tf
import numpy as np
import cv2
import os
import time
import keras
# model

model = keras.models.load_model(os.path.join(
    os.getcwd(), 'DenseNet121.h5'))  # load .h5 Model

# variables
normal = 0
covid = 0
peumonia = 0
formData = dict()
UPLOAD_FOLDER = './upload'

# flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# routes


@app.route("/")
def home_view():
    return render_template('index.html')


@app.route("/userDetail", methods=['GET', 'POST'])
def userDetails_view():
    if request.method == 'GET':
        return render_template('userDetails.html')
    else:
        global formData
        formData = request.form
        print(formData)
        return redirect('/prediction')


@app.route("/prediction")
def pred_view():
    params = {'normal': 0,	'covid': 0, 'pneumonia': 0}
    return render_template('prediction.html', params=params)


@app.route('/upload', methods=['POST'])
def upload():
    params = {'normal': 70, 'covid': 20, 'pneumonia': 10}
    if (request.method == 'POST'):
        try:
            img = request.files['imageIN']  # Get Images from user

            # preprocessing on image
            img = cv2.imdecode(np.fromstring(
                img.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img = np.array(img) / 255.0
            img = img.reshape(-1, 224, 224, 3)

            prediction = model.predict(img)  # get predictions on image
            print('\n')
            print('*'*50)
            print(f'prediction->{prediction}')
            print('\n')
            global normal, covid, peumonia
            normal = prediction[0, 1] * 100
            covid = prediction[0, 0] * 100
            peumonia = prediction[0, 2] * 100
            print(f'normal->{normal}')
            print(f'covid->{covid}')
            print(f'peumonia->{peumonia}')
            print('*'*50)

            params['normal'] = prediction[0, 1] * \
                100  # add predictions on params dict
            params['covid'] = prediction[0, 0] * 100
            params['pneumonia'] = prediction[0, 2] * 100

            params['status'] = True
        except:
            params['status'] = False
    return jsonify(params)


@app.route("/report")
def report_view():
    try:
        global formData, peumonia, normal, covid
        fname = formData['fname']
        lname = formData['lname']
        age = formData['age']
        gender = formData['gender']
        blood = formData['blood']
        address = formData['address']
        zipcode = formData['zip']
        mobile = formData['mobile']
        if covid < 80 and peumonia < 80:
            cov = 'Negative'
            pnu = 'Negative'
        elif covid > 80 and peumonia < 80:
            cov = 'Positive'
            pnu = 'Negative'
        elif covid > 80 and peumonia > 80:
            cov = 'Positive'
            pnu = 'Positive'
        elif covid < 80 and peumonia > 80:
            cov = 'Negative'
            pnu = 'Positive'
        params = {'fname': fname, 'lname': lname, 'age': age,
              'gender': gender, 'blood': blood, 
              'address': address, 'zip': zipcode, 
              'mobile': mobile, 'cov': cov,'pnu':pnu,'rep_view':'inline','err_view':'none'}
    except:
        params={'rep_view':'none','err_view':'block'}
    return render_template('report.html',params=params)

if __name__ == "__main__": 
	app.run()

