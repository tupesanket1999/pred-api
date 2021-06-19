import flask
from flask import Flask, request, Response
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin


import cv2
import numpy as np

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True

@cross_origin()
@app.route('/', methods=['POST'])
def home():
        model=load_model('./model.h5')
        imageFile = request.files['imagefile']
        image_path = './images/' + imageFile.filename
        imageFile.save(image_path);
        img_arr = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (150, 150))
        resized_arr=np.array(resized_arr)
        x2=[]
        x2.append(resized_arr)
        x2=np.array(x2) / 255
        x2 = x2.reshape(-1, 150, 150, 1)
        result=model.predict_classes(x2)
        if result == 0:
                return "yes"
        return "no"
app.run()
