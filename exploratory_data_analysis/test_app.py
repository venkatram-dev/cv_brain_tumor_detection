# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request
from keras.models import load_model
import cv2

model = None
app = Flask(__name__)

#python test_app.py 
# check in browser http://0.0.0.0:80/
#displays hello world
# in another terminal run this 
# curl -X POST    0.0.0.0:80/predict    -H 'Content-Type: application/json'
#returns output classification 1 confidence 99.98050332069397

def load_saved_model():
    global model
    # model variable refers to the global variable
    #with open('iris_trained_model.pkl', 'rb') as f:
        #model = pickle.load(f)
    model = load_model("cnn_model.h5")


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        #data = request.get_json()  # Get data posted as a json
        #data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        #prediction = model.predict(data)  # runs globally loaded model on the data
        
        import numpy as np
        desired_image_shape = (250,250)
        pred_folder_location = "../data/pred"
        #pred_folder_location = "/content/drive/MyDrive/Colab_Notebooks/Brain_Tumor_Detection/pred"
        from random import randint
        rand_int = randint(0,50)
        pred_file_name = f"{pred_folder_location}/pred{rand_int}.jpg"
        #pred_file_name = f"{pred_folder_location}/pred49.jpg"

        print ('pred_file_name',pred_file_name)
        ##pred_folder_location = "/content/drive/MyDrive/Colab_Notebooks/Brain_Tumor_Detection/yes"
        #pred_file_name = f"{pred_folder_location}/y3.jpg"
        pred_img_sample = cv2.imread(pred_file_name)
        pred_img_sample= pred_img_sample/255
        resized_pred_img_sample = cv2.resize(pred_img_sample,desired_image_shape)
        reshaped_pred_img_sample = resized_pred_img_sample.reshape(1,250, 250, 3)

        prediction_pred_image = model.predict(reshaped_pred_img_sample)
        prediction_pred_image = model.predict_on_batch(reshaped_pred_img_sample)
        print(prediction_pred_image)
        classification = np.where(prediction_pred_image == np.amax(prediction_pred_image))[1][0]
        confidence = str(prediction_pred_image[0][classification]*100)

        print ('classification',classification)
        print ('confidence',str(prediction_pred_image[0][classification]*100))
        ret_str = f"classification {classification} confidence {confidence}"
    return (ret_str)


if __name__ == '__main__':
    load_saved_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
