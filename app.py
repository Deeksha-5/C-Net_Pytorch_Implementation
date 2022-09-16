import os
import shutil
import requests
import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
from c_net_main import cnet

app = Flask(__name__)
CORS(app)

cnet_40x = None
cnet_100x = None
cnet_200x = None
cnet_400x = None
UPLOAD_FOLDER = '/data/TestHistopath'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_MAGNIFICATIONS = {'40x', '100x', '200x', '400x'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_magnification(mag):
    return mag.lower() in ALLOWED_MAGNIFICATIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return ('No file part')
            # return redirect(request.url)
        file = request.files['file']
        mag = request.form['mag']
        if file.filename == '':
            return ('No selected file')
            # return redirect(request.url)
        if mag not in ALLOWED_MAGNIFICATIONS:
            return ('Incorrect magnification entered')
            # return redirect(request.url)
        if file and allowed_file(file.filename) and allowed_magnification(mag):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_in = Image.open(UPLOAD_FOLDER + '/' + filename)
            if mag == '40x':
                res = cnet_40x.test(img_in)
            elif mag == '100x':
                res = cnet_100x.test(img_in)
            elif mag == '200x':
                res = cnet_200x.test(img_in)
            elif mag == '400x':
                res = cnet_400x.test(img_in)
            print(res.item())
            final_res = int(np.round(res.item()))
            idx_to_class = {0: 'benign', 1: 'malignant'}
            res_class = idx_to_class[final_res]
            # response_obj = {"output_class": res_class}
            # return Response(response=json.dumps(response_obj), status=200)
            return res_class

@app.route('/test')
def index():
    response_obj = {"status": "Success", "message": "Hi -- Service Running Successfully"}
    return Response(response=json.dumps(response_obj), status=200)

@app.before_first_request
def loadmodel():
    global cnet_40x 
    global cnet_100x
    global cnet_200x
    global cnet_400x
    if cnet_40x is None:
        cnet_40x = cnet('/data/BreakHis/40X_Splitted/model_64.pth')
    if cnet_100x is None:
        cnet_100x = cnet('/data/BreakHis/100X_Splitted/model_32.pth')
    if cnet_200x is None:
        cnet_200x = cnet('/data/BreakHis/200X_Splitted/model_32.pth')
    if cnet_400x is None:
        cnet_400x = cnet('/data/BreakHis/400X_Splitted/run_1/model_32.pth')

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     return response

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5000'))
    except ValueError:
        PORT = 5000
    app.run(HOST, PORT)