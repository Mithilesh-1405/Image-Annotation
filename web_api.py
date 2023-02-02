import flask
from flask_cors import CORS
from flask import request, jsonify,send_file
from flask import Flask, render_template, request, session
import os
import skimage.io
from werkzeug.utils import secure_filename
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")
import cv2
import requests
import numpy as np
import os

UPLOAD_FOLDER ="/Users/rohankurdekar/Downloads/MLMINIPROJ/detectron2/static/uploads"
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 

def score_image(predictor: DefaultPredictor, image_url: str):
    # load an image of Lionel Messi with a ball
    image_reponse = requests.get(image_url)
    image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)

    # make prediction
    return image,predictor(image)

def custom_score_image(predictor: DefaultPredictor,img):
    # load an image of Lionel Messi with a ball
    
    # image_as_np_array = np.frombuffer(img, np.uint8)
    # image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(predictor(img))

    # make prediction
    return img,predictor(img)

def prepare_pridctor():
    # create config
    cfg = get_cfg()
    add_pointrend_config(cfg)
    # below path applies to current installation location of Detectron2
    cfg.merge_from_file("../detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy


    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    return (predictor, classes)

app = flask.Flask(__name__)
CORS(app)
predictor, classes = prepare_pridctor()



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
         files = request.files.getlist("uploaded-file");

         for file in files:
             uploaded_img = file
             img_filename = secure_filename(uploaded_img.filename)
             uploaded_img.save(os.path.join(UPLOAD_FOLDER, img_filename))

        # Upload file flask
       # uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        #img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
       #uploaded_img.save(os.path.join(UPLOAD_FOLDER, img_filename))
        # Storing uploaded file path in flask session
        # session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    return render_template('res.html')


@app.route("/upload/api/score-image", methods=["POST"])
def custom():
     if request.method == 'POST':
         files = request.files.getlist("uploaded-file");

         for file in files:
             uploaded_img = file
             img_filename = secure_filename(uploaded_img.filename)
             uploaded_img.save(os.path.join(UPLOAD_FOLDER, img_filename))
             read=skimage.io.imread(os.path.join(UPLOAD_FOLDER, img_filename))
             im,scoring_result = custom_score_image(predictor,read)
             
             instances = scoring_result["instances"] 
             scores = instances.get_fields()["scores"].tolist()
             pred_classes = instances.get_fields()["pred_classes"].tolist()
             pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
    # pred_masks=instances.get_fields()["pred_masks"].tensor.tolist()
             v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
             point_rend_result = v.draw_instance_predictions(scoring_result["instances"].to("cpu")).get_image()
             os.chdir("/Users/rohankurdekar/Downloads/MLMINIPROJ/detectron2/static/uploads")
             cv2.imwrite(img_filename,point_rend_result)
    
 
     return render_template('res.html')
             

    

    
   



    
   

@app.route("/api/score-image", methods=["POST"])
def process_score_image_request():

    image_url = request.json["imageUrl"]
    im,scoring_result = score_image(predictor, image_url)

    instances = scoring_result["instances"]
 
    scores = instances.get_fields()["scores"].tolist()
    pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
    # pred_masks=instances.get_fields()["pred_masks"].tensor.tolist()
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    point_rend_result = v.draw_instance_predictions(scoring_result["instances"].to("cpu")).get_image()
    os.chdir("/Users/rohankurdekar/Downloads/MLMINIPROJ/Annotation/uploads")
    cv2.imwrite("img.jpg",point_rend_result)
    cv2.imwrite("Image.jpg",point_rend_result)
    
    


    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
        "classes": classes,
        
    }

    return jsonify(response)
  

app.run(host="0.0.0.0", port=3000)