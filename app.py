import flask
from flask_cors import CORS
from flask import request, jsonify,send_file
from flask import Flask, render_template, request, session,redirect,url_for,flash
from PIL import Image
import base64
import io
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
import zipfile

#for windows
UPLOAD_FOLDER_ANOT ="F:/Minor_web/detectron2/static/annotation"
UPLOAD_FOLDER_KEY ="F:/Minor_web/detectron2/static/keypoint"

#for mac
# UPLOAD_FOLDER="/Users/rohankurdekar/Downloads/MLMINIPROJ/forkeddetecton/static"
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name for template path
# The default folder name for static files should be "static" else need to mention custom folder for static path
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_ANOT
 

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
    #For Windows
    cfg.merge_from_file("F:/Minor_web/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

    #For Mac
    # cfg.merge_from_file("/Users/rohankurdekar/Downloads/MLMINIPROJ/forkeddetecton/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    
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

@app.route('/uploadanot')
def uploadimganot():
    return render_template('annotation.html')

@app.route('/uploadkeypoint')
def uploadimgkey():
    return render_template('keypoint.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
         files = request.files.getlist("uploaded-file");

         for file in files:
             uploaded_img = file
             img_filename = secure_filename(uploaded_img.filename)
             uploaded_img.save(os.path.join(UPLOAD_FOLDER_ANOT, img_filename))

        # Upload file flask
       # uploaded_img = request.files['uploaded-file']
        # Extracting uploaded data file name
        #img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
       #uploaded_img.save(os.path.join(UPLOAD_FOLDER, img_filename))
        # Storing uploaded file path in flask session
        # session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    return render_template('res.html')



@app.route("/upload/api/score-image/keypoint", methods=["POST"])
def keypoint():
        if request.method == 'POST':
            files = request.files.getlist("uploaded-file");


        for file in files:
            uploaded_img = file
            img_filename = secure_filename(uploaded_img.filename)
            uploaded_img.save(os.path.join(UPLOAD_FOLDER_KEY, img_filename))
            im=skimage.io.imread(os.path.join(UPLOAD_FOLDER_KEY, img_filename))
            cfg = get_cfg()   # get a fresh new config
            cfg.merge_from_file("F:/Minor_web/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
            cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
            cfg.MODEL.DEVICE = "cpu" 
            predictor = DefaultPredictor(cfg)
            outputs = predictor(im)
            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.custom_draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
            os.chdir(UPLOAD_FOLDER_KEY)
            cv2.imwrite(img_filename,out)
             
        # return render_template('res.html')
        return redirect(url_for("display"))


@app.route("/upload/api/score-image", methods=["POST"])
def custom():
    if request.method == 'POST':
         files = request.files.getlist("uploaded-file");

         for file in files:
             uploaded_img = file
             print(file)
             img_filename = secure_filename(uploaded_img.filename)
             uploaded_img.save(os.path.join(UPLOAD_FOLDER_ANOT+"/Original", img_filename))
             read=skimage.io.imread(os.path.join(UPLOAD_FOLDER_ANOT+"/Original", img_filename))
             im,scoring_result = custom_score_image(predictor,read)
             
             instances = scoring_result["instances"] 
             scores = instances.get_fields()["scores"].tolist()
             pred_classes = instances.get_fields()["pred_classes"].tolist()
             pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
             
             # pred_masks=instances.get_fields()["pred_masks"].tensor.tolist()
             
             v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
             point_rend_result = v.draw_instance_predictions(False,scoring_result["instances"].to("cpu")).get_image()
             masked = v.draw_instance_predictions(True,scoring_result["instances"].to("cpu")).get_image()
           
           

             os.chdir(UPLOAD_FOLDER_ANOT+"/Box")
             cv2.imwrite(img_filename,point_rend_result)
             #new = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)

            # withmask= new.draw_instance_predictions(True,scoring_result["instances"].to("cpu")).get_image()
             
             os.chdir(UPLOAD_FOLDER_ANOT+"/Mask")
             cv2.imwrite(img_filename,masked)     
        
    # return redirect(url_for("display"))
    return  render_template('res.html')
    # return f"{img_filename}"
    
# @app.route('/download')
def makeZip(option):
    # maskFolder=os.listdir(UPLOAD_FOLDER_ANOT+"/Mask")
    # os.chdir(UPLOAD_FOLDER_ANOT+"/zips")
    
    path=UPLOAD_FOLDER_ANOT+"/"+option
    Folder=os.listdir(path)
    
    
    #Check if Box.zip || Mask.zip already exists
    #If yes, return the generated zip
    
    
    if(os.path.exists(UPLOAD_FOLDER_ANOT+'/zips/'+'Annotation_'+option+'.zip')):
        p=UPLOAD_FOLDER_ANOT+'/zips/'+'Annotation_'+option+'.zip'
        return send_file(p,as_attachment=True)
    else:
        handle=zipfile.ZipFile('Annotation_'+option+'.zip','w')
        for files in Folder:
            print(files)
            handle.write(files,compress_type=zipfile.ZIP_DEFLATED)
                
        handle.close()
   
        
# def Annotation_Box():
#     p=UPLOAD_FOLDER_ANOT+"/Box/Annotation_Box.zip"
    
        
        
    
@app.route("/api/choice",methods=["POST"])
def choice():
    option = request.form['options']
    print(option)
    
    folder=UPLOAD_FOLDER_ANOT+"/"+option; #path
    # print(folder)
    files=os.listdir(folder) #Files in the path
    os.chdir(folder)
    # handle=zipfile.ZipFile('Annotation'+'_'+option+'.zip','w')
    filelist=[]
    list1=[]
    for img_filename in files:
        img=Image.open(folder+"/"+img_filename)
        data=io.BytesIO()
        img.save(data,"JPEG")
        filelist.append(img_filename)
        encode_img_data = base64.b64encode(data.getvalue())
        list1.append(encode_img_data.decode("UTF-8"))
        # print(img_filename)
        # handle.write(img_filename,compress_type=zipfile.ZIP_DEFLATED)
    
    makeZip(option)
    # handle.close()
    # variable=list1[2];    
    # return render_template('res.html', len=len(list1),names=list1)
    return render_template('resultdisplay.html',len=len(list1),images=list1,files=filelist,opt=option)
    # filename=encode_img_data.decode("UTF-8")


  

@app.route("/api/preview", methods=["GET"])
def display():
    # return f"<h1>Hello</h1>"

    files=os.listdir(UPLOAD_FOLDER_KEY)
    # handle=zipfile.ZipFile('Annotation_output.zip','w')
    filelist=[]
    list1=[]
    
    for img_filename in files:
        img=Image.open(UPLOAD_FOLDER_KEY+"/"+img_filename)
        data=io.BytesIO()
        img.save(data,"JPEG")
        filelist.append(img_filename)
        encode_img_data = base64.b64encode(data.getvalue())
        list1.append(encode_img_data.decode("UTF-8"))
        # handle.write(img_filename,compress_type=zipfile.ZIP_DEFLATED)
    
    # handle.close()
    # variable=list1[2];    
    # return render_template('res.html', len=len(list1),names=list1)
    return render_template('resultdisplay.html',len=len(list1),images=list1,files=filelist)
    # filename=encode_img_data.decode("UTF-8")


@app.route('/downloadBox')
def Annotation_Box():
    p=UPLOAD_FOLDER_ANOT+"/Box/Annotation_Box.zip"
    return send_file(p,as_attachment=True)


@app.route('/downloadMask')
def Annotation_Mask():
    p=UPLOAD_FOLDER_ANOT+"/Mask/Annotation_Mask.zip"
    return send_file(p,as_attachment=True)

# @app.route("/api/score-image", methods=["POST"])
# def process_score_image_request():

#     image_url = request.json["imageUrl"]
#     im,scoring_result = score_image(predictor,image_url)

#     instances = scoring_result["instances"]
 
#     scores = instances.get_fields()["scores"].tolist()
#     pred_classes = instances.get_fields()["pred_classes"].tolist()
#     pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
#     # pred_masks=instances.get_fields()["pred_masks"].tensor.tolist()
#     v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
#     point_rend_result = v.draw_instance_predictions(scoring_result["instances"].to("cpu")).get_image()

#     #For Windows
#     # os.chdir("C:/Users/Manjunath/Downloads/uploads")

#     #For mac
#     os.chdir("/Users/rohankurdekar/Downloads/MLMINIPROJ/forkeddetecton/uploads")
#     cv2.imwrite("img.jpg",point_rend_result)
#     cv2.imwrite("Image.jpg",point_rend_result)
    
    


    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_boxes" : pred_boxes,
        "classes": classes,
        
    }

    return jsonify(response)
  

app.run(host="0.0.0.0", port=3000,debug=True)


