import os
from google.colab import files
import argparse
import main
from flask import Flask,jsonify, send_from_directory,request
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
from urllib.parse import quote
import cv2, imutils
import matplotlib.pyplot as plt
app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})
# app.config['CORS_HEADER'] = 'Content-Type'
# app.config['CORS_HEADERS'] = 'Content-Type'
run_with_ngrok(app)  
root=os.getcwd()

@app.route("/")
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def home():
    return "<h1>GFG is great platform to learn</h1>"
@app.route('/styling')
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def styling():
  main.transfom(args)
@app.route('/status')
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def status():
  if main.success_transform_flag:
    return jsonify({'success': 'Done'}), 200
  else:
    return  jsonify({'unknown': 'unknown'}), 400

@app.route('/list_images/<folder_images>')
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# /list_images/
def list_images(folder_images):
    try:
        # Get the absolute path of the folder
        folder_path = os.path.join(root,'ffhq_image',folder_images)
        print(folder_path)
        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # List all files in the folder
            files = os.listdir(folder_path)
            output=[]
            _url=request.url
            _url=_url.replace(f"/list_images/{folder_images}","")
            # Filter only image files (e.g., jpg, png, etc.)
            image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            for file in files:
              output.append({
                "name":file.split(".")[0],
                "image_path":f"{_url}/get_image/hair_resource/{file}"
                })
            response= jsonify(output)
            # response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        else:
            return 404

    except Exception as e:
        response=jsonify({'error': str(e)})
        # response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

# Define a route to serve individual images by filename
@app.route('/get_image/<folder_path>/<filename>')
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
# example: /get_image/ffhq_image/customer_pic/customer.png
def get_image(folder_path, filename):
    try:
      # Create the complete path to the image file
      image_path = os.path.join(root,'ffhq_image', folder_path, filename)
      print(image_path)
      # Check if the image file exists
      if os.path.exists(image_path) and os.path.isfile(image_path):
          # Send the image from the directory
        return send_from_directory(os.path.join(root,'ffhq_image', folder_path), filename)
      else:
          return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def upload_image():
    for r in request.files:
      print(r)
    if 'image' not in request.files:
        response=jsonify({'error': 'No image provided'})
        # response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400

    image = request.files['image']
    # image=scale_image(image)
    image.save(image.filename)
    # Move the image to the Colab directory
    colab_directory = os.path.join(root,'ffhq_image/customer_pic/temp')
    new_name=os.path.join(colab_directory,'customer.png')
    # new_path = os.path.join(colab_directory, image.filename)
    print(f"new_path {new_name}")
    os.rename(image.filename, new_name)
    # print(f"rename {new_path}")
    # center_and_resize_portrait_image(new_name,os.path.join(root,'ffhq_image/customer_pic/output/croppted_customer.png'))
    response=jsonify({'colabPath': new_name})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response
def center_and_resize_portrait_image(image_path,output_path):
    # Load the image
    print(f"loading image: {image_path}")
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=1024)
    # Load a face detection model (e.g., Haar Cascade or DNN-based face detector)
    # Replace 'haarcascade_frontalface_default.xml' with the appropriate model path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # No face detected, return None
        return None

    # Assuming you want to use the first detected face (you can modify this logic)
    (x, y, w, h) = faces[0]
    print(f"{x},{y},{w},{h}")
    # Calculate the center of the detected face
    center_x = x + w // 2
    center_y = y + h // 2
    print(f"Center:{center_x},{center_y}")
    # Define the crop size (512x512)
    crop_size = 1024

    # Calculate the cropping box
    
    left = max(center_x - crop_size // 2, 0) if center_x - crop_size>0 else 0
    top = max(center_y - crop_size // 2, 0) if center_y - crop_size>0 else 0
    right = min(left + crop_size, image.shape[1])
    bottom = min(top + crop_size, image.shape[0])

    # Crop the image
    cropped_image = image[top:bottom, left:right]
    # Save the resulting image
    cv2.imwrite(output_path, cropped_image)
    # Resize the cropped image to 512x512 without stretching
    # resized_image = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)
    print(f"saved at:{output_path} ")
    return cropped_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Style Your Hair')

    # flip
    parser.add_argument('--flip_check', action='store_true', help='image2 might be flipped')

    # warping and alignment
    parser.add_argument('--warp_front_part', default=True,
                        help='optimize warped_trg img from W+ space and only optimized [:6] part')
    parser.add_argument('--warped_seg', default=True, help='create aligned mask from warped seg')
    parser.add_argument('--align_src_first', default=True, help='align src with trg mask before blending')
    parser.add_argument('--optimize_warped_trg_mask', default=True, help='optimize warped_trg_mask')
    parser.add_argument('--mean_seg', default=True, help='use mean seg when alignment')

    parser.add_argument('--kp_type', type=str, default='3D', help='kp_type')
    parser.add_argument('--kp_loss', default=True, help='use keypoint loss when alignment')
    parser.add_argument('--kp_loss_lambda', type=float, default=1000, help='kp_loss_lambda')

    # blending
    parser.add_argument('--blend_with_gram', default=True, help='add gram matrix loss in blending step')
    parser.add_argument('--blend_with_align', default=True,
                        help='optimization of alignment process with blending')


    # hair related loss
    parser.add_argument('--warp_loss_with_prev_list', nargs='+', help='select among delta_w, style_hair_slic_large',default=None)
    parser.add_argument('--sp_hair_lambda', type=float, default=5.0, help='Super pixel hair loss when embedding')


    # utils
    parser.add_argument('--version', type=str, default='v1', help='version name')
    parser.add_argument('--save_all', action='store_true',help='save all output from whole process')
    parser.add_argument('--embedding_dir', type=str, default='./output/', help='embedding vector directory')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='./image/',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='The directory to save the output images')
    parser.add_argument('--im_path1', type=str, default='16.png', help='Identity image')
    parser.add_argument('--im_path2', type=str, default='15.png', help='Structure image')
    parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
    parser.add_argument('--verbose', action='store_true', help='Print loss information')
    parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')

    # Embedding loss options
    parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
    parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
    parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
    parser.add_argument('--l_F_lambda', type=float, default=0.1, help='L_F loss multiplier factor')
    parser.add_argument('--W_steps', type=int, default=1100, help='Number of W space optimization steps')
    parser.add_argument('--FS_steps', type=int, default=250, help='Number of W space optimization steps')

    # Alignment loss options
    parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
    parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
    parser.add_argument('--align_steps1', type=int, default=400, help='')
    parser.add_argument('--align_steps2', type=int, default=100, help='')
    parser.add_argument('--warp_steps', type=int, default=100, help='')

    # Blend loss options
    parser.add_argument('--face_lambda', type=float, default=1.0, help='')
    parser.add_argument('--hair_lambda', type=str, default=1.0, help='')
    parser.add_argument('--blend_steps', type=int, default=400, help='')

    
    args = parser.parse_args()
    app.run()