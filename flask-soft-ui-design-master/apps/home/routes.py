# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request , jsonify
from flask_login import login_required
from jinja2 import TemplateNotFound
from flask import Flask, render_template, request, send_file
import requests
import io
from PIL import Image
from PIL import UnidentifiedImageError  # Import the UnidentifiedImageError class
import requests
import io
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
#from chat import get_response

API_URL = "https://api-inference.huggingface.co/models/yahyasmt/brain_tumor_2"
headers = {"Authorization": "Bearer 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"}


app = Flask(__name__)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@blueprint.route('/index',methods=['GET', 'POST'])
@login_required
def index():
    return render_template('home/index.html', segment='index')

##################################################################
@blueprint.route('/indeximage',methods=['GET', 'POST'])
@login_required
def indeximage():
    if request.method == "POST":
        prompt = request.form["input-text"]

        image_bytes = query({
            "inputs": prompt,
        })
        with open("./apps/static/assets/mdl/aa.jpeg", "wb") as image_file:
            image_file.write(image_bytes)

    return render_template('home/indexImage.html')

#########################


# Process the uploaded image and display the 3D plot
@blueprint.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load your image (color or grayscale)
        image = plt.imread(filename)

        # Check if it's a color image
        is_color_image = len(image.shape) == 3 and image.shape[2] in [3, 4]

        # Convert color image to grayscale if needed
        if is_color_image:
            gray_image = np.mean(image, axis=2)  # Convert to grayscale
        else:
            gray_image = image

        # Normalize the image to the range [0, 1]
        gray_image = gray_image / np.max(gray_image)

        # Apply a Gaussian filter to smooth the image (adjust sigma for smoothing effect)
        sigma = 2.0
        smoothed_image = gaussian_filter(gray_image, sigma=sigma)

        # Scale the smoothed image to control the elevation magnitude
        elevation_scale = 50.0  # Adjust this value as needed
        elevated_image = smoothed_image * elevation_scale

        # Create a grid of X and Y coordinates
        x, y = np.meshgrid(np.arange(gray_image.shape[1]), np.arange(gray_image.shape[0]))

        # Create the Z coordinates using the elevated image
        z = elevated_image

        # Create a 3D surface plot of the inflated image using Plotly
        fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis')])
        fig.update_layout(scene=dict(aspectmode="data"))
        plot_div = fig.to_html(full_html=False)

        return render_template('home/result.html', plot_div=plot_div)

    return redirect(request.url)

# Serve uploaded files
@blueprint.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@blueprint.route('/3D')
@login_required
def ThreeD():
    return render_template('home/3D.html')


@blueprint.route('/chat')
# @login_required
def chat():
    return render_template('home/chat.html')

#@blueprint.route("/predict")
#@login_required
#def predict():
#    text = request.get_json().get("message")
#    response = get_response(text)
#    message = {"answer" : response}
#    return jsonify(message)
##################################################################




@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
