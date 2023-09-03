import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Render the HTML form for image upload
@app.route('/')
def index():
    return render_template('index.html')

# Process the uploaded image and display the 3D plot
@app.route('/upload', methods=['POST'])
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

        # Load your grayscale image
        gray_image = plt.imread(filename)

        # Normalize the grayscale image to the range [0, 1]
        gray_image = gray_image / 255.0

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

        return render_template('result.html', plot_div=plot_div)

    return redirect(request.url)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)