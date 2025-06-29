from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model('vgg16.h5')

class_names = ['biodegradable', 'recyclable', 'trash']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        
        img_file = request.files['image']
        if img_file.filename == '':
            return "No selected file"
        
        # Ensure the uploads directory exists
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        img_path = os.path.join(upload_folder, img_file.filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Use url_for to generate the correct image URL for the template
        img_url = url_for('static', filename=f'uploads/{img_file.filename}')
        return render_template('result.html', prediction=predicted_class, image_path=img_url)

    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template("blog.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        return render_template("blog-single.html", success=True)
    return render_template("blog-single.html", success=False)

if __name__ == '__main__':
    app.run(debug=True)