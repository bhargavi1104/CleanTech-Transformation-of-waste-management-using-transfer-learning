 Cleantech: Transforming Waste Management with Transfer Learning

This project is developed as part of an AI/ML internship focused on solving real-world problems through intelligent automation. The system uses **Transfer Learning with VGG16** to classify waste into categories like **biodegradable**, **recyclable**, and **trash**, helping promote sustainable waste management practices.

 🚀 Features
- 🌿 Classify waste images into categories
- 🧠 Powered by VGG16 Transfer Learning
- 📷 Upload your image and get instant predictions
- 💻 Flask-based interactive web application
- 🎨 Clean and responsive UI with multiple pages (Home, Predict, Contact, About)

🛠️ Technologies Used
- Python
- TensorFlow & Keras
- Flask
- HTML, CSS
- Bootstrap (for design elements)
- Git & GitHub

📁 Folder Structure
├── app.py
├── split\_data.py
├── train\_model.py
├── vgg16.h5
├── static/
│   └── assets/
│       ├── bg.png
│       ├── logo.png
│       └── other images
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── blog.html
│   └── blog-single.html

📷 Sample Usage
1. Run `app.py`
2. Open browser and visit `http://127.0.0.1:5000/`
3. Navigate to **Predict** page
4. Upload an image of waste (like banana peel or plastic bottle)
5. Get classification result instantly

📌 Note
* The dataset used is from Kaggle: [Municipal Solid Waste Dataset](https://www.kaggle.com/datasets/elinachen717/municipal-solid-waste-dataset)
* Ensure `vgg16.h5` (trained model) is in the root directory before running the app.
