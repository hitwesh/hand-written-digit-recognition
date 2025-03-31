# Handwritten Digit Recognition & Model Comparison

This project compares multiple machine learning models (SVM, RFC, KNN, and CNN) for handwritten digit recognition. A user uploads a handwritten digit image, and all models predict the digit, allowing for a comparison of their performance.

## Features
- Upload a handwritten digit image
- Process the image for all models
- Get predictions from SVM, RFC, KNN, and CNN
- Compare accuracy of different models

## Tech Stack
- **Backend**: Flask, OpenCV, NumPy, TensorFlow/Keras, Scikit-learn
- **Frontend**: HTML, CSS, JavaScript

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd handwritten-digit-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the machine learning models:
   ```bash
   python train_models.py
   python train_cnn.py
   ```
4. Start the Flask server:
   ```bash
   python app.py
   ```
5. Open `index.html` in your browser and upload an image to test.

## Folder Structure
```
├── backend/
│   ├── app.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── train_models.py
│   ├── train_cnn.py
│   ├── generate_models.py
│   └── models/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── scripts.js
├── requirements.txt
└── README.md
```

## Notes
- Ensure that all models are trained before running the application.
- The Flask server should be running while testing the frontend.
- Images should be **28x28 pixels grayscale** for accurate predictions.

## License
This project is open-source and available under the MIT License.
