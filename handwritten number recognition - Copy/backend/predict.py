import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import preprocess_image

# Load all trained models
svm_model = joblib.load("models/svm_model.pkl")
rfc_model = joblib.load("models/rfc_model.pkl")
knn_model = joblib.load("models/knn_model.pkl")
cnn_model = load_model("models/cnn_model.h5")  # Load CNN model

def predict_digit(image_file):
    """Predicts the digit from an image using SVM, RFC, KNN, and CNN models."""
    
    # Preprocess the image
    img_flat, img_cnn = preprocess_image(image_file)
    
    # Get predictions from each model
    svm_pred = svm_model.predict([img_flat])[0]
    rfc_pred = rfc_model.predict([img_flat])[0]
    knn_pred = knn_model.predict([img_flat])[0]
    
    # CNN prediction (returns probabilities, so take the class with highest probability)
    cnn_probs = cnn_model.predict(np.expand_dims(img_cnn, axis=0))  # Add batch dimension
    cnn_pred = np.argmax(cnn_probs)

    # Return all predictions
    return {
        "SVM Prediction": int(svm_pred),
        "RFC Prediction": int(rfc_pred),
        "KNN Prediction": int(knn_pred),
        "CNN Prediction": int(cnn_pred),
        "CNN Confidence Scores": cnn_probs.tolist()  # Show confidence for each digit (0-9)
    }
