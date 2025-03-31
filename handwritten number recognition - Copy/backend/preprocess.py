import numpy as np
import cv2

def preprocess_image(file):
    """Preprocess an uploaded image for model predictions."""
    
    # Read image as grayscale
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28 (same as MNIST dataset)
    img = cv2.resize(img, (28, 28))
    
    # Invert colors (MNIST format is white digits on black background)
    img = cv2.bitwise_not(img)
    
    # Normalize pixel values (0-255 → 0-1)
    img = img / 255.0
    
    # Flatten the image for traditional ML models (SVM, RFC, KNN)
    img_flat = img.flatten()
    
    # Reshape for CNN (28x28x1)
    img_cnn = img.reshape(28, 28, 1)
    
    return img_flat, img_cnn
