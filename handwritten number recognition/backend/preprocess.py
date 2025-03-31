import numpy as np
import cv2

def preprocess_image(file):
    """Preprocesses an image file for ONNX model prediction."""
    # Read the image as grayscale
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 (same as MNIST dataset)
    img = cv2.resize(img, (28, 28))

    # Invert colors (if necessary)
    img = cv2.bitwise_not(img)

    # Normalize pixel values
    img = img.astype(np.float32) / 255.0

    # Reshape for ONNX (1, 1, 28, 28)
    img = img.reshape(1, 1, 28, 28)

    return img
