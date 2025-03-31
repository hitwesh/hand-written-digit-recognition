import numpy as np
import onnxruntime as ort
import cv2

# Load the ONNX model using ONNX Runtime
onnx_model_path = "models/cnn_model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["DmlExecutionProvider"])  # DirectML for AMD GPUs

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

    # Reshape for the ONNX model (1, 1, 28, 28)
    img = img.reshape(1, 1, 28, 28)

    return img

def predict_digit(file):
    """Predicts the digit in the given image using ONNX Runtime."""
    img = preprocess_image(file)

    # Run inference
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)

    # Get the predicted class
    predicted_class = np.argmax(outputs[0])
    return predicted_class

