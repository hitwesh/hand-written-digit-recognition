# 📝 Handwritten Digit Recognition & Model Comparison

This project compares multiple machine learning models (**SVM, RFC, KNN, and CNN**) for **handwritten digit recognition**. It takes a handwritten digit image as input, processes it, and displays predictions from all four models along with an accuracy comparison.

---

## 📌 Features
- ✅ **Supports 4 Models**: CNN, SVM, Random Forest, and KNN  
- ✅ **Web-Based Interface** (Upload an image and get predictions)  
- ✅ **Pre-Trained Models Available** (Fast inference without retraining)  
- ✅ **Works with or without GPU**  
- ✅ **Simple & Clean UI**  

---

## 🏗️ Installation  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition

2️⃣ Create a Virtual Environment (Recommended)

# For Windows (CMD)
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Running the Application
Option 1: Run the Pre-Trained Models (Recommended)

    🚀 Fastest way to test the project (No need to train models)

python app.py

Now, open http://127.0.0.1:5000/ in your browser and upload an image!
Option 2: Train the Models (If Needed)

    If you want to retrain the models from scratch, follow these steps.

Step 1: Train SVM, RFC, KNN Models

python training/generate_models.py

⏳ Training Time:

    SVM: ~2 min

    RFC: ~3 min

    KNN: ~1 min

Step 2: Train CNN Model

python training/train_cnn.py

⏳ Training Time:

    With GPU: ~5-10 min

    Without GPU: ~30-60 min

🔹 Once trained, the models will be saved inside the models/ folder.
📂 Project Structure

📦 handwritten-digit-recognition
│-- 📂 models/              # Saved trained models (SVM, RFC, KNN, CNN)
│-- 📂 static/              # CSS & JavaScript files
│-- 📂 templates/           # HTML files for frontend
│-- 📂 training/            # Training scripts for all models
│   ├── generate_models.py  # Train SVM, RFC, KNN
│   ├── train_cnn.py        # Train CNN model separately
│-- app.py                  # Main Flask backend
│-- preprocess.py           # Image preprocessing for models
│-- predict.py              # Predict function using trained models
│-- requirements.txt        # Required dependencies
│-- README.md               # Documentation (You are here!)

🎯 How the Code Works
1️⃣ Preprocessing (preprocess.py)

    Converts the input image to grayscale

    Resizes it to 28x28 pixels (same as MNIST)

    Inverts colors if needed (to match MNIST format)

    Normalizes pixel values

2️⃣ Model Training

    SVM, Random Forest, and KNN models are trained using generate_models.py

    CNN model is trained separately using train_cnn.py

3️⃣ Predictions (predict.py)

    Loads the pre-trained models

    Runs inference using all four models

    Displays each model's prediction and confidence score

💻 Running on Different Systems
On a PC with GPU (For Faster CNN Training)

    Train all models on your GPU machine

    Copy the models/ folder to your main PC

    Run app.py to test predictions

On a PC without GPU (Run Inference Only)

    Use pre-trained models (No training needed)

    Simply run

    python app.py

🔧 Troubleshooting
❌ "Module Not Found" Error?

Run:

pip install -r requirements.txt

❌ "Models Not Found" Error?

Run:

python training/generate_models.py

❌ Slow Performance?

If you have a GPU, TensorFlow will automatically use it for faster CNN inference.
🤖 Technologies Used

    Python (Flask, NumPy, OpenCV)

    Machine Learning (SVM, RFC, KNN, CNN)

    Deep Learning (TensorFlow, Keras)

    Frontend (HTML, CSS, JavaScript, Bootstrap)

