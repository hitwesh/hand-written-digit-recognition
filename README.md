Handwritten Number Recognition/
│── backend/
│   ├── app.py                    # Main Flask backend handling API requests
│── data/                          # Placeholder for dataset files
│── frontend/
│   ├── static/
│   │   ├── scripts.js             # JavaScript for frontend interactivity
│   │   ├── styles.css             # CSS for frontend styling
│   ├── templates/
│   │   ├── index.html             # Main webpage for user interaction
│── hand-written-digit-recognition/
│   ├── .gitattributes             # Git configuration file
│── models/                        # Pre-trained models
│   ├── cnn_model.onnx             # CNN model in ONNX format
│   ├── cnn_model.pth              # CNN model in PyTorch format
│   ├── knn_model.pkl              # KNN trained model
│   ├── rfc_model.pkl              # Random Forest trained model
│   ├── svm_model.pkl              # SVM trained model
│── training/                      # Model training scripts
│   ├── generate_models.py         # Script to generate models
│   ├── train_cnn.py               # Trains the CNN model
│   ├── train_models.py            # Trains ML models (SVM, KNN, RFC)
│── .gitignore                      # Specifies ignored files for Git
│── README.md                       # Project documentation
│── requirements.txt                 # List of dependencies
