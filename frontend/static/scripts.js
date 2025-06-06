document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const dropArea = document.getElementById("drop-area");
    const imagePreview = document.getElementById("imagePreview");
    const resultBox = document.getElementById("result");
    const predictionText = document.getElementById("predictionText");
    const loadingSpinner = document.getElementById("loading");

    // Handle File Selection
    fileInput.addEventListener("change", handleFile);

    function handleFile(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove("hidden");
                predictBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }

    // Handle Drag & Drop
    dropArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropArea.classList.add("dragging");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragging");
    });

    dropArea.addEventListener("drop", (event) => {
        event.preventDefault();
        dropArea.classList.remove("dragging");
        fileInput.files = event.dataTransfer.files;
        handleFile({ target: fileInput });
    });

    // Predict Button Click
    predictBtn.addEventListener("click", function () {
        const file = fileInput.files[0];
        if (!file) {
            alert("Please upload an image first!"); 
            return;
        }

        resultBox.classList.add("hidden");
        loadingSpinner.classList.remove("hidden");

        const formData = new FormData();
        formData.append("file", file);

        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            loadingSpinner.classList.add("hidden");
            resultBox.classList.remove("hidden");

            if (data.error) {
                predictionText.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
                return;
            }

            predictionText.innerHTML = `
                🧠 CNN Prediction: ${data.cnn_prediction} <br>
                🤖 SVM Prediction: ${data.svm_prediction} <br>
                🌲 RFC Prediction: ${data.rfc_prediction} <br>
                🔍 KNN Prediction: ${data.knn_prediction}
            `;
        })
        .catch(error => {
            loadingSpinner.classList.add("hidden");
            predictionText.innerHTML = `<span style="color:red;">Error: ${error.message}</span>`;
        });
    });
});
