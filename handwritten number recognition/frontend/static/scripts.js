document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const imagePreview = document.getElementById("imagePreview");
    const resultDiv = document.getElementById("result");
    const predictionText = document.getElementById("predictionText");

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove("hidden");
                predictBtn.disabled = false;
            };

            reader.readAsDataURL(file);
        }
    });

    predictBtn.addEventListener("click", function () {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            predictionText.innerHTML = `
                <strong>🔢 CNN:</strong> ${data.cnn} <br>
                <strong>🤖 SVM:</strong> ${data.svm} <br>
                <strong>🌲 RFC:</strong> ${data.rfc} <br>
                <strong>📍 KNN:</strong> ${data.knn} <br>
            `;
            resultDiv.classList.remove("hidden");
        })
        .catch(error => console.error("Error:", error));
    });
});
