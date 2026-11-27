
const imageInput = document.getElementById("leafImage");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const result = document.getElementById("result");
const cropInput = document.getElementById("crop");
const chooseImageBtn = document.getElementById("chooseImageBtn");
const fileName = document.getElementById("fileName");

let selectedFile = null;
let previewUrl = null;

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp"
];

// Escape text before inserting API values into HTML.
function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };

        return entities[character];
    });
}

// Open image picker.
chooseImageBtn.addEventListener("click", () => {
    imageInput.click();
});

// Display uploaded image preview.
imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
        alert("Please select a JPG, PNG or WEBP image.");
        imageInput.value = "";
        selectedFile = null;
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        alert("Image must be smaller than 10 MB.");
        imageInput.value = "";
        selectedFile = null;
        return;
    }

    selectedFile = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    preview.style.display = "block";

    fileName.textContent = `Selected file: ${file.name}`;

    result.style.display = "none";
    result.innerHTML = "";
});

// Format confidence value.
function formatConfidence(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "N/A";
    }

    return `${number.toFixed(2)}%`;
}

// Display prediction results.
function displayResults(data) {

    const predictions = Array.isArray(data.top_predictions)
        ? data.top_predictions
        : [];

    const bestPrediction = predictions[0] || {};

    // Support common response field names from the API.
    const diseaseName =
        data.predicted_disease ||
        data.disease_name ||
        bestPrediction.disease_name ||
        bestPrediction.class_name ||
        "Prediction unavailable";

    const confidence =
        data.confidence ??
        bestPrediction.confidence;

    const selectedCrop =
        data.selected_crop ||
        bestPrediction.crop ||
        cropInput.value;

    const modelName = data.model_name || "MobileNetV2";

    const dimensions = data.original_width && data.original_height
        ? `${data.original_width} × ${data.original_height}`
        : "Not available";

    let predictionsHTML = "";

    if (predictions.length > 0) {

        predictionsHTML = predictions.map((item, index) => {

            const name =
                item.disease_name ||
                item.class_name ||
                "Unknown";

            const score = formatConfidence(item.confidence);

            return `
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    gap:12px;
                    padding:12px;
                    margin-top:8px;
                    background:#f1f8f2;
                    border-radius:8px;
                ">
                    <span>
                        ${index + 1}. ${escapeHTML(name)}
                    </span>

                    <strong>${escapeHTML(score)}</strong>
                </div>
            `;
        }).join("");
    } else {
        predictionsHTML = "<p>Top predictions are not available.</p>";
    }

    result.innerHTML = `
        <div style="
            margin-top:24px;
            padding:22px;
            background:#ffffff;
            border:1px solid #dcebdd;
            border-radius:14px;
            color:#20352a;
        ">

            <h2 style="color:#218447; margin-top:0;">
                AI Analysis Successful!
            </h2>

            <p>
                <strong>Predicted Disease:</strong>
                ${escapeHTML(diseaseName)}
            </p>

            <p>
                <strong>Confidence:</strong>
                ${escapeHTML(formatConfidence(confidence))}
            </p>

            <p>
                <strong>Selected Crop:</strong>
                ${escapeHTML(selectedCrop)}
            </p>

            <p>
                <strong>AI Model:</strong>
                ${escapeHTML(modelName)}
            </p>

            <p>
                <strong>Original Image Size:</strong>
                ${escapeHTML(dimensions)}
            </p>

            <h3 style="margin-top:22px;">
                Top 3 Predictions
            </h3>

            ${predictionsHTML}

            <p style="
                margin-top:18px;
                font-size:13px;
                color:#64746a;
            ">
                Note: AI predictions are decision-support information.
                Confirm the result with a qualified agricultural expert
                before taking crop treatment decisions.
            </p>

        </div>
    `;

    result.style.display = "block";
}

// Send image to FastAPI and receive AI prediction.
analyzeBtn.addEventListener("click", async () => {

    if (!selectedFile) {
        alert("Please upload a leaf image first.");
        return;
    }

    const formData = new FormData();

    formData.append("file", selectedFile);
    formData.append("crop", cropInput.value);

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analysing Leaf...";

    result.style.display = "block";
    result.textContent = "Please wait. MobileNetV2 is analysing your leaf image...";

    try {

        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        
const responseText = await response.text();

let data;

try {
    data = JSON.parse(responseText);
} catch {
    data = { detail: responseText };
}

if (!response.ok) {

    let errorMessage = "Unknown API error";

    if (typeof data.detail === "string") {
        errorMessage = data.detail;

    } else if (data.detail) {
        errorMessage = JSON.stringify(data.detail, null, 2);
    }

    throw new Error(
        `HTTP ${response.status}: ${errorMessage}`
    );
}

displayResults(data);


    } catch (error) {

        result.style.display = "block";

        result.innerHTML = `
            <div style="
                padding:18px;
                background:#fff0f0;
                border:1px solid #f0caca;
                border-radius:10px;
                color:#a52a2a;
            ">
                <h3>Analysis Failed</h3>
                <p>${escapeHTML(error.message)}</p>
                <p>Please check your image and try again.</p>
            </div>
        `;

    } finally {

        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyse Leaf";

    }
});
