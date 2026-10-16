from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from collections import deque
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort
import numpy as np
import secrets
import uuid
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per scan
# Set MAIA_SECRET_KEY / MAIA_PASSWORD in your environment for anything beyond local use
app.secret_key = os.environ.get("MAIA_SECRET_KEY") or secrets.token_hex(32)

DASHBOARD_PASSWORD = os.environ.get("MAIA_PASSWORD", "81412")
VERSION = "2.1.0"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Same normalisation the models were trained with (ImageNet mean / std)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

history = deque(maxlen=15)

# Every model lives in models/<folder>/ with a resnet18.onnx and a labels.txt.
# "pretty" maps the raw label to what we actually show people.
MODELS = [
    {
        "id": "tumor_detection",
        "name": "Brain Tumor Detection",
        "scan": "Brain MRI",
        "about": "Checks a brain MRI slice for the presence of a tumor.",
        "version": "V2",
        "pretty": {"no": "No tumor found", "yes": "Tumor present"},
    },
    {
        "id": "tumor_classification",
        "name": "Brain Tumor Classification",
        "scan": "Brain MRI",
        "about": "Sorts a brain MRI slice into glioma, meningioma, pituitary tumor or no tumor.",
        "version": "V2",
        "pretty": {"glioma": "Glioma", "meningioma": "Meningioma", "none": "No tumor", "pituitary": "Pituitary tumor"},
    },
    {
        "id": "alzheimers",
        "name": "Alzheimer's Classification",
        "scan": "Brain MRI",
        "about": "Estimates the level of cognitive impairment from Alzheimer's disease.",
        "version": "V2",
        "pretty": {"none": "No impairment", "verymild": "Very mild impairment", "mild": "Mild impairment", "moderate": "Moderate impairment"},
    },
    {
        "id": "covid",
        "name": "COVID-19 Detection",
        "scan": "Chest X-ray",
        "about": "Looks for signs of COVID-19 in a chest X-ray.",
        "version": "V1",
        "pretty": {"covid": "COVID-19 signs", "nocovid": "No COVID-19 signs"},
    },
    {
        "id": "pneumonia",
        "name": "Pneumonia Detection",
        "scan": "Chest X-ray",
        "about": "Looks for pneumonia in a chest X-ray.",
        "version": "V1",
        "pretty": {"NORMAL": "Normal", "PNEUMONIA": "Pneumonia"},
    },
    {
        "id": "fractures",
        "name": "Fracture Detection",
        "scan": "Bone X-ray",
        "about": "Checks a bone X-ray for fractures.",
        "version": "V1",
        "pretty": {"fracture": "Fracture", "none": "No fracture"},
    },
    {
        "id": "EDC",
        "name": "Eye Disease Classification",
        "scan": "Retinal photo",
        "about": "Sorts a fundus photo into cataract, diabetic retinopathy, glaucoma or normal.",
        "version": "V1",
        "pretty": {"cataract": "Cataract", "diabetic_retinopathy": "Diabetic retinopathy", "glaucoma": "Glaucoma", "normal": "Normal"},
    },
]

# Load models and labels once at startup
for m in MODELS:
    folder = os.path.join(BASE_DIR, "models", m["id"])
    m["session"] = ort.InferenceSession(os.path.join(folder, "resnet18.onnx"), providers=["CPUExecutionProvider"])
    m["input"] = m["session"].get_inputs()[0].name
    with open(os.path.join(folder, "labels.txt")) as f:
        m["labels"] = [line.strip() for line in f if line.strip()]

nets = {m["id"]: m for m in MODELS}
model_enabled = {m["id"]: True for m in MODELS}


def public_info(m):
    info = {k: m[k] for k in ("id", "name", "scan", "about", "version")}
    info["accuracy"] = m.get("accuracy")
    return info


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    img_data = np.asarray(img, dtype=np.float32) / 255.0
    img_data = (img_data - MEAN) / STD
    img_data = np.transpose(img_data, (2, 0, 1))  # C,H,W
    return np.expand_dims(img_data, 0)  # batch dimension


def logged_in():
    return session.get("logged_in", False)


@app.route("/")
def index():
    # Group the switched-on models by scan type, keeping the order of MODELS
    groups = {}
    for m in MODELS:
        if model_enabled[m["id"]]:
            groups.setdefault(m["scan"], []).append(public_info(m))
    return render_template("index.html", groups=groups, version=VERSION)


@app.route("/info")
def info():
    return render_template("info.html", models=[public_info(m) for m in MODELS], version=VERSION)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), DASHBOARD_PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        return render_template("login.html", error=True, version=VERSION)
    return render_template("login.html", version=VERSION)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html", version=VERSION)


@app.route("/models")
def list_models():
    # Public: only what's currently switched on
    return jsonify([public_info(m) for m in MODELS if model_enabled[m["id"]]])


@app.route("/classify", methods=["POST"])
def classify():
    file = request.files.get("image")
    mode = request.form.get("mode", "")

    if mode not in nets:
        return jsonify({"error": "Pick a model first."}), 400
    if not model_enabled[mode]:
        return jsonify({"error": "That model is switched off right now."}), 400
    if not file or not file.filename:
        return jsonify({"error": "No image was uploaded."}), 400

    original_name = secure_filename(file.filename) or "scan"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "MAIA only reads .png and .jpg images."}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    # Prefix with a short id so two scans called "scan.png" don't overwrite each other
    saved_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    filename = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
    file.save(filename)

    try:
        img_input = preprocess_image(filename)
    except (UnidentifiedImageError, OSError):
        os.remove(filename)
        return jsonify({"error": "That file couldn't be opened as an image."}), 400

    try:
        m = nets[mode]
        probs = m["session"].run(None, {m["input"]: img_input})[0][0]
    except Exception as e:
        return jsonify({"error": f"The model failed to run: {e}"}), 500

    order = np.argsort(probs)[::-1]
    scores = [
        {"label": m["pretty"].get(m["labels"][i], m["labels"][i]), "raw": m["labels"][i], "p": round(float(probs[i]), 4)}
        for i in order
    ]
    top = scores[0]

    result = {
        "model": m["name"],
        "model_id": mode,
        "prediction": top["label"],
        "confidence": f"{top['p'] * 100:.2f}%",
        "scores": scores,
        "filename": original_name,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    history.appendleft(result)
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "That image is over 16 MB."}), 413


@app.route("/history")
def get_history():
    if not logged_in():
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(list(history))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if not logged_in():
        return jsonify({"error": "Unauthorized"}), 403
    history.clear()
    return jsonify({"status": "cleared"})


@app.route("/model_status", methods=["GET"])
def get_model_status():
    if not logged_in():
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify([dict(public_info(m), enabled=model_enabled[m["id"]]) for m in MODELS])


@app.route("/toggle_model/<model_id>", methods=["POST"])
def toggle_model(model_id):
    if not logged_in():
        return jsonify({"error": "Unauthorized"}), 403
    if model_id not in model_enabled:
        return jsonify({"error": "Invalid model id"}), 400
    model_enabled[model_id] = not model_enabled[model_id]
    return jsonify({"model_id": model_id, "enabled": model_enabled[model_id]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
