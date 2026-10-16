# MAIA — Medical AI Assistant

MAIA (Medical Artificial Intelligence Assistant) is a web-based platform that uses AI to analyze medical scans quickly. Whether it's detecting brain tumors, classifying their type, assessing Alzheimer's severity, spotting signs of COVID-19 or pneumonia, identifying bone fractures or sorting eye diseases, MAIA gives a result in seconds. With a clean, minimal interface and easy model switching, it makes complex diagnostics simple and accessible.

> **MAIA is a learning project, not a medical device.** It has not been clinically validated. Never use it to make a medical decision. If you're worried about a scan, talk to a doctor.

```
 __  __    _    ___    _
|  \/  |  /_\  |_ _|  /_\
| |\/| | //_\\  | |  //_\\
| |  | |/ ___ \ | | / ___ \
|_|  |_/_/   \_\___/_/   \_\
```

---

## Contents

- [What's new in 2.1](#whats-new-in-21)
- [Quick start](#quick-start)
- [How to use it](#how-to-use-it)
- [The models](#the-models)
- [Admin dashboard](#admin-dashboard)
- [Configuration](#configuration)
- [Hosting MAIA for free](#hosting-maia-for-free)
- [Project layout](#project-layout)
- [How a scan is processed](#how-a-scan-is-processed)
- [HTTP API](#http-api)
- [Retraining a model](#retraining-a-model)
- [Known limitations](#known-limitations)
- [Changelog](#changelog)

---

## What's new in 2.1

**Accuracy fix (the big one).** The models were trained on images normalised with the ImageNet mean and standard deviation, but `app.py` never applied that step before inference. Every model was effectively looking at the wrong colours. On a clean, de-duplicated brain MRI test set:

| Model | Before (v2.0.2) | After the fix alone |
|---|---|---|
| Brain Tumor Classification | 30.7% accuracy: called almost everything "meningioma" | 95.6% |
| Brain Tumor Detection | 53.7% balanced accuracy: missed 92% of healthy scans | 93.8% |

On the sample eye images in `uploads/`, the eye model used to call every image "diabetic retinopathy". It now gets the glaucoma and cataract samples right.

**Other bug fixes**
- Label files were looked up under the wrong folder names (`models/brain_tumor_detection/...`), so results showed a class number instead of a name.
- The home page loaded its model list from an admin-only endpoint, so the dropdown was empty unless you were logged in.
- Uploaded filenames weren't sanitised: a crafted name could write outside `uploads/`. Two uploads with the same name also overwrote each other.
- The secret key was hard-coded; the dashboard password can now be set from the environment.
- Non-image files, oversized files and unreadable images now get a clear error instead of a stack trace.

**New look.** Same warm tan palette, typewriter type and ASCII logo. It's tidied into one shared stylesheet:
- Models are grouped by scan type. Each shows a one-line description and a version tag.
- Scans preview right inside the drop zone.
- The result is typed up like a lab slip, with a confidence bar for every class, not just the winner.
- MAIA flags low-confidence results and reminds you to match the model to the scan.
- Dark mode is remembered between visits and follows your system setting by default.
- The layout works on phones, and the result slip prints cleanly.
- The dashboard has proper on/off switches, timestamps and a refresh button.

---

## Quick start

You need Python 3.9 or newer.

```bash
git clone https://github.com/oliverbarnet/MAIA.git
cd MAIA
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app.py
```

Then open <http://localhost:5000> (or `http://<your-ipv4-address>:5000` from another device on the same network).

All seven models load at startup. That takes a few seconds and about 350 MB of RAM.

---

## How to use it

1. **Pick a model** that matches your scan. Models are grouped by scan type: Brain MRI, Chest X-ray, Bone X-ray and Retinal photo.
2. **Add a scan.** Click the box to choose a file, or drag one in. It must be **.png or .jpg**, up to 16 MB. A preview appears in the box.
3. **Analyze scan.** The report on the right shows:
   - the top answer and its confidence
   - a bar for every class the model knows about
   - a note if the model is unsure (top answer below 60%)

Use **Print** for a paper copy of the slip, or **New scan** to start again.

Every model gives an answer for any picture, even a photo of a dog. It only means something if the scan matches the model.

---

## The models

All seven are ResNet-18 image classifiers that take a 224×224 RGB image. They're stored as ONNX in `models/<id>/resnet18.onnx`, with class names in `models/<id>/labels.txt`.

| Model | id | Scan | Classes | Version | Clean test result |
|---|---|---|---|---|---|
| Brain Tumor Detection | `tumor_detection` | Brain MRI | no, yes | V1 | 93.8% balanced acc. |
| Brain Tumor Classification | `tumor_classification` | Brain MRI | glioma, meningioma, none, pituitary | V1 | 95.6% accuracy |
| Alzheimer's Classification | `alzheimers` | Brain MRI | mild, moderate, none, verymild | V2 | not measured |
| COVID-19 Detection | `covid` | Chest X-ray | covid, nocovid | V1 | not measured |
| Pneumonia Detection | `pneumonia` | Chest X-ray | NORMAL, PNEUMONIA | V1 | not measured |
| Fracture Detection | `fractures` | Bone X-ray | fracture, none | V1 | not measured |
| Eye Disease Classification | `EDC` | Retinal photo | cataract, diabetic_retinopathy, glaucoma, normal | V1 | not measured |

"Not measured" means there was no clean held-out test set to check that model against. The datasets for those five live on Kaggle, Mendeley, Zenodo and Figshare. That's covered in [Retraining a model](#retraining-a-model).

<!-- RESULTS -->

---

## Admin dashboard

Go to **Dashboard** in the top bar and log in. From there you can:

- **switch models on or off.** Switched-off models disappear from the scan page. This resets when the server restarts.
- **see the last 15 results:** time, model, file, prediction and confidence.
- **clear the history.**

The default password is `81412`. **Change it** (see below) before you run MAIA anywhere other people can reach it.

---

## Configuration

Everything is optional and set through environment variables:

| Variable | Default | What it does |
|---|---|---|
| `MAIA_PASSWORD` | `81412` | Dashboard password |
| `MAIA_SECRET_KEY` | random on each start | Signs the login cookie. Set it if you want logins to survive a restart. |
| `PORT` | `5000` | Port to listen on |
| `MAIA_MAX_LOADED` | `0` (all) | How many models to keep in memory at once. `0` loads all seven at startup (~600 MB RAM, fastest). `2` loads models on demand and keeps the two most recently used (about 370 MB at peak). Each switch to a model that isn't loaded costs about a third of a second. |
| `MAIA_KEEP_UPLOADS` | `1` | `0` deletes each scan straight after it's analysed. Use this on any public server. |

```bash
MAIA_PASSWORD='something-better' MAIA_SECRET_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')" python3 app.py
```

---

## Hosting MAIA for free

MAIA runs on [Render](https://render.com)'s free plan. The repo already contains the config (`render.yaml`), so setup takes about five minutes. The free plan usually doesn't need a credit card. Render sometimes asks for one as an anti-fraud check, which places a $1 hold that is reversed straight away.

**Why Render?** As of October 2026, Hugging Face Spaces requires a paid plan for Docker and CPU Gradio apps. Render, Koyeb and similar hosts give 512 MB of RAM for free. MAIA normally uses ~600 MB with every model loaded, so `render.yaml` turns on low-memory mode (`MAIA_MAX_LOADED=2`). In a load test (84 scans, 4 at a time, cycling through all seven models, the same gunicorn settings as `render.yaml`), memory peaked at about 370 MB and every request succeeded.

### Steps

1. Sign up at <https://dashboard.render.com/register>. Choose **GitHub** so Render can see your repos.
2. Click **New +** → **Blueprint**.
3. Pick the MAIA repository. If it isn't listed, click **Configure account** and give Render access to it.
4. Set **Branch** to the branch that contains `render.yaml`.
5. Render reads `render.yaml` and shows one service called `maia`. It asks for **`MAIA_PASSWORD`**: type the dashboard password you want. Don't reuse `81412`, because it's in this README.
6. Click **Apply** (or **Deploy Blueprint**). The first build takes a few minutes while it installs packages and pulls the ~300 MB of models.
7. When the status turns **Live**, your site is at `https://maia-xxxx.onrender.com` (shown at the top of the service page).

Every push to that branch redeploys automatically.

### What to expect on the free plan

- **It sleeps.** After 15 minutes with no visitors the service spins down. The next visitor waits about a minute while it wakes up, then it's normal speed.
- **750 free hours a month** per account. That's enough to run one service all month.
- **Memory is reset on restart.** That includes the dashboard history and any models you switched off.
- **Scans are not stored.** `MAIA_KEEP_UPLOADS=0` deletes each image straight after analysis.
- **The first scan with each model is slower**, about a third of a second, because only two models stay loaded. After that it's quick.

To change a setting later: open the service on Render → **Environment** → edit → **Save changes**. Render redeploys on its own.

---

## Project layout

```
MAIA/
├── app.py                  Flask app: routes, model loading, inference
├── requirements.txt        What the web app needs
├── render.yaml             One-click config for free hosting on Render
├── models/
│   └── <id>/
│       ├── resnet18.onnx   The model
│       ├── labels.txt      Class names, one per line, in output order
│       └── *.engine        TensorRT caches for Jetson (optional, ignored by app.py)
├── templates/
│   ├── base.html           Shared header, footer and dark-mode toggle
│   ├── index.html          Scan page
│   ├── info.html           About page
│   ├── login.html          Dashboard login
│   └── dashboard.html      Admin dashboard
├── static/
│   ├── css/maia.css        All the styling, in one place
│   └── images/             Logo / favicon
├── uploads/                Scans people upload (git-ignored, apart from the samples)
└── training/
    ├── prepare_brain_mri.py   Download + de-duplicate + split the brain MRI data
    ├── train.py               Fine-tune ResNet-18 and export to ONNX
    ├── evaluate.py            Score old vs new models on a test set
    ├── requirements.txt       Extra packages for training (PyTorch etc.)
    └── results/               Saved evaluation numbers
```

To **add a new model**: drop a `resnet18.onnx` and `labels.txt` into `models/<new_id>/`, then add an entry to the `MODELS` list at the top of `app.py`. That entry sets the name, scan type, description, version and friendly label names. The page picks it up automatically.

---

## How a scan is processed

1. The upload is checked (extension, size, can it be opened as an image) and saved to `uploads/` under a sanitised name with a short random prefix.
2. The image is converted to RGB and resized straight to 224×224 (bilinear, no crop).
3. Pixel values are scaled to 0–1 and then normalised with the ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. This is the step v2.0 was missing.
4. ONNX Runtime runs the model. Its last layer is a softmax, so the output is already a set of probabilities that add up to 1.
5. Classes are sorted by probability and mapped to friendly names, and the result is returned as JSON and added to the dashboard history.

---

## HTTP API

The web page uses these. You can call them yourself too.

| Method | Path | Auth | What it does |
|---|---|---|---|
| GET | `/models` | none | Models that are currently switched on |
| POST | `/classify` | none | Form fields: `image` (file), `mode` (model id). Returns the prediction. |
| GET | `/history` | dashboard | Last 15 results |
| POST | `/clear_history` | dashboard | Empties the history |
| GET | `/model_status` | dashboard | Every model and whether it's on |
| POST | `/toggle_model/<id>` | dashboard | Switches a model on/off |

```bash
curl -F mode=EDC -F image=@uploads/Glaucoma_014.png http://localhost:5000/classify
```

```json
{
  "model": "Eye Disease Classification",
  "model_id": "EDC",
  "prediction": "Glaucoma",
  "confidence": "86.23%",
  "scores": [
    {"label": "Glaucoma", "raw": "glaucoma", "p": 0.8623},
    {"label": "Cataract", "raw": "cataract", "p": 0.1254},
    {"label": "Diabetic retinopathy", "raw": "diabetic_retinopathy", "p": 0.0075},
    {"label": "Normal", "raw": "normal", "p": 0.0048}
  ],
  "filename": "Glaucoma_014.png",
  "time": "17:17:25"
}
```

---

## Retraining a model

Training happens outside the web app, and only needs doing when you want a better model. It runs on a normal CPU: the brain models took about 40 minutes each on 4 cores. A GPU is much faster.

```bash
pip install -r training/requirements.txt
```

### 1. Get the data

`prepare_brain_mri.py` clones the dataset, cleans it and writes train/val/test folders for both brain tumor models:

```bash
python training/prepare_brain_mri.py --out data
```

**De-duplication matters here.** About 75% of the dataset's official "Testing" images are exact or pixel-identical copies of "Training" images, including every "no tumor" test image. Accuracy measured on the official split therefore mostly measures memory. The script:

1. pools both splits;
2. hashes every image, with md5 for exact copies and a difference hash for resized copies;
3. keeps one copy of each image;
4. makes a fresh stratified split: 15% test, 12% validation, the rest training.

That took the 3,264 images down to 2,867 unique ones. A spot check confirmed every "near duplicate" was pixel-identical to its match: the largest difference was under 1 grey level out of 255, and no pair had different labels.

### 2. Train

```bash
python training/train.py --data data/tumor_classification \
    --labels models/tumor_classification/labels.txt \
    --out runs/tumor_classification --epochs 15
```

What it does:

- **Starting weights:** ImageNet ResNet-18 from timm's "ResNet strikes back" release (`resnet18_a1_0-d63eafa0.pth`), downloaded from GitHub on first run. Pass `--weights` to use a local copy.
- **Augmentation:** kept mild, since scans are fairly standardised: small random crops (75–100%), horizontal flips, ±10° rotation and ±20% brightness/contrast.
- **Imbalance:** the loss uses class weights, so small classes still count. "No tumor" is only 11% of the detection data.
- **Optimiser:** AdamW with a one-cycle learning-rate schedule and label smoothing of 0.1.
- **Which epoch is kept:** the one with the best validation *balanced* accuracy (the average of the per-class recalls).
- **Export:** the model goes to `runs/<task>/resnet18.onnx` with a softmax added and the same `input_0` / `output_0` names as before.

### 3. Check it's actually better

```bash
python training/evaluate.py --data data/tumor_classification/test \
    --labels models/tumor_classification/labels.txt \
    --model old=models/tumor_classification/resnet18.onnx \
    --model new=runs/tumor_classification/resnet18.onnx
```

`evaluate.py` uses exactly the same preprocessing as `app.py`. Only copy the new model into `models/<id>/` if it wins. If you're on a Jetson, also delete the old `*.engine` and `*.sha256sum` files in that folder so TensorRT rebuilds them.

### Retraining the other five models

The same `train.py` / `evaluate.py` work for any ImageFolder-style dataset (`train/<label>/`, `val/<label>/`, `test/<label>/`). The folder names must match the lines in that model's `labels.txt`. Good public sources:

| Model | Dataset | Where |
|---|---|---|
| Pneumonia | Kermany et al. 2018, *Cell*, ~5,900 paediatric chest X-rays, CC BY 4.0 | Mendeley Data, doi:10.17632/rscbjbr9sj |
| COVID-19 | COVID-QU-Ex (Qatar University / Univ. of Dhaka) | Kaggle |
| Fractures | FracAtlas, CC BY 4.0 | Figshare |
| Alzheimer's | OASIS-derived 4-class MRI set | Kaggle (watch for the same patient appearing in both train and test) |
| Eye disease | Cataract / DR / glaucoma / normal fundus set | Kaggle |

These hosts weren't reachable from the environment the 2.1 models were trained in, which is why only the brain models were retrained. Whatever the dataset, run a de-duplication pass like `prepare_brain_mri.py` before trusting a test score.

---

## Known limitations

- **Not clinically validated.** Test scores come from public research datasets. Scans from a different hospital, scanner or patient group will usually score lower.
- **No "not a scan" detection.** Every model always picks one of its classes, even for an unrelated picture.
- **Confidence isn't certainty.** A model can be confidently wrong, especially on unusual images.
- **One slice at a time.** The brain models look at a single 2D MRI slice, not a full 3D study.
- **History and model on/off state are kept in memory** and reset when the server restarts.
- **Uploaded scans are kept on disk** in `uploads/`. Clear the folder yourself if that matters.
- The dashboard is protected by a single shared password. It's fine on your own network, but it isn't real user management.

---

## Changelog

- **2.1.0:** fixed missing input normalisation (large accuracy gain on every model); fixed label loading, public model list, filename safety and secret key; redesigned all pages; added training / evaluation scripts; detailed README.
- **2.0.2:** previous release.
