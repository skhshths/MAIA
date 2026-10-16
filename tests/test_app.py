"""
Quick checks for the web app. Run from the repo root:
    pip install pytest
    python -m pytest tests
"""
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as maia  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(maia.__file__), "uploads", "Glaucoma_014.png")


@pytest.fixture
def client(tmp_path):
    maia.app.config["TESTING"] = True
    maia.app.config["UPLOAD_FOLDER"] = str(tmp_path)
    for k in maia.model_enabled:
        maia.model_enabled[k] = True
    maia.history.clear()
    with maia.app.test_client() as c:
        yield c


def png_bytes(color=(120, 60, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    buf.seek(0)
    return buf


def login(client):
    return client.post("/login", data={"password": maia.DASHBOARD_PASSWORD})


def test_pages_load(client):
    for path in ["/", "/info", "/login"]:
        assert client.get(path).status_code == 200


def test_every_model_has_labels_matching_its_output():
    for m in maia.MODELS:
        out = maia.get_session(m["id"]).get_outputs()[0].shape[-1]
        assert len(m["labels"]) == out, m["id"]
        assert set(m["pretty"]) == set(m["labels"]), m["id"]


def test_preprocess_is_imagenet_normalised(tmp_path):
    p = tmp_path / "grey.png"
    Image.new("RGB", (300, 200), (124, 116, 104)).save(p)  # roughly the ImageNet mean
    x = maia.preprocess_image(str(p))
    assert x.shape == (1, 3, 224, 224)
    assert np.abs(x).max() < 0.05


def test_public_model_list_needs_no_login(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert {m["id"] for m in r.get_json()} == set(maia.nets)


def test_classify_returns_named_scores(client):
    r = client.post("/classify", data={"mode": "tumor_classification", "image": (png_bytes(), "scan.png")})
    data = r.get_json()
    assert r.status_code == 200
    assert data["prediction"] in maia.nets["tumor_classification"]["pretty"].values()
    assert len(data["scores"]) == 4
    assert abs(sum(s["p"] for s in data["scores"]) - 1) < 1e-3
    assert data["scores"] == sorted(data["scores"], key=lambda s: -s["p"])


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample image not present")
def test_sample_glaucoma_image_is_recognised(client):
    with open(SAMPLE, "rb") as f:
        r = client.post("/classify", data={"mode": "EDC", "image": (f, "Glaucoma_014.png")})
    assert r.get_json()["prediction"] == "Glaucoma"


def test_rejects_bad_requests(client):
    assert client.post("/classify", data={"mode": "nope", "image": (png_bytes(), "a.png")}).status_code == 400
    assert client.post("/classify", data={"mode": "EDC"}).status_code == 400
    assert client.post("/classify", data={"mode": "EDC", "image": (io.BytesIO(b"hello"), "a.txt")}).status_code == 400
    assert client.post("/classify", data={"mode": "EDC", "image": (io.BytesIO(b"not really"), "a.png")}).status_code == 400


def test_upload_name_cannot_escape_folder(client, tmp_path):
    r = client.post("/classify", data={"mode": "EDC", "image": (png_bytes(), "../../evil.png")})
    assert r.status_code == 200
    saved = os.listdir(tmp_path)
    if maia.KEEP_UPLOADS:
        assert len(saved) == 1 and saved[0].endswith("evil.png") and "/" not in saved[0]
    else:
        assert saved == []  # deleted straight after analysis
    assert not os.path.exists(os.path.join(tmp_path, "..", "..", "evil.png"))


def test_admin_routes_need_login(client):
    assert client.get("/history").status_code == 403
    assert client.get("/model_status").status_code == 403
    assert client.post("/toggle_model/EDC").status_code == 403
    assert client.get("/dashboard").status_code == 302


def test_wrong_password_is_rejected(client):
    r = client.post("/login", data={"password": "wrong"})
    assert r.status_code == 200 and b"didn" in r.data
    assert client.get("/history").status_code == 403


def test_toggle_hides_model_and_blocks_it(client):
    login(client)
    assert client.post("/toggle_model/covid").get_json()["enabled"] is False
    assert "covid" not in {m["id"] for m in client.get("/models").get_json()}
    r = client.post("/classify", data={"mode": "covid", "image": (png_bytes(), "x.png")})
    assert r.status_code == 400
    client.post("/toggle_model/covid")


def test_history_records_results(client):
    login(client)
    client.post("/classify", data={"mode": "EDC", "image": (png_bytes(), "x.png")})
    hist = client.get("/history").get_json()
    assert len(hist) == 1 and hist[0]["model_id"] == "EDC"
    client.post("/clear_history")
    assert client.get("/history").get_json() == []
