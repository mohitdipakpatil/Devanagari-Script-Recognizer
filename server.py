import io
import os
import base64
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Lazy import to reduce cold start cost if TF isn't needed for index
import tensorflow as tf
from docx import Document

APP_TITLE = "Devanagari Character Classifier"
MODEL_PATH = os.environ.get("MODEL_PATH", "devnagri_model.h5")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "classes.npy")

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend from ./static
if not os.path.isdir("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

model: Optional[tf.keras.Model] = None
classes: Optional[np.ndarray] = None


def load_model_and_classes() -> None:
    global model, classes
    if model is None:
        # Only accept the explicit path (defaults to devnagri_model.h5)
        path = MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at '{path}'. Place 'devnagri_model.h5' in the project root or set $env:MODEL_PATH to its full path."
            )
        model = tf.keras.models.load_model(path)
    if classes is None:
        if not os.path.exists(CLASSES_PATH):
            # Try reconstruct from joblib if available
            le_path = "devanagari_label_encoder.joblib"
            if os.path.exists(le_path):
                try:
                    import joblib  # type: ignore
                    le = joblib.load(le_path)
                    np.save(CLASSES_PATH, le.classes_)
                except Exception:
                    pass
        if os.path.exists(CLASSES_PATH):
            classes = np.load(CLASSES_PATH, allow_pickle=True)
            # Map known label keys to Devanagari glyphs if needed
            map_dict = {
                'character_01_ka': 'क','character_02_kha': 'ख','character_03_ga': 'ग','character_04_gha': 'घ',
                'character_05_kna': 'ङ','character_06_cha': 'च','character_07_chha': 'छ','character_08_ja': 'ज',
                'character_09_jha': 'झ','character_10_yna': 'ञ','character_11_taamatar': 'ट','character_12_thaa': 'ठ',
                'character_13_daa': 'ड','character_14_dhaa': 'ढ','character_15_adna': 'ण','character_16_tabala': 'त',
                'character_17_tha': 'थ','character_18_da': 'द','character_19_dha': 'ध','character_20_na': 'न',
                'character_21_pa': 'प','character_22_pha': 'फ','character_23_ba': 'ब','character_24_bha': 'भ',
                'character_25_ma': 'म','character_26_yaw': 'य','character_27_ra': 'र','character_28_la': 'ल',
                'character_29_waw': 'व','character_30_motosaw': 'श','character_31_petchiryakha': 'ष','character_32_patalosaw': 'स',
                'character_33_ha': 'ह','character_34_chhya': 'क्ष','character_35_tra': 'त्र','character_36_gya': 'ज्ञ',
                'digit_0':'०','digit_1':'१','digit_2':'२','digit_3':'३','digit_4':'४','digit_5':'५',
                'digit_6':'६','digit_7':'७','digit_8':'८','digit_9':'९'
            }
            try:
                mapped = []
                for v in classes:
                    s = v.item() if hasattr(v, 'item') else v
                    mapped.append(map_dict.get(s, s))
                classes = np.array(mapped, dtype=object)
            except Exception:
                # If anything goes wrong, keep original classes
                pass
        else:
            raise FileNotFoundError(
                "Classes file not found at '" + CLASSES_PATH + "'. Save label classes via np.save('classes.npy', label_encoder.classes_)."
            )


def _label_to_glyph(s: str) -> str:
    """Best-effort mapping from dataset label keys to Devanagari glyphs."""
    map_dict = {
        'character_01_ka': 'क','character_02_kha': 'ख','character_03_ga': 'ग','character_04_gha': 'घ',
        'character_05_kna': 'ङ','character_06_cha': 'च','character_07_chha': 'छ','character_08_ja': 'ज',
        'character_09_jha': 'झ','character_10_yna': 'ञ','character_11_taamatar': 'ट','character_12_thaa': 'ठ',
        'character_13_daa': 'ड','character_14_dhaa': 'ढ','character_15_adna': 'ण','character_16_tabala': 'त',
        'character_17_tha': 'थ','character_18_da': 'द','character_19_dha': 'ध','character_20_na': 'न',
        'character_21_pa': 'प','character_22_pha': 'फ','character_23_ba': 'ब','character_24_bha': 'भ',
        'character_25_ma': 'म','character_26_yaw': 'य','character_27_ra': 'र','character_28_la': 'ल',
        'character_29_waw': 'व','character_30_motosaw': 'श','character_31_petchiryakha': 'ष','character_32_patalosaw': 'स',
        'character_33_ha': 'ह','character_34_chhya': 'क्ष','character_35_tra': 'त्र','character_36_gya': 'ज्ञ',
        'digit_0':'०','digit_1':'१','digit_2':'२','digit_3':'३','digit_4':'४','digit_5':'५',
        'digit_6':'६','digit_7':'७','digit_8':'८','digit_9':'९'
    }
    return map_dict.get(s, s)


def preprocess_image_to_tensor(img: Image.Image) -> np.ndarray:
    # 1) Convert to grayscale
    img_gray = img.convert("L")
    arr = np.asarray(img_gray, dtype=np.uint8)

    # 2) Keep training-like polarity by default (black strokes on light background)
    # If your canvas produces white strokes on black, flip here; otherwise leave as-is
    # Toggle via env: INVERT_INPUT=1 to force inversion
    invert_env = os.environ.get("INVERT_INPUT", "0").strip()
    force_invert = invert_env in ("1", "true", "True")
    if force_invert:
        arr = 255 - arr

    # 3) Binarize using Otsu threshold to stabilize stroke/background separation
    hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, thresh = 0.0, 0.0, 0.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    bin_fg = (arr >= thresh).astype(np.uint8)  # foreground where bright

    # 4) Crop to the bounding box of the foreground
    coords = np.argwhere(bin_fg > 0)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        arr_cropped = arr[y0:y1, x0:x1]
    else:
        arr_cropped = arr  # fallback if nothing detected

    # 4.5) Deskew using the principal axis of the foreground
    coords_c = np.argwhere((arr_cropped > 0))
    if coords_c.size > 0:
        # Compute covariance eigenvectors
        coords_f = coords_c.astype(np.float32)
        coords_f -= coords_f.mean(axis=0, keepdims=True)
        cov = np.cov(coords_f.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        # Principal direction
        principal = eigvecs[:, int(np.argmax(eigvals))]
        angle_rad = np.arctan2(principal[0], principal[1])  # rows vs cols
        angle_deg = np.degrees(angle_rad)
        # Limit rotation to reasonable range
        if -25 <= angle_deg <= 25:
            img_deskew = Image.fromarray(arr_cropped).rotate(-angle_deg, resample=Image.BILINEAR, expand=True, fillcolor=0)
            arr_cropped = np.asarray(img_deskew, dtype=np.uint8)

    # 5) Optional morphology to normalize stroke thickness
    fg_ratio = float((arr_cropped > 0).sum()) / float(arr_cropped.size)
    pil_tmp = Image.fromarray(arr_cropped)
    if fg_ratio < 0.02:
        # Very thin strokes → dilate slightly
        pil_tmp = pil_tmp.filter(ImageFilter.MaxFilter(3))
        arr_cropped = np.asarray(pil_tmp, dtype=np.uint8)
    elif fg_ratio > 0.35:
        # Very thick strokes → erode slightly
        pil_tmp = pil_tmp.filter(ImageFilter.MinFilter(3))
        arr_cropped = np.asarray(pil_tmp, dtype=np.uint8)

    # 6) Pad to square with margins, then resize to 32x32
    h, w = arr_cropped.shape
    side = max(h, w)
    margin = max(2, side // 12)  # small margin around glyph
    side_with_margin = side + 2 * margin
    canvas = np.zeros((side_with_margin, side_with_margin), dtype=np.uint8)
    # center the cropped glyph
    y_off = (side_with_margin - h) // 2
    x_off = (side_with_margin - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = arr_cropped

    img_square = Image.fromarray(canvas)
    img_resized = img_square.resize((32, 32), Image.BILINEAR)

    # 7) Normalize to [0,1] and add channel and batch dims → (1, 32, 32, 1)
    arr_f = np.asarray(img_resized, dtype=np.float32) / 255.0
    arr_f = arr_f.reshape(1, 32, 32, 1)
    return arr_f


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
) -> JSONResponse:
    load_model_and_classes()

    if file is None and not image_base64:
        return JSONResponse({"error": "Provide an image file or base64 payload."}, status_code=400)

    try:
        if file is not None:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        else:
            # Expect data URL or raw base64
            data = image_base64
            if "," in data:
                data = data.split(",", 1)[1]
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return JSONResponse({"error": "Invalid image payload."}, status_code=400)

    tensor = preprocess_image_to_tensor(image)
    probs = model.predict(tensor, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    raw = classes[top_idx].item() if hasattr(classes[top_idx], "item") else classes[top_idx]
    label = _label_to_glyph(str(raw))

    # Also return top-5 for transparency
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [
        {
            "label": _label_to_glyph(str(classes[i].item() if hasattr(classes[i], "item") else classes[i])),
            "probability": float(probs[i]),
        }
        for i in top5_idx
    ]

    return JSONResponse(
        {
            "prediction": label,
            "probability": top_prob,
            "top5": top5,
        }
    )


# --- Simple segmentation-based OCR for words/lines ---
def _binarize(arr: np.ndarray) -> np.ndarray:
    hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, thresh = 0.0, 0.0, 0.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return (arr >= thresh).astype(np.uint8)


def segment_characters(img: Image.Image) -> list[Image.Image]:
    # Pre-binarize consistently with single-character path (invert if needed)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    if gray.mean() > 128:
        gray = 255 - gray
    bin_fg = _binarize(gray)
    # Ensure strokes are foreground (foreground should be minority of pixels)
    if bin_fg.mean() > 0.5:
        bin_fg = 1 - bin_fg

    # Horizontal projection: keep dominant text rows (optional line cropping)
    rows = bin_fg.sum(axis=1)
    if rows.max() > 0:
        row_idx = np.where(rows > 0)[0]
        top, bottom = row_idx.min(), row_idx.max() + 1
        bin_fg = bin_fg[top:bottom, :]
        gray = gray[top:bottom, :]

    # Remove Devanagari headline (shirorekha): zero out very full rows
    seg_mask = bin_fg.copy()
    width = seg_mask.shape[1]
    heavy_rows = np.where((seg_mask.sum(axis=1) / max(1, width)) > 0.7)[0]
    for r in heavy_rows:
        r0 = max(0, r - 1)
        r1 = min(seg_mask.shape[0], r + 2)
        seg_mask[r0:r1, :] = 0

    # Vertical projection: find gaps to split into characters on headline-removed mask
    cols = seg_mask.sum(axis=0)
    # Identify sequences of zero columns as gaps
    gaps = []
    start = None
    for i, v in enumerate(cols):
        if v == 0 and start is None:
            start = i
        elif v > 0 and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, len(cols)))

    gap_widths = [b - a for a, b in gaps] or [0]
    median_gap = float(np.median(gap_widths)) if gap_widths else 0.0
    split_threshold = max(2, int(median_gap * 1.5))

    # Walk across columns, carve runs of non-zero separated by big gaps
    char_boxes = []
    in_run = False
    run_start = 0
    i = 0
    while i < len(cols):
        if cols[i] > 0 and not in_run:
            in_run = True
            run_start = i
        if in_run:
            # Count consecutive zero columns from here
            j = i
            zero_count = 0
            while j < len(cols) and cols[j] == 0:
                zero_count += 1
                if zero_count >= split_threshold:
                    in_run = False
                    char_boxes.append((run_start, j - zero_count + 1))
                    i = j  # continue after gap
                    break
                j += 1
            else:
                i += 1
                continue
        else:
            i += 1
    if in_run:
        char_boxes.append((run_start, len(cols)))

    # Connected-components fallback to better handle touching glyphs
    def cc_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        boxes: list[tuple[int, int, int, int]] = []
        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0 or visited[y, x]:
                    continue
                # BFS
                stack = [(y, x)]
                visited[y, x] = True
                y_min, y_max, x_min, x_max = y, y, x, x
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    count += 1
                    if cy < y_min: y_min = cy
                    if cy > y_max: y_max = cy
                    if cx < x_min: x_min = cx
                    if cx > x_max: x_max = cx
                    for ny in (cy-1, cy, cy+1):
                        for nx in (cx-1, cx, cx+1):
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] == 1:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                # Filter tiny noise
                if count >= max(10, (h*w)//2000):
                    boxes.append((y_min, y_max+1, x_min, x_max+1))
        # Merge boxes that are extremely close (broken parts)
        boxes.sort(key=lambda b: b[2])
        merged: list[tuple[int,int,int,int]] = []
        for b in boxes:
            if not merged:
                merged.append(b)
                continue
            y0,y1,x0,x1 = b
            py0,py1,px0,px1 = merged[-1]
            if x0 - px1 <= 2:  # close horizontally
                merged[-1] = (min(py0,y0), max(py1,y1), px0, max(px1,x1))
            else:
                merged.append(b)
        return merged

    if len(char_boxes) <= 1:
        # Use connected components when projection is not reliable
        comps = cc_boxes(seg_mask)
        char_boxes = [(x0, x1) for (y0, y1, x0, x1) in comps]

    # Extract crops per box, tighten vertically, convert to PIL
    chars: list[Image.Image] = []
    for x0, x1 in char_boxes:
        crop_bin = seg_mask[:, x0:x1]
        crop_gray = gray[:, x0:x1]
        if crop_bin.sum() == 0:
            continue
        r = crop_bin.sum(axis=1)
        r_idx = np.where(r > 0)[0]
        y0, y1 = int(r_idx.min()), int(r_idx.max() + 1)
        crop = crop_gray[y0:y1, :]
        if crop.size == 0:
            continue
        chars.append(Image.fromarray(crop))
    # Fallback: if we failed to segment, treat entire image as one character/word chunk
    if not chars:
        chars = [Image.fromarray(gray)]
    return chars


@app.post("/predict_text")
async def predict_text(
    file: UploadFile | None = File(default=None),
    image_base64: str | None = Form(default=None),
) -> JSONResponse:
    load_model_and_classes()

    if file is None and not image_base64:
        return JSONResponse({"error": "Provide an image file or base64 payload."}, status_code=400)

    # Load image
    try:
        if file is not None:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
        else:
            data = image_base64
            if "," in data:
                data = data.split(",", 1)[1]
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return JSONResponse({"error": "Invalid image payload."}, status_code=400)

    # Segment
    char_images = segment_characters(image)
    if not char_images:
        return JSONResponse({"text": "", "characters": []})

    # Predict each character using existing preprocessing path
    predicted = []
    text_out = []
    # Compute gap-based spacing
    boxes_for_gap = []
    # Recreate boxes for spacing (x ranges)
    tmp_mask = _binarize(np.asarray(image.convert("L"), dtype=np.uint8))
    if tmp_mask.mean() > 0.5:
        tmp_mask = 1 - tmp_mask
    # Use segmentation mask pipeline to get final x boxes
    boxes_for_gap = []
    seg = segment_characters(image)
    # segment_characters returns crops already sorted; approximate widths from crops
    widths = [c.size[0] for c in seg]
    median_w = float(np.median(widths)) if widths else 0.0

    prev_right = None
    for idx_img, ch_img in enumerate(char_images):
        tensor = preprocess_image_to_tensor(ch_img)
        probs = model.predict(tensor, verbose=0)[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        raw = classes[idx].item() if hasattr(classes[idx], "item") else classes[idx]
        label = _label_to_glyph(str(raw))
        # Insert space based on large gaps between components
        if idx_img > 0 and median_w > 0:
            left_gap = 0
            try:
                left_gap = (char_images[idx_img].size[0])  # placeholder per-crop width
            except Exception:
                left_gap = 0
            # Heuristic: if the gap between components exceeds 0.8 * median width, add space
            if left_gap > 0.8 * median_w:
                text_out.append(" ")
        text_out.append(label)
        predicted.append({"label": label, "probability": prob})

    return JSONResponse({"text": "".join(text_out), "characters": predicted})


@app.post("/text_to_docx")
async def text_to_docx(text: str = Form(...)) -> StreamingResponse:
    # Create a DOCX document in memory
    doc = Document()
    # Use a Unicode-capable default font; Word will render Devanagari if available on system
    doc.add_paragraph(text)
    mem = io.BytesIO()
    doc.save(mem)
    mem.seek(0)
    headers = {
        "Content-Disposition": "attachment; filename=prediction.docx"
    }
    return StreamingResponse(
        mem,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )


# Entry point for `python server.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

