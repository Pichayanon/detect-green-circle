import cv2
import os
import warnings
import numpy as np
import pytesseract
from pytesseract import Output

warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

input_dir = 'pdf_pages'
output_dir = 'pdf_images'
debug_dir = 'debug_symbols'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)

#reader = easyocr.Reader(['en'], gpu=True)


def ocr_with_easyocr(image_region):
    def get_text_and_conf(results):
        if not results:
            return "", 0.0
        results = sorted(results, key=lambda r: min(p[0] for p in r[0]))  # left to right
        full_text = " ".join([r[1] for r in results])
        avg_conf = np.mean([r[2] for r in results])
        return full_text.strip(), avg_conf

    result_beam = reader.readtext(image_region, detail=1, decoder='beamsearch')
    text_beam, conf_beam = get_text_and_conf(result_beam)

    if conf_beam < 0.9:
        print("EasyOCR conf < 0.9 → retrying with greedy")
        result_greedy = reader.readtext(image_region, detail=1, decoder='greedy')
        text_greedy, conf_greedy = get_text_and_conf(result_greedy)

        if conf_greedy > conf_beam:
            return text_greedy, conf_greedy
        else:
            return text_beam, conf_beam
    else:
        return text_beam, conf_beam


def ocr_with_tesseract(image_region):
    config = '--psm  7'
    result = pytesseract.image_to_data(image_region, output_type=Output.DICT, config=config)

    texts = []
    confs = []
    for text, conf in zip(result['text'], result['conf']):
        if text.strip() and conf != '-1':
            texts.append(text.strip())
            confs.append(float(conf))

    if texts:
        full_text = " ".join(texts)
        avg_conf = np.mean(confs) / 100.0
        return full_text.strip(), avg_conf
    else:
        return "", 0.0


# ---------------- MAIN LOOP -------------------
for i in range(182, 186):
    in_path = os.path.join(input_dir, f'page_{i}.png')
    out_path = os.path.join(output_dir, f'cropped_page_{i}.png')

    img = cv2.imread(in_path)
    if img is None:
        print(f'File not found: {in_path}')
        continue

    # ==== Crop Main Graph ====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f'No content found: {in_path}')
        continue

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = img[y:y + h, x:x + w]
    cv2.imwrite(out_path, cropped)
    print(f'Saved: {out_path}')

    # ==== Process Header ====
    header_crop = img[0:y, :]
    scale_factor = 4.75
    header_zoom = cv2.resize(header_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray_header = cv2.cvtColor(header_zoom, cv2.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9.5, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(gray_header, -1, sharpen_kernel)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5, 5))
    contrast = clahe.apply(sharpened)

    de_noised = cv2.bilateralFilter(contrast, d=9, sigmaColor=75, sigmaSpace=75)

    _, binary = cv2.threshold(de_noised, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"[{i}]No header text region found.")
        continue

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    pad = 25
    height, width = de_noised.shape
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, width)
    y2 = min(y + h + pad, height)

    text_region = de_noised[y1:y2, x1:x2]
    debug_path = os.path.join(debug_dir, f'header_page_{i}.png')
    cv2.imwrite(debug_path, text_region)

    # ==== OCR EasyOCR vs Tesseract ====
    # text_easy, conf_easy = ocr_with_easyocr(text_region)
    text_tess, conf_tess = ocr_with_tesseract(text_region)

    text_tess = text_tess.replace('l', 'I')
    if text_tess.endswith("US") and not text_tess.endswith(" US"):
        if len(text_tess) > 2:
            text_tess = text_tess[:-2].strip() + " US"
    # if text_tess.startswith("/"):
    #     text_tess = "7" + text_tess[1:]

    print(f"[{i}] Tesseract: {text_tess} (conf: {conf_tess:.3f})")
    print()
