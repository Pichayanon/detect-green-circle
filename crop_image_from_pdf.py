import cv2
import csv
import os
import re
import warnings
import numpy as np
import pytesseract
from pytesseract import Output

stock_list = []
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

input_dir = 'pdf_page_output'

graph_dir = 'pdf_graph_output'
header_dir = 'pdf_header_output'
csv_dir = "csv"

csv_header_file = os.path.join(csv_dir, "stock_symbols.csv")

os.makedirs(graph_dir, exist_ok=True)
os.makedirs(header_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)


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
        print("EasyOCR conf < 0.9 retrying with greedy")
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


for i in range(2, 350):
    in_path = os.path.join(input_dir, f'page_{i}.png')
    out_path = os.path.join(graph_dir, f'cropped_page_{i}.png')

    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f'NO STOCK SYMBOL AND GRAPH IN IMAGE: {in_path}')
        continue

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = img[y:y + h, x:x + w]
    cv2.imwrite(out_path, cropped)
    print(f'[{i}] GRAPH PATH: {out_path}')

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

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    pad = 25
    height, width = de_noised.shape
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, width)
    y2 = min(y + h + pad, height)

    text_region = de_noised[y1:y2, x1:x2]
    debug_path = os.path.join(header_dir, f'header_page_{i}.png')
    cv2.imwrite(debug_path, text_region)

    text_tess, conf_tess = ocr_with_tesseract(text_region)

    text_tess = re.sub(r'\bl', 'I', text_tess)
    if text_tess.endswith("US") and not text_tess.endswith(" US"):
        if len(text_tess) > 2:
            text_tess = text_tess[:-2].strip() + " US"

    stock_name = text_tess.strip()
    stock_list.append({
        "page": i,
        "stock_name": stock_name,
        "file_saved": f"cropped_page_{i}.png"
    })

    with open(csv_header_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "stock_name", "file_saved"])
        writer.writeheader()
        writer.writerows(stock_list)

    print(f"[{i}] STOCK SYMBOLS: {text_tess} [CONFIDENT {conf_tess:.3f}]")
    print(f"SAVE STOCK SYMBOL TO CSV: {csv_header_file}")
    print()
