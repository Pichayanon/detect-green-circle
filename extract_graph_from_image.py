import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pytesseract
import easyocr
import time
import warnings
import os
import re

warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
start_time = time.time()
MONTH_ABBR = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


# ------------------- Utility Functions -------------------
def convert_bgr_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def crop_image(image, crop_top, crop_bottom, crop_left, crop_right):
    return image[crop_top:image.shape[0] - crop_bottom, crop_left:image.shape[1] - crop_right]


def mask_white_area(hsv_image, lower_white, upper_white):
    mask_white_image = cv2.inRange(hsv_image, lower_white, upper_white)
    mask_white_image = cv2.morphologyEx(mask_white_image, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask_white_image


# ------------------- Detect Lines -------------------
def detect_lines(mask_image, min_length=150, max_gap=30):
    edges = cv2.Canny(mask_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=min_length, maxLineGap=max_gap)
    return lines


def find_x_axis(lines, img_width):
    horizontal = []
    x_axis = None

    if lines is not None:
        min_length = img_width * 0.5
        max_length = img_width * 0.9
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 10 or angle > 170:
                horizontal.append((length, (x1, y1, x2, y2)))
        if not horizontal:
            return x_axis

        horizontal = sorted(horizontal, reverse=True)
        if len(horizontal) > 1:
            length_horizontal = horizontal[1][0]
            if min_length < length_horizontal < max_length:
                x_axis = horizontal[1][1]
        else:
            length_horizontal = horizontal[1][0]
            if min_length < length_horizontal < max_length:
                x_axis = horizontal[1][1]
    return x_axis


def find_y_axis(lines, x_axis):
    vertical = []
    y_axis = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = np.sqrt(dx ** 2 + dy ** 2)
            angle = np.degrees(np.arctan2(dy, dx))

            if 80 < angle < 100:
                vertical.append((length, (x1, y1, x2, y2)))
            vertical = sorted(vertical, reverse=True)

            vertical_left = []
            if len(vertical) > 0:
                x_axis_left = x_axis[0]
                for length, (x1, y1, x2, y2) in vertical:
                    if max(x1, x2) < x_axis_left + 10:
                        vertical_left.append((length, (x1, y1, x2, y2)))
            if vertical_left:
                y_axis = max(vertical_left, key=lambda t: t[0])[1]
    return y_axis


# ------------------- Find x minor ticks -------------------
def find_x_minor_ticks(mask_image, x_axis_left, x_axis_top, x_axis_right, x_axis_bot,
                       pad_left=5, pad_right=5, strip_height=12):
    strip_image = mask_image[
                  max(0, int((x_axis_top + x_axis_bot) / 2) + 12 - strip_height // 2):
                  min(mask_image.shape[0], int((x_axis_top + x_axis_bot) / 2) + 12 + strip_height // 2),
                  max(0, min(x_axis_left, x_axis_right) - pad_left):
                  min(mask_image.shape[1], max(x_axis_left, x_axis_right) + pad_right)
                  ]

    proj = np.sum(strip_image, axis=0)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    peaks, _ = find_peaks(proj, height=0.18, distance=10)
    minor_tick_coords = []
    for peak in peaks:
        tx = max(0, min(x_axis_left, x_axis_right) - pad_left) + peak
        ty = int((x_axis_top + x_axis_bot) / 2)
        if x_axis_left + 10 < tx < x_axis_right - 10:
            minor_tick_coords.append((tx, ty))

    return sorted(minor_tick_coords, key=lambda t: t[0])


def auto_add_x_minor_ticks(minor_tick_coords, x_axis_left, x_axis_top, x_axis_right, x_axis_bot
                           , y_axis_left, y_axis_right):
    if len(minor_tick_coords) >= 2:
        steps = [minor_tick_coords[i + 1][0] - minor_tick_coords[i][0] for i in range(len(minor_tick_coords) - 1)]
        avg_step = int(np.median(steps))

        right_x = minor_tick_coords[-1][0]
        while right_x + avg_step < max(x_axis_left, x_axis_right) + 50:
            right_x = right_x + avg_step
            minor_tick_coords.append((right_x, int((x_axis_top + x_axis_bot) / 2)))

        left_x = minor_tick_coords[0][0]
        while left_x - avg_step > min(x_axis_left, x_axis_right) - 50:
            left_x = left_x - avg_step
            minor_tick_coords.append((left_x, int((x_axis_top + x_axis_bot) / 2)))

    return minor_tick_coords


# ------------------- OCR X Labels -------------------
def ocr_with_pytesseract(image):
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)


def ocr_with_easyocr(reader, image):
    results = reader.readtext(image, detail=1)
    detected = []
    for (bbox, text, conf) in results:
        text_clean = text.strip().replace(' ', '')
        if text_clean.isdigit() and conf > 0.6:
            left = int(bbox[0][0])
            right = int(bbox[1][0])
            center_x = (left + right) // 2
            detected.append((text_clean, center_x))
    return detected


def ocr_month_with_easyocr(reader, image):
    results = reader.readtext(image, detail=1)
    detected = []
    for (bbox, text, conf) in results:
        text_clean = text.strip().replace(' ', '').upper()
        if text_clean in MONTH_ABBR and conf > 0.5:
            left = int(bbox[0][0])
            right = int(bbox[1][0])
            center_x = (left + right) // 2
            detected.append((text_clean, center_x))
    return detected


def is_year_x_label_valid_pair(labels):
    try:
        years = [int(lbl) for lbl in labels if lbl.isdigit()]
        if len(years) < 2:
            return False
        num_pairs = len(years) // 2
        if num_pairs == 0:
            return False

        first_diff = years[1] - years[0]
        for i in range(1, num_pairs):
            diff = years[2 * i + 1] - years[2 * i]
            if diff != first_diff:
                return False
        return True
    except Exception:
        return False


def is_year_x_label_valid_range(gaps):
    if not gaps:
        return False
    if len(gaps) == 1:
        return True
    if len(gaps) <= 3 and not (all(g == 1 for g in gaps)):
        return False
    if all(g == 1 for g in gaps):
        return True
    if all((g == 1 if i % 2 == 0 else g != 1) for i, g in enumerate(gaps)):
        return True
    if all((g != 1 if i % 2 == 0 else g == 1) for i, g in enumerate(gaps)):
        return True
    return False


def year_label_gaps(labels):
    if len(labels) == 1:
        return [1]
    years = [int(lbl) for lbl in labels]
    return [years[i + 1] - years[i] for i in range(len(years) - 1)]


def detect_labels_on_x_axis_pytesseract(image, strip_parm, x_axis_left, x_axis_right):
    labels, label_x = [], []
    day = False
    for top, bot in strip_parm:
        label_strip = image[top:bot, min(x_axis_left, x_axis_right):max(x_axis_left, x_axis_right)]
        label_strip_gray = cv2.cvtColor(label_strip, cv2.COLOR_BGR2GRAY)
        ocr_result = ocr_with_pytesseract(label_strip_gray)
        labels_temp, label_x_temp = [], []
        for i, text in enumerate(ocr_result['text']):
            if text.strip().isdigit() and int(ocr_result['conf'][i]) > 60:
                left = ocr_result['left'][i] - 30
                width = ocr_result['width'][i] + 10
                center_x = left + width // 2
                labels_temp.append(text)
                label_x_temp.append(center_x)
        if labels_temp and not (is_year_x_label_valid_range(year_label_gaps(labels_temp))):
            continue
        if labels_temp and any(len(label) == 2 for label in labels_temp):
            day = False
            continue
        if labels_temp:
            return labels_temp, label_x_temp, day, (top, bot)
    return [], [], day, None


def detect_labels_on_x_axis_easyocr(image, strip_parm, x_axis_left, x_axis_right, reader):
    labels, label_x = [], []
    day = False
    for top, bot in strip_parm:
        label_strip = image[top:bot, min(x_axis_left, x_axis_right):max(x_axis_left, x_axis_right)]
        easy_labels = ocr_with_easyocr(reader, label_strip)
        easy_label_vals, easy_label_x = zip(*easy_labels) if easy_labels else ([], [])
        if easy_labels and not (is_year_x_label_valid_range(year_label_gaps(list(easy_label_vals)))):
            continue
        if easy_labels and any(len(label) == 2 for label in easy_label_vals):
            day = False
            continue
        if easy_labels:
            return list(easy_label_vals), list(easy_label_x), day, (top, bot)
    return [], [], day, None


def detect_labels_on_x_axis(image, strip_parm, x_axis_left, x_axis_right, reader):
    labels, label_x, day, rng = detect_labels_on_x_axis_pytesseract(
        image, strip_parm, x_axis_left, x_axis_right)
    easy_labels, easy_label_x, easy_day, easy_rng = None, None, None, None
    if len(labels) <= 3:
        easy_labels, easy_label_x, easy_day, easy_rng = detect_labels_on_x_axis_easyocr(
            image, strip_parm, x_axis_left, x_axis_right, reader)
        if len(labels) < len(easy_labels):
            labels, label_x, day, rng = easy_labels, easy_label_x, easy_day, easy_rng
    return labels, label_x, day, rng


def detect_month_labels_on_x_axis_pytesseract(image, strip_parm, x_axis_left, x_axis_right):
    months, months_x = [], []
    for top, bot in strip_parm:
        label_strip = image[top:bot, min(x_axis_left, x_axis_right):max(x_axis_left, x_axis_right)]
        label_strip_gray = cv2.cvtColor(label_strip, cv2.COLOR_BGR2GRAY)
        ocr_result = ocr_with_pytesseract(label_strip_gray)
        for i, text in enumerate(ocr_result['text']):
            text_clean = text.strip().upper()
            if text_clean in MONTH_ABBR and int(ocr_result['conf'][i]) > 70:
                left = ocr_result['left'][i] - 10
                width = ocr_result['width'][i] + 10
                center_x = left + width // 2

                already = False
                for m, x in zip(months, months_x):
                    if m == text_clean and abs(x - center_x) < 15:
                        already = True
                        break
                if not already:
                    months.append(text_clean)
                    months_x.append(center_x)
    return months, months_x


def detect_month_labels_on_x_axis_easyocr(image, strip_parm, x_axis_left, x_axis_right, reader):
    months, months_x = [], []
    for top, bot in strip_parm:
        label_strip = image[top:bot, min(x_axis_left, x_axis_right):max(x_axis_left, x_axis_right)]
        easy_labels = ocr_month_with_easyocr(reader, label_strip)
        for month, cx in easy_labels:
            already = False
            for m, x in zip(months, months_x):
                if m == month and abs(x - cx) < 15:
                    already = True
                    break
            if not already:
                months.append(month)
                months_x.append(cx)
    return months, months_x


def detect_month_labels_on_x_axis(image, strip_parm, x_axis_left, x_axis_right, reader):
    # Try pytesseract first
    months, months_x = detect_month_labels_on_x_axis_pytesseract(
        image, strip_parm, x_axis_left, x_axis_right)
    if not months:
        # fallback to EasyOCR
        months, months_x = detect_month_labels_on_x_axis_easyocr(
            image, strip_parm, x_axis_left, x_axis_right, reader)
    return months, months_x


def auto_fill_missing_months(month_labels, months_x):
    if not month_labels or not months_x or len(month_labels) != len(months_x):
        return month_labels, months_x

    zipped = list(zip(months_x, month_labels))
    zipped_sorted = sorted(zipped, key=lambda t: t[0])
    sorted_x, sorted_labels = zip(*zipped_sorted)
    sorted_x = list(sorted_x)
    sorted_labels = list(sorted_labels)

    filled_labels = [sorted_labels[0]]
    filled_x = [sorted_x[0]]
    prev_idx = MONTH_ABBR.index(sorted_labels[0])
    prev_x = sorted_x[0]

    for i in range(1, len(sorted_labels)):
        curr_label = sorted_labels[i]
        curr_x = sorted_x[i]
        curr_idx = MONTH_ABBR.index(curr_label)
        num_missing = (curr_idx - prev_idx - 1) % 12
        if num_missing > 0:
            step = (curr_x - prev_x) / (num_missing + 1)
            for j in range(num_missing):
                missing_idx = (prev_idx + j + 1) % 12
                filled_labels.append(MONTH_ABBR[missing_idx])
                filled_x.append(int(prev_x + step * (j + 1)))
        filled_labels.append(curr_label)
        filled_x.append(curr_x)
        prev_idx = curr_idx
        prev_x = curr_x

    return filled_labels, filled_x


# ------------------- Find x major ticks -------------------
def count_consecutive_pairs(years):
    if not years:
        return 0
    years = sorted([int(y) for y in years])
    count = 0
    for i in range(1, len(years)):
        if years[i] == years[i - 1] + 1:
            count += 1
    return count


def find_x_major_ticks(image, strip_heights_parm, x_axis_left, x_axis_right, x_axis_top, x_axis_bot,
                       minor_tick_x_coords, labels, x_label, minor_tick_coords, months,
                       pad_left=100, pad_right=12, match_threshold=10):
    minor_ticks_x = [tx for tx, ty in minor_tick_x_coords]

    if len(months) > 0:
        offset_list = [80, 75, 70, 65]
    else:
        offset_list = [75]

    for strip_height_major in strip_heights_parm:
        for y_offset in offset_list:
            y_base_major = (x_axis_top + x_axis_bot) / 2
            y_base_major = int(y_base_major)
            if len(months) > 0:
                y_strip_top_major = y_base_major + y_offset
            else:
                y_strip_top_major = y_base_major + 25

            y_strip_bot_major = y_strip_top_major + strip_height_major
            strip_img = image[
                        max(y_strip_top_major, 0):min(y_strip_bot_major, image.shape[0]),
                        max(0, min(x_axis_left, x_axis_right) - pad_left):min(image.shape[1], max(x_axis_left,  x_axis_right) + pad_right)
                        ]
            cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_test/axis_{strip_height_major}.png", strip_img)
            col_proj = np.sum(strip_img, axis=0)
            col_proj = (col_proj - col_proj.min()) / (col_proj.max() - col_proj.min() + 1e-6)
            peaks_major, _ = find_peaks(col_proj, height=0.4, distance=25)
            detected_major_ticks_x = [max(0, min(x_axis_left, x_axis_right) - pad_left) + int(px) for px in peaks_major]

            if len(x_label) > 1:
                detected_major_ticks_x = auto_add_x_major_ticks(detected_major_ticks_x, x_axis_left, x_axis_right,
                                                                minor_tick_coords)
            if (len(detected_major_ticks_x) <= 2) and len(labels) > 3:
                continue
            if len(detected_major_ticks_x) == 0:
                continue
            if len(x_label) == 1 and len(detected_major_ticks_x) > 2:
                continue
            if len(x_label) == 2 and len(detected_major_ticks_x) > 3:
                continue

            num_not_matched = 0
            for mx in detected_major_ticks_x:
                if min([abs(mx - mnx) for mnx in minor_ticks_x]) > match_threshold:
                    num_not_matched += 1
            if num_not_matched > 2:
                continue

            if len(labels) >= 2:
                if float(labels[1]) - float(labels[0]) == 1:
                    if abs(len(labels) - len(detected_major_ticks_x)) > 1:
                        continue

            if len(detected_major_ticks_x) < count_consecutive_pairs(labels):
                continue

            return sorted(detected_major_ticks_x), y_strip_bot_major
    return []


def auto_add_x_major_ticks(detected_major_ticks_x, x_axis_left, x_axis_right, minor_tick_x_coords):
    if len(detected_major_ticks_x) < 2:
        return detected_major_ticks_x

    major_ticks_x = sorted(detected_major_ticks_x)
    steps = np.diff(major_ticks_x)
    min_step = np.min(steps)
    minor_ticks_x = [tx for tx, _ in minor_tick_x_coords]

    while True:
        next_tick = major_ticks_x[-1] + min_step
        if next_tick > max(x_axis_left, x_axis_right) + 5:
            break
        major_ticks_x.append(next_tick)

    while True:
        next_tick = major_ticks_x[0] - min_step
        if next_tick < min(x_axis_left, x_axis_right) - 5:
            break
        if next_tick in major_ticks_x:
            break
        major_ticks_x.append(next_tick)

    major_ticks_x = sorted(major_ticks_x)
    steps = np.diff(major_ticks_x)
    insert_positions = []
    for i, gap in enumerate(steps):
        if gap > min_step:
            num_to_add = int(round(gap / min_step)) - 1
            left = major_ticks_x[i]
            right = major_ticks_x[i + 1]
            minor_in_gap = [mnx for mnx in minor_ticks_x if left < mnx < right]
            for n in range(num_to_add):
                target_pos = left + (n + 1) * min_step
                if minor_in_gap:
                    nearest_minor = min(minor_in_gap, key=lambda mnx: abs(mnx - target_pos))
                    insert_positions.append(
                        nearest_minor if abs(nearest_minor - target_pos) <= min_step else int(target_pos))
                else:
                    insert_positions.append(int(target_pos))

    return sorted(set(major_ticks_x + insert_positions))


# ------------------- Map X Major tick and label -------------------
def map_labels_to_x_major_ticks(label_x, labels, major_tick_coords, x_axis_left, x_axis_right):
    tick_relative_x_position = np.array(major_tick_coords) - min(x_axis_left, x_axis_right)
    mapping = []
    if len(label_x) <= 2:
        closest_idx = np.argmin(np.abs(tick_relative_x_position - label_x[0]))
        mapping.append({
            'text': labels[0],
            'between_tick': (major_tick_coords[closest_idx], major_tick_coords[closest_idx]),
            'approx_tick_index': closest_idx
        })
    else:
        for lx, ltxt in zip(label_x, labels):
            idx = np.searchsorted(tick_relative_x_position, lx)
            if 0 < idx < len(major_tick_coords):
                mapping.append({
                    'text': ltxt,
                    'between_tick': (major_tick_coords[idx - 1], major_tick_coords[idx]),
                    'approx_tick_index': idx - 1
                })
    return mapping


# ------------------- Map X minor tick and label -------------------
def compress_x_minor_tick(mapping):
    result = []
    used_tick = set()
    for m in mapping:
        tick = m['between_tick']
        if tick not in used_tick:
            result.append(m)
            used_tick.add(tick)
    return result


def is_10_year_range(mapping):
    for m in mapping:
        for m2 in mapping:
            if m != m2 and m['between_tick'] == m2['between_tick']:
                if abs(int(m['text']) - int(m2['text'])) >= 8:
                    return True
    return False


def is_single_year_range(mapping):
    tick_pair_count = {}
    for m in mapping:
        key = m['between_tick']
        tick_pair_count.setdefault(key, 0)
        tick_pair_count[key] += 1
    for count in tick_pair_count.values():
        if count > 1:
            return True
    return False


def is_day_range(month_labels, months_x):
    month_pos_pairs = sorted(zip(months_x, month_labels))
    sorted_month_labels = [label for _, label in month_pos_pairs]

    for i in range(1, len(sorted_month_labels)):
        if sorted_month_labels[i] == sorted_month_labels[i - 1]:
            return True
    return False


def map_labels_to_minor_ticks_single_year_with_quarter(mapping, tick_coords_sorted, y_strip):
    label_results = []
    for i in range(len(mapping) - 1):
        year = int(mapping[i]['text'])
        next_year = int(mapping[i + 1]['text'])
        tick_x1 = mapping[i]['between_tick'][0]
        tick_x2 = mapping[i]['between_tick'][1]
        ticks_in_segment = [tx for (tx, ty) in tick_coords_sorted if tick_x1 <= tx < tick_x2]
        num_ticks_in_segment = len(ticks_in_segment)
        if num_ticks_in_segment > 0:
            for quarter_idx, tx in enumerate(ticks_in_segment, 1):
                label = f"{year}.{quarter_idx * 3 - 2:02d}"
                label_results.append({'tick_pos': (tx, y_strip), 'label': label})

    if len(mapping) >= 1:
        year = int(mapping[-1]['text'])
        tick_x1 = mapping[-1]['between_tick'][0]
        tick_x2 = mapping[-1]['between_tick'][1]
        ticks_in_segment = [tx for (tx, ty) in tick_coords_sorted if tick_x1 <= tx < tick_x2]
        num_ticks_in_segment = len(ticks_in_segment)
        if num_ticks_in_segment > 0:
            for quarter_idx, tx in enumerate(ticks_in_segment, 1):
                if quarter_idx * 3 - 2 >= 12:
                    continue
                label = f"{year}.{quarter_idx * 3 - 2:02d}"
                label_results.append({'tick_pos': (tx, y_strip), 'label': label})

    first_major_tick_x = mapping[0]['between_tick'][0]
    first_year = int(mapping[0]['text'])
    ticks_before = [tx for (tx, ty) in tick_coords_sorted if tx < first_major_tick_x]
    first_segment_num_ticks = len(
        [tx for (tx, ty) in tick_coords_sorted if first_major_tick_x <= tx < mapping[1]['between_tick'][0]])
    if first_segment_num_ticks > 0 and ticks_before:
        for idx, tx in enumerate(reversed(ticks_before), 1):
            total_month = 12 - 3 * (idx - 1)
            prev_year = first_year - 1
            while total_month <= 0:
                total_month += 12
                prev_year -= 1
            label = f"{prev_year}.{total_month - 2:02d}"
            label_results.append({'tick_pos': (tx, y_strip), 'label': label})

    last_major_tick_x = mapping[-1]['between_tick'][1]
    last_year = int(mapping[-1]['text'])
    ticks_after = [tx for (tx, ty) in tick_coords_sorted if tx >= last_major_tick_x]
    prev_segment_num_ticks = len(
        [tx for (tx, ty) in tick_coords_sorted if mapping[-2]['between_tick'][1] <= tx < last_major_tick_x])
    if prev_segment_num_ticks > 0 and ticks_after:
        for idx, tx in enumerate(ticks_after, 1):
            label = f"{last_year + 1}.{idx * 3 - 2:02d}"
            label_results.append({'tick_pos': (tx, y_strip), 'label': label})
    return label_results


def map_labels_to_minor_ticks_range_5_year_with_quarter(mapping, tick_coords_sorted, y_strip):
    print("Case 4 Year")
    label_results = []
    mapping = compress_x_minor_tick(mapping)

    if len(mapping) > 1:
        first_major_tick_x = mapping[0]['between_tick'][0]
        first_year = int(mapping[0]['text'])
        ticks_before_first_major = [tx for (tx, ty) in tick_coords_sorted if tx <= first_major_tick_x]

        for idx, tx in enumerate(reversed(ticks_before_first_major), 1):
            year_label = f"{first_year - idx + 1}.01"
            label_results.append({'tick_pos': (tx, y_strip), 'label': year_label, 'is_quarter': False})
            if idx == 1 and len(ticks_before_first_major) != 1:
                label_results.pop()

        if len(ticks_before_first_major) >= 2:
            for j in range(len(ticks_before_first_major) - 1):
                tick_left = ticks_before_first_major[-(j + 2)]
                tick_right = ticks_before_first_major[-(j + 1)]
                year = first_year - (j + 1)
                for q in range(1, 4):
                    frac = q / 4
                    quarter_tick_x = int(tick_left + frac * (tick_right - tick_left))
                    quarter_label = f"{year}.{(q * 3) + 1:02d}"
                    label_results.append(
                        {'tick_pos': (quarter_tick_x, y_strip), 'label': quarter_label, 'is_quarter': True})

            tick_left = ticks_before_first_major[0]
            tick_right = ticks_before_first_major[1]
            year = first_year - len(ticks_before_first_major)
            tick_step = tick_right - tick_left
            max_extra = 1

            for i in range(max_extra):
                extra_tick_left = tick_left - tick_step
                extra_tick_right = tick_left
                if extra_tick_left < 0:
                    break

                for q in range(1, 4):
                    frac = q / 4
                    quarter_tick_x = int(extra_tick_left + frac * (extra_tick_right - extra_tick_left))
                    quarter_label = f"{year}.{(q * 3) + 1:02d}"
                    label_results.append(
                        {'tick_pos': (quarter_tick_x, y_strip), 'label': quarter_label, 'is_quarter': True})

                label_results.append({'tick_pos': (extra_tick_left, y_strip), 'label': str(year), 'is_quarter': False})
                tick_left = extra_tick_left
                year -= 1

    for i in range(len(mapping)):
        year = int(mapping[i]['text'])
        tick_x1 = mapping[i]['between_tick'][0] - 3
        tick_x2 = mapping[i]['between_tick'][1] + 3
        ticks_in_segment = [tx for (tx, ty) in tick_coords_sorted if tick_x1 <= tx < tick_x2 - 3]
        if ticks_in_segment and ticks_in_segment[-1] != tick_x2:
            ticks_in_segment.append(tick_x2)
        elif not ticks_in_segment:
            ticks_in_segment = [tick_x1, tick_x2]

        for j, tx in enumerate(ticks_in_segment):
            label = f"{year + j}.01"
            label_results.append({'tick_pos': (tx, y_strip), 'label': label, 'is_quarter': False})
        if i != len(mapping):
            label_results.pop()

        for j in range(len(ticks_in_segment) - 1):
            tx1 = ticks_in_segment[j]
            tx2 = ticks_in_segment[j + 1]
            for q in range(1, 4):
                frac = q / 4
                tq = int(tx1 + frac * (tx2 - tx1))
                q_label = f"{year + j}.{(q * 3) + 1:02d}"
                label_results.append({'tick_pos': (tq, y_strip), 'label': q_label, 'is_quarter': True})

    if len(mapping) >= 1:
        year = int(mapping[-1]['text']) + 1
        tick_x1 = mapping[-1]['between_tick'][1]

        if len(mapping) >= 2:
            prev_tick_x1 = mapping[-2]['between_tick'][0]
            prev_tick_x2 = mapping[-2]['between_tick'][1]
            step = prev_tick_x2 - prev_tick_x1
        else:
            step = mapping[-1]['between_tick'][1] - mapping[-1]['between_tick'][0]

        tick_x2 = tick_x1 + step
        ticks_after = [tx for (tx, ty) in tick_coords_sorted if tick_x1 <= tx <= tick_x2]
        if not ticks_after:
            ticks_after = [tick_x1, tick_x2]

        for tick_in_segment_idx in range(len(ticks_after) - 1):
            tick_left = ticks_after[tick_in_segment_idx]
            tick_right = ticks_after[tick_in_segment_idx + 1]
            for q in range(1, 4):
                frac = q / 4
                tq = int(tick_left + frac * (tick_right - tick_left))
                q_label = f"{year + tick_in_segment_idx + 4}.{(q * 3) + 1:02d}"
                label_results.append({'tick_pos': (tq, y_strip), 'label': q_label, 'is_quarter': True})

        for j, tx in enumerate(ticks_after):
            label = f"{year + j + 4}.01"
            label_results.append({'tick_pos': (tx, y_strip), 'label': label, 'is_quarter': False})

    return label_results


def map_labels_to_minor_ticks_range_10_year_with_quarter(mapping, tick_coords_sorted, y_strip, output=None):
    label_results = []
    mapping = compress_x_minor_tick(mapping)

    print("Case 10 Year")
    if len(mapping) > 1:
        first_major_tick_x = mapping[0]['between_tick'][0]
        first_year = int(mapping[0]['text'])
        ticks_before = [tx for (tx, ty) in tick_coords_sorted if tx <= first_major_tick_x]

        for tick_idx, tick_x in enumerate(reversed(ticks_before), 1):
            year_label = f"{first_year - tick_idx + 1}.01"
            label_results.append({'tick_pos': (tick_x, y_strip), 'label': year_label, 'is_quarter': False})
            if tick_idx == 1 and len(ticks_before) != 1:
                label_results.pop()

        if len(ticks_before) >= 2:
            for segment_idx in range(len(ticks_before) - 1):
                tick_left = ticks_before[-(segment_idx + 2)]
                tick_right = ticks_before[-(segment_idx + 1)]
                year = first_year - (segment_idx + 1)
                for quarter_idx in range(1, 8):
                    frac = quarter_idx / 8
                    quarter_tick_x = int(tick_left + frac * (tick_right - tick_left))
                    if quarter_idx < 4:
                        quarter_label = f"{year + (segment_idx * 2) - 1}.{(quarter_idx + 1) * 3 - 2:02d}"
                    else:
                        quarter_label = f"{year + (segment_idx * 2)}.{((quarter_idx - 4) + 1) * 3 - 2:02d}"
                    label_results.append(
                        {'tick_pos': (quarter_tick_x, y_strip), 'label': quarter_label, 'is_quarter': True})

            tick_left = ticks_before[0]
            tick_right = ticks_before[1]
            year = first_year - len(ticks_before)
            step = tick_right - tick_left
            max_extra = 1

            for _ in range(max_extra):
                next_tick_left = tick_left - step
                next_tick_right = tick_left
                if next_tick_left < 0:
                    break
                for quarter_idx in range(1, 4):
                    frac = quarter_idx / 4
                    quarter_tick_x = int(next_tick_left + frac * (next_tick_right - next_tick_left))
                    quarter_label = f"{year}.{(quarter_idx + 1) * 3 - 2:02d}"
                    label_results.append(
                        {'tick_pos': (quarter_tick_x, y_strip), 'label': quarter_label, 'is_quarter': True})

                label_results.append({'tick_pos': (next_tick_left, y_strip), 'label': str(year), 'is_quarter': False})
                tick_left = next_tick_left
                year -= 1

    for mapping_idx in range(len(mapping)):
        year = int(mapping[mapping_idx]['text'])
        tick_x1 = mapping[mapping_idx]['between_tick'][0] - 3
        tick_x2 = mapping[mapping_idx]['between_tick'][1] + 3

        ticks_in_segment = [tx for (tx, ty) in tick_coords_sorted if tick_x1 <= tx < tick_x2 - 3]

        # if ticks_in_segment and abs(ticks_in_segment[-1] - tick_x2) > 5:
        if ticks_in_segment and ticks_in_segment[-1] != tick_x2:
            ticks_in_segment.append(tick_x2)
        elif not ticks_in_segment:
            ticks_in_segment = [tick_x1, tick_x2]

        for tick_in_segment_idx, tick_x in enumerate(ticks_in_segment):
            label = f"{year + tick_in_segment_idx * 2}.01"
            label_results.append({'tick_pos': (tick_x, y_strip), 'label': label, 'is_quarter': False})
        if mapping_idx != len(mapping):
            label_results.pop()

        for tick_in_segment_idx in range(len(ticks_in_segment) - 1):
            tick_left = ticks_in_segment[tick_in_segment_idx]
            tick_right = ticks_in_segment[tick_in_segment_idx + 1]
            for quarter_idx in range(1, 8):
                frac = quarter_idx / 8
                quarter_tick_x = int(tick_left + frac * (tick_right - tick_left))
                if quarter_idx < 4:
                    quarter_label = f"{year + (tick_in_segment_idx * 2)}.{(quarter_idx + 1) * 3 - 2:02d}"
                else:
                    quarter_label = f"{year + (tick_in_segment_idx * 2) + 1}.{((quarter_idx - 4) + 1) * 3 - 2:02d}"
                label_results.append(
                    {'tick_pos': (quarter_tick_x, y_strip), 'label': quarter_label, 'is_quarter': True})

    if len(mapping) >= 1:
        year = int(mapping[-1]['text']) + 1
        tick_x1 = mapping[-1]['between_tick'][1]

        if len(mapping) >= 2:
            prev_tick_x1 = mapping[-2]['between_tick'][0]
            prev_tick_x2 = mapping[-2]['between_tick'][1]
            step = prev_tick_x2 - prev_tick_x1
        else:
            step = mapping[-1]['between_tick'][1] - mapping[-1]['between_tick'][0]

        tick_x2 = tick_x1 + step
        ticks_after = [tx for (tx, ty) in tick_coords_sorted if tick_x1 - 10 <= tx <= tick_x2]
        if not ticks_after:
            ticks_after = [tick_x1, tick_x2]

        for tick_in_segment_idx, tick_x in enumerate(ticks_after):
            label = f"{year + (tick_in_segment_idx * 2) + 9}.01"
            label_results.append({'tick_pos': (tick_x, y_strip), 'label': label, 'is_quarter': False})

        for tick_in_segment_idx in range(len(ticks_after) - 1):
            tick_left = ticks_after[tick_in_segment_idx]
            tick_right = ticks_after[tick_in_segment_idx + 1]
            for quarter_idx in range(1, 8):
                frac = quarter_idx / 8
                quarter_tick_x = int(tick_left + frac * (tick_right - tick_left))
                if quarter_idx < 4:
                    quarter_label = f"{year + (tick_in_segment_idx * 2) + 9}.{(quarter_idx + 1) * 3 - 2:02d}"
                else:
                    quarter_label = f"{year + (tick_in_segment_idx * 2) + 10}.{((quarter_idx - 4) + 1) * 3 - 2:02d}"
                label_results.append(
                    {'tick_pos': (quarter_tick_x, y_strip), 'label': quarter_label, 'is_quarter': True})
    return label_results


def map_labels_to_minor_ticks_range_months(mapping, tick_coords_sorted, y_strip, output=None):
    print("Case Month")
    if not mapping:
        return []

    label_results = []

    start_year = int(mapping[0]['text'])
    tick_start, tick_end = mapping[0]['between_tick']

    segment_ticks = [tx for (tx, ty) in tick_coords_sorted if tick_start <= tx <= tick_end]
    segment_ticks = sorted(segment_ticks)
    if not segment_ticks:
        segment_ticks = [tick_start, tick_end]
    elif segment_ticks[-1] != tick_end:
        segment_ticks.append(tick_end)

    ticks_before_segment = [tx for (tx, ty) in tick_coords_sorted if tx < tick_start]
    ticks_before_segment = sorted(ticks_before_segment)

    ticks_after_segment = [tx for (tx, ty) in tick_coords_sorted if tx > tick_end]
    ticks_after_segment = sorted(ticks_after_segment)

    for i, tx in enumerate(reversed(ticks_before_segment), 1):
        month = 12 - (i - 1)
        year = start_year
        while month <= 0:
            month += 12
            year -= 1
        label = f"{year}.{month:02d}"
        label_results.append({'tick_pos': (tx, y_strip), 'label': label, 'is_quarter': False})

    for i, tx in enumerate(segment_ticks):
        month = i + 1
        if month <= 12:
            year = start_year + 1
        else:
            year = start_year + 1 + (month - 1) // 12
        month_in_year = (month - 1) % 12 + 1
        label = f"{year}.{month_in_year:02d}"
        label_results.append({'tick_pos': (tx, y_strip), 'label': label, 'is_quarter': False})

    base_index = len(segment_ticks)
    for i, tx in enumerate(ticks_after_segment, 1):
        month = base_index + i
        year = start_year + 1 + (month - 1) // 12
        month_in_year = (month - 1) % 12 + 1
        label = f"{year}.{month_in_year:02d}"
        label_results.append({'tick_pos': (tx, y_strip), 'label': label, 'is_quarter': False})

    # # Optional: วาดลงรูปสำหรับ debug
    # if output is not None:
    #     for result in label_results:
    #         cv2.putText(output, result['label'], (result['tick_pos'][0] - 10, y_strip - 12),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

    return label_results


def map_labels_to_minor_ticks_range_day(mapping, tick_coords_sorted, y_strip, month_labels, month_x, output=None):
    label_results = []

    if not mapping or len(tick_coords_sorted) < 2 or not month_labels or not month_x:
        return label_results

    zipped = list(zip(month_x, month_labels))
    zipped_sorted = sorted(zipped, key=lambda x: x[0])
    sorted_month_labels = [m for _, m in zipped_sorted]

    base_year = int(mapping[0]['text'])
    months_in_year = 12

    num_groups = (len(tick_coords_sorted) + 3) // 4
    gen_months = []
    cur_month_idx = MONTH_ABBR.index(sorted_month_labels[0])
    cur_year = base_year

    for i in range(num_groups):
        gen_months.append((cur_year, cur_month_idx))
        cur_month_idx += 1
        if cur_month_idx >= 12:
            cur_month_idx = 0
            cur_year += 1

    for i, (tx, _) in enumerate(tick_coords_sorted):
        month_group = i // 4
        label_year, label_month_idx = gen_months[month_group]
        label = f"{label_year}.{(label_month_idx + 1):02d}"
        label_results.append({'tick_pos': (tx, y_strip), 'label': label, 'is_quarter': False})

        # if output is not None:
        #     cv2.putText(output, label, (tx - 15, y_strip - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 50), 1)

    return label_results


# ------------------- Find y major tick -------------------
def find_y_major_ticks(mask_image, y_axis_top, y_axis_left, y_axis_bottom, y_axis_right,
                       pad_top=5, pad_bottom=5, strip_width=5, shift_left=10):
    x_strip = int((y_axis_left + y_axis_right) / 2) - shift_left
    strip_image = mask_image[
                  max(0, min(y_axis_top, y_axis_bottom) - pad_top):min(mask_image.shape[0],
                                                                       max(y_axis_top, y_axis_bottom) + pad_bottom),
                  max(0, x_strip - strip_width // 2):min(mask_image.shape[1], x_strip + strip_width // 2)]

    proj = np.sum(strip_image, axis=1)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    peaks, _ = find_peaks(proj, height=0.18, distance=10)
    minor_tick_coords = []
    for peak in peaks:
        ty = max(0, min(y_axis_top, y_axis_bottom) - pad_top) + peak
        tx = int((y_axis_left + y_axis_right) / 2)
        if ty < max(y_axis_top, y_axis_bottom) - 10:
            minor_tick_coords.append((tx, ty))

    return sorted(minor_tick_coords, key=lambda t: t[1])


def parse_y_label_value(label):
    if not isinstance(label, str):
        return None

    label = label.strip().lower()

    multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
    match = re.fullmatch(r'(\d+(\.\d+)?)([kmb]?)', label)

    if not match:
        return None

    num_part = float(match.group(1))
    suffix = match.group(3)

    multiplier = multipliers.get(suffix, 1)
    return num_part * multiplier


def format_y_label_value(val):
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}B" if val % 1e9 != 0 else f"{int(val/1e9)}B"
    elif abs(val) >= 1e6:
        return f"{val/1e6:.2f}M" if val % 1e6 != 0 else f"{int(val/1e6)}M"
    elif abs(val) >= 1e3:
        return f"{val/1e3:.2f}k" if val % 1e3 != 0 else f"{int(val/1e3)}k"
    else:
        return str(int(val)) if val == int(val) else str(val)

def correct_misread_labels(y_labels):
    for i, tick in enumerate(y_labels):
        label = tick['label']

        if label is None:
            continue

        if label == '15M':
            neighbors = []
            if i > 0:
                neighbors.append(y_labels[i - 1]['label'])
            if i < len(y_labels) - 1:
                neighbors.append(y_labels[i + 1]['label'])
            if any(n in ['1M', '2M'] for n in neighbors):
                tick['label'] = '1.5M'

        if label in ['15B']:
            neighbors = []
            if i > 0:
                neighbors.append(y_labels[i - 1]['label'])
            if i < len(y_labels) - 1:
                neighbors.append(y_labels[i + 1]['label'])
            if any(n in ['1B', '2B'] for n in neighbors):
                tick['label'] = '1.5B'

    return y_labels


# ------------------- OCR Y label -------------------
def ocr_y_white_label(y_axis_left, y_axis_right, y_axis_top, y_axis_bot, img_cropped,
                      pad_top=5, pad_bottom=5):
    for shift_left in (5, 10):
        x_strip = int((y_axis_left + y_axis_right) / 2) - shift_left
        ocr_attempts = [(x_strip - 110, x_strip - 20), (x_strip - 110, x_strip - 15), (x_strip - 110, x_strip - 10), (x_strip - 110, x_strip - 5),
                        (x_strip - 120, x_strip - 20), (x_strip - 120, x_strip - 15), (x_strip - 120, x_strip - 10), (x_strip - 120, x_strip - 5),
                        (x_strip - 130, x_strip - 20), (x_strip - 130, x_strip - 15), (x_strip - 130, x_strip - 10), (x_strip - 130, x_strip - 5),
                        (x_strip - 140, x_strip - 20), (x_strip - 140, x_strip - 15), (x_strip - 140, x_strip - 10), (x_strip - 140, x_strip - 5),
                        (x_strip - 150, x_strip - 20), (x_strip - 150, x_strip - 15), (x_strip - 150, x_strip - 10), (x_strip - 150, x_strip - 5)]
        valid_suffix = ('M', 'm', 'B', 'b', 'K', 'k', '8', '0')
        ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.MBkK'
        for attempt_idx, (x_left, x_right) in enumerate(ocr_attempts):
            ocr_roi = img_cropped[
                      max(0, min(y_axis_top, y_axis_bot) - pad_top):min(img_cropped.shape[0], max(y_axis_top, y_axis_bot) + pad_bottom),
                      max(0, x_left):min(img_cropped.shape[1], x_right)
                      ]

            cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_test/ocr_y_{attempt_idx}.png",ocr_roi)
            ocr_result_y = pytesseract.image_to_data(
                ocr_roi,
                output_type=pytesseract.Output.DICT,
                config=ocr_config
            )

            y_labels = []
            for i, text in enumerate(ocr_result_y['text']):
                label = text.strip()
                if label.endswith('.') or label.endswith('-'):
                    label = label.rstrip('.-')

                top = ocr_result_y['top'][i]
                height = ocr_result_y['height'][i]
                center_y = max(0, min(y_axis_top, y_axis_bot) - pad_top) + top + height // 2
                if label != "":
                    y_labels.append({'y': center_y, 'label': label})

            y_labels = correct_misread_labels(y_labels)
            non_empty_labels = [label for label in y_labels if label['label'].strip() != '']
            label_texts = [label['label'] for label in non_empty_labels]

            label_with_values = []
            invalid_values = False
            for lbl in non_empty_labels:
                val = parse_y_label_value(lbl['label'])
                if val is None:
                    invalid_values = True
                    break
                label_with_values.append((lbl['y'], val))

            if invalid_values:
                continue

            sorted_by_y = sorted(label_with_values, key=lambda x: x[0])
            values = [v for _, v in sorted_by_y]
            new_labels = [{'y': y, 'label': str(l)} for (y, l) in
                          zip([y for (y, _) in sorted_by_y], [l for (_, l) in sorted_by_y])]

            if not all(values[i] > values[i + 1] for i in range(len(values) - 1)):
                continue

            if not new_labels:
                continue

            if len(new_labels) < 2:
                continue

            ends_with_suffix = [lbl[-1] in valid_suffix for lbl in label_texts if lbl]
            if any(ends_with_suffix):
                if not all(ends_with_suffix):
                    continue
            else:
                pass

            edge_threshold = 2.0
            diffs = [abs(values[i] - values[i + 1]) for i in range(len(values) - 1)]
            if len(diffs) >= 2:
                median_diff = np.median(diffs)
                if diffs[0] > edge_threshold * median_diff:
                    new_labels = new_labels[1:]
                    values = values[1:]
                    diffs = diffs[1:]
                if diffs and diffs[-1] > edge_threshold * median_diff:
                    new_labels = new_labels[:-1]
                    values = values[:-1]
            result = []
            for l in new_labels:
                label_str = l['label']
                val = parse_y_label_value(label_str)
                if val is not None:
                    formatted_label = format_y_label_value(val)
                    result.append({'y': l['y'], 'label': formatted_label})
            return result

        return []


def ocr_y_green_label(img_cropped, best_line_y, y_axis_labels):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_cropped, -1, sharpen_kernel)
    hsv = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 50, 50])
    upper_green = np.array([85, 255, 255])

    x_y = int((best_line_y[0] + best_line_y[2]) / 2)
    y1_y = min(best_line_y[1], best_line_y[3])
    y2_y = max(best_line_y[1], best_line_y[3]) + 10

    strip_settings = [
        (200, 12), (200, 10), (200, 15), (200, 20), (200, 25),
        (180, 12), (180, 10), (180, 15), (180, 20), (180, 25),
        (160, 12), (160, 10), (160, 15), (160, 20), (160, 25),
        (140, 12), (140, 10), (140, 15), (140, 20), (140, 25),
    ]

    prev_int_labels = None

    for attempt_idx, (strip_width, pad_right) in enumerate(strip_settings):
        strip_left = max(0, x_y - strip_width)
        strip_right = max(0, x_y - pad_right)

        green_strip = hsv[y1_y:y2_y, strip_left:strip_right]
        mask_green = cv2.inRange(green_strip, lower_green, upper_green)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_results = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 8:
                abs_x = strip_left + x
                abs_y = y1_y + y
                roi = img_cropped[abs_y:abs_y + h, abs_x:abs_x + w]
                # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"img/debug/green_label_Y.png", roi)
                text = pytesseract.image_to_string(roi, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.MBkK').strip()

                if text == "." or text == "-":
                    text = "0"

                if text:
                    green_results.append({'y': abs_y + h // 2, 'label': text})
                    #cv2.rectangle(img_cropped, (abs_x, abs_y), (abs_x + w, abs_y + h), (0, 255, 0), 1)
                    #cv2.putText(img_cropped, text, (abs_x, abs_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not green_results:
            continue

        has_invalid_text = all(not any(c.isdigit() for c in r['label']) for r in green_results)
        if has_invalid_text:
            continue

        has_invalid_decimal = any(('.' in r['label']) and (r['label'][-1] not in 'kKmMbB') for r in green_results)
        if has_invalid_decimal:
            continue

        valid_suffixes = tuple("0123456789MmKkBb")
        green_results = [g for g in green_results if g['label'] and g['label'][-1] in valid_suffixes]


        if not green_results:
            continue

        int_labels = []
        for g in green_results:
            try:
                val = float(g['label'].replace('K', '').replace('k', '').replace('M', '').replace('B', ''))
                if val.is_integer():
                    int_labels.append(g['label'])
            except Exception:
                pass

        if int_labels:
            if int_labels != prev_int_labels:
                prev_int_labels = int_labels
                continue
        prev_int_labels = int_labels

        combined = y_axis_labels + green_results
        combined = [item for item in combined if item['label'].strip() != '']
        combined = sorted(combined, key=lambda x: x['y'])

        parsed_values = [parse_y_label_value(l['label']) for l in combined]
        if None in parsed_values:
            continue

        if not all(parsed_values[i] > parsed_values[i + 1] for i in range(len(parsed_values) - 1)):
            continue

        return green_results

    return []


# ------------------- Map Y and label -------------------
def find_y_nearest_with_label(tick_y, y_tick_labels):
    if isinstance(tick_y, tuple):
        tick_y = tick_y[1]
    tick_y = int(tick_y)

    min_dist = float('inf')
    nearest_label = None

    for item in y_tick_labels:
        item_y = int(item['y'])
        dist = abs(item_y - tick_y)
        if dist < min_dist:
            min_dist = dist
            nearest_label = item
    return nearest_label


def map_ticks_to_labels(major_ticks_y, y_tick_labels, max_distance=20):
    mapped_tick_labels = []
    for tick_y in major_ticks_y:
        if isinstance(tick_y, tuple):
            tick_y_val = int(tick_y[1])
        else:
            tick_y_val = int(tick_y)

        nearest = find_y_nearest_with_label(tick_y, y_tick_labels)
        if nearest is not None and abs(int(nearest['y']) - tick_y_val) < max_distance:
            mapped_tick_labels.append({'y': tick_y_val, 'label': nearest['label']})
        else:
            mapped_tick_labels.append({'y': tick_y_val, 'label': None})
    return mapped_tick_labels


def auto_fill_missing_tick_label(mapped_tick_labels):
    anchor_ticks = []
    for i, tick in enumerate(mapped_tick_labels):
        parsed_val = parse_y_label_value(tick['label'])
        if tick['label'] is not None and parsed_val is not None:
            anchor_ticks.append((i, tick['y'], parsed_val))

    if len(anchor_ticks) < 2:

        if len(anchor_ticks) == 1:
            val = anchor_ticks[0][2]
            label_str = ""
            if abs(val) >= 1e9:
                label_str = f"{val / 1e9:.0f}B"
            elif abs(val) >= 1e6:
                label_str = f"{val / 1e6:.0f}M"
            elif abs(val) >= 1e3:
                label_str = f"{val / 1e3:.0f}k"
            else:
                label_str = f"{int(val)}" if val == int(val) else f"{val:.0f}"

            for tick in mapped_tick_labels:
                if tick['label'] is None:
                    tick['label'] = label_str
        else:
            for tick in mapped_tick_labels:
                tick['label'] = "N/A"
        return mapped_tick_labels

    anchor_ticks_sorted = sorted(anchor_ticks, key=lambda x: x[1], reverse=True)
    y0, y1 = anchor_ticks_sorted[0][1], anchor_ticks_sorted[1][1]
    v0, v1 = anchor_ticks_sorted[0][2], anchor_ticks_sorted[1][2]

    delta_y = y0 - y1
    delta_val = v0 - v1

    if delta_y == 0:
        return mapped_tick_labels

    value_per_pixel = delta_val / delta_y

    anchor_y = y0
    anchor_value = v0

    for tick in mapped_tick_labels:
        if tick['label'] is None:
            dy_from_anchor = tick['y'] - anchor_y
            estimated_val = anchor_value + value_per_pixel * dy_from_anchor

            if abs(estimated_val) >= 1e9:
                label_str = f"{estimated_val / 1e9:.2f}B"
            elif abs(estimated_val) >= 1e6:
                label_str = f"{estimated_val / 1e6:.2f}M"
            elif abs(estimated_val) >= 1e3:
                label_str = f"{estimated_val / 1e3:.2f}k"
            else:
                label_str = f"{int(estimated_val)}" if estimated_val == int(estimated_val) else f"{estimated_val:.0f}"

            tick['label'] = label_str

    return mapped_tick_labels


# ------------------- Detect Green Circle -------------------
def interpolate_y_label(target_y, mapped_tick_labels, green_labels=None):
    tick_positions = [tick['y'] for tick in mapped_tick_labels]
    tick_values = [parse_y_label_value(tick['label']) for tick in mapped_tick_labels]

    valid_ticks = [(y, val, tick['label']) for y, val, tick in zip(tick_positions, tick_values, mapped_tick_labels) if
                   val is not None]

    if not valid_ticks or len(valid_ticks) == 1:
        for tick in mapped_tick_labels:
            return tick['label']
        return "N/A"

    if green_labels is not None and len(green_labels) > 0:
        for green in green_labels:
            if abs(green['y'] - target_y) <= 5:
                return green['label']

    for y, val, label in valid_ticks:
        if abs(target_y - y) <= 3:
            return label

    valid_ticks.sort(key=lambda t: t[0], reverse=True)
    y_coords, values, labels = zip(*valid_ticks)

    base_y = y_coords[0]
    base_value = values[0]
    dy = y_coords[0] - y_coords[1]
    dv = values[0] - values[1]

    if dy == 0:
        return "N/A"

    slope = dv / dy
    delta_y = target_y - base_y
    estimated_value = base_value + slope * delta_y

    if estimated_value < 0 and (not green_labels or len(green_labels) == 0):
        return "0"

    if estimated_value < 0 and green_labels and len(green_labels) != 0:
        for green in green_labels:
            dist = abs(green['y'] - target_y)
            if dist <= 8:
                return green['label']
            else:
                return "0"

    if estimated_value >= 1e9:
        return f"{estimated_value / 1e9:.2f}B"
    elif estimated_value >= 1e6:
        return f"{estimated_value / 1e6:.2f}M"
    elif estimated_value >= 1e3:
        return f"{estimated_value / 1e3:.2f}k"
    else:
        return f"{estimated_value:.0f}"


def scan_green_balls_by_x_ticks(hsv_image, mask_output_img, tick_labels, mapped_tick_labels, x_axis, y_axis,
                                y_green_label, day, green_threshold=12):
    print("\n===== Scan Green Ball By X Tick =====")
    x_axis_y = int((x_axis[1] + x_axis[3]) / 2)
    y_axis_x = int((y_axis[0] + y_axis[2]) / 2)
    x_axis_xmax = max(x_axis[0], x_axis[2])

    ignore_rect = (190, 420, 270, 450)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    print("╔════════════╤══════════════╗")
    print("║ Tick Label │ Y-axis Value ║")
    print("╟────────────┼──────────────╢")

    found_flags = []
    found_ys = []

    sorted_ticks = sorted(tick_labels, key=lambda t: t['tick_pos'][0])
    for tick in sorted_ticks:
        tx, ty = tick['tick_pos']

        if tx > x_axis_xmax or tx < y_axis_x - 5:
            found_flags.append(False)
            found_ys.append(None)
            continue

        found = False
        found_y = None
        y_axis_top = min(y_axis[1] - 5, y_axis[3] - 5)
        for y in range(x_axis_y - 1, 0, -1):
            if y <= y_axis_top:
                break
            if abs(tx - y_axis_x) < 5:
                window = mask_green[
                         max(0, y + 2):min(mask_green.shape[0], y + 17),
                         max(0, tx):min(mask_green.shape[1], tx + 1)
                         ]
                _green_threshold = 10
            elif day:
                window = mask_green[
                         max(0, y + 2):min(mask_green.shape[0], y + 17),
                         max(0, tx - 8):min(mask_green.shape[1], tx + 8)
                         ]
                _green_threshold = 20
            else:
                window = mask_green[
                         max(0, y):min(mask_green.shape[0], y + 15),
                         max(0, tx):min(mask_green.shape[1], tx + 1)
                         ]
                _green_threshold = green_threshold

            green_count = np.sum(window > 128)
            if green_count > _green_threshold:
                if ignore_rect[0] <= tx <= ignore_rect[2] and ignore_rect[1] <= y <= ignore_rect[3]:
                    break
                found = True
                found_y = y
                break
        found_flags.append(found)
        found_ys.append(found_y)

    last_green_idx = -1
    for idx, flag in enumerate(found_flags):
        if flag:
            last_green_idx = idx

    last_y_label = None
    for i, tick in enumerate(sorted_ticks):
        tx, ty = tick['tick_pos']

        if tx > x_axis_xmax or tx < y_axis_x - 5:
            found_flags.append(False)
            found_ys.append(None)
            continue

        label = tick['label']
        found = found_flags[i]
        y = found_ys[i]
        result_y_label = None

        if found:
            if day:
                y_label = interpolate_y_label(y + 5, mapped_tick_labels, y_green_label)
            else:
                y_label = interpolate_y_label(y + 5, mapped_tick_labels, y_green_label)
            print(f"║ {label:<10} │ {y_label:<12} ║")
            last_y_label = y_label
            result_y_label = y_label
        else:
            if last_y_label is not None and i < last_green_idx:
                print(f"║ {label:<10} │ {last_y_label:<12} ║")
                result_y_label = last_y_label
            else:
                print(f"║ {label:<10} │ {'-':<12} ║")
                result_y_label = None

    print("╚════════════╧══════════════╝")


# ------------------- Main -------------------
def main1():
    start_time = time.time()
    reader = easyocr.Reader(['en'])
    input_dir = "pdf_images"
    output_dir = "image_final_test"
    os.makedirs(output_dir, exist_ok=True)
    save = 0
    for i in range(3, 183):
        if i == 173:
            continue
        filename = f"cropped_page_{i}.png"
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"{filename} not found or cannot open.")
            continue

        cropped_image = crop_image(img, 25, 25, 50, 50)
        hsv_image = convert_bgr_to_hsv(cropped_image)
        mask_white_image = mask_white_area(hsv_image, np.array([0, 0, 200]), np.array([180, 50, 255]))

        # ===== Axes =====
        detected_line = detect_lines(mask_white_image)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_white_image_y_line = cv2.dilate(mask_white_image, kernel_line, iterations=1)
        mask_white_image_y_line = cv2.morphologyEx(mask_white_image_y_line, cv2.MORPH_CLOSE, kernel_line)
        detected_blur_line = detect_lines(mask_white_image_y_line)

        x_axis = find_x_axis(detected_line, cropped_image.shape[1])
        if x_axis is None:
            x_axis = find_x_axis(detected_blur_line, cropped_image.shape[1])

        y_axis = find_y_axis(detected_blur_line, x_axis)
        if y_axis is None:
            y_axis = find_y_axis(detected_line, x_axis)

        if x_axis is None:
            print(f"{filename} could not detect X axes.")
            continue

        if y_axis is None:
            print(f"{filename} could not detect Y axes.")
            continue

        x_axis_left, x_axis_top, x_axis_right, x_axis_bot = x_axis
        y_axis_left, y_axis_top, y_axis_right, y_axis_bot = y_axis

        # cv2.line(cropped_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 5)
        # cv2.line(cropped_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 255, 0), 5)
        #
        # cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_axis/axis_{i-1}.png", cropped_image)

        # ===== X Minor ticks =====
        minor_tick_coords = find_x_minor_ticks(mask_white_image, x_axis_left, x_axis_top, x_axis_right, x_axis_bot)
        minor_tick_coords = auto_add_x_minor_ticks(minor_tick_coords, x_axis_left, x_axis_top, x_axis_right, x_axis_bot,
                                                   y_axis_left, y_axis_right)

        if minor_tick_coords is None:
            print(f"{filename} could not detect X minor tick.")
            continue

        # for (tx, ty) in minor_tick_coords:
        #     cv2.line(cropped_image, (tx, ty - 10), (tx, ty + 10), (255, 100, 0), 2)
        #
        # cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_minor/minor_{i-1}.png", cropped_image)

        # ===== X labels =====
        strip_ocr_parm = [
            (x_axis_top + 5, x_axis_bot + 75), (x_axis_top + 6, x_axis_bot + 75), (x_axis_top + 7, x_axis_bot + 60),
            (x_axis_top + 8, x_axis_bot + 75), (x_axis_top + 7, x_axis_bot + 165), (x_axis_top + 50, x_axis_bot + 165),
            (x_axis_top + 50, x_axis_bot + 200)
        ]
        labels, label_x, day, success_range = detect_labels_on_x_axis(
            cropped_image, strip_ocr_parm, x_axis_left, x_axis_right, reader
        )

        if labels is None:
            print(f"{filename} could not detect X labels.")
            continue

        # for l, x in zip(labels, label_x):
        #     cv2.putText(cropped_image, l, (x + 180, x_axis_bot + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 50, 255),2)
        #
        # cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_x_label/x_label_{i-1}.png", cropped_image)

        # ===== X month =====
        if len(labels) <= 3:
            months, months_x = detect_month_labels_on_x_axis(cropped_image, strip_ocr_parm, x_axis_left, x_axis_right,
                                                             reader)
            if len(labels) > 1:
                months, months_x = auto_fill_missing_months(months, months_x)
            day = is_day_range(months, months_x)
            for m, x in zip(months, months_x):
                cv2.putText(cropped_image, m, (x + 150, x_axis_bot + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 50, 255),2)
        else:
            months = []

        # cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_month/month_{i-1}.png", cropped_image)

        # ===== Major ticks =====
        strip_x_major_parm = [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 19, 21]
        try:
            major_tick_coords, y_x_major_coord = find_x_major_ticks(
                mask_white_image, strip_x_major_parm,
                x_axis_left, x_axis_right, x_axis_top, x_axis_bot,
                minor_tick_coords, labels, label_x, minor_tick_coords, months
            )
        except Exception:
            print(f"{filename} could not detect major.")
            continue

        # for tx in major_tick_coords:
        #     cv2.line(cropped_image, (tx, y_x_major_coord - 10), (tx, y_x_major_coord + 10), (0, 250, 255), 3)
        #
        # cv2.imwrite(f"/Users/pichayanon/extract_graph/final_version_update/debug_x_major/x_major_{i - 1}.png", cropped_image)

        mapping = map_labels_to_x_major_ticks(label_x, labels, major_tick_coords, x_axis_left, x_axis_right)

        tick_labels = []
        tick_coords_sorted = sorted(minor_tick_coords, key=lambda t: t[0])
        y_strip = int((x_axis_top + x_axis_bot) / 2)

        if is_10_year_range(mapping):
            tick_labels = map_labels_to_minor_ticks_range_10_year_with_quarter(mapping, tick_coords_sorted, y_strip,
                                                                               cropped_image)
        elif is_single_year_range(mapping):
            tick_labels = map_labels_to_minor_ticks_range_5_year_with_quarter(mapping, tick_coords_sorted, y_strip)
        elif day:
            tick_labels = map_labels_to_minor_ticks_range_day(mapping, tick_coords_sorted, y_strip, months, months_x)
        elif len(mapping) == 1:
            tick_labels = map_labels_to_minor_ticks_range_months(mapping, tick_coords_sorted, y_strip, cropped_image)
        else:
            tick_labels = map_labels_to_minor_ticks_single_year_with_quarter(mapping, tick_coords_sorted, y_strip)
        major_tick_coords = auto_add_x_major_ticks(major_tick_coords, x_axis_left, x_axis_right, minor_tick_coords)

        y_major_tick_coords = find_y_major_ticks(mask_white_image, y_axis_top, y_axis_left, y_axis_bot, y_axis_right)
        mask_gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        y_tick_labels = ocr_y_white_label(y_axis_left, y_axis_right, y_axis_top, y_axis_bot, mask_gray_image)
        if len(y_tick_labels) == 0:
            y_tick_labels = ocr_y_white_label(y_axis_left, y_axis_right, y_axis_top, y_axis_bot, mask_white_image)

        y_green_labels = ocr_y_green_label(cropped_image, y_axis, y_tick_labels)
        mapped_tick_labels = map_ticks_to_labels(y_major_tick_coords, y_tick_labels)
        mapped_tick_labels = auto_fill_missing_tick_label(mapped_tick_labels)

        if len(mapped_tick_labels) <= 1:
            mapped_tick_labels = y_green_labels
        scan_green_balls_by_x_ticks(hsv_image, cropped_image, tick_labels, mapped_tick_labels, x_axis, y_axis,
                                    y_green_labels, day)
        cv2.line(cropped_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 5)
        cv2.line(cropped_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 255, 0), 5)

        for (tx, ty) in minor_tick_coords:
            cv2.line(cropped_image, (tx, ty - 10), (tx, ty + 10), (255, 100, 0), 2)

        for tx in major_tick_coords:
            cv2.line(cropped_image, (tx, y_x_major_coord - 10), (tx, y_x_major_coord + 10), (0, 250, 255), 3)

        # for (tx, ty) in y_major_tick_coords:
        #     cv2.line(cropped_image, (0, ty), (cropped_image.shape[1] - 1, ty), (255, 100, 0), 2)

        # for n, tick in enumerate(sorted(tick_labels, key=lambda t: t['tick_pos'][0]), 1):
        #     tx, ty = tick['tick_pos']
        #     label = tick['label']
        #
        #     if tx > x_axis_right - 5:
        #         continue
        #
        #     if tx < y_axis_left - 5:
        #         continue

        plt.figure(figsize=(14, 7))
        plt.title(f'Detected All Ticks & Labels (Image {i-1})')
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'crop_image_{i-1}.png')
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")
        save+=1
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Total Time {elapsed:.2f} sec")
        print(f"Saved Image {save}")


if __name__ == "__main__":
    main1()
