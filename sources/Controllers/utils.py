import cv2
import numpy as np


def class_Order(boxes, categories):
    Z = []
    # Z = [x for _,x in sorted(zip(categories, boxes))]
    cate = np.argsort(categories)
    for index in cate:
        Z.append(boxes[index])

    return Z


def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

        # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked using the
    # integer data type
    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")
    return final_boxes, final_labels


def get_center_point(box):
    left, top, right, bottom = box
    return left + ((right - left) // 2), top + (
        (bottom - top) // 2
    )  # (x_c, y_c) # Need to fix bottom_left and bottom_right


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    image = np.asarray(image)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# def getMissingCorner(categories, boxes): # boxes: top_left, top_right, bottom_left, bottom_right
# 	if 0 not in categories: # Missing top_left
# 		delta_vertical = boxes[3][2] - boxes[1][2]
# 		delta_horizon = boxes[3][3] - boxes[2][3]
# 		x_miss =


import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional
from PIL import Image

PLACE_BO_CONG_AN = "Bộ Công an"
PLACE_C06 = "Cục Cảnh sát quản lý hành chính về trật tự xã hội"
PLACE_C72 = "Cục Cảnh sát Đăng ký, quản lý cư trú và Dữ liệu quốc gia về dân cư"


def remove_accents(input_str: str) -> str:
    """Normalize Vietnamese characters to unaccented lowercase ASCII."""
    if not input_str:
        return ""
    cleaned = input_str.replace("0", "o")
    nfkd_form = unicodedata.normalize("NFKD", cleaned)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


def remove_red_stamp(pil_img: Image.Image) -> Image.Image:
    """
    Tiền xử lý ảnh vùng văn bản để làm mờ/khử con dấu đỏ:
    Kênh Red (R) của con dấu màu đỏ có cường độ sáng rất cao gần với màu nền giấy thẻ,
    trong khi nét chữ in màu đen có giá trị R rất thấp.
    Bằng cách trích xuất kênh Red và tăng tương phản, con dấu đỏ sẽ bị triệt tiêu,
    làm nổi bật nét chữ đen cho mô hình OCR nhận diện chính xác.
    """
    try:
        arr = np.array(pil_img.convert("RGB"))
        r = arr[:, :, 0]
        p_low, p_high = np.percentile(r, (2, 98))
        if p_high > p_low:
            r_norm = np.clip((r.astype(np.float32) - p_low) / (p_high - p_low) * 255.0, 0, 255).astype(np.uint8)
        else:
            r_norm = r
        return Image.fromarray(r_norm).convert("RGB")
    except Exception:
        return pil_img


def extract_issuer_text_from_image(aligned: Image.Image, W: int, H: int, detector) -> str:
    """
    Cố gắng trích xuất chuỗi văn bản nơi cấp từ ảnh mặt sau bằng nhiều kỹ thuật:
    1. Crop từng dòng chức danh với margin an toàn (tránh chém đứt đầu/chân chữ).
    2. Áp dụng tiền xử lý khử con dấu đỏ đè lên chữ.
    3. Thử OCR cả ảnh gốc và ảnh đã lọc dấu đỏ để chọn văn bản rõ nhất.
    4. Quét cả khối bao quanh và vùng đáy nếu là mẫu thẻ mới.
    """
    candidates = []

    # Dòng 1: CỤC TRƯỞNG CỤC CẢNH SÁT...
    crop_1 = aligned.crop((int(W * 0.10), int(H * 0.17), int(W * 0.55), int(H * 0.25)))
    # Dòng 2: QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI...
    crop_2 = aligned.crop((int(W * 0.05), int(H * 0.22), int(W * 0.60), int(H * 0.31)))

    t1_orig = detector.predict(crop_1).strip()
    t2_orig = detector.predict(crop_2).strip()

    t1_clean = detector.predict(remove_red_stamp(crop_1)).strip()
    t2_clean = detector.predict(remove_red_stamp(crop_2)).strip()

    line1 = t1_clean if len(t1_clean) >= len(t1_orig) else t1_orig
    line2 = t2_clean if len(t2_clean) >= len(t2_orig) else t2_orig
    combined = f"{line1} {line2}".strip()
    if combined:
        candidates.append(combined)

    # Nếu dòng đơn quá ngắn, thử quét cả khối chức danh
    if len(combined) < 10:
        crop_full = aligned.crop((int(W * 0.05), int(H * 0.17), int(W * 0.60), int(H * 0.32)))
        t_full = detector.predict(crop_full).strip()
        t_full_clean = detector.predict(remove_red_stamp(crop_full)).strip()
        best_full = t_full_clean if len(t_full_clean) > len(t_full) else t_full
        if len(best_full) > len(combined):
            candidates.append(best_full)

    # Quét thêm vùng nửa dưới (đối với thẻ Căn cước mới 2024 có chữ 'Nơi cấp: BỘ CÔNG AN')
    try:
        crop_bottom = aligned.crop((int(W * 0.05), int(H * 0.65), int(W * 0.65), int(H * 0.85)))
        t_bottom = detector.predict(remove_red_stamp(crop_bottom)).strip()
        if any(kw in remove_accents(t_bottom) for kw in ["bo cong an", "cong an", "noi cap"]):
            candidates.append(t_bottom)
    except Exception:
        pass

    return " ".join(candidates).strip()


def infer_place_from_date(date_str: str) -> Optional[str]:
    """
    Suy luận nơi cấp theo căn cứ pháp luật Việt Nam:
    1. Trước 10/10/2018 (Thông tư 61/2015/TT-BCA): Cục Cảnh sát Đăng ký, quản lý cư trú và Dữ liệu quốc gia về dân cư (C72).
    2. Từ 10/10/2018 đến 30/06/2024 (Thông tư 33/2018/TT-BCA & Thông tư 06/2021/TT-BCA): Cục Cảnh sát quản lý hành chính về trật tự xã hội (C06).
    3. Từ 01/07/2024 trở đi (Luật Căn cước 2023 & Thông tư 17/2024/TT-BCA): Bộ Công an.
    """
    if not date_str:
        return None

    match = re.search(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})", date_str)
    if match:
        try:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            dt = datetime(year, month, day)
            if dt < datetime(2018, 10, 10):
                return PLACE_C72
            elif dt < datetime(2024, 7, 1):
                return PLACE_C06
            else:
                return PLACE_BO_CONG_AN
        except Exception:
            pass

    match_y = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
    if match_y:
        y = int(match_y.group(1))
        if y < 2018:
            return PLACE_C72
        elif y > 2024:
            return PLACE_BO_CONG_AN
        elif y < 2024:
            return PLACE_C06

    return None


def infer_place_from_text(raw_text: str) -> Optional[str]:
    """Phân loại nơi cấp từ chuỗi OCR thô sử dụng keyword matching và fuzzy distance."""
    if not raw_text:
        return None

    clean = remove_accents(raw_text)

    # 1. Keyword matching
    if any(k in clean for k in ["bo cong an", "ministry of public security"]):
        return PLACE_BO_CONG_AN
    if "cong an" in clean and "canh sat" not in clean:
        return PLACE_BO_CONG_AN

    if any(k in clean for k in ["cu tru", "dan cu", "du lieu", "quoc gia", "c72"]):
        return PLACE_C72

    if any(k in clean for k in ["hanh chinh", "trat tu", "xa hoi", "qlhc", "ttxh"]):
        return PLACE_C06

    # 2. Fuzzy similarity matching
    candidates = [
        (
            PLACE_BO_CONG_AN,
            [
                "bo cong an",
                "noi cap bo cong an",
                "ministry of public security",
            ],
        ),
        (
            PLACE_C06,
            [
                "cuc truong cuc canh sat quan ly hanh chinh ve trat tu xa hoi",
                "cuc canh sat quan ly hanh chinh ve trat tu xa hoi",
                "quan ly hanh chinh ve trat tu xa hoi",
                "cuc canh sat quan ly hanh chinh",
            ],
        ),
        (
            PLACE_C72,
            [
                "cuc truong cuc canh sat dang ky quan ly cu tru va du lieu quoc gia ve dan cu",
                "cuc canh sat dang ky quan ly cu tru va du lieu quoc gia ve dan cu",
                "dang ky quan ly cu tru va du lieu quoc gia ve dan cu",
            ],
        ),
    ]

    best_score = 0.0
    best_place = None

    for place_name, phrases in candidates:
        for phrase in phrases:
            score = SequenceMatcher(None, clean, phrase).ratio()
            for word in phrase.split():
                if len(word) >= 3 and word in clean:
                    score += 0.08
            if score > best_score:
                best_score = score
                best_place = place_name

    if best_score >= 0.25:
        return best_place

    return None


def determine_issue_place(issue_date_str: str, ocr_raw_text: str) -> str:
    """
    Kết hợp giữa việc cố gắng trích xuất nội dung từ ảnh (OCR text)
    và quy tắc ngày cấp:
    1. Ưu tiên nội dung trích xuất từ ảnh (ocr_raw_text):
       Nếu OCR nhận diện được từ khóa / độ tương đồng cao với 1 trong 3 cơ quan,
       hệ thống sẽ lấy kết quả trích xuất đó và sửa lỗi/chuẩn hóa về tên cơ quan đúng.
    2. Nếu OCR vùng nơi cấp bị mờ hoặc không trích xuất được chữ có nghĩa:
       Hệ thống áp dụng quy tắc ngày cấp (căn cứ pháp lý 100%) để suy luận chính xác nơi cấp.
    3. Fallback: Mặc định là C06 (chiếm đại đa số thẻ hiện nay).
    """
    place_by_text = infer_place_from_text(ocr_raw_text)
    place_by_date = infer_place_from_date(issue_date_str)

    # Ưu tiên 1: Kết quả trích xuất thực tế từ ảnh nếu đọc được nội dung có nghĩa
    if place_by_text:
        return place_by_text

    # Ưu tiên 2: Quy tắc ngày cấp nếu OCR trích xuất không ra chữ
    if place_by_date:
        return place_by_date

    # Mặc định fallback
    return PLACE_C06

