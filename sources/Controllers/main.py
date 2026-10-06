import os
import re
import json
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

import databases
import numpy as np
import torch

# Fix for PyTorch 2.6+ default weights_only=True
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import yolov5
from fastapi import Depends, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from pylibsrtp import Session
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import sources.Controllers.config as cfg
from sources import app, templates
from sources.Controllers import utils
from sources.Models import models
from sources.Models.database import SQLALCHEMY_DATABASE_URL, SessionLocal, engine
from sources.Models.models import Feedback

""" ---- Setup ---- """
# Init Database
database = databases.Database(SQLALCHEMY_DATABASE_URL)
models.Base.metadata.create_all(bind=engine)


async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Startup database server before start app
@app.on_event("startup")
async def startup_database():
    await database.connect()


# Shutdown database sever after closed app
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Init yolov5 model
CORNER_MODEL = yolov5.load(cfg.CORNER_MODEL_PATH)
CONTENT_MODEL = yolov5.load(cfg.CONTENT_MODEL_PATH)
FACE_MODEL = yolov5.load(cfg.FACE_MODEL_PATH)

# Set conf and iou threshold -> Remove overlap and low confident bounding boxes
CONTENT_MODEL.conf = cfg.CONF_CONTENT_THRESHOLD
CONTENT_MODEL.iou = cfg.IOU_CONTENT_THRESHOLD

# CORNER_MODEL.conf = cfg.CONF_CORNER_THRESHOLD
# CORNER_MODEL.iou = cfg.IOU_CORNER_THRESHOLD

# Config directory
UPLOAD_FOLDER = cfg.UPLOAD_FOLDER
SAVE_DIR = cfg.SAVE_DIR
FACE_CROP_DIR = cfg.FACE_DIR

""" ---- ##### -----"""


class feedback_Request(BaseModel):
    content: str
    rating: int

    class Config:
        orm_mode = True


class contact_Request(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    message: str

    class Config:
        orm_mode = True


""" Recognizion detected parts in ID """
config = Cfg.load_config_from_name(
    "vgg_seq2seq"
)  # OR vgg_transformer -> acc || vgg_seq2seq -> time
# config = Cfg.load_config_from_file(cfg.OCR_CFG)
# config['weights'] = cfg.OCR_MODEL_PATH
config["cnn"]["pretrained"] = False
config["device"] = cfg.DEVICE
config["predictor"]["beamsearch"] = False
detector = Predictor(config)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/home")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/id_card")
async def id_extract_page(request: Request):
    return templates.TemplateResponse(request=request, name="idcard.html")


@app.get("/ekyc")
async def ekyc_page(request: Request):
    return templates.TemplateResponse(request=request, name="ekyc.html")


@app.get("/feedback")
async def feedback_page(request: Request):
    return templates.TemplateResponse(request=request, name="feedback.html")


@app.get("/contact")
async def contact_page(request: Request):
    return templates.TemplateResponse(request=request, name="contact.html")


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


def extract_issuer_text_from_image(aligned: Image.Image, W: int, H: int, ocr_detector) -> str:
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

    t1_orig = ocr_detector.predict(crop_1).strip()
    t2_orig = ocr_detector.predict(crop_2).strip()

    t1_clean = ocr_detector.predict(remove_red_stamp(crop_1)).strip()
    t2_clean = ocr_detector.predict(remove_red_stamp(crop_2)).strip()

    line1 = t1_clean if len(t1_clean) >= len(t1_orig) else t1_orig
    line2 = t2_clean if len(t2_clean) >= len(t2_orig) else t2_orig
    combined = f"{line1} {line2}".strip()
    if combined:
        candidates.append(combined)

    # Nếu dòng đơn quá ngắn, thử quét cả khối chức danh
    if len(combined) < 10:
        crop_full = aligned.crop((int(W * 0.05), int(H * 0.17), int(W * 0.60), int(H * 0.32)))
        t_full = ocr_detector.predict(crop_full).strip()
        t_full_clean = ocr_detector.predict(remove_red_stamp(crop_full)).strip()
        best_full = t_full_clean if len(t_full_clean) > len(t_full) else t_full
        if len(best_full) > len(combined):
            candidates.append(best_full)

    # Quét thêm vùng nửa dưới (đối với thẻ Căn cước mới 2024 có chữ 'Nơi cấp: BỘ CÔNG AN')
    try:
        crop_bottom = aligned.crop((int(W * 0.05), int(H * 0.65), int(W * 0.65), int(H * 0.85)))
        t_bottom = ocr_detector.predict(remove_red_stamp(crop_bottom)).strip()
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


# Gán vào module utils để đảm bảo utils.extract_issuer_text_from_image luôn khả dụng
utils.extract_issuer_text_from_image = extract_issuer_text_from_image
utils.determine_issue_place = determine_issue_place
utils.infer_place_from_date = infer_place_from_date
utils.infer_place_from_text = infer_place_from_text
utils.remove_red_stamp = remove_red_stamp


def extract_back_info(img_path):
    if not os.path.exists(img_path):
        return {"issue_date": "", "issue_place": ""}

    img_pil = Image.open(img_path).convert("RGB")

    # Corner detection & perspective transform if corners detected
    try:
        CORNER = CORNER_MODEL(img_pil)
        predictions = CORNER.pred[0]
        categories = predictions[:, 5].tolist()
        if len(categories) == 4:
            boxes = utils.class_Order(predictions[:, :4].tolist(), categories)
            center_points = list(map(utils.get_center_point, boxes))
            center_points = np.asarray(center_points)
            aligned = utils.four_point_transform(img_pil, center_points)
            aligned = Image.fromarray(aligned).convert("RGB")
        else:
            aligned = img_pil
    except Exception:
        aligned = img_pil

    W, H = aligned.size

    # 1. Date of issue (Ngày, tháng, năm / Date, month, year)
    crop_date = aligned.crop((int(W * 0.02), int(H * 0.12), int(W * 0.58), int(H * 0.23)))
    text_date_raw = detector.predict(crop_date)

    issue_date = ""
    match_date = re.search(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})", text_date_raw)
    if match_date:
        d, m, y = match_date.group(1), match_date.group(2), match_date.group(3)
        issue_date = f"{int(d):02d}/{int(m):02d}/{y}"
    else:
        match_date_text = re.search(
            r"(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2})\s*(?:năm|nam)\s*(\d{4})",
            text_date_raw,
            re.IGNORECASE,
        )
        if match_date_text:
            d, m, y = match_date_text.group(1), match_date_text.group(2), match_date_text.group(3)
            issue_date = f"{int(d):02d}/{int(m):02d}/{y}"
        else:
            issue_date = text_date_raw.strip()

    # 2. Cố gắng trích xuất nội dung Nơi cấp từ ảnh (Tiền xử lý lọc con dấu + quét đa vùng)
    ocr_raw_place = extract_issuer_text_from_image(aligned, W, H, detector)

    # 3. Chuẩn hóa nơi cấp: Ưu tiên nội dung OCR trích xuất được từ ảnh, fallback sang quy tắc ngày cấp
    issue_place = determine_issue_place(issue_date, ocr_raw_place)

    return {
        "issue_date": issue_date,
        "issue_place": issue_place,
    }



@app.post("/uploader")
async def upload(
    file: Optional[UploadFile] = File(None),
    file_back: Optional[UploadFile] = File(None)
):
    if not os.path.isdir(cfg.UPLOAD_FOLDER):
        os.makedirs(cfg.UPLOAD_FOLDER, exist_ok=True)

    # Clean previous uploads
    for uploaded_img in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        except Exception:
            pass

    front_img_path = None
    back_img_path = None

    if file is not None and getattr(file, "filename", None) not in [None, "", "NULL", "undefined"]:
        if file.filename == "WRONG_EXTS":
            return JSONResponse(status_code=404, content={"message": "This file is not supported!"})
        front_img_path = f"./{UPLOAD_FOLDER}/front_{file.filename}"
        contents = await file.read()
        with open(front_img_path, "wb") as f:
            f.write(contents)

    if file_back is not None and getattr(file_back, "filename", None) not in [None, "", "NULL", "undefined"]:
        if file_back.filename == "WRONG_EXTS":
            return JSONResponse(status_code=404, content={"message": "This file is not supported!"})
        back_img_path = f"./{UPLOAD_FOLDER}/back_{file_back.filename}"
        contents_back = await file_back.read()
        with open(back_img_path, "wb") as f:
            f.write(contents_back)

    if front_img_path is None and back_img_path is None:
        return JSONResponse(status_code=403, content={"message": "No file selected!"})

    FIELDS_DETECTED = []
    if front_img_path is not None:
        res_front = await extract_info(path_id=front_img_path)
        if isinstance(res_front, JSONResponse) and res_front.status_code >= 400:
            if back_img_path is None:
                return res_front
            FIELDS_DETECTED = []
        else:
            try:
                body = json.loads(res_front.body.decode())
                FIELDS_DETECTED = body.get("data", [])
            except Exception:
                FIELDS_DETECTED = []

    issue_date = ""
    issue_place = ""
    if back_img_path is not None:
        res_back = extract_back_info(back_img_path)
        issue_date = res_back.get("issue_date", "")
        issue_place = res_back.get("issue_place", "")

    response = {
        "data": FIELDS_DETECTED,
        "issue_date": issue_date,
        "issue_place": issue_place,
    }
    return JSONResponse(content=jsonable_encoder(response))


@app.post("/extract")
# @app.api_route("/extract", methods=["GET", "POST"])
async def extract_info(ekyc=False, path_id=None):
    """Check if uploaded image exist"""
    if not os.path.isdir(cfg.UPLOAD_FOLDER):
        os.mkdir(cfg.UPLOAD_FOLDER)

    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if path_id is not None:
        img = path_id
    elif not ekyc and INPUT_IMG:
        img = os.path.join(UPLOAD_FOLDER, INPUT_IMG[0])
    else:
        img = path_id

    CORNER = CORNER_MODEL(img)
    # CORNER.save(save_dir='results/')
    predictions = CORNER.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if len(categories) != 4:
        error = "Detecting corner failed!"
        return JSONResponse(status_code=401, content={"message": error})
    boxes = utils.class_Order(predictions[:, :4].tolist(), categories)  # x1, x2, y1, y2
    IMG = Image.open(img).convert("RGB")
    center_points = list(map(utils.get_center_point, boxes))

    """ Temporary fixing """
    c2, c3 = center_points[2], center_points[3]
    c2_fix, c3_fix = (c2[0], c2[1] + 30), (c3[0], c3[1] + 30)
    center_points = [center_points[0], center_points[1], c2_fix, c3_fix]
    center_points = np.asarray(center_points)
    aligned = utils.four_point_transform(IMG, center_points)
    # Convert from OpenCV to PIL format
    aligned = Image.fromarray(aligned).convert("RGB")
    # aligned.save('res.jpg')
    # CORNER.show()

    CONTENT = CONTENT_MODEL(aligned)
    # CONTENT.save(save_dir='results/')
    predictions = CONTENT.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if 7 not in categories:
        if len(categories) < 9:
            error = "Missing fields! Detecting content failed!"
            return JSONResponse(status_code=402, content={"message": error})
    elif 7 in categories:
        if len(categories) < 10:
            error = "Missing fields! Detecting content failed!"
            return JSONResponse(status_code=402, content={"message": error})

    boxes = predictions[:, :4].tolist()

    """ Non Maximum Suppression """
    boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)
    boxes = utils.class_Order(boxes, categories)  # x1, x2, y1, y2
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    else:
        for f in os.listdir(SAVE_DIR):
            os.remove(os.path.join(SAVE_DIR, f))

    for index, box in enumerate(boxes):
        left, top, right, bottom = box
        if 5 < index < 9:
            # right = c3[0]
            right = right + 100
        cropped_image = aligned.crop((left, top, right, bottom))
        cropped_image.save(os.path.join(SAVE_DIR, f"{index}.jpg"))

    FIELDS_DETECTED = []  # Collecting all detected parts
    for idx, img_crop in enumerate(sorted(os.listdir(SAVE_DIR))):
        if idx > 0:
            img_ = Image.open(os.path.join(SAVE_DIR, img_crop))
            s = detector.predict(img_)
            FIELDS_DETECTED.append(s)

    if 7 in categories:
        FIELDS_DETECTED = (
            FIELDS_DETECTED[:6]
            + [FIELDS_DETECTED[6] + ", " + FIELDS_DETECTED[7]]
            + [FIELDS_DETECTED[8]]
        )

    response = {"data": FIELDS_DETECTED}

    response = jsonable_encoder(response)
    return JSONResponse(content=response)


@app.post("/download")
async def download(file: str = Form(...)):
    if file != "undefined":
        noti = "Download file successfully!"
        return JSONResponse(status_code=201, content={"message": noti})
    else:
        error = "No file to download!"
        return JSONResponse(status_code=405, content={"message": error})


@app.post("/feedback")
async def save_feedback(
    content: str = Form(...), rating: int = Form(...), db: Session = Depends(get_db)
):
    feedback = Feedback()
    feedback.content = content
    feedback.rating = rating
    db.add(feedback)
    db.commit()

    response = {"code": "200", "content": "save successfully"}

    return JSONResponse(content=response)


@app.post("/contact")
async def contact(request: contact_Request):
    # print(request.name)
    pass


@app.post("/ekyc/uploader")
async def get_id_card(id: UploadFile = File(...), img: UploadFile = File(...)):
    INPUT_IMG = os.listdir(UPLOAD_FOLDER)
    if INPUT_IMG is not None:
        for uploaded_img in INPUT_IMG:
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))

    id_location = f"./{UPLOAD_FOLDER}/{id.filename}"
    id_contents = await id.read()

    with open(id_location, "wb") as f:
        f.write(id_contents)

    img_location = f"./{UPLOAD_FOLDER}/{img.filename}"
    img_contents = await img.read()
    with open(img_location, "wb") as f_:
        f_.write(img_contents)

    # Validating file
    INPUT_FILE = os.listdir(UPLOAD_FOLDER)
    if "NULL_1" in INPUT_FILE and "NULL_2" not in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing ID card image!"
        return JSONResponse(status_code=410, content={"message": error})
    elif "NULL_2" in INPUT_FILE and "NULL_1" not in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing person image!"
        return JSONResponse(status_code=411, content={"message": error})
    elif "NULL_1" in INPUT_FILE and "NULL_2" in INPUT_FILE:
        for uploaded_img in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, uploaded_img))
        error = "Missing ID card and person images!"
        return JSONResponse(status_code=412, content={"message": error})
    else:
        id_name = id.filename.split(".")
        new_id_name = f"./{UPLOAD_FOLDER}/id.{id_name[-1]}"
        os.rename(id_location, new_id_name)
        img_name = img.filename.split(".")
        new_img_name = f"./{UPLOAD_FOLDER}/person.{img_name[-1]}"
        os.rename(img_location, new_img_name)

    FACE = FACE_MODEL(new_img_name)
    predictions = FACE.pred[0]
    categories = predictions[:, 5].tolist()  # Class
    if 0 not in categories:
        error = "No face detected!"
        return JSONResponse(status_code=413, content={"message": error})
    elif categories.count(0) > 1:
        error = "Multiple faces detected!"
        return JSONResponse(status_code=414, content={"message": error})

    boxes = predictions[:, :4].tolist()

    """ Non Maximum Suppression """
    # boxes, categories = utils.non_max_suppression_fast(np.array(boxes), categories, 0.7)

    if not os.path.isdir(FACE_CROP_DIR):
        os.mkdir(FACE_CROP_DIR)
    else:
        for f in os.listdir(FACE_CROP_DIR):
            os.remove(os.path.join(FACE_CROP_DIR, f))

    FACE_IMG = Image.open(new_img_name).convert("RGB")
    # left, top, right, bottom = boxes[0]
    cropped_image = FACE_IMG.crop((boxes[0]))
    cropped_image.save(os.path.join(FACE_CROP_DIR, "face_crop.jpg"))

    return await extract_info(ekyc=True, path_id=new_id_name)
