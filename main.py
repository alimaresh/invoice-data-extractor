"""
Invoice Visual Analyzer

A Computer Vision–based system for automatic invoice data extraction.

This project applies classical image preprocessing techniques using OpenCV,
followed by OCR with Tesseract to extract structured information such as
invoice date and total amount from real-world invoice images.

Includes:
- Adjustable preprocessing (contrast, brightness, blur)
- CLAHE adaptive contrast enhancement
- OCR text extraction
- Rule-based data parsing
- Web interface with real-time preview
- JSON-based persistent storage

Developed as part of a Computer Vision academic project.
Author: Ali Maresh
"""

from flask import Flask, request, jsonify, send_from_directory
import pytesseract
import re
import os
import tempfile 
import json 
import cv2
import numpy as np
from flask import make_response


app = Flask(__name__)  # إنشاء تطبيق Flask

# Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = os.path.join( 
    os.getcwd(), "tesseract", "tesseract.exe"
)

# ---------------------------------------
# OpenCV Preprocessing
# ---------------------------------------

def preprocess_image(image_path, contrast=1.0, brightness=0, blur=0):  # دالة لمعالجة الصورة قبل OCR
    image = cv2.imread(image_path)  
    if image is None:  
        return None  

    # تطبيق التباين والسطوع
    # alpha = التباين, beta = السطوع
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    #  تطبيق التنعيم (Blur)
    if blur > 0:
        ksize = int(blur) * 2 + 1  # Ensure odd kernel size (3, 5, 7...)
        adjusted = cv2.GaussianBlur(adjusted, (ksize, ksize), 0)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY) 

    # إزالة الضوضاء
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)  

    # 5. CLAHE (تحسين التباين التكيفي)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  #تحسين التباين عبر CLAHE
    enhanced = clahe.apply(denoised)  # تطبيق تحسين التباين على الصورة
    
    # CLAHE :
    # يقسم الصورة إلى مربعات صغيرة
    # يحسب Histogram لكل مربع على حدة
    # يعيد توزيع الإضاءة داخل كل مربع لزيادة التباين
    # يضع حدًا أقصى للتباين حتى لا تتضخم الضوضاء
    # يدمج المربعات معًا بسلاسة حتى لا تظهر حدود بينها

    return enhanced 


# ---------------------------------------
# OCR
# ---------------------------------------

def extract_text(image):  # دالة لاستخراج النص من الصورة
    return pytesseract.image_to_string(image, lang="eng")


# ---------------------------------------
# Date Extraction
# ---------------------------------------

def extract_date(text): 
    numeric = re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', text)  # البحث عن تاريخ بصيغة رقمية

    if numeric: 
        return numeric.group() 

    month_names = ( 
        "january|february|march|april|may|june|"
        "july|august|september|october|november|december"
    )

    text_lower = text.lower() 
    textual = re.search(  
        rf'({month_names})\s+\d{{1,2}}(,\s*\d{{4}})?',
        text_lower
    )

    if textual:  # في حال العثور على تاريخ نصي
        return textual.group().title()  

    return None

# ---------------------------------------
# Total Extraction (multiple keywords)
# ---------------------------------------

def extract_total(text): 
    text = text.lower().replace(",", ".") 

    total_patterns = [  
        r'\btotal\b',
        r'\bgrand\s+total\b',
        r'\bamount\s+due\b',
        r'\bbalance\s+due\b',
        r'\binvoice\s+total\b'
    ]

    for pattern in total_patterns: 
        regex = rf'{pattern}[^0-9$]*\$?\s*([0-9]+(\.[0-9]{{2}})?)'  
        match = re.search(regex, text)  # البحث في النص
        if match:
            return match.group(1) 

    subtotal_patterns = [ 
        r'\bsubtotal\b',
        r'\bsub\s+total\b'
    ]

    for pattern in subtotal_patterns:
        regex = rf'{pattern}[^0-9$]*\$?\s*([0-9]+(\.[0-9]{{2}})?)'  # بناء التعبير النمطي
        match = re.search(regex, text)  
        if match:  
            return match.group(1) 

    numbers = re.findall(r'\$?\s*([0-9]+\.[0-9]{2})', text)  # استخراج جميع القيم العشرية
    if numbers:  # في حال وجود قيم
        return max(numbers, key=lambda x: float(x))  # إرجاع أعلى قيمة

    return None 


def debug_print_ocr_text(text):  
    print("\n========== OCR DEBUG OUTPUT ==========\n")  
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):  # المرور على كل سطر مع رقم
        clean_line = line.strip()  
        if clean_line: 
            print(f"{i:03d}: {clean_line}") 

    print("\n========== END OCR DEBUG ==========\n") 

DATA_FILE = "data/invoices.json"  

def load_saved_data():  # دالة تحميل البيانات المخزنة
    if not os.path.exists(DATA_FILE): 
        return [] 
    with open(DATA_FILE, "r", encoding="utf-8") as f:  # فتح الملف للقراءة
        try:
            return json.load(f)  # تحميل البيانات من JSON
        except:
            return []

def save_invoice_data(new_entries):  # دالة حفظ بيانات جديدة
    data = load_saved_data()  
    data.extend(new_entries)  
    with open(DATA_FILE, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=4, ensure_ascii=False) 

# ---------------------------------------
# Routes
# ---------------------------------------

@app.route("/")  # المسار الرئيسي
def home():
    return send_from_directory(".", "index.html") 

@app.route("/invoices", methods=["GET"])  # مسار جلب الفواتير المحفوظة
def get_saved_invoices():
    return jsonify(load_saved_data())  # إرجاع البيانات بصيغة JSON

@app.route("/preview", methods=["POST"])
def preview_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        contrast = float(request.form.get("contrast", 1.0))
        brightness = int(float(request.form.get("brightness", 0)))
        blur = int(float(request.form.get("blur", 0)))
    except ValueError:
        return jsonify({"error": "Invalid parameters"}), 400

    # Create temp file, close it, then write/read
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp.close()
    
    try:
        file.save(temp.name)
        processed = preprocess_image(temp.name, contrast, brightness, blur)
    finally:
        if os.path.exists(temp.name):
            os.remove(temp.name)

    if processed is None:
         return jsonify({"error": "Processing failed"}), 500

    # Encode processed image (which is grayscale) to PNG for display
    _, buffer = cv2.imencode(".png", processed)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route("/analyze", methods=["POST"])
def analyze_invoice():
    images = request.files.getlist("images")
    
    # Get preprocessing params from form
    try:
        contrast = float(request.form.get("contrast", 1.0))
        brightness = int(float(request.form.get("brightness", 0)))
        blur = int(float(request.form.get("blur", 0)))
    except ValueError:
        contrast, brightness, blur = 1.0, 0, 0

    results = []  

    for img in images: 
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp.close()

        try:
            img.save(temp.name)  # حفظ الصورة مؤقتًا
            processed = preprocess_image(temp.name, contrast, brightness, blur)  # معالجة الصورة

            if processed is None:  # التحقق من نجاح المعالجة
                continue

            text = extract_text(processed)  # استخراج النص من الصورة
            debug_print_ocr_text(text)  # طباعة النص للتصحيح

            date = extract_date(text)  
            total = extract_total(text)  
            
            if date and total:  # التحقق من وجود تاريخ وإجمالي
                results.append({  # إضافة النتيجة
                    "filename": img.filename,  
                    "date": date,  
                    "total": total,  
                    "note": text.strip()  
                })
        
        finally:
            if os.path.exists(temp.name):
                os.remove(temp.name)

    if results: 
        save_invoice_data(results)  

    return jsonify(results) 


if __name__ == "__main__": 
    app.run(debug=False, use_reloader=False)
