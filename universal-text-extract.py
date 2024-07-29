import fitz  # PyMuPDF
import docx
import subprocess
from bs4 import BeautifulSoup
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import json
import os
import logging
from selenium import webdriver
from langdetect import detect
from google.cloud import vision
from pdfminer.high_level import extract_text as pdf_extract_text
from textblob import TextBlob

# Setup logging
logging.basicConfig(level=logging.INFO, filename='text_extraction.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update path as needed

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.medianBlur(image, 5)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def ocr_image(image_path, use_google_vision=False):
    processed_image = preprocess_image(image_path)
    if not use_google_vision:
        return pytesseract.image_to_string(processed_image)
    else:
        client = vision.ImageAnnotatorClient()
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        return response.full_text_annotation.text

def ocr_pdf(pdf_path, use_google_vision=False):
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        image_path = 'temp_page.png'
        page.save(image_path, 'PNG')
        text += ocr_image(image_path, use_google_vision)
    return text

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logging.error(f"Error extracting text from DOCX {docx_path}: {str(e)}")
        return None

def extract_text_from_doc(doc_path):
    try:
        result = subprocess.run(['antiword', doc_path], stdout=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except Exception as e:
        logging.error(f"Error extracting text from DOC {doc_path}: {str(e)}")
        return None

def extract_text_from_html(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        logging.error(f"Error extracting text from HTML content: {str(e)}")
        return None

def extract_text_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df.to_string(index=False)
    except Exception as e:
        logging.error(f"Error extracting text from CSV {csv_path}: {str(e)}")
        return None

def extract_text_from_excel(excel_path):
    try:
        df = pd.read_excel(excel_path)
        return df.to_string(index=False)
    except Exception as e:
        logging.error(f"Error extracting text from Excel {excel_path}: {str(e)}")
        return None

def extract_text_from_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        return json.dumps(data, indent=4)
    except Exception as e:
        logging.error(f"Error extracting text from JSON {json_path}: {str(e)}")
        return None

def extract_text(file_path, lang='en', use_google_vision=False):
    _, file_extension = os.path.splitext(file_path)

    try:
        if file_extension.lower() == '.pdf':
            return ocr_pdf(file_path, use_google_vision)
        elif file_extension.lower() in ['.doc', '.docx']:
            return extract_text_from_doc(file_path) if file_extension.lower() == '.doc' else extract_text_from_docx(file_path)
        elif file_extension.lower() in ['.html', '.htm']:
            with open(file_path, 'r', encoding='utf-8') as file:
                return extract_text_from_html(file.read())
        elif file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            return ocr_image(file_path, use_google_vision)
        elif file_extension.lower() == '.csv':
            return extract_text_from_csv(file_path)
        elif file_extension.lower() in ['.xls', '.xlsx']:
            return extract_text_from_excel(file_path)
        elif file_extension.lower() == '.json':
            return extract_text_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        return None

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return 'unknown'

def translate_text(text, target_language='en'):
    try:
        return str(TextBlob(text).translate(to=target_language))
    except Exception:
        return text

def integrated_text_extraction(file_path, lang='en', use_google_vision=False):
    text = extract_text(file_path, lang, use_google_vision)
    detected_language = detect_language(text)
    if detected_language != lang:
        text = translate_text(text, target_language=lang)
    return text

# Example usage
if __name__ == "__main__":
    file_path = 'example.pdf'
    extracted_text = integrated_text_extraction(file_path)
    print(extracted_text)
