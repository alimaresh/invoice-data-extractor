# Invoice Visual Analyzer

Automatic extraction of invoice data using OpenCV and OCR.

------------------------------------------------------------------------

## Overview

Invoice Visual Analyzer is a computer vision--based system that extracts
structured information from invoice images.

The system processes uploaded invoice images using OpenCV preprocessing
techniques, applies OCR with Tesseract, and extracts key fields such as:

-   Invoice Date\
-   Total Amount

A web interface allows real-time adjustment of preprocessing parameters
to improve OCR accuracy.

------------------------------------------------------------------------

## Project Goal

The objective of this project is to demonstrate how classical computer
vision techniques combined with OCR can automatically extract meaningful
information from real-world invoice images without training deep
learning models.

------------------------------------------------------------------------

## Technologies Used

-   Python\
-   OpenCV\
-   Tesseract OCR\
-   Flask\
-   HTML / CSS / JavaScript

------------------------------------------------------------------------

## System Architecture

### 1. Image Preprocessing (OpenCV)

-   Brightness adjustment\
-   Contrast adjustment\
-   Gaussian blur\
-   Grayscale conversion\
-   CLAHE (Adaptive contrast enhancement)

### 2. Text Extraction (OCR)

Tesseract OCR converts the processed image into raw text.

### 3. Information Extraction

-   Invoice date detection (multiple formats supported)\
-   Total amount extraction (keyword-based + fallback logic)

------------------------------------------------------------------------

## Web Interface Features

-   Drag & drop invoice upload\
-   Real-time preprocessing preview\
-   Adjustable contrast, brightness, and blur\
-   Structured result display\
-   JSON-based persistent storage

------------------------------------------------------------------------

## Data Storage

Extracted invoices are stored in:

data/invoices.json

------------------------------------------------------------------------

## How to Run

1.  Install dependencies:

pip install flask opencv-python pytesseract

2.  Ensure Tesseract OCR is installed on your system.

3.  Run the application:

python app.py

4.  Open in browser:

http://127.0.0.1:5000

------------------------------------------------------------------------

## Limitations

-   Works best on printed invoices\
-   Handwritten invoices may reduce accuracy\
-   English language only

------------------------------------------------------------------------

## Future Improvements

-   Table structure detection\
-   Multilingual OCR support\
-   Deep learning--based layout analysis\
-   Confidence scoring

------------------------------------------------------------------------

## Academic Context

Developed as part of a Computer Vision course to demonstrate practical
application of classical image processing combined with OCR-based text
extraction.

------------------------------------------------------------------------

Author: Ali Maresh
