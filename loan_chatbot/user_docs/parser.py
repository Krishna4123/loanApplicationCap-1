import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import pypdf

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        str: Extracted text.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error reading image: {e}"

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file. First tries direct text extraction,
    then falls back to OCR if the PDF contains scanned images.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text.
    """
    try:
        # Try direct text extraction first
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # If we got meaningful text, return it
            if text.strip():
                return text.strip()
        
        # If no text found, use OCR
        print("No text found in PDF, using OCR...")
        images = convert_from_path(pdf_path)
        text = ""
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            text += pytesseract.image_to_string(image) + "\n"
        
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def parse_user_application(file_path: str) -> str:
    """
    Reads the content of a user-uploaded file (image or PDF) using OCR.
    
    Args:
        file_path (str): Path to the uploaded file.
        
    Returns:
        str: The extracted content.
    """
    if not os.path.exists(file_path):
        return "Error: File not found."
    
    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    
    # Route to appropriate handler
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']:
        return extract_text_from_image(file_path)
    else:
        return f"Error: Unsupported file type '{ext}'. Please upload a PDF or image file."
