# OCR Setup Instructions for Windows

## Prerequisites

This application uses Tesseract OCR for extracting text from images and PDFs.

### Install Tesseract OCR

1. **Download Tesseract**:
   - Visit: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)

2. **Install**:
   - Run the installer
   - **Important**: Note the installation path (default: `C:\Program Files\Tesseract-OCR`)
   - Make sure to check "Add to PATH" during installation

3. **Verify Installation**:
   ```powershell
   tesseract --version
   ```

### If PATH is not set automatically

Add Tesseract to your PATH manually:
```powershell
$env:Path += ";C:\Program Files\Tesseract-OCR"
```

Or set it in your Python code (add to `user_docs/parser.py`):
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Install Python Dependencies

```powershell
pip install -r loan_chatbot/requirements.txt
```

### Additional Requirement for PDF to Image Conversion

Install Poppler (required by pdf2image):
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler`
3. Add `C:\Program Files\poppler\Library\bin` to PATH
