import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from PIL import Image
import tabula 
import base64
import math

def extract_text(pdf_path):
    """Extract text from a PDF file"""
    document = fitz.open(pdf_path)
    text = ""
    # iterate through each page and extract
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def extract_and_compress_images(pdf_path, target_base64_size_bytes=40000):
    # Calculate the target size for the compressed image to account for base64 encoding overhead
    target_size_bytes = target_base64_size_bytes * 3 // 4
    
    document = fitz.open(pdf_path)
    image_list = []
    
    # Retrieve images
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        image_list.extend(page.get_images(full=True))

    def compress_image_to_target_size(image, target_size_bytes):
        buffer = BytesIO()
        quality = 85  # Starting quality
        image.save(buffer, format="JPEG", quality=quality)
        
        while buffer.tell() > target_size_bytes and quality > 10:
            buffer = BytesIO()
            quality -= 5
            # Resize image if quality reduction is not enough
            if buffer.tell() > target_size_bytes:  # Resize if way over target
                width, height = image.size
                refactor_size = buffer.tell() / target_size_bytes
                image = image.resize((width // math.sqrt(refactor_size), height // math.sqrt(refactor_size)), Image.ANTIALIAS)
            image.save(buffer, format="JPEG", quality=quality)
            
        return buffer.getvalue()

    # Convert images to base64
    base64_images = []
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = document.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(BytesIO(image_bytes))
        
        # Compress the image
        compressed_image_bytes = compress_image_to_target_size(image, target_size_bytes)
        
        # Encode the compressed image to base64
        img_str = base64.b64encode(compressed_image_bytes).decode("utf-8")
        if len(img_str) <= target_base64_size_bytes:
            base64_images.append(img_str)
        else:
            print(f"Warning: Image {img_index} exceeds the target base64 size after compression.")

    return base64_images


def parse_pdf(pdf_path):
    # Extract text
    text = extract_text(pdf_path)
    
    # Extract images
    images = extract_and_compress_images(pdf_path, target_base64_size_bytes=37000)

    return text, images




